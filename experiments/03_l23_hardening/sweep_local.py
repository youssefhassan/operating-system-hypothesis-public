"""
Experiment 03 (local) — multi-prompt CFG sweep for the Klüver L2/3 hardening study.

Adapted from Exp 01's sweep_local.py. Same diffusers generation machinery
(device/dtype gating, SD 3.5 embed caching, MPS-stable VAE decode), generalized
to the **6-prompt set** pre-registered in preregistration.json. The prompts, the
guidance grid, and the seeds are read from the pre-registration so this script
cannot silently drift from the committed design (the prompts are a
`forbidden_to_automate` field).

Filenames encode the prompt: `{prompt_id}_g{g}_s{s}.png` for conditioned images
and `uncond_s{s}.png` for the (prompt-independent) empty-prompt baseline.
Guidance is blinded at judging and un-blinded only at analysis.

Host: designed for a single Apple M5 Pro 64GB (no sharding needed — 64GB unified
memory runs SD 3.5 on-GPU without cpu-offload). The Exp 01 --num-shards path is
retained for portability but unnecessary here. `--skip-existing` makes any run
resumable.

Usage:
    python sweep_local.py --model sdxl --prompts all --unconditional --skip-existing
    python sweep_local.py --model sd35 --prompts all --unconditional --skip-existing
    python sweep_local.py --model sdxl --prompts p2_portrait p3_bicycle   # subset
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(_PROJECT_ROOT / ".env")
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
except ModuleNotFoundError:
    pass

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
PREREG = HERE / "preregistration.json"

MODEL_REGISTRY: dict[str, dict] = {
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "arch": "UNet (latent)",
        "cfg_type": "true CFG",
        "gated": False,
    },
    "sd35": {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "arch": "MMDiT",
        "cfg_type": "true CFG",
        "gated": True,
    },
}

HIGH_UNIFIED_RAM_GB = 48


def _load_prereg() -> dict:
    return json.loads(PREREG.read_text())


def _prompt_map(prereg: dict) -> dict[str, str]:
    """prompt_id -> prompt text, from the committed pre-registration."""
    return {p["id"]: p["text"] for p in prereg["prompts"]}


# --------------------------- device / pipeline ---------------------------------
# (device/dtype gating, pipeline load, VAE decode, and SD 3.5 handling are ported
#  verbatim from Exp 01's sweep_local.py — they are load-bearing MPS-stability
#  fixes; see that file's comments for the full rationale.)


def _system_ram_gb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and size > 0:
            return pages * size / (1024**3)
    except (ValueError, OSError, AttributeError):
        pass
    return None


def _mps_has_headroom(device: str) -> bool:
    if device != "mps":
        return False
    ram = _system_ram_gb()
    return ram is not None and ram >= HIGH_UNIFIED_RAM_GB


def _pick_device_and_dtype(device_arg: str, dtype_arg: str, model_key: str):
    import torch

    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_arg

    if dtype_arg == "auto":
        # MPS + SDXL UNet: fp16/bf16 goes all-NaN once CFG>1 (two UNet passes);
        # float32 is stable. SD 3.5 MMDiT: float32 full load is heavy; fp16 fits
        # and is CFG-stable. On a 64GB host fp16 SD 3.5 stays fully on-GPU.
        if device == "cpu":
            dtype = torch.float32
        elif device == "mps" and model_key == "sd35":
            dtype = torch.float16
        elif device == "mps":
            dtype = torch.float32
        else:
            dtype = torch.float16
    else:
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[dtype_arg]
    return device, dtype


def _load_pipeline(model_id, dtype, device, scheduler, low_memory, model_key, variant=None):
    from diffusers import AutoPipelineForText2Image

    # variant="fp16" loads the fp16-precision weight files; torch_dtype casts them
    # (e.g. up to float32 for MPS-stable SDXL). Halves the download vs fp32 files.
    kw = {"torch_dtype": dtype}
    if variant:
        kw["variant"] = variant
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kw)
    if device == "mps" and model_key == "sd35":
        try:
            pipe = pipe.to(device)
            print("[exp03] SD 3.5 on MPS: full fp16 GPU load")
        except RuntimeError as exc:
            print(f"[exp03] SD 3.5 full MPS load failed ({exc}); using model CPU offload")
            pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    if scheduler == "dpm":
        try:
            from diffusers import DPMSolverMultistepScheduler

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True,
            )
        except Exception as e:
            print(f"[exp03] scheduler swap failed ({e}); keeping model default.")

    if getattr(pipe, "vae", None) is not None and hasattr(pipe.vae, "config"):
        if hasattr(pipe.vae.config, "force_upcast"):
            pipe.vae.config.force_upcast = True

    if device != "cpu" and low_memory:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
    return pipe


def _is_sd3_pipeline(pipe) -> bool:
    return pipe.__class__.__name__ == "StableDiffusion3Pipeline"


def _build_sd3_embed(pipe, prompt: str, device: str) -> tuple:
    """Encode one prompt (+ empty negative) once. Cached per prompt across seeds."""
    import torch

    dev = torch.device(device)
    with torch.no_grad():
        return pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            device=dev, do_classifier_free_guidance=True,
        )


def _sd3_exec_device(pipe):
    if getattr(pipe, "_execution_device", None) is not None:
        return pipe._execution_device
    return next(pipe.transformer.parameters()).device


def _move_sd3_embeds(embeds, device):
    import torch

    dev = torch.device(device)
    return tuple(t.to(dev) if t is not None else None for t in embeds)


def _generate(pipe, prompt, guidance, seed, steps, w, h, device, *, sd3_embed=None):
    import numpy as np
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    common = dict(guidance_scale=guidance, num_inference_steps=steps,
                  width=w, height=h, generator=generator)
    if sd3_embed is not None and _is_sd3_pipeline(pipe):
        pe, npe, ppe, nppe = _move_sd3_embeds(sd3_embed, _sd3_exec_device(pipe))
        out = pipe(prompt_embeds=pe, negative_prompt_embeds=npe,
                   pooled_prompt_embeds=ppe, negative_pooled_prompt_embeds=nppe, **common)
    else:
        out = pipe(prompt=prompt, **common)
    img = out.images[0]
    if np.asarray(img).max() == 0:
        raise RuntimeError(
            f"all-black image at guidance={guidance} seed={seed} — VAE/UNet NaNs "
            "(MPS fp16 + CFG>1); retry with --dtype float32.")
    return img


# ---------------------------------- run ----------------------------------------


def run(args: argparse.Namespace) -> None:
    prereg = _load_prereg()
    pmap = _prompt_map(prereg)
    guidance = args.guidance if args.guidance is not None else prereg["guidance_grid"]
    seeds = args.seeds if args.seeds is not None else prereg["seeds"]
    steps = args.steps if args.steps is not None else prereg["fixed"]["steps"]

    if args.prompts == ["all"]:
        prompt_ids = list(pmap)
    else:
        unknown = [p for p in args.prompts if p not in pmap]
        if unknown:
            raise SystemExit(f"unknown prompt id(s): {unknown}. known: {list(pmap)}")
        prompt_ids = args.prompts

    spec = MODEL_REGISTRY[args.model]
    model_id = args.model_path or spec["model_id"]  # local dir override (flaky-download workaround)
    out_dir = Path(args.outdir) if args.outdir else HERE / "results-local" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    device, dtype = _pick_device_and_dtype(args.device, args.dtype, args.model)

    # Build job list, grouped by prompt so the SD 3.5 embed cache is built once
    # per prompt (T5-XXL encoding dominates otherwise).
    jobs: list[dict] = []
    for pid in prompt_ids:
        for g in guidance:
            for s in seeds:
                jobs.append({"kind": "conditioned", "prompt_id": pid, "prompt": pmap[pid],
                             "guidance": g, "seed": s, "filename": f"{pid}_g{g:g}_s{s}.png"})
    if args.unconditional:
        for s in seeds:  # empty prompt is prompt-independent → one baseline set
            jobs.append({"kind": "unconditional", "prompt_id": "uncond", "prompt": "",
                         "guidance": 1.0, "seed": s, "filename": f"uncond_s{s}.png"})

    if args.num_shards > 1:
        jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard]
    if args.skip_existing:
        jobs = [j for j in jobs if not (out_dir / j["filename"]).exists()]

    scheduler = args.scheduler
    if scheduler == "dpm" and not spec["arch"].startswith("UNet"):
        scheduler = "keep"  # SD 3.5 MMDiT uses its native flow-matching sampler
    elif scheduler == "dpm" and device != "cuda":
        scheduler = "keep"  # DPM++ Karras is NaN-unstable on MPS/CPU in low precision

    print(f"[exp03] model={args.model} ({model_id}) arch={spec['arch']}")
    print(f"[exp03] device={device} dtype={dtype} steps={steps} scheduler={scheduler}")
    print(f"[exp03] prompts={prompt_ids}")
    print(f"[exp03] guidance={guidance} seeds={seeds}")
    ram = _system_ram_gb()
    high_mem = _mps_has_headroom(device)
    if ram is not None:
        print(f"[exp03] unified RAM {ram:.0f} GB" + (" — SD 3.5 fast path" if high_mem else ""))
    if spec["gated"]:
        print(f"[exp03] gated model — HF_TOKEN {'set' if os.environ.get('HF_TOKEN') else 'MISSING'}")
    if not jobs:
        print("[exp03] nothing to do (all jobs filtered out).")
        return
    print(f"[exp03] {len(jobs)} images to generate -> {out_dir}")

    low_mem = (args.low_memory or (device == "mps" and not high_mem)) and not args.no_low_memory
    pipe = _load_pipeline(model_id, dtype, device, scheduler, low_mem, args.model, args.variant)

    is_sd3 = args.model == "sd35" and _is_sd3_pipeline(pipe)
    embed_cache: dict[str, tuple] = {}
    if is_sd3:
        print("[exp03] caching SD 3.5 prompt embeddings per prompt (+ empty baseline)")
        for pid in prompt_ids:
            embed_cache[pid] = _build_sd3_embed(pipe, pmap[pid], device)
        if args.unconditional:
            embed_cache["uncond"] = _build_sd3_embed(pipe, "", device)

    import torch

    runs: list[dict] = []
    total = len(jobs)
    for done, job in enumerate(jobs, start=1):
        tag = "UNCOND " if job["kind"] == "unconditional" else ""
        print(f"[exp03] ({done}/{total}) {tag}{job['prompt_id']} g={job['guidance']} "
              f"s={job['seed']} -> {job['filename']}")
        sd3_embed = embed_cache.get(job["prompt_id"]) if is_sd3 else None
        image = _generate(pipe, job["prompt"], job["guidance"], job["seed"],
                          steps, args.width, args.height, device, sd3_embed=sd3_embed)
        image.save(out_dir / job["filename"])
        if device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        runs.append({k: job[k] for k in ("kind", "prompt_id", "guidance", "seed", "filename")})

    metadata = {
        "experiment": "03_l23_hardening (local)",
        "model_key": args.model, "model_id": model_id, "arch": spec["arch"],
        "cfg_type": spec["cfg_type"], "device": device, "dtype": str(dtype),
        "scheduler": scheduler, "rubric_version": prereg["rubric_version"],
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "fixed": {"steps": steps, "width": args.width, "height": args.height},
        "sweep": {"prompt_ids": prompt_ids, "guidance": guidance, "seeds": seeds,
                  "unconditional": args.unconditional},
        "prompts": {pid: pmap[pid] for pid in prompt_ids},
        "shard": {"index": args.shard, "num_shards": args.num_shards},
        "runs": runs,
    }
    # Merge with any existing metadata (resumed runs) so the record is complete.
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text())
            seen = {(r["filename"]) for r in runs}
            runs = runs + [r for r in prev.get("runs", []) if r["filename"] not in seen]
            metadata["runs"] = sorted(runs, key=lambda r: r["filename"])
        except Exception:
            pass
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"[exp03] done. {len(metadata['runs'])} images recorded -> {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experiment 03 multi-prompt CFG sweep.")
    p.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="sdxl")
    p.add_argument("--prompts", nargs="+", default=["all"],
                   help="prompt ids from preregistration.json, or 'all'")
    p.add_argument("--guidance", type=float, nargs="+", default=None,
                   help="override the pre-registered guidance grid")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="override the pre-registered seed set")
    p.add_argument("--steps", type=int, default=None, help="default: prereg fixed.steps")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--unconditional", action="store_true",
                   help="also generate empty-prompt baselines (base-layer anchor)")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    p.add_argument("--scheduler", choices=["dpm", "keep"], default="dpm")
    p.add_argument("--low-memory", action="store_true")
    p.add_argument("--no-low-memory", action="store_true")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true", help="safe resume")
    p.add_argument("--outdir", default=None)
    p.add_argument("--model-path", default=None,
                   help="local directory to load weights from (overrides the hub id; "
                        "use with a pre-downloaded `hf download --local-dir` snapshot)")
    p.add_argument("--variant", default=None,
                   help="weight variant to load, e.g. 'fp16' (loaded then cast by --dtype)")
    return p


if __name__ == "__main__":
    _args = build_parser().parse_args()
    if _args.num_shards < 1 or not (0 <= _args.shard < _args.num_shards):
        raise SystemExit(f"invalid sharding: shard={_args.shard} num_shards={_args.num_shards}")
    run(_args)
