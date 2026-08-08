"""
Experiment 01 (local) — Multi-model classifier-free-guidance sweep via diffusers.

This is the plan-aligned version of Exp 01:
the guidance gradient runs on open-weights latent-diffusion models, not on the
BFL FLUX.2-flex API (whose guidance floor of 1.5 can't reach the low regime).

Why local + open weights:
- The independent variable is the CFG / guidance dial. Only models that expose
  it can be swept. Closed products (Midjourney, Gemini/"Nano Banana") don't, so
  they can't be part of the controlled comparison.
- Running across architectures (UNet vs DiT, true-CFG vs guidance-distilled)
  separates "architecture-general property" from "single-model quirk."

The base-layer subtlety:
- CFG is eps = eps_uncond + g * (eps_cond - eps_uncond).
  g=1 -> pure conditional (the prompt, no amplification);
  g=0 -> pure unconditional (the model's own prior — the "base layer");
  g>1 -> amplified prompt adherence.
- Standard diffusers pipelines disable the unconditional pass when
  guidance_scale <= 1, so you cannot reach 0<g<1 by lowering the dial alone.
  To anchor the true base layer we therefore *also* generate an unconditional
  (empty-prompt) baseline and ask whether low-guidance conditioned output drifts
  toward it, and whether that unconditional output carries structured geometry
  (form constants) rather than featureless noise. (A custom sub-1 guidance hook
  is a noted follow-up; see README.)

Usage:
    python sweep_local.py --model sdxl
    python sweep_local.py --model sd35 --steps 40 --unconditional
    python sweep_local.py --model sdxl --guidance 1.0 2.0 4.5 7.0 11.0 15.0 \
        --seeds 42 43 44 --prompt "a coral reef teeming with fish"

Requires the diffusers stack (see ../../requirements.txt). Some models are
gated on the Hugging Face Hub (SD 3.5, FLUX): set HF_TOKEN in the project-root
.env (read automatically below) and accept the model license once on the Hub.
`huggingface-cli login` also works if you prefer not to use .env.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Load HF_TOKEN (and any other keys) from the project-root .env so gated
# models download without a separate `huggingface-cli login`. huggingface_hub
# reads HF_TOKEN from the environment automatically once it's set here.
try:
    from dotenv import load_dotenv

    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(_PROJECT_ROOT / ".env")
    # Older tooling expects HUGGING_FACE_HUB_TOKEN; mirror HF_TOKEN to it.
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
except ModuleNotFoundError:
    pass

# Architecture / CFG-type spans the matrix that distinguishes
# architecture-general from model-specific. AutoPipeline picks the class.
MODEL_REGISTRY: dict[str, dict] = {
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "default_steps": 25,  # default sampler on MPS; DPM++ 2M Karras on CUDA
        "arch": "UNet (latent)",
        "cfg_type": "true CFG",
        "gated": False,
    },
    "sd15": {
        "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "default_steps": 30,  # DPM++ 2M Karras
        "arch": "UNet (latent, older)",
        "cfg_type": "true CFG",
        "gated": False,
    },
    "sd35": {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "default_steps": 40,
        "arch": "MMDiT",
        "cfg_type": "true CFG",
        "gated": True,
    },
    "flux": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "default_steps": 28,
        "arch": "DiT",
        "cfg_type": "guidance-distilled (confound)",
        "gated": True,
    },
}

# Mixed materials/colors so any decomposition into lower-layer primitives is
# visible rather than hidden inside one object (matches run.py's rationale).
DEFAULT_PROMPT = (
    "a watermelon, a glass half-filled with water, and a set of keys on a wooden table"
)
DEFAULT_GUIDANCE = [1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0, 11.0, 15.0]
DEFAULT_SEEDS = [42, 43, 44]

REPO_ROOT = Path(__file__).resolve().parents[2]

# Apple unified-memory headroom: below this, SD 3.5 defaults to attention
# slicing + VAE tiling + parking text encoders after embed cache. At 48 GB+
# (e.g. M5 64 GB laptop) keep the full pipeline on GPU for speed.
HIGH_UNIFIED_RAM_GB = 48


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


def _pick_device_and_dtype(device_arg: str, dtype_arg: str, model_key: str | None = None):
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
        # MPS + SDXL UNet: bf16/fp16 is stable at guidance<=1 (no CFG pass) but
        # produces all-NaN / black images once guidance>1 enables classifier-
        # free guidance (two UNet passes + scaled combination). float32 on MPS
        # is slower but stable across the full SDXL guidance grid.
        # MPS + SD 3.5 MMDiT: float32 full load OOMs (~30 GiB); fp16 fits on
        # 24 GB unified memory and is stable with CFG (different arch than UNet).
        if device == "cpu":
            dtype = torch.float32
        elif device == "mps" and model_key == "sd35":
            dtype = torch.float16
        elif device == "mps":
            dtype = torch.float32
        else:
            dtype = torch.float16
    else:
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
            dtype_arg
        ]
    return device, dtype


def _load_pipeline(
    model_id: str,
    dtype,
    device,
    scheduler: str = "dpm",
    low_memory: bool = False,
    model_key: str | None = None,
):
    import torch
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)
    # SD 3.5 on MPS: prefer fp16 full GPU (fast). float32 OOMs at ~30 GiB;
    # enable_model_cpu_offload() on unified memory shuffles every component
    # through the bus each step (~1 hr/image). Fall back to offload only if
    # full load fails. SDXL UNet stays fully on-device in float32.
    if device == "mps" and model_key == "sd35":
        try:
            pipe = pipe.to(device)
            print("[exp01] SD 3.5 on MPS: full fp16 GPU load")
        except RuntimeError as exc:
            print(f"[exp01] SD 3.5 full MPS load failed ({exc}); using model CPU offload")
            pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    # Faster sampler: DPM++ 2M with Karras sigmas reaches comparable quality in
    # far fewer steps than the default scheduler. "keep" leaves the model default.
    if scheduler == "dpm":
        try:
            from diffusers import DPMSolverMultistepScheduler

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )
        except Exception as e:
            print(f"[exp01] scheduler swap failed ({e}); keeping model default.")

    # SDXL VAE decode can NaN in bf16/fp16 on MPS (partial corruption or cast
    # warnings). Do NOT move VAE weights to float32 — that breaks MPS decode
    # entirely (all-NaN frames). Instead set force_upcast so the pipeline
    # upcasts latents to fp32 only during the decode step (diffusers-native).
    if getattr(pipe, "vae", None) is not None and hasattr(pipe.vae, "config"):
        if hasattr(pipe.vae.config, "force_upcast"):
            pipe.vae.config.force_upcast = True

    # On MPS, attention slicing + VAE tiling are ON by default (stable decode,
    # lower peak memory). --low-memory forces them on other devices too;
    # --no-low-memory disables them even on MPS.
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


def _build_sd3_embed_cache(pipe, prompt: str, device: str) -> dict:
    """Encode fixed prompts once; T5-XXL dominates per-image time if re-run."""
    import torch

    dev = torch.device(device)
    with torch.no_grad():
        cond = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt="",
            negative_prompt_2="",
            negative_prompt_3="",
            device=dev,
            do_classifier_free_guidance=True,
        )
        empty = pipe.encode_prompt(
            prompt="",
            prompt_2="",
            prompt_3="",
            negative_prompt="",
            negative_prompt_2="",
            negative_prompt_3="",
            device=dev,
            do_classifier_free_guidance=True,
        )
    return {"cond": cond, "empty": empty}


def _sd3_exec_device(pipe):
    import torch

    if getattr(pipe, "_execution_device", None) is not None:
        return pipe._execution_device
    return next(pipe.transformer.parameters()).device


def _move_sd3_embeds(embeds, device):
    import torch

    dev = torch.device(device)

    def _one(t):
        return t.to(dev) if t is not None else None

    return tuple(_one(t) for t in embeds)


def _drop_sd3_text_encoders(pipe) -> None:
    """Drop encoders after caching embeds — frees RAM without breaking MPS device routing."""
    import gc

    import torch

    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        enc = getattr(pipe, name, None)
        if enc is not None:
            del enc
            setattr(pipe, name, None)
    gc.collect()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _generate(
    pipe,
    prompt: str,
    guidance: float,
    seed: int,
    steps: int,
    w: int,
    h: int,
    device,
    *,
    kind: str = "conditioned",
    embed_cache: dict | None = None,
):
    import numpy as np
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    common = dict(
        guidance_scale=guidance,
        num_inference_steps=steps,
        width=w,
        height=h,
        generator=generator,
    )
    if embed_cache and _is_sd3_pipeline(pipe):
        exec_dev = _sd3_exec_device(pipe)
        pe, npe, ppe, nppe = _move_sd3_embeds(
            embed_cache["empty" if kind == "unconditional" else "cond"],
            exec_dev,
        )
        out = pipe(
            prompt_embeds=pe,
            negative_prompt_embeds=npe,
            pooled_prompt_embeds=ppe,
            negative_pooled_prompt_embeds=nppe,
            **common,
        )
    else:
        out = pipe(prompt=prompt, **common)
    img = out.images[0]
    arr = np.asarray(img)
    if arr.max() == 0:
        raise RuntimeError(
            f"all-black image at guidance={guidance} seed={seed} — "
            "VAE/UNet produced NaNs (common on MPS in bf16/fp16 when CFG>1; "
            "retry with --dtype float32 or use CUDA)"
        )
    return img


def run(args: argparse.Namespace) -> None:
    spec = MODEL_REGISTRY[args.model]
    steps = args.steps if args.steps is not None else spec["default_steps"]
    out_dir = (
        Path(args.outdir)
        if args.outdir
        else REPO_ROOT / "experiments" / "01_image_test" / "results-local" / args.model
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device, dtype = _pick_device_and_dtype(args.device, args.dtype, args.model)

    # Build the full job list first, then keep only this machine's shard. Shards
    # partition deterministically (job index % num_shards == shard), so three
    # Mac minis can each run a third with --num-shards 3 --shard 0|1|2, all
    # writing gX_sY.png into a shared/merged results dir.
    jobs: list[dict] = []
    for g in args.guidance:
        for s in args.seeds:
            jobs.append({"kind": "conditioned", "guidance": g, "seed": s,
                         "prompt": args.prompt, "filename": f"g{g:.1f}_s{s}.png"})
    if args.unconditional:
        for s in args.seeds:
            jobs.append({"kind": "unconditional", "guidance": 1.0, "seed": s,
                         "prompt": "", "filename": f"uncond_s{s}.png"})

    if args.num_shards > 1:
        jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard]

    if args.skip_existing:
        jobs = [j for j in jobs if not (out_dir / j["filename"]).exists()]

    # Scheduler gating. DPM++ 2M Karras is fast but only safe in some setups:
    #  - non-UNet models (SD 3.5 MMDiT, FLUX DiT) use flow-matching samplers, so
    #    swapping would silently degrade them.
    #  - on MPS/CPU in low precision (bf16/fp16) the Karras sigma schedule
    #    overflows and produces all-NaN images; the model's default sampler is
    #    stable there. DPM++ is kept only on CUDA, where it is reliable.
    scheduler = args.scheduler
    fallback_reason = None
    if scheduler == "dpm" and not spec["arch"].startswith("UNet"):
        scheduler, fallback_reason = "keep", f"{spec['arch']} uses its native sampler"
    elif scheduler == "dpm" and device != "cuda":
        scheduler, fallback_reason = "keep", f"DPM++ is NaN-unstable on {device} in {dtype}"

    print(f"[exp01] model={args.model} ({spec['model_id']})")
    print(f"[exp01] arch={spec['arch']} cfg={spec['cfg_type']} gated={spec['gated']}")
    print(f"[exp01] device={device} dtype={dtype} steps={steps} scheduler={scheduler}")
    high_mem_mac = _mps_has_headroom(device)
    ram_gb = _system_ram_gb()
    if high_mem_mac and ram_gb is not None:
        print(f"[exp01] unified RAM {ram_gb:.0f} GB — SD 3.5 fast path (no slicing/park)")
    if fallback_reason:
        print(f"[exp01] (requested scheduler '{args.scheduler}' -> 'keep': {fallback_reason})")
    if args.num_shards > 1:
        print(f"[exp01] shard {args.shard}/{args.num_shards}: {len(jobs)} jobs on this machine")
    if spec["gated"]:
        tok = "set" if os.environ.get("HF_TOKEN") else "MISSING"
        print(f"[exp01] note: gated model — HF_TOKEN {tok}; accept the model license on the Hub.")

    if not jobs:
        print("[exp01] nothing to do (all jobs filtered out).")
        return

    low_mem = (args.low_memory or (device == "mps" and not high_mem_mac)) and not args.no_low_memory
    pipe = _load_pipeline(
        spec["model_id"], dtype, device, scheduler,
        low_mem,
        model_key=args.model,
    )

    embed_cache = None
    if args.model == "sd35" and _is_sd3_pipeline(pipe):
        print("[exp01] caching SD 3.5 prompt embeddings (fixed prompt + empty baseline)")
        embed_cache = _build_sd3_embed_cache(pipe, args.prompt, device)
        if not getattr(pipe, "_hf_hook", None) and not high_mem_mac:
            _drop_sd3_text_encoders(pipe)
            print("[exp01] text encoders dropped after embed cache; DiT+VAE stay on GPU")

    runs: list[dict] = []
    total = len(jobs)
    for done, job in enumerate(jobs, start=1):
        label = "UNCONDITIONAL (empty prompt) " if job["kind"] == "unconditional" else ""
        print(f"[exp01] ({done}/{total}) {label}guidance={job['guidance']} seed={job['seed']} -> {job['filename']}")
        image = _generate(
            pipe, job["prompt"], job["guidance"], job["seed"],
            steps, args.width, args.height, device,
            kind=job["kind"], embed_cache=embed_cache,
        )
        image.save(out_dir / job["filename"])
        import torch

        if device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        runs.append({"kind": job["kind"], "guidance": job["guidance"],
                     "seed": job["seed"], "filename": job["filename"]})

    metadata = {
        "experiment": "01_image_test (local)",
        "model_key": args.model,
        "model_id": spec["model_id"],
        "arch": spec["arch"],
        "cfg_type": spec["cfg_type"],
        "device": device,
        "dtype": str(dtype),
        "scheduler": scheduler,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "prompt": args.prompt,
        "fixed": {"steps": steps, "width": args.width, "height": args.height},
        "sweep": {"guidance": args.guidance, "seeds": args.seeds, "unconditional": args.unconditional},
        "shard": {"index": args.shard, "num_shards": args.num_shards},
        "runs": runs,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[exp01] done. {len(runs)} images + metadata -> {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experiment 01 local CFG sweep (diffusers).")
    p.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="sdxl")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--guidance", type=float, nargs="+", default=DEFAULT_GUIDANCE)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--steps", type=int, default=None, help="default: model-specific")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument(
        "--unconditional",
        action="store_true",
        help="also generate empty-prompt baselines (the true base layer anchor)",
    )
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument(
        "--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto"
    )
    p.add_argument(
        "--scheduler", choices=["dpm", "keep"], default="dpm",
        help="dpm = DPM++ 2M Karras (fast, fewer steps look like more); keep = model default",
    )
    p.add_argument(
        "--low-memory", action="store_true",
        help="enable attention slicing + VAE tiling (also the MPS default)",
    )
    p.add_argument(
        "--no-low-memory", action="store_true",
        help="disable attention slicing + VAE tiling even on MPS (faster, less stable)",
    )
    p.add_argument(
        "--num-shards", type=int, default=1,
        help="split the job list across N machines (run one shard per Mac mini)",
    )
    p.add_argument(
        "--shard", type=int, default=0, help="this machine's shard index in [0, num-shards)",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="skip jobs whose output PNG already exists (safe resume)",
    )
    p.add_argument("--outdir", default=None)
    return p


if __name__ == "__main__":
    _args = build_parser().parse_args()
    if _args.num_shards < 1 or not (0 <= _args.shard < _args.num_shards):
        raise SystemExit(f"invalid sharding: shard={_args.shard} num_shards={_args.num_shards} "
                         f"(need num_shards>=1 and 0<=shard<num_shards)")
    run(_args)
