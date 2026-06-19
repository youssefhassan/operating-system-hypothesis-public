"""
Experiment 01 (local) — Multi-model classifier-free-guidance sweep via diffusers.

This is the plan-aligned version of Exp 01 (see the project plan Phase I and §8):
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
gated on the Hugging Face Hub (SD 3.5, FLUX); run `huggingface-cli login` and
accept the model license first.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

# Architecture / CFG-type spans the matrix that distinguishes
# architecture-general from model-specific. AutoPipeline picks the class.
MODEL_REGISTRY: dict[str, dict] = {
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "default_steps": 40,
        "arch": "UNet (latent)",
        "cfg_type": "true CFG",
        "gated": False,
    },
    "sd15": {
        "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "default_steps": 50,
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


def _pick_device_and_dtype(device_arg: str, dtype_arg: str):
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
        # fp16 on MPS frequently produces all-black SDXL images (overflow ->
        # NaN); bf16 avoids it. CUDA is fine on fp16; CPU needs fp32.
        if device == "cpu":
            dtype = torch.float32
        elif device == "mps":
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
            dtype_arg
        ]
    return device, dtype


def _load_pipeline(model_id: str, dtype, device):
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    # Save memory on 24GB unified-memory Macs / smaller GPUs.
    if device != "cpu":
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
    return pipe


def _generate(pipe, prompt: str, guidance: float, seed: int, steps: int, w: int, h: int, device):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    out = pipe(
        prompt=prompt,
        guidance_scale=guidance,
        num_inference_steps=steps,
        width=w,
        height=h,
        generator=generator,
    )
    return out.images[0]


def run(args: argparse.Namespace) -> None:
    spec = MODEL_REGISTRY[args.model]
    steps = args.steps if args.steps is not None else spec["default_steps"]
    out_dir = (
        Path(args.outdir)
        if args.outdir
        else REPO_ROOT / "experiments" / "01_image_test" / "results-local" / args.model
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device, dtype = _pick_device_and_dtype(args.device, args.dtype)
    print(f"[exp01] model={args.model} ({spec['model_id']})")
    print(f"[exp01] arch={spec['arch']} cfg={spec['cfg_type']} gated={spec['gated']}")
    print(f"[exp01] device={device} dtype={dtype} steps={steps}")
    if spec["gated"]:
        print("[exp01] note: gated model — run `huggingface-cli login` and accept the license.")

    pipe = _load_pipeline(spec["model_id"], dtype, device)

    runs: list[dict] = []
    pairs = [(g, s) for g in args.guidance for s in args.seeds]
    total = len(pairs) + (len(args.seeds) if args.unconditional else 0)
    done = 0

    for guidance, seed in pairs:
        done += 1
        filename = f"g{guidance:.1f}_s{seed}.png"
        print(f"[exp01] ({done}/{total}) guidance={guidance} seed={seed} -> {filename}")
        image = _generate(pipe, args.prompt, guidance, seed, steps, args.width, args.height, device)
        image.save(out_dir / filename)
        runs.append({"kind": "conditioned", "guidance": guidance, "seed": seed, "filename": filename})

    if args.unconditional:
        for seed in args.seeds:
            done += 1
            filename = f"uncond_s{seed}.png"
            print(f"[exp01] ({done}/{total}) UNCONDITIONAL (empty prompt) seed={seed} -> {filename}")
            image = _generate(pipe, "", 1.0, seed, steps, args.width, args.height, device)
            image.save(out_dir / filename)
            runs.append({"kind": "unconditional", "guidance": 1.0, "seed": seed, "filename": filename})

    metadata = {
        "experiment": "01_image_test (local)",
        "model_key": args.model,
        "model_id": spec["model_id"],
        "arch": spec["arch"],
        "cfg_type": spec["cfg_type"],
        "device": device,
        "dtype": str(dtype),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "prompt": args.prompt,
        "fixed": {"steps": steps, "width": args.width, "height": args.height},
        "sweep": {"guidance": args.guidance, "seeds": args.seeds, "unconditional": args.unconditional},
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
    p.add_argument("--outdir", default=None)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
