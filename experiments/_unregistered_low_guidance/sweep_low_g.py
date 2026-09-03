"""
Unregistered scratch — sub-CFG guidance sweep (g = 0 .. 2, step 0.25).

Thin wrapper over Exp 03's `sweep_local.py`: same prompts, same pipeline loading,
same MPS dtype gating, same resumable job list. The only change is that
classifier-free guidance is forced ON for every guidance value.

Why the patch is necessary
--------------------------
diffusers gates CFG at `guidance_scale > 1` on both pipelines, so without it
g = 0, 0.25, 0.5, 0.75 and 1.0 would all take the CFG-off branch and return the
same conditional-only image. Forcing it on lets the ordinary combination

    pred = pred_uncond + g * (pred_cond - pred_uncond)

run at every g, which is well defined below 1 and yields the pure unconditional
prediction at g = 0. See README.md.

The patch is applied to the pipeline *classes*, before any pipeline is built, so
it covers whatever `AutoPipelineForText2Image` hands back. It costs a second
forward pass at every g (both branches are always evaluated), which is the price
of having a real low-guidance axis at all.

Usage:
    python sweep_low_g.py --model sdxl --unconditional --skip-existing
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EXP03 = HERE.parent / "03_l23_hardening"
sys.path.insert(0, str(EXP03))

import sweep_local as S  # noqa: E402  (path must be set first)

GUIDANCE = [round(0.25 * i, 2) for i in range(9)]  # 0.00 .. 2.00
SEEDS = [42, 43]


def force_cfg_always_on() -> list[str]:
    """Override `do_classifier_free_guidance` to True on every text-to-image
    pipeline class we might load. Returns the names patched, for the log."""
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
        StableDiffusion3Pipeline,
    )
    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
        StableDiffusionXLPipeline,
    )

    patched = []
    for cls in (StableDiffusionXLPipeline, StableDiffusion3Pipeline):
        prop = cls.__dict__.get("do_classifier_free_guidance")
        if not isinstance(prop, property):
            raise SystemExit(
                f"{cls.__name__}.do_classifier_free_guidance is not a property in this "
                "diffusers version — the CFG patch would silently do nothing. Inspect "
                "the pipeline source before running."
            )
        cls.do_classifier_free_guidance = property(lambda self: True)
        patched.append(cls.__name__)
    return patched


def _sanitize_metadata(out_dir: Path) -> None:
    """Rewrite absolute paths in metadata.json to repo-relative ones.

    sweep_local.py records `model_id` as whatever was passed to --model-path, and
    this wrapper passes an absolute path to Exp 03's local SDXL snapshot. That
    absolute path contains the developer's home directory, which is exactly what
    scripts/sync_public.sh's content audit blocks from crossing to the public
    export -- and it caught this on the first sync attempt. Fix it at the source
    so the file is clean the moment it is written, not by hand afterwards.
    """
    import json

    meta = out_dir / "metadata.json"
    if not meta.exists():
        return
    root = str(REPO_ROOT) + "/"
    text = meta.read_text()
    if root not in text:
        return
    meta.write_text(text.replace(root, ""))
    print(f"[low-g] metadata.json: rewrote absolute paths as repo-relative")


def build_parser():
    p = S.build_parser()
    p.set_defaults(
        guidance=GUIDANCE,
        seeds=SEEDS,
        unconditional=True,
        skip_existing=True,
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.outdir is None:
        args.outdir = str(HERE / "results-local" / args.model)
    # Reuse Exp 03's local fp16 SDXL snapshot: hub downloads of SDXL were flaky on
    # this host (see ../03_l23_hardening/README.md section 4), and reusing the exact
    # weights Exp 03 ran on keeps the two sweeps comparable.
    if args.model == "sdxl" and args.model_path is None:
        local = EXP03 / "sdxl_local"
        if (local / "model_index.json").exists():
            args.model_path = str(local)
            if args.variant is None:
                args.variant = "fp16"
            print(f"[low-g] using Exp 03's local SDXL snapshot: {local}")
    names = force_cfg_always_on()
    print(f"[low-g] CFG forced on for: {', '.join(names)}")
    print(f"[low-g] guidance grid: {args.guidance}")
    S.run(args)
    _sanitize_metadata(Path(args.outdir))
