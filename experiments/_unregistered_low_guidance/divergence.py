"""
How far has the image travelled from the prior, as a function of guidance?

A judge-free, purely mechanical companion to analyze_low_g.py. For each prompt and
seed it measures mean absolute pixel distance from that seed's g=0 image (the pure
negative-branch render, which is identical across prompts — see check_g0.py).

This isolates *how much the prompt is moving the image at all*, separately from
whether the result looks like a Kluver form constant. If the distance curve is flat
below some g and then jumps, the prompt has a threshold rather than a gradual onset,
and any rubric trend across that region is really tracking the threshold.

Usage:
    python divergence.py --model sdxl
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def run(model: str) -> dict:
    from PIL import Image

    d = HERE / "results-local" / model
    fre = re.compile(r"^(?P<pid>p\d+_[a-z0-9]+)_g(?P<g>[0-9.]+)_s(?P<s>\d+)\.png$")
    imgs: dict[tuple[str, float, int], Path] = {}
    for png in d.glob("*.png"):
        if m := fre.match(png.name):
            imgs[(m.group("pid"), float(m.group("g")), int(m.group("s")))] = png
    if not imgs:
        raise SystemExit(f"no images in {d}")

    seeds = sorted({k[2] for k in imgs})
    gs = sorted({k[1] for k in imgs})
    pids = sorted({k[0] for k in imgs})

    # the g=0 render is the prior, identical for every prompt at a given seed
    base = {}
    for s in seeds:
        p = next((imgs[(pid, 0.0, s)] for pid in pids if (pid, 0.0, s) in imgs), None)
        if p is not None:
            base[s] = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)

    per_g: dict[float, list[float]] = defaultdict(list)
    per_prompt: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (pid, g, s), path in imgs.items():
        if s not in base:
            continue
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        dist = float(np.abs(arr - base[s]).mean())
        per_g[g].append(dist)
        per_prompt[pid][g].append(dist)

    print(f"\n=== {model}: mean |pixel| distance from the g=0 prior (0-255) ===")
    print(f"  {'g':>5}  {'mean':>7}  {'min':>7}  {'max':>7}")
    gs = [g for g in gs if per_g[g]]  # a partial/resumed run has empty columns
    for g in gs:
        v = np.array(per_g[g])
        print(f"  {g:>5.2f}  {v.mean():>7.2f}  {v.min():>7.2f}  {v.max():>7.2f}")

    print("\n  per-prompt means:")
    print("    prompt        " + "".join(f"{g:>7.2f}" for g in gs))
    for pid in pids:
        row = "".join(f"{np.mean(per_prompt[pid][g]):>7.2f}" if per_prompt[pid][g]
                      else f"{'-':>7}" for g in gs)
        print(f"    {pid.split('_')[1][:12]:<12}  {row}")

    return {"guidance": gs,
            "mean_distance": {str(g): float(np.mean(per_g[g])) for g in gs},
            "per_prompt": {p: {str(g): float(np.mean(v[g])) for g in v}
                           for p, v in per_prompt.items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sdxl")
    a = ap.parse_args()
    run(a.model)
