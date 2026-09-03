"""
Contact sheet: one row per prompt, one column per guidance value.

The point of this run is a *shape* — whether the Klüver phenomena rise as the image
falls toward the prior as well as when it is over-guided. A per-guidance mean table
can hide a U with a near-zero pooled correlation, so look at the grid too.

Usage:
    python contact_sheet.py --model sdxl --seed 42
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "03_l23_hardening"))

CELL = 224
LABEL_W, LABEL_H = 120, 22


def build(model: str, seed: int, out: Path) -> Path:
    from PIL import Image, ImageDraw

    d = HERE / "results-local" / model
    fre = re.compile(rf"^(?P<pid>p\d+_[a-z0-9]+)_g(?P<g>[0-9.]+)_s{seed}\.png$")
    found: dict[tuple[str, float], Path] = {}
    for png in d.glob("*.png"):
        if m := fre.match(png.name):
            found[(m.group("pid"), float(m.group("g")))] = png
    if not found:
        raise SystemExit(f"no images for {model} seed {seed} in {d}")

    pids = sorted({k[0] for k in found})
    gs = sorted({k[1] for k in found})
    W = LABEL_W + CELL * len(gs)
    H = LABEL_H + CELL * len(pids)
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)

    for j, g in enumerate(gs):
        draw.text((LABEL_W + j * CELL + 4, 6), f"g={g:g}", fill="black")
    for i, pid in enumerate(pids):
        draw.text((4, LABEL_H + i * CELL + CELL // 2), pid.split("_")[1][:14], fill="black")
        for j, g in enumerate(gs):
            p = found.get((pid, g))
            if p is None:
                continue
            im = Image.open(p).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
            sheet.paste(im, (LABEL_W + j * CELL, LABEL_H + i * CELL))

    sheet.save(out)
    print(f"[sheet] {model} seed {seed}: {len(pids)} prompts x {len(gs)} g -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sdxl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else (
        HERE / "results-local" / a.model / "figures" / f"sheet_{a.model}_s{a.seed}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    build(a.model, a.seed, out)
