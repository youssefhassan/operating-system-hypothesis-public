"""
Experiment 01b (EXPLORATORY) — objective, judge-free texture metrics.

Motivation. Exp 01 tested Kluver *Level-1* form constants (lattice, cobweb,
tunnel, spiral) via a VLM and got a clean null on SDXL and SD 3.5. But Kluver
names three levels of hallucinatory constant, and the qualitative thing we saw
at low guidance — watermelon flesh tiling, repeated/fragmented objects — is
*Level-2 reduplication*, not the four Level-1 shapes. The post's own table also
predicts a convolution-dominant UNet should fall back toward "repeating texture
/ periodic pattern" at low conditioning. Both claims are about PERIODICITY, and
periodicity is measurable from pixels alone — no judge, no new generation.

This module re-measures the EXISTING Exp 01 images with two objective metrics:

  periodicity   secondary-peak height of the normalized 2-D autocorrelation,
                with the zero-lag core masked out. ~0 for a unique scene, ~1 for
                a perfectly tiled/repeating image. Directly operationalizes
                "the image repeats itself at some spatial offset" (reduplication
                / tiling).

  spectral_order  1 - normalized entropy of the radial power spectrum. Higher
                when energy concentrates at a few spatial frequencies (ordered /
                periodic texture); lower for broadband 1/f natural-image content.

Both are computed on grayscale, mean-subtracted, Hann-windowed images, so flat
gradients and overall composition do not dominate.

This is EXPLORATORY and post-hoc: it does NOT touch the pre-registered
confirmatory metric M (geometric_intensity/3). It reuses the same images and the
same Spearman / permutation-p machinery from loop.py, and reports separately.

Usage:
    python texture_metrics.py --model sdxl
    python texture_metrics.py --model sdxl --plot
    python texture_metrics.py --model sd35 --size 512
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from loop import RESULTS_ROOT, spearman, spearman_perm_p

FNAME_RE = re.compile(r"^g(?P<g>[0-9.]+)_s(?P<s>\d+)\.png$")
UNCOND_RE = re.compile(r"^uncond_s(?P<s>\d+)\.png$")

FIELDS = ("periodicity", "spectral_order")


# ------------------------------- metrics ---------------------------------------

def _gray(path: Path, size: int) -> np.ndarray:
    from PIL import Image

    im = Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float64)


def _windowed(a: np.ndarray) -> np.ndarray:
    """Mean-subtract and apply a separable Hann window (kills edge wrap + DC tilt)."""
    a = a - a.mean()
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    return a * w


def periodicity(a: np.ndarray, core_frac: float = 0.06) -> float:
    """Height in [0,1] of the strongest autocorrelation peak outside the zero-lag core.

    A unique natural scene has a single sharp central peak and little else (~0.0-0.3).
    A tiled / periodic image repeats itself, producing strong off-center peaks (~0.6-1.0).
    `core_frac` excludes lags smaller than core_frac * size, so mere smoothness
    (high autocorrelation at tiny lags) does not count as periodicity.
    """
    aw = _windowed(a)
    power = np.abs(np.fft.fft2(aw)) ** 2
    ac = np.fft.ifft2(power).real
    ac = np.fft.fftshift(ac)
    peak = ac.max()
    if peak <= 0:
        return 0.0
    ac = ac / peak  # zero-lag (center) == 1.0
    h, w = ac.shape
    cy, cx = h // 2, w // 2
    r = max(2, int(round(core_frac * min(h, w))))
    yy, xx = np.ogrid[:h, :w]
    core = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    ac[core] = 0.0
    return float(np.clip(ac.max(), 0.0, 1.0))


def spectral_order(a: np.ndarray, n_bins: int = 128) -> float:
    """1 - normalized entropy of the radial power spectrum, in [0,1].

    Energy spread evenly across spatial frequencies (broadband, 1/f-ish natural
    image) -> high entropy -> low order. Energy piled at a few frequencies
    (periodic texture) -> low entropy -> high order.
    """
    aw = _windowed(a)
    power = np.abs(np.fft.fftshift(np.fft.fft2(aw))) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = rad.max()
    bins = np.clip((rad / rmax * (n_bins - 1)).astype(int), 0, n_bins - 1)
    radial = np.zeros(n_bins)
    np.add.at(radial, bins.ravel(), power.ravel())
    radial = radial[1:]  # drop the DC/lowest bin
    total = radial.sum()
    if total <= 0:
        return 0.0
    p = radial / total
    p = p[p > 0]
    ent = -(p * np.log(p)).sum()
    ent_max = np.log(len(radial))
    return float(np.clip(1.0 - ent / ent_max, 0.0, 1.0)) if ent_max > 0 else 0.0


def measure(path: Path, size: int) -> dict[str, float]:
    a = _gray(path, size)
    return {
        "periodicity": round(periodicity(a), 4),
        "spectral_order": round(spectral_order(a), 4),
    }


# ------------------------------- aggregation -----------------------------------

def analyze_model(model: str, size: int) -> dict:
    d = RESULTS_ROOT / model
    if not d.exists():
        raise SystemExit(f"missing results dir {d}")

    by_field: dict[str, dict[float, list[float]]] = {f: defaultdict(list) for f in FIELDS}
    uncond: dict[str, list[float]] = {f: [] for f in FIELDS}
    per_image: dict[str, dict[str, float]] = {}

    pngs = sorted(p for p in d.glob("*.png"))
    for p in pngs:
        mc = FNAME_RE.match(p.name)
        mu = UNCOND_RE.match(p.name)
        if not (mc or mu):
            continue
        vals = measure(p, size)
        per_image[p.name] = vals
        if mc:
            g = float(mc["g"])
            for f in FIELDS:
                by_field[f][g].append(vals[f])
        else:
            for f in FIELDS:
                uncond[f].append(vals[f])

    out: dict = {
        "model": model,
        "size": size,
        "n_images": len(per_image),
        "fields": {},
        "uncond_mean": {f: round(float(np.mean(uncond[f])), 4) if uncond[f] else None for f in FIELDS},
    }
    for f in FIELDS:
        gs, vs = [], []
        for g, lst in by_field[f].items():
            gs += [g] * len(lst)
            vs += lst
        ag, av = np.array(gs), np.array(vs)
        means = {str(g): round(float(np.mean(by_field[f][g])), 4) for g in sorted(by_field[f])}
        out["fields"][f] = {
            "means_by_guidance": means,
            "spearman_rho_vs_g": round(spearman(ag, av), 4) if len(ag) >= 3 else None,
            "spearman_p": round(spearman_perm_p(ag, av), 4) if len(ag) >= 3 else None,
        }
    out["per_image"] = per_image
    return out


def make_plot(model: str, report: dict) -> Path:
    import matplotlib.pyplot as plt

    fig_dir = RESULTS_ROOT / model / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(FIELDS), figsize=(6 * len(FIELDS), 4))
    if len(FIELDS) == 1:
        axes = [axes]
    for ax, f in zip(axes, FIELDS):
        fd = report["fields"][f]
        means = fd["means_by_guidance"]
        gs = sorted(float(g) for g in means)
        ys = [means[str(g)] for g in gs]
        rho = fd["spearman_rho_vs_g"]
        ax.plot(gs, ys, "o-", color="#7c3aed", linewidth=2, markersize=6, label="conditioned")
        ub = report["uncond_mean"].get(f)
        if ub is not None:
            ax.axhline(ub, color="#ef4444", linestyle="--", linewidth=1.5,
                       label=f"uncond baseline ({ub:.3f})")
        title_rho = f"{rho:+.2f}" if rho is not None else "n/a"
        ax.set_title(f"{f}  (rho={title_rho})")
        ax.set_xlabel("guidance (CFG)")
        ax.set_ylabel(f)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Exp 01b objective texture metrics — {model}", y=1.03)
    fig.tight_layout()
    out = fig_dir / "texture_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run(args: argparse.Namespace) -> None:
    report = analyze_model(args.model, args.size)
    out_json = RESULTS_ROOT / args.model / "texture_report.json"
    out_json.write_text(json.dumps(report, indent=2))

    summary = {f: {"rho_vs_g": report["fields"][f]["spearman_rho_vs_g"],
                   "p": report["fields"][f]["spearman_p"],
                   "uncond": report["uncond_mean"][f]}
               for f in FIELDS}
    print(json.dumps({"model": args.model, "n_images": report["n_images"],
                      "summary": summary}, indent=2))
    print(f"\n[texture] wrote {out_json}")

    if args.plot:
        out = make_plot(args.model, report)
        print(f"[texture] curve -> {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 01b objective texture/periodicity metrics.")
    p.add_argument("--model", default="sdxl")
    p.add_argument("--size", type=int, default=512, help="resize edge before FFT")
    p.add_argument("--plot", action="store_true", help="write figures/texture_curves.png")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
