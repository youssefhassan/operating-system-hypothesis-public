"""
Experiment 01 — aggregate stats, exploratory rubric fields, contact sheets.

Confirmatory: reuses loop.aggregate / loop.verdict (metric M = geometric_intensity).
Exploratory: veridicality, spontaneity, coherent_scene, complexity vs guidance
(not pre-registered; report separately from M).

Usage:
    python analyze.py --model sdxl
    python analyze.py --model sdxl --contact-sheet
    python analyze.py --model sdxl --contact-sheet --contact-sheet-layout by-seed
    python analyze.py --model sdxl --contact-sheet --no-organize-views
    python analyze.py --model sdxl --contact-sheet --contact-sheet-split 3 3 4
    python analyze.py --model sdxl --gifs
    python analyze.py --model sdxl --gifs --gif-size 768 --gif-duration 600
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from loop import PREREG, RESULTS_ROOT, aggregate, spearman, spearman_perm_p, verdict

HERE = Path(__file__).resolve().parent
FNAME_RE = re.compile(r"^g(?P<g>[0-9.]+)_s(?P<s>\d+)\.png$")
# Default contact-sheet layout when N=10 (seeds 42–51): three readable panels.
DEFAULT_CONTACT_SHEET_SPLIT = [3, 3, 4]

EXPLORATORY_FIELDS = (
    "veridicality",
    "spontaneity",
    "coherent_scene",
    "complexity",
    "geometric_intensity",
)


def _load_judgements(model: str) -> dict:
    path = RESULTS_ROOT / model / "judgements.json"
    if not path.exists():
        raise SystemExit(f"missing {path} — run judge.py first")
    return json.loads(path.read_text())


def _claude_scores(judgements: dict) -> list[tuple[float, int, dict]]:
    """Return [(guidance, seed, claude_record), ...] for conditioned images."""
    rows = []
    for fn, rec in judgements["images"].items():
        m = FNAME_RE.match(fn)
        if not m:
            continue
        c = rec.get("claude", {})
        if "error" in c:
            continue
        rows.append((float(m["g"]), int(m["s"]), c))
    return rows


def exploratory_stats(model: str, prereg: dict) -> dict:
    rows = _claude_scores(_load_judgements(model))
    if not rows:
        return {"ready": False}

    by_field: dict[str, dict[float, list[float]]] = {
        f: defaultdict(list) for f in EXPLORATORY_FIELDS
    }
    all_g: dict[str, list[float]] = {f: [] for f in EXPLORATORY_FIELDS}
    all_v: dict[str, list[float]] = {f: [] for f in EXPLORATORY_FIELDS}

    for g, _s, c in rows:
        for f in EXPLORATORY_FIELDS:
            v = float(c.get(f, 0))
            by_field[f][g].append(v)
            all_g[f].append(g)
            all_v[f].append(v)

    out: dict = {"model": model, "ready": True, "fields": {}}
    for f in EXPLORATORY_FIELDS:
        ag, av = np.array(all_g[f]), np.array(all_v[f])
        means = {str(g): round(float(np.mean(vs)), 3) for g, vs in sorted(by_field[f].items())}
        out["fields"][f] = {
            "means_by_guidance": means,
            "spearman_rho_vs_g": round(spearman(ag, av), 4),
            "spearman_p": round(spearman_perm_p(ag, av), 4),
        }
    return out


def coverage(model: str, prereg: dict) -> dict:
    d = RESULTS_ROOT / model
    grid = prereg["guidance_grid"]
    target = prereg["samples_per_setting"]
    counts: dict[str, int] = {}
    for g in grid:
        n = len(list(d.glob(f"g{g:.1f}_s*.png")))
        counts[f"g={g}"] = n
    uncond = len(list(d.glob("uncond_s*.png")))
    return {
        "model": model,
        "target_per_setting": target,
        "by_guidance": counts,
        "unconditional": uncond,
        "complete": all(n >= target for n in counts.values()) and uncond >= target,
    }


def _chunk_seeds(seeds: list[int], chunk_sizes: list[int]) -> list[list[int]]:
    if sum(chunk_sizes) != len(seeds):
        raise SystemExit(
            f"--contact-sheet-split {chunk_sizes} sums to {sum(chunk_sizes)} "
            f"but {len(seeds)} seeds were given"
        )
    groups: list[list[int]] = []
    i = 0
    for size in chunk_sizes:
        groups.append(seeds[i : i + size])
        i += size
    return groups


def _load_cell(model_dir: Path, kind: str, g: float | None, seed: int):
    from PIL import Image

    if kind == "uncond":
        p = model_dir / f"uncond_s{seed}.png"
    else:
        p = model_dir / f"g{g:.1f}_s{seed}.png"
    return Image.open(p).convert("RGB") if p.exists() else None


def _format_dose_label(g: float | None) -> str:
    if g is None:
        return "uncond"
    if g == int(g):
        return str(int(g))
    return f"{g:.1f}".rstrip("0").rstrip(".")


def _dose_sequence(grid: list[float]) -> list[tuple[str, float | None, str]]:
    """Playback order: uncond, then guidance ascending (1, 1.5, 2, …)."""
    return [("uncond", None, "uncond")] + [
        ("cond", g, _format_dose_label(g)) for g in grid
    ]


def _dose_symlink_name(index: int, label: str) -> str:
    return f"{index:02d}_{label}.png"


def _render_contact_sheet(
    model_dir: Path,
    out_path: Path,
    *,
    row_labels: list[str],
    col_labels: list[str],
    rows_imgs: list[list],
    thumb: int = 256,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nrows, ncols = len(rows_imgs), len(col_labels)
    label_w, label_h = 72, 28
    cell = thumb + label_h
    sheet = Image.new("RGB", (label_w + ncols * cell, nrows * cell), (240, 240, 240))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for ri, (rlabel, row) in enumerate(zip(row_labels, rows_imgs)):
        draw.text((4, ri * cell + thumb // 2), rlabel, fill=(0, 0, 0), font=font)
        for ci, img in enumerate(row[:ncols]):
            x = label_w + ci * cell
            y = ri * cell
            if img is None:
                draw.rectangle([x, y, x + thumb, y + thumb], outline=(180, 180, 180))
                draw.text((x + 8, y + thumb // 2), "missing", fill=(120, 120, 120), font=font)
                continue
            im = img.resize((thumb, thumb), Image.Resampling.LANCZOS)
            sheet.paste(im, (x, y))
            draw.text((x, y + thumb), col_labels[ci], fill=(0, 0, 0), font=font)

    sheet.save(out_path)
    return out_path


def make_contact_sheet_by_g(
    model: str,
    prereg: dict,
    seeds: list[int],
    thumb: int = 256,
    part: int | None = None,
) -> Path:
    """Rows = guidance (uncond, then g↑); cols = seeds. Compare seeds at fixed g."""
    d = RESULTS_ROOT / model
    grid = prereg["guidance_grid"]
    rows_imgs: list[list] = [[_load_cell(d, "uncond", None, s) for s in seeds]]
    for g in grid:
        rows_imgs.append([_load_cell(d, "cond", g, s) for s in seeds])
    row_labels = ["uncond"] + [f"g={g}" for g in grid]
    col_labels = [f"s{s}" for s in seeds]
    seed_tag = "-".join(str(s) for s in seeds)
    part_tag = f"_part{part}" if part is not None else ""
    out = d / "figures" / "by_g" / f"dose_by_g{part_tag}_seeds{seed_tag}.png"
    return _render_contact_sheet(d, out, row_labels=row_labels, col_labels=col_labels,
                                   rows_imgs=rows_imgs, thumb=thumb)


def make_contact_sheet_by_seed(
    model: str,
    prereg: dict,
    seeds: list[int],
    thumb: int = 256,
    part: int | None = None,
) -> Path:
    """Rows = seeds; cols = dose (uncond, then g↑). Follow one seed across guidance."""
    d = RESULTS_ROOT / model
    grid = prereg["guidance_grid"]
    col_specs = _dose_sequence(grid)
    col_labels = [label for _kind, _g, label in col_specs]
    rows_imgs: list[list] = []
    for s in seeds:
        rows_imgs.append([_load_cell(d, kind, g, s) for kind, g, _label in col_specs])
    row_labels = [f"s{s}" for s in seeds]
    seed_tag = "-".join(str(s) for s in seeds)
    part_tag = f"_part{part}" if part is not None else ""
    out = d / "figures" / "by_seed" / f"dose_by_seed{part_tag}_seeds{seed_tag}.png"
    return _render_contact_sheet(d, out, row_labels=row_labels, col_labels=col_labels,
                                   rows_imgs=rows_imgs, thumb=thumb)


def make_contact_sheet(
    model: str,
    prereg: dict,
    seeds: list[int],
    thumb: int = 256,
    cols: int | None = None,
    part: int | None = None,
) -> Path:
    """Alias for by-g layout (backward compatible)."""
    return make_contact_sheet_by_g(model, prereg, seeds, thumb=thumb, part=part)


def make_contact_sheets(
    model: str,
    prereg: dict,
    seed_groups: list[list[int]],
    layout: str = "both",
    **kwargs,
) -> list[Path]:
    paths: list[Path] = []
    multi = len(seed_groups) > 1
    for i, seeds in enumerate(seed_groups, start=1):
        part = i if multi else None
        if layout in ("both", "by-g"):
            paths.append(make_contact_sheet_by_g(model, prereg, seeds, part=part, **kwargs))
        if layout in ("both", "by-seed"):
            paths.append(make_contact_sheet_by_seed(model, prereg, seeds, part=part, **kwargs))
    return paths


def organize_view_folders(model: str, prereg: dict, seeds: list[int]) -> tuple[Path, Path]:
    """Symlink trees for browsing: by_seed/s42/g1.0.png and by_guidance/g1.0/s42.png."""
    import os

    d = RESULTS_ROOT / model
    by_seed_root = d / "by_seed"
    by_g_root = d / "by_guidance"
    grid = prereg["guidance_grid"]

    for s in seeds:
        seed_dir = by_seed_root / f"s{s}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for old in seed_dir.iterdir():
            if old.is_symlink():
                old.unlink()
        for i, (kind, g, label) in enumerate(_dose_sequence(grid)):
            if kind == "uncond":
                target = f"uncond_s{s}.png"
            else:
                target = f"g{g:.1f}_s{s}.png"
            link = seed_dir / _dose_symlink_name(i, label)
            if link.exists() or link.is_symlink():
                link.unlink()
            if (d / target).exists():
                os.symlink(f"../../{target}", link)

    for g in grid:
        g_dir = by_g_root / f"g{g:.1f}"
        g_dir.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            target = f"g{g:.1f}_s{s}.png"
            link = g_dir / f"s{s}.png"
            if link.exists() or link.is_symlink():
                link.unlink()
            if (d / target).exists():
                os.symlink(f"../../{target}", link)

    uncond_dir = by_g_root / "uncond"
    uncond_dir.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        target = f"uncond_s{s}.png"
        link = uncond_dir / f"s{s}.png"
        if link.exists() or link.is_symlink():
            link.unlink()
        if (d / target).exists():
            os.symlink(f"../../{target}", link)

    return by_seed_root, by_g_root


def _label_frame(img, text: str, bar_h: int = 36):
    from PIL import Image, ImageDraw, ImageFont

    w, h = img.size
    out = Image.new("RGB", (w, h + bar_h), (240, 240, 240))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    draw.text((8, h + 10), text, fill=(0, 0, 0), font=font)
    return out


def make_dose_gifs(
    model: str,
    prereg: dict,
    seeds: list[int],
    *,
    size: int = 512,
    duration_ms: int = 800,
    label_frames: bool = True,
) -> list[Path]:
    """One GIF per seed: uncond → g ascending (same order as by_seed contact sheets)."""
    from PIL import Image

    d = RESULTS_ROOT / model
    grid = prereg["guidance_grid"]
    out_dir = d / "figures" / "by_seed" / "gifs"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_specs = _dose_sequence(grid)
    paths: list[Path] = []

    for seed in seeds:
        frames = []
        for kind, g, label in frame_specs:
            img = _load_cell(d, kind, g, seed)
            if img is None:
                continue
            im = img.resize((size, size), Image.Resampling.LANCZOS)
            if label_frames:
                im = _label_frame(im, f"s{seed}  {label}")
            frames.append(im)

        if len(frames) < 2:
            continue

        out = out_dir / f"s{seed}_dose.gif"
        frames[0].save(
            out,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        paths.append(out)

    return paths


def make_m_curve(model: str, prereg: dict) -> Path | None:
    import matplotlib.pyplot as plt

    agg = aggregate(model, prereg)
    if not agg.get("ready"):
        return None
    fig_dir = RESULTS_ROOT / model / "figures"
    fig_dir.mkdir(exist_ok=True)

    gs = agg["guidance"]
    means = agg["means"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gs, means, "o-", color="#2563eb", linewidth=2, markersize=6)
    ax.set_xlabel("guidance (CFG)")
    ax.set_ylabel("M(g) — mean geometric_intensity / 3")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Exp 01 confirmatory metric — {model}")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="#999", linewidth=0.8)
    fig.tight_layout()
    out = fig_dir / "m_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def make_exploratory_curves(model: str, prereg: dict) -> Path | None:
    import matplotlib.pyplot as plt

    exp = exploratory_stats(model, prereg)
    if not exp.get("ready"):
        return None
    fig_dir = RESULTS_ROOT / model / "figures"
    fig_dir.mkdir(exist_ok=True)

    fields = ["veridicality", "spontaneity", "coherent_scene", "geometric_intensity"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for ax, f in zip(axes.flat, fields):
        means = exp["fields"][f]["means_by_guidance"]
        gs = sorted(float(g) for g in means)
        ys = [means[str(g)] for g in gs]
        rho = exp["fields"][f]["spearman_rho_vs_g"]
        ax.plot(gs, ys, "o-", linewidth=2, markersize=5)
        ax.set_title(f"{f}  (ρ={rho:+.2f})")
        ax.set_xlabel("guidance")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Exp 01 exploratory rubric fields — {model}", y=1.02)
    fig.tight_layout()
    out = fig_dir / "exploratory_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run(args: argparse.Namespace) -> None:
    prereg = json.loads(PREREG.read_text())
    cov = coverage(args.model, prereg)
    agg = aggregate(args.model, prereg)
    exp = exploratory_stats(args.model, prereg)
    report = {
        "coverage": cov,
        "confirmatory": agg,
        "verdict": verdict(agg, prereg) if agg.get("ready") else "inconclusive",
        "exploratory": exp,
    }
    out_json = RESULTS_ROOT / args.model / "analysis_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\n[analyze] wrote {out_json}")

    if args.contact_sheet:
        n = prereg["samples_per_setting"]
        all_seeds = args.seeds or list(range(42, 42 + n))
        split = args.contact_sheet_split
        if split is None and args.seeds is None and n == sum(DEFAULT_CONTACT_SHEET_SPLIT):
            split = DEFAULT_CONTACT_SHEET_SPLIT
        if split:
            groups = _chunk_seeds(all_seeds, split)
            sheets = make_contact_sheets(
                args.model, prereg, groups, layout=args.contact_sheet_layout
            )
        else:
            sheets = make_contact_sheets(
                args.model, prereg, [all_seeds], layout=args.contact_sheet_layout
            )
        for cs in sheets:
            print(f"[analyze] contact sheet -> {cs}")
        if args.organize_views:
            by_seed, by_g = organize_view_folders(args.model, prereg, all_seeds)
            print(f"[analyze] browse folders -> {by_seed}/  and  {by_g}/")

    if args.gifs:
        n = prereg["samples_per_setting"]
        gif_seeds = args.seeds or list(range(42, 42 + n))
        for gif in make_dose_gifs(
            args.model,
            prereg,
            gif_seeds,
            size=args.gif_size,
            duration_ms=args.gif_duration,
            label_frames=not args.no_gif_labels,
        ):
            print(f"[analyze] dose gif -> {gif}")

    if args.plots:
        mc = make_m_curve(args.model, prereg)
        if mc:
            print(f"[analyze] M(g) curve -> {mc}")
        ec = make_exploratory_curves(args.model, prereg)
        if ec:
            print(f"[analyze] exploratory curves -> {ec}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 01 analysis + figures.")
    p.add_argument("--model", default="sdxl")
    p.add_argument("--contact-sheet", action="store_true")
    p.add_argument("--plots", action="store_true", help="write M(g) and exploratory PNG curves")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="seeds for contact sheet columns")
    p.add_argument(
        "--contact-sheet-split",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="multiple contact sheets with N seeds each (default 3 3 4 when N=10 and --seeds omitted)",
    )
    p.add_argument(
        "--contact-sheet-layout",
        choices=["both", "by-seed", "by-g"],
        default="both",
        help="by-seed: rows=seeds, cols=dose (uncond→high g); by-g: rows=dose, cols=seeds",
    )
    p.add_argument(
        "--organize-views",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="symlink trees by_seed/s{N}/ and by_guidance/g{X}/ (default on with --contact-sheet)",
    )
    p.add_argument(
        "--gifs",
        action="store_true",
        help="write one dose-sweep GIF per seed under figures/by_seed/gifs/",
    )
    p.add_argument("--gif-size", type=int, default=512, help="GIF frame edge length in px")
    p.add_argument("--gif-duration", type=int, default=800, help="ms per frame")
    p.add_argument("--no-gif-labels", action="store_true", help="omit s/g labels on GIF frames")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
