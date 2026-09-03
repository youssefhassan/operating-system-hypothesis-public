"""
Unregistered scratch — descriptive read-out of the sub-CFG sweep.

Deliberately thin. There is no pre-registered hypothesis here, so there is no
confirmatory test to run and no multiple-comparison correction to apply: reporting
a p-value would dress an exploratory look up as a result. What this prints is a
profile — each Klüver field's mean against guidance, per model and per judge, with
bootstrap CIs so the eye can tell a trend from two seeds' worth of noise.

Two judges from different families: Qwen3-VL-32B (local, open-weight) and Claude
Sonnet 5 (Exp 03's Judge A). That crossing matters — it is the same circularity
break Exp 03 used, so their agreement is evidence about the images rather than
about one lineage's shared blind spots. Where they disagree, believe neither.

Expect the kappa to be flattered at low guidance on SDXL: that region is pinned at
the rubric maximum, and two raters agreeing on a saturated ceiling is not the same
as two raters agreeing on a measurement.

The Spearman rho is a *descriptive* monotonicity summary over the 0-2 window only.
It is not comparable to Exp 03's rho, which was fitted over g = 1..15 with 10 seeds
and a pre-registered model.

Usage:
    python analyze_low_g.py                 # -> report.json + stdout tables
"""

from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP03 = HERE.parent / "03_l23_hardening"
sys.path.insert(0, str(EXP03))

import rubric as R  # noqa: E402
import statlib as SL  # noqa: E402

# tag -> the judgements file that judge writes into each results-local/<model> dir
JUDGE_FILES = {
    "32b": "judgements_qwen_32b.json",       # Qwen3-VL-32B, local MLX
    "claude": "judgements_claude.json",      # Claude Sonnet 5, Exp 03's judge.py
}
JUDGE_TAGS = tuple(JUDGE_FILES)
Q_ORDINAL, Q_BINARY = 4, 2  # ordinal fields are 0..3; tiling is 0/1


def _q(field: str) -> int:
    return Q_BINARY if field in R.BIN_FIELDS else Q_ORDINAL


def _boot_mean_ci(vals: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean. Two seeds is thin; the width is the point."""
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


@lru_cache(maxsize=None)
def _sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def _dedupe_pseudoreplicates(rows: list[dict], field: str) -> list[dict]:
    """Collapse rows that are the *same image* judged under different prompt ids.

    At g=0 the prompt is arithmetically cancelled, so all six prompts render one
    identical image per seed (that is exactly what check_g0.py asserts). Counting
    those as six independent observations is pseudo-replication: it would shrink the
    bootstrap CI by roughly sqrt(6) and make a column containing two distinct images
    look like twelve. One row per (seed, sha) survives.

    Above g=0 the images genuinely differ, so nothing is dropped and this is a no-op.
    """
    seen: dict[tuple, dict] = {}
    for r in rows:
        key = (r["seed"], r.get("sha"))
        if r.get("sha") is None:
            return rows  # no hashes available; leave the data alone
        if key not in seen:
            seen[key] = r
    return list(seen.values())


def load(model_dir: Path, tag: str) -> list[dict]:
    """One record per successfully judged conditioned image, for one judge."""
    path = model_dir / JUDGE_FILES[tag]
    if not path.exists():
        return []
    blob = json.loads(path.read_text())
    rows = []
    for fname, rec in blob["images"].items():
        if "error" in rec:
            continue
        meta = R.parse_filename(fname)
        if meta is None or meta.get("kind") != "conditioned":
            continue  # uncond baselines have no guidance axis to sit on
        rows.append({"file": fname, "prompt_id": meta["prompt_id"],
                     "guidance": float(meta["guidance"]), "seed": int(meta["seed"]),
                     "sha": _sha(model_dir / fname),
                     "missing_fields": rec.get("missing_fields", []),
                     **{f: rec[f] for f in R.ALL_FIELDS if f in rec}})
    return rows


def report_judge(model: str, tag: str, rows: list[dict]) -> dict:
    print(f"\n=== {model} / judge {tag} — {len(rows)} judged images ===")
    defaulted = sum(1 for r in rows if r["missing_fields"])
    if defaulted:
        print(f"  note: {defaulted} images had >=1 field defaulted to 0 by coerce() — "
              "those zeros are not real judgements")
    grid = sorted({r["guidance"] for r in rows})
    out: dict = {"n": len(rows), "n_with_defaulted_fields": defaulted,
                 "guidance": grid, "fields": {}}

    for field in R.ALL_FIELDS:
        vals = [r for r in rows if field in r]
        if not vals:
            continue
        print(f"\n{field}")
        print(f"  {'g':>5}  {'n':>3}  {'mean':>6}   95% CI")
        per_g = {}
        for g in grid:
            sub = _dedupe_pseudoreplicates(
                [r for r in vals if r["guidance"] == g], field)
            arr = np.array([r[field] for r in sub], dtype=float)
            if not len(arr):
                continue
            lo, hi = _boot_mean_ci(arr, seed=int(g * 100))
            per_g[g] = {"n": int(len(arr)), "mean": float(arr.mean()), "ci": [lo, hi]}
            print(f"  {g:>5.2f}  {len(arr):>3}  {arr.mean():>6.3f}   [{lo:.3f}, {hi:.3f}]")

        # Shape, not just direction. A U (high at the prior, low in the middle, rising
        # again with guidance) would make the pooled rho near zero and invisible, so
        # say explicitly where the minimum sits.
        means = {g: per_g[g]["mean"] for g in per_g}
        g_lo, g_hi = min(means), max(means)
        g_min = min(means, key=lambda k: means[k])
        # A bare "the minimum is not at an endpoint" test calls almost everything
        # U-shaped: with 12 images per cell, the last point sits a hundredth above
        # the minimum purely by noise. Require the rebound to clear a quarter of a
        # rubric point (an eighth for the 0/1 tiling field) before saying "U".
        tol = 0.125 if field in R.BIN_FIELDS else 0.25
        rebound = means[g_hi] - means[g_min]
        interior = g_min not in (g_lo, g_hi) and rebound > tol
        shape = (f"U-shaped (min at g={g_min:g}, rebound {rebound:+.2f})" if interior
                 else f"monotone-ish (no rebound above {tol:g}; "
                      f"min at g={g_min:g}, rebound {rebound:+.2f})")
        print(f"  shape: {shape} — mean at g={g_lo:g} is {means[g_lo]:.3f}, "
              f"min at g={g_min:g} is {means[g_min]:.3f}, "
              f"at g={g_hi:g} is {means[g_hi]:.3f}")

        # Pooled rho on the deduplicated set, for the same pseudo-replication reason:
        # six copies of the g=0 image would triple-count the anchor end of the axis.
        pooled = [r for g in grid
                  for r in _dedupe_pseudoreplicates(
                      [v for v in vals if v["guidance"] == g], field)]
        x = np.array([r["guidance"] for r in pooled], dtype=float)
        y = np.array([r[field] for r in pooled], dtype=float)
        rho = SL.spearman(x, y) if len(set(y.tolist())) > 1 else float("nan")
        print(f"  descriptive rho over g=0..2: {rho:+.3f}  "
              f"(n={len(pooled)} after dedupe, exploratory, no p-value)")

        # Per-prompt, because Exp 03 section 5.1 found these effects are strongly
        # prompt-dependent: a pooled number alone would be misleading.
        by_prompt = {}
        for pid in sorted({r["prompt_id"] for r in vals}):
            sub = [r for r in vals if r["prompt_id"] == pid]
            xs = np.array([r["guidance"] for r in sub], dtype=float)
            ys = np.array([r[field] for r in sub], dtype=float)
            by_prompt[pid] = (SL.spearman(xs, ys)
                              if len(set(ys.tolist())) > 1 else float("nan"))
        print("  per-prompt rho: " +
              "  ".join(f"{p.split('_')[0]}={v:+.2f}" for p, v in by_prompt.items()))

        out["fields"][field] = {"shape": shape, "g_at_min": g_min,
                                "interior_minimum": interior,
                                "per_guidance": {str(k): v for k, v in per_g.items()},
                                "rho_pooled": rho, "rho_by_prompt": by_prompt}
    return out


def cross_judge(model: str, per_tag: dict[str, list[dict]]) -> dict:
    """Quadratic-weighted kappa between the two judges on the images both scored."""
    tags = [t for t in JUDGE_TAGS if per_tag.get(t)]
    if len(tags) < 2:
        print(f"\n=== {model} / cross-judge: only {len(tags)} judge(s) — skipped ===")
        return {}
    a_by = {r["file"]: r for r in per_tag[tags[0]]}
    b_by = {r["file"]: r for r in per_tag[tags[1]]}
    shared = sorted(set(a_by) & set(b_by))
    print(f"\n=== {model} / cross-judge {tags[0]} vs {tags[1]} — "
          f"{len(shared)} shared images ===")
    if not shared:
        return {}

    out: dict = {"judges": tags, "n_shared": len(shared), "fields": {}}
    paired = []
    for field in R.ALL_FIELDS:
        a = [a_by[f][field] for f in shared if field in a_by[f] and field in b_by[f]]
        b = [b_by[f][field] for f in shared if field in a_by[f] and field in b_by[f]]
        if not a:
            continue
        q = _q(field)
        k = SL.weighted_cohens_kappa(a, b, q)
        ac2 = SL.gwet_ac2(a, b, q)
        pa = SL.percent_agreement(a, b)
        print(f"  {field:<15} weighted kappa {k:+.3f}   Gwet AC2 {ac2:+.3f}   "
              f"exact agreement {pa:.1%}")
        out["fields"][field] = {"weighted_kappa": k, "gwet_ac2": ac2,
                                "percent_agreement": pa}
        if field in R.INT_FIELDS:
            paired.append((a, b))

    # Split the kappa by regime. A headline kappa can be carried by cells where the
    # rubric is pinned at its maximum — two raters agreeing on a ceiling is not two
    # raters agreeing on a measurement. Reporting both halves shows which it is, and
    # the ceiling counts say whether saturation is a property of the images or of one
    # judge (on SDXL it turned out to be the latter: Qwen pinned 26/48 low-g images at
    # 3-on-all-four-fields, Claude pinned none).
    lo = [f for f in shared if a_by[f]["guidance"] < 1.0]
    hi = [f for f in shared if a_by[f]["guidance"] >= 1.0]
    print(f"  -- by regime (g<1: n={len(lo)}, g>=1: n={len(hi)}) --")
    out["by_regime"] = {}
    for field in R.ALL_FIELDS:
        row = {}
        for label, files in (("g<1", lo), ("g>=1", hi)):
            x = [a_by[f][field] for f in files if field in a_by[f] and field in b_by[f]]
            y = [b_by[f][field] for f in files if field in a_by[f] and field in b_by[f]]
            row[label] = (SL.weighted_cohens_kappa(x, y, _q(field))
                          if x and (len(set(x)) > 1 or len(set(y)) > 1) else float("nan"))
        print(f"  {field:<15} g<1 {row['g<1']:+.3f}   g>=1 {row['g>=1']:+.3f}")
        out["by_regime"][field] = row

    for tag, src in zip(tags, (a_by, b_by)):
        pin = sum(1 for f in lo if all(src[f].get(k) == 3 for k in R.INT_FIELDS))
        print(f"  ceiling check: {tag} scored 3 on all four ordinal fields for "
              f"{pin}/{len(lo)} images at g<1")
        out.setdefault("ceiling_at_low_g", {})[tag] = {"pinned": pin, "n": len(lo)}

    if paired:
        comp = SL.composite_kappa_ci(paired, Q_ORDINAL)
        ci = comp.get("ci95")
        ci_s = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "n/a"
        print(f"  composite over the 4 ordinal fields: {comp['point']:+.3f}  "
              f"95% CI {ci_s}")
        out["composite_ordinal"] = comp
    return out


if __name__ == "__main__":
    dirs = [p for p in sorted((HERE / "results-local").glob("*")) if p.is_dir()]
    report = {"note": "EXPLORATORY, unregistered. Not comparable to Exp 03's grid: "
                      "CFG was forced on below g=1, which Exp 01/03 did not do.",
              "judges": {"32b": "Qwen3-VL-32B-Instruct-4bit (local MLX)",
                         "claude": "claude-sonnet-5 (Batch API)"},
              "judge_caveat": "Two judges across model families (open-weight Qwen "
                              "vs Claude), the same circularity break Exp 03 used. "
                              "Agreement in SDXL's low-g region is inflated by "
                              "rubric saturation at the 3.000 ceiling.",
              "rubric_version": R.rubric_version(), "models": {}}
    any_rows = False
    for d in dirs:
        per_tag = {t: load(d, t) for t in JUDGE_TAGS}
        if not any(per_tag.values()):
            print(f"[analyze] {d.name}: nothing judged yet — skipping")
            continue
        any_rows = True
        entry: dict = {"judges": {}}
        for tag, rows in per_tag.items():
            if rows:
                entry["judges"][tag] = report_judge(d.name, tag, rows)
        entry["cross_judge"] = cross_judge(d.name, per_tag)
        report["models"][d.name] = entry
    if not any_rows:
        raise SystemExit("nothing judged yet")
    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    print("\n[analyze] wrote report.json")  # relative: absolute paths leak $HOME
