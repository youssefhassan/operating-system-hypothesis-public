"""Exp 03 — POST-HOC / EXPLORATORY sensitivity analyses.

Everything in this file was written AFTER the confirmatory gates were computed
and reported. None of it changes, rescues or supersedes the pre-registered
verdict in `analyze.py` (SDXL confirm, SD 3.5 null on slope, overall not
confirmed under `both_models_required`). It exists to characterise two things
the confirmatory analysis surfaced but was not designed to answer:

  1. SHAPE. The pre-registered endpoint is a *linear* slope in guidance. The
     observed curves are not linear: most of the movement is in the bottom one
     or two steps of the dial. Quantify how much, and refit with the bottom
     point dropped, to see whether the dose-response is a genuine gradient or a
     single cliff at g=1.

  2. THE FOREST CONTROL. `preregistration.json` designates p6_forest a
     low-objecthood control that "should show tiling but LOW
     reduplication/condensation if the effect is object-bound". On SD 3.5 it
     behaved (composite rho -0.02). On SDXL it did not (-0.62). Break the forest
     down per field to see whether the SDXL signal runs through the
     object-bound fields (reduplication, condensation) or the generic ones
     (fragmentation, distortion, tiling).

Reuses `analyze.py`'s record loading and metric construction verbatim so the
composite here is the same composite the confirmatory analysis used.

    python posthoc.py --judges claude,qwen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze as A
import rubric as R
import statlib as S

HERE = Path(__file__).resolve().parent
INT = list(R.INT_FIELDS)


def _load(model: str, judges: list[str]):
    cond, uncond, notes = A._load_records(model, judges)
    A._build_metric(cond, uncond, notes)
    return cond, uncond, notes


# ------------------------------- 1. shape --------------------------------------


def shape(cond: list[dict]) -> dict:
    """How much of the dose-response lives in the bottom step(s) of the dial?"""
    gs = sorted({r["guidance"] for r in cond})
    means = {g: float(np.mean([r["composite"] for r in cond if r["guidance"] == g]))
             for g in gs}

    top, bottom = means[gs[0]], min(means.values())
    total_drop = top - bottom
    first_step = means[gs[0]] - means[gs[1]]

    out = {
        "composite_mean_by_guidance": {str(g): round(means[g], 4) for g in gs},
        "total_drop_top_to_min": round(total_drop, 4),
        "first_step_drop": round(first_step, 4),
        "share_of_drop_in_first_step": round(first_step / total_drop, 4) if total_drop else None,
    }

    # Refit with the bottom point dropped: is there a gradient above g=1?
    for drop_below in (1.0, 2.0):
        sub = [r for r in cond if r["guidance"] > drop_below]
        if len(sub) < 30:
            continue
        g = np.array([r["guidance"] for r in sub], float)
        c = np.array([r["composite"] for r in sub], float)
        rho = S.spearman(g, c)
        key = f"excluding_g_le_{drop_below:g}"
        out[key] = {
            "n": len(sub),
            "guidance_values": sorted({float(x) for x in g}),
            "spearman_composite_vs_g": round(float(rho), 4),
            "perm_p": round(float(S.spearman_perm_p(g, c)), 4),
            "lmm_slope_standardized": round(float(A._lmm_slope(sub)["slope_standardized"]), 4),
            "lmm_ci95": [round(float(v), 4) for v in A._lmm_slope(sub)["ci95"]],
        }
    return out


# --------------------------- 2. the forest control -----------------------------


def per_prompt_per_field(cond: list[dict]) -> dict:
    """Spearman of each raw field against guidance, within each prompt."""
    out: dict = {}
    prompts = sorted({r["prompt_id"] for r in cond})
    for p in prompts:
        recs = [r for r in cond if r["prompt_id"] == p]
        g = np.array([r["guidance"] for r in recs], float)
        block = {}
        for f in INT + list(R.BIN_FIELDS):
            v = np.array([r[f] for r in recs], float)
            block[f] = {
                "rho": round(float(S.spearman(g, v)), 4),
                "mean": round(float(v.mean()), 4),
                "mean_at_lowest_g": round(
                    float(np.mean([r[f] for r in recs if r["guidance"] == min(g)])), 4),
            }
        c = np.array([r["composite"] for r in recs], float)
        block["composite_rho"] = round(float(S.spearman(g, c)), 4)
        out[p] = block
    return out


def matched_quality_without_g1(cond: list[dict]) -> dict:
    """Re-run the matched-quality contrast with g=1 removed from the low arm.

    `analyze.py` defines the low arm as g <= 3. If a model's whole effect lives
    at g=1, the matched-quality de-confound inherits that dependency even though
    the arm nominally spans three dose levels. This isolates it.
    """
    out = {}
    for label, low_levels in (("low_arm_g<=3 (pre-registered)", (1.0, 2.0, 3.0)),
                              ("low_arm_g in {2,3} (g=1 removed)", (2.0, 3.0))):
        low = [r for r in cond if r["guidance"] in low_levels and r.get("Q") is not None]
        high = [r for r in cond if r["guidance"] > 8 and r.get("Q") is not None]
        if not low or not high:
            continue
        ql, qh = [r["Q"] for r in low], [r["Q"] for r in high]
        band = (max(min(ql), min(qh)), min(max(ql), max(qh)))
        lo_c = np.array([r["composite"] for r in low if band[0] <= r["Q"] <= band[1]])
        hi_c = np.array([r["composite"] for r in high if band[0] <= r["Q"] <= band[1]])
        if len(lo_c) < 3 or len(hi_c) < 3:
            continue
        ci = S.bootstrap_delta_ci(lo_c, hi_c)
        out[label] = {
            "n_low": int(len(lo_c)), "n_high": int(len(hi_c)),
            "cliffs_delta_low_vs_high": round(float(S.cliffs_delta(lo_c, hi_c)), 4),
            "ci95": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        }
    return out


def tiling_rate(cond: list[dict]) -> dict:
    """Fraction of images flagged as tiling (judge-mean > 0.5), per prompt."""
    out = {}
    for p in sorted({r["prompt_id"] for r in cond}):
        recs = [r for r in cond if r["prompt_id"] == p]
        lo = [r for r in recs if r["guidance"] == min(x["guidance"] for x in recs)]
        out[p] = {
            "overall": round(float(np.mean([r["tiling"] > 0.5 for r in recs])), 4),
            "at_lowest_g": round(float(np.mean([r["tiling"] > 0.5 for r in lo])), 4),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judges", default="claude,qwen")
    ap.add_argument("--models", default="sdxl,sd35")
    ap.add_argument("--out", default="posthoc_report.json")
    args = ap.parse_args()

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    report = {"status": "POST-HOC / EXPLORATORY — does not affect the confirmatory verdict",
              "judges": judges, "models": {}}

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        cond, uncond, notes = _load(model, judges)
        report["models"][model] = {
            "n_conditioned": notes["n_conditioned"],
            "shape": shape(cond),
            "matched_quality_without_g1": matched_quality_without_g1(cond),
            "per_prompt_per_field": per_prompt_per_field(cond),
            "tiling_rate": tiling_rate(cond),
        }

    (HERE / args.out).write_text(json.dumps(report, indent=2))

    for model, blk in report["models"].items():
        sh = blk["shape"]
        print(f"\n=== {model} (n={blk['n_conditioned']}) ===")
        print(" composite by g:", sh["composite_mean_by_guidance"])
        print(f" first step = {sh['first_step_drop']} of {sh['total_drop_top_to_min']} total"
              f" ({sh['share_of_drop_in_first_step']:.0%} of the drop)")
        for k in ("excluding_g_le_1", "excluding_g_le_2"):
            if k in sh:
                s = sh[k]
                print(f" {k}: rho={s['spearman_composite_vs_g']:+.3f} p={s['perm_p']:.4f}"
                      f" lmm={s['lmm_slope_standardized']:+.3f} {s['lmm_ci95']}")
        print(" matched-quality contrast:")
        for lbl, mq in blk["matched_quality_without_g1"].items():
            print(f"   {lbl:34s} delta={mq['cliffs_delta_low_vs_high']:+.3f} {mq['ci95']}"
                  f" (n {mq['n_low']}/{mq['n_high']})")
        print(" per-prompt per-field rho vs g:")
        for p, b in blk["per_prompt_per_field"].items():
            fields = "  ".join(f"{f[:5]} {b[f]['rho']:+.2f}" for f in INT + ["tiling"])
            print(f"   {p:16s} comp {b['composite_rho']:+.2f} | {fields}")
        print(" tiling rate:", {p: v["at_lowest_g"] for p, v in blk["tiling_rate"].items()})

    print(f"\nwrote {HERE / args.out}")


if __name__ == "__main__":
    main()
