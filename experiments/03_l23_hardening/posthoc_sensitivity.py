"""
Exp 03 sensitivity analyses, pre-specified in posthoc_sensitivity_prespec.md
(2026-09-03). EXPLORATORY: same data, same panel, no gate.

  S1  covariate scale: linear (pre-registered) vs z(log2 g) vs rank(g)
  S2  three-field composite (no distortion) vs the four-field composite

Usage: python posthoc_sensitivity.py --judges claude,qwen
Writes posthoc_sensitivity.json.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import analyze as A

HERE = Path(__file__).resolve().parent
PREREG = json.loads((HERE / "preregistration.json").read_text())


def _with_covariate(cond: list[dict], transform) -> list[dict]:
    """Copy of the records with `guidance` replaced by the transformed value,
    so analyze._lmm_slope (which z-scores whatever it finds under 'guidance')
    fits the same model on a different scale."""
    out = copy.deepcopy(cond)
    g = np.array([r["guidance"] for r in cond], float)
    t = transform(g)
    for r, v in zip(out, t):
        r["guidance"] = float(v)
    return out


def _rank(g: np.ndarray) -> np.ndarray:
    levels = sorted(set(g.tolist()))
    return np.array([levels.index(v) for v in g], float)


def _three_field_composite(cond: list[dict], uncond: list[dict]) -> None:
    fields = [f for f in A.INT if f != "distortion"]
    for r in cond + uncond:
        r["composite"] = float(np.mean([r["z_" + f] for f in fields]))


def run_model(model: str, judges: list[str]) -> dict:
    cond, uncond, notes = A._load_records(model, judges)
    A._build_metric(cond, uncond, notes)
    out = {"n_conditioned": len(cond), "judges": notes["judge_names"]}

    # ---- S1: covariate scale, four-field composite (pre-registered metric)
    out["S1_covariate_scale"] = {
        "linear_prereg": A._lmm_slope(cond),
        "log2_g": A._lmm_slope(_with_covariate(cond, np.log2)),
        "rank_g": A._lmm_slope(_with_covariate(cond, _rank)),
    }

    # ---- S2: three-field composite
    cond3, uncond3 = copy.deepcopy(cond), copy.deepcopy(uncond)
    _three_field_composite(cond3, uncond3)
    s2 = {
        "fields": [f for f in A.INT if f != "distortion"],
        "lmm_slope_linear": A._lmm_slope(cond3),
        "lmm_slope_log2_g": A._lmm_slope(_with_covariate(cond3, np.log2)),
        "per_prompt": A._per_prompt_slopes(cond3),
    }
    if notes.get("quality_available"):
        s2["deconfound"] = A._deconfound(cond3, notes)
    out["S2_three_field_composite"] = s2
    out["four_field_reference"] = {
        "lmm_slope_linear": out["S1_covariate_scale"]["linear_prereg"],
        "per_prompt": A._per_prompt_slopes(cond),
        "deconfound": A._deconfound(cond, notes) if notes.get("quality_available") else None,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="claude,qwen")
    args = ap.parse_args()
    judges = args.judges.split(",")
    report = {"label": "EXPLORATORY (posthoc_sensitivity_prespec.md, 2026-09-03)",
              "models": {m: run_model(m, judges) for m in PREREG["models"]["confirmatory"]}}
    (HERE / "posthoc_sensitivity.json").write_text(json.dumps(report, indent=2, default=str))
    for m, o in report["models"].items():
        s1 = o["S1_covariate_scale"]
        print(f"\n[{m}] S1 four-field composite slope by covariate scale")
        for k in ("linear_prereg", "log2_g", "rank_g"):
            v = s1[k]
            print(f"   {k:13s} {v['slope_standardized']:+.3f}  CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
        s2 = o["S2_three_field_composite"]
        print(f"[{m}] S2 three-field composite (no distortion)")
        for k in ("lmm_slope_linear", "lmm_slope_log2_g"):
            v = s2[k]
            print(f"   {k:17s} {v['slope_standardized']:+.3f}  CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
        pp = s2["per_prompt"]
        print(f"   per-prompt negative: {pp['n_negative']}/{pp['n_prompts']}  {pp['per_prompt_spearman_composite_vs_g']}")
        if "deconfound" in s2:
            print(f"   deconfound: {json.dumps(s2['deconfound'])[:300]}")
        ref = o["four_field_reference"]
        print(f"   four-field reference per-prompt: {ref['per_prompt']['n_negative']}/{ref['per_prompt']['n_prompts']}")


if __name__ == "__main__":
    main()
