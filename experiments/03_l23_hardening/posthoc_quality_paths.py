"""Exp 03: POST-HOC / EXPLORATORY. Why the quality adjustment moves the two models apart.

Written 2026-09-28, after every confirmatory gate was computed and reported. It changes,
rescues or supersedes nothing in `analyze.py`. It answers one question the pre-registered
quality adjustment raised: partialling out the quality score Q weakened the composite-guidance
correlation on SDXL (-0.463 to -0.433) and strengthened it on SD 3.5 (-0.216 to -0.245).
The partial Spearman is built from three pairwise correlations,

    partial = (r_cg - r_cq * r_qg) / sqrt((1 - r_cq^2) (1 - r_qg^2))

with c = composite, g = guidance, q = Q. This file reports the three, splits the change
from raw to partial into the numerator term (what Q shares with both) and the denominator
term (the rescaling after g's quality-predicting part is removed), and bootstraps the
change itself so it can be read against noise.

Correction, 2026-09-29: the pooled composite-vs-Q correlation must not be read alone. On
SD 3.5 it is near zero (+0.04) because two opposite patterns cancel: across guidance
levels, lower quality goes with more breakdown; within each level from g = 2 up, higher
quality goes with more breakdown. Q acts as a suppressor there, which is why adjusting for
it slightly strengthens the correlation. An earlier reading ("Q is unrelated to the
composite on SD 3.5, so quality explains none of the effect") was wrong and was removed
from the paper. `within_guidance_level` and `between_level_means` below record the split.
The quality-component check is still cited by the paper's limitations.

Reuses `analyze.py`'s record loading and metric construction verbatim.

    python posthoc_quality_paths.py --judges claude,qwen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze as A
import statlib as S

HERE = Path(__file__).resolve().parent


def paths(cond: list[dict], n_boot: int = 5000, seed: int = 0) -> dict:
    c = np.array([r["composite"] for r in cond], float)
    g = np.array([r["guidance"] for r in cond], float)
    q = np.array([r["Q"] for r in cond], float)
    clip_iqa = np.array([r["clip_iqa"] for r in cond], float)
    aesthetic = np.array([r["aesthetic"] for r in cond], float)
    r_cg, r_qg, r_cq = S.spearman(c, g), S.spearman(q, g), S.spearman(c, q)
    partial = S.partial_spearman(c, g, q)
    numerator = r_cg - r_cq * r_qg
    denom = float(np.sqrt((1 - r_cq**2) * (1 - r_qg**2)))

    # image-level bootstrap of the change from raw to partial
    rng = np.random.default_rng(seed)
    n = len(c)
    deltas, rcqs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        deltas.append(S.partial_spearman(c[i], g[i], q[i]) - S.spearman(c[i], g[i]))
        rcqs.append(S.spearman(c[i], q[i]))
    d_lo, d_hi = np.percentile(deltas, [2.5, 97.5])
    q_lo, q_hi = np.percentile(rcqs, [2.5, 97.5])

    levels = sorted(set(g))
    within = {str(int(gl)): round(S.spearman(c[g == gl], q[g == gl]), 4) for gl in levels}
    above1 = g > 1
    between = S.spearman(np.array([c[g == gl].mean() for gl in levels]),
                         np.array([q[g == gl].mean() for gl in levels]))

    return {
        "n_images": n,
        "within_guidance_level": {
            "rho_composite_vs_Q_by_g": within,
            "rho_composite_vs_Q_g_above_1_pooled": round(S.spearman(c[above1], q[above1]), 4),
            "note": "composite vs Q inside each guidance level; read with between_level_means",
        },
        "between_level_means": {
            "rho_mean_composite_vs_mean_Q": round(float(between), 4),
            "n_levels": len(levels),
        },
        "rho_composite_vs_g": round(r_cg, 4),
        "rho_Q_vs_g": round(r_qg, 4),
        "rho_Q_vs_g_perm_p": round(S.spearman_perm_p(q, g), 4),
        "rho_composite_vs_Q": round(r_cq, 4),
        "rho_composite_vs_Q_ci95": [round(float(q_lo), 4), round(float(q_hi), 4)],
        "rho_composite_vs_Q_perm_p": round(S.spearman_perm_p(c, q), 4),
        "partial_spearman_controlling_Q": round(partial, 4),
        "change_raw_to_partial": round(partial - r_cg, 4),
        "change_raw_to_partial_ci95": [round(float(d_lo), 4), round(float(d_hi), 4)],
        "quality_component_check": {
            "rho_clip_iqa_vs_aesthetic": round(S.spearman(clip_iqa, aesthetic), 4),
            "partial_spearman_controlling_clip_iqa": round(
                S.partial_spearman(c, g, clip_iqa), 4),
            "partial_spearman_controlling_aesthetic": round(
                S.partial_spearman(c, g, aesthetic), 4),
            "note": "The components are separate no-reference proxies. Their agreement "
                    "and separate adjustments are construct checks, not a causal decomposition.",
        },
        "decomposition": {
            "numerator_shared_term": round(-r_cq * r_qg, 4),
            "numerator": round(numerator, 4),
            "denominator": round(denom, 4),
            "numerator_only_partial": round(numerator, 4),
            "denominator_only_partial": round(r_cg / denom, 4),
            "note": "Algebraic rank-correlation decomposition only: numerator_only = raw "
                    "minus the shared correlation term before rescaling; denominator_only = "
                    "raw rescaled as if Q had zero correlation with the composite.",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="claude,qwen")
    args = ap.parse_args()
    judges = [j.strip() for j in args.judges.split(",")]
    out = {"status": "POST-HOC / EXPLORATORY, written 2026-09-28; does not affect the confirmatory verdict",
           "judges": judges, "bootstrap": "image-level, 5000 resamples, seed 0", "models": {}}
    for m in ["sdxl", "sd35"]:
        cond, uncond, notes = A._load_records(m, judges)
        A._build_metric(cond, uncond, notes)
        out["models"][m] = paths(cond)
    (HERE / "posthoc_quality_paths.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
