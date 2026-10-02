"""
Exp 03 sensitivity analyses, round 2. Pre-specified in posthoc_sensitivity2_prespec.md
(2026-09-17). EXPLORATORY: same data, same panel, no gate.

  S3  model x guidance interaction (pooled LMM), on the z-composite and on the raw composite
  S4  random slopes by prompt
  S5  cluster bootstrap (by seed, by prompt) for Spearman and the LMM slope
  S6  Claude vs Qwen3-VL-32B kappa on the 28 human-subset images

Usage: python posthoc_sensitivity2.py --judges claude,qwen
Writes posthoc_sensitivity2.json.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import analyze as A
import statlib as S

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PREREG = json.loads((HERE / "preregistration.json").read_text())
INT = list(A.INT)


def _df(cond: list[dict], model: str) -> pd.DataFrame:
    g = np.array([r["guidance"] for r in cond], float)
    return pd.DataFrame({
        "composite": [r["composite"] for r in cond],
        "raw_composite": [float(np.mean([r[f] for f in INT])) for r in cond],
        "guidance": g, "guidance_std": A._z(g),
        "prompt": [r["prompt_id"] for r in cond],
        "seed": [str(r["seed"]) for r in cond],
        "model": model, "grp": 0,
    })


def _fit_slope(df: pd.DataFrame, y: str = "composite") -> dict:
    vcf = {"prompt": "0 + C(prompt)", "seed": "0 + C(seed)"}
    md = smf.mixedlm(f"{y} ~ guidance_std", df, groups="grp", vc_formula=vcf, re_formula="0")
    m = md.fit(reml=True, method="lbfgs", maxiter=200)
    ci = m.conf_int().loc["guidance_std"]
    return {"slope_standardized": float(m.fe_params["guidance_std"]),
            "ci95": [float(ci[0]), float(ci[1])], "p": float(m.pvalues["guidance_std"])}


# ------------------------------------------------------------------ S3
def s3_interaction(dfs: dict[str, pd.DataFrame]) -> dict:
    pooled = pd.concat(dfs.values(), ignore_index=True)
    pooled["grp"] = 0
    # guidance_std is z-scored within model already (same grid, so identical); model coded
    # with SD 3.5 as reference so the interaction reads "SDXL slope minus SD 3.5 slope".
    pooled["model"] = pd.Categorical(pooled["model"], categories=["sd35", "sdxl"])
    out = {}
    for label, y in (("z_composite_prereg", "composite"), ("raw_composite", "raw_composite")):
        vcf = {"prompt": "0 + C(prompt)", "seed": "0 + C(seed)"}
        md = smf.mixedlm(f"{y} ~ guidance_std * model", pooled, groups="grp",
                         vc_formula=vcf, re_formula="0")
        m = md.fit(reml=True, method="lbfgs", maxiter=300)
        term = [k for k in m.fe_params.index if ":" in k][0]
        ci = m.conf_int().loc[term]
        out[label] = {"interaction_term": term,
                      "slope_sd35": float(m.fe_params["guidance_std"]),
                      "interaction_sdxl_minus_sd35": float(m.fe_params[term]),
                      "ci95": [float(ci[0]), float(ci[1])], "p": float(m.pvalues[term]),
                      "n": int(len(pooled))}
    return out


# ------------------------------------------------------------------ S4
def s4_random_slopes(df: pd.DataFrame) -> dict:
    md = smf.mixedlm("composite ~ guidance_std", df, groups="prompt", re_formula="~guidance_std")
    m = md.fit(reml=True, method="lbfgs", maxiter=300)
    ci = m.conf_int().loc["guidance_std"]
    cov = m.cov_re
    sd_slope = float(np.sqrt(cov.loc["guidance_std", "guidance_std"]))
    return {"method": "statsmodels MixedLM, groups=prompt, random intercept + slope; seed RE dropped",
            "slope_standardized": float(m.fe_params["guidance_std"]),
            "ci95": [float(ci[0]), float(ci[1])], "p": float(m.pvalues["guidance_std"]),
            "random_slope_sd": sd_slope,
            "random_intercept_sd": float(np.sqrt(cov.iloc[0, 0])),
            "converged": bool(m.converged)}


# ------------------------------------------------------------------ S5
def s5_cluster_bootstrap(df: pd.DataFrame, n_spear: int = 500, n_lmm: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for cluster in ("seed", "prompt"):
        levels = sorted(df[cluster].unique())
        rhos, slopes = [], []
        for b in range(n_spear):
            pick = rng.choice(levels, size=len(levels), replace=True)
            sub = pd.concat([df[df[cluster] == lv] for lv in pick], ignore_index=True)
            rhos.append(S.spearman(sub["composite"].to_numpy(), sub["guidance"].to_numpy()))
            if b < n_lmm:
                # relabel duplicated clusters so the RE structure stays sane
                sub = sub.copy()
                sub[cluster] = np.repeat([f"{lv}_{i}" for i, lv in enumerate(pick)],
                                         [int((df[cluster] == lv).sum()) for lv in pick])
                try:
                    slopes.append(_fit_slope(sub)["slope_standardized"])
                except Exception:  # noqa: BLE001
                    pass
        out[f"by_{cluster}"] = {
            "n_clusters": len(levels), "n_resamples_spearman": n_spear, "n_resamples_lmm": len(slopes),
            "spearman_ci95": [float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))],
            "lmm_slope_ci95": [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))],
        }
    out["point"] = {"spearman": S.spearman(df["composite"].to_numpy(), df["guidance"].to_numpy()),
                    "lmm_slope": _fit_slope(df)["slope_standardized"]}
    return out


# ------------------------------------------------------------------ S6
def s6_same28(records: dict[str, list[dict]], sfx: tuple[str, str]) -> dict:
    subset = json.loads((HERE / "human_subset.json").read_text())["items"]
    idx = {(m, r["filename"]): r for m, recs in records.items() for r in recs}
    pairs = {f: ([], []) for f in INT + ["tiling"]}
    used = 0
    for it in subset:
        r = idx.get((it["model"], it["filename"]))
        if not r:
            continue
        used += 1
        for f in pairs:
            pairs[f][0].append(int(round(r[f + "_" + sfx[0]])))
            pairs[f][1].append(int(round(r[f + "_" + sfx[1]])))
    q = lambda f: 2 if f == "tiling" else 4
    wk = {f: round(S.weighted_cohens_kappa(*pairs[f], q(f)), 4) for f in pairs}
    return {"n_images": used, "judges": ["claude", "qwen"],
            "weighted_kappa": wk,
            "mean_per_field_kappa": round(float(np.nanmean([wk[f] for f in INT])), 4),
            "composite_bootstrap": S.composite_kappa_ci([pairs[f] for f in INT], 4),
            "gwet_ac2": {f: round(S.gwet_ac2(*pairs[f], q(f)), 4) for f in pairs},
            "percent_agreement": {f: round(S.percent_agreement(*pairs[f]), 4) for f in pairs}}


# ------------------------------------------------------------------ S7, S8 (added 2026-09-25, external review)
def s7_trimmed_arm_balance(cond: list[dict]) -> dict:
    """Residual quality imbalance inside the pre-registered 'matched-quality' arms. The arms
    are common-support trimmed (both arms restricted to the overlapping Q band), not matched:
    this reports the standardised mean difference in Q that remains between them."""
    low = [r for r in cond if r["guidance"] <= 3]; high = [r for r in cond if r["guidance"] > 8]
    ql, qh = [r["Q"] for r in low], [r["Q"] for r in high]
    band = (max(min(ql), min(qh)), min(max(ql), max(qh)))
    lo = np.array([r["Q"] for r in low if band[0] <= r["Q"] <= band[1]])
    hi = np.array([r["Q"] for r in high if band[0] <= r["Q"] <= band[1]])
    smd = float((hi.mean() - lo.mean()) / np.sqrt((lo.var(ddof=1) + hi.var(ddof=1)) / 2))
    return {"n_low": int(len(lo)), "n_high": int(len(hi)), "band": [float(band[0]), float(band[1])],
            "mean_Q_low": float(lo.mean()), "mean_Q_high": float(hi.mean()),
            "residual_Q_smd_high_minus_low": smd,
            "note": "arms are common-support trimmed, not balanced; a positive SMD means the high-guidance arm is still of higher quality"}


def s8_partial_rho_prompt_bootstrap(cond: list[dict], n: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    prompts = sorted(set(r["prompt_id"] for r in cond)); byp = {p: [r for r in cond if r["prompt_id"] == p] for p in prompts}
    def part(rs):
        return S.partial_spearman(np.array([r["composite"] for r in rs]), np.array([r["guidance"] for r in rs]), np.array([r["Q"] for r in rs]))
    vals = []
    for _ in range(n):
        pick = rng.choice(prompts, len(prompts), replace=True)
        vals.append(part([r for p in pick for r in byp[p]]))
    return {"point": float(part(cond)), "n_resamples": n, "n_clusters": len(prompts),
            "ci95_prompt_cluster": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="claude,qwen")
    args = ap.parse_args()
    judges = args.judges.split(",")
    dfs, recs, sfx = {}, {}, None
    for m in PREREG["models"]["confirmatory"]:
        cond, uncond, notes = A._load_records(m, judges)
        A._build_metric(cond, uncond, notes)
        dfs[m] = _df(cond, m)
        recs[m] = cond
        sfx = tuple(s for (s, _n) in notes["judges"])[:2]
    report = {"label": "EXPLORATORY (posthoc_sensitivity2_prespec.md, 2026-09-17)",
              "S3_model_by_guidance_interaction": s3_interaction(dfs),
              "models": {}}
    for m, df in dfs.items():
        report["models"][m] = {"n_conditioned": int(len(df)),
                               "S4_random_slopes_by_prompt": s4_random_slopes(df),
                               "S5_cluster_bootstrap": s5_cluster_bootstrap(df),
                               "S7_trimmed_arm_balance": s7_trimmed_arm_balance(recs[m]),
                               "S8_partial_rho_prompt_bootstrap": s8_partial_rho_prompt_bootstrap(recs[m])}
    report["S6_model_vs_model_on_human_subset"] = s6_same28(recs, sfx)
    (HERE / "posthoc_sensitivity2.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "models"}, indent=1)[:3000])
    for m, o in report["models"].items():
        print(m, json.dumps(o, indent=1)[:1500])


if __name__ == "__main__":
    main()
