"""Exp 03 sensitivity analyses, round 3: answers to the xsci review of Preprint I v1.

Pre-specified in posthoc_review_xsci_prespec.md (2026-10-22). Exploratory; no gate.
  R1  within-judge z composite (z per judge and field, then average judges)
  R2  model x guidance interaction with checkpoint-specific prompt/seed intercepts
  R3  partial rho controlling CLIP-IQA and aesthetic as two covariates
  R4  prompt-generality tally on the five test prompts
  R6  quality-adjusted guidance slope inside the LMM
  R7  everything registered, with the 28 screening (= human subset) images dropped
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import analyze as A
import statlib as S

HERE = Path(__file__).resolve().parent
PREREG = json.loads((HERE / "preregistration.json").read_text())
INT = list(A.INT)
CONTROL = {p["id"] for p in PREREG["prompts"] if p.get("is_control")}


def _df(cond: list[dict], model: str) -> pd.DataFrame:
    g = np.array([r["guidance"] for r in cond], float)
    d = {"composite": [r["composite"] for r in cond],
         "guidance_std": A._z(g),
         "prompt": [r["prompt_id"] for r in cond],
         "seed": [str(r["seed"]) for r in cond],
         "model": model, "grp": 0}
    if all(r.get("Q") is not None for r in cond):
        d["Q"] = [r["Q"] for r in cond]
        d["z_iqa"] = A._z([r["clip_iqa"] for r in cond])
        d["z_aes"] = A._z([r["aesthetic"] for r in cond])
    return pd.DataFrame(d)


def _fit(formula: str, df: pd.DataFrame, vcf: dict, term: str) -> dict:
    md = smf.mixedlm(formula, df, groups="grp", vc_formula=vcf, re_formula="0")
    m, used = None, None
    for opt in ("lbfgs", "powell"):
        m = md.fit(reml=True, method=opt, maxiter=1000, disp=False)
        used = opt
        if m.converged:
            break
    ci = m.conf_int().loc[term]
    return {"formula": formula, "vc": list(vcf), "optimizer": used, "converged": bool(m.converged),
            "estimate": float(m.fe_params[term]), "ci95": [float(ci[0]), float(ci[1])],
            "p": float(m.pvalues[term]), "n": int(len(df))}


VCF = {"prompt": "0 + C(prompt)", "seed": "0 + C(seed)"}


# ------------------------------------------------------------------ R1
def r1_within_judge_z(cond: list[dict], sfx: tuple[str, str]) -> dict:
    c = copy.deepcopy(cond)
    for f in INT:
        per_judge = {}
        for s in sfx:
            col = np.array([r[f + "_" + s] for r in c], float)
            per_judge[s] = (col - col.mean()) / (col.std() or 1.0)
        for i, r in enumerate(c):
            r["z_" + f] = float(np.mean([per_judge[s][i] for s in sfx]))
    for r in c:
        r["composite"] = float(np.mean([r["z_" + f] for f in INT]))
    lmm = A._lmm_slope(c)
    pp = A._per_prompt_slopes(c)
    return {"lmm_slope": lmm["slope_standardized"], "ci95": lmm["ci95"], "p": lmm["p"],
            "converged": lmm["converged"], "optimizer": lmm["optimizer"],
            "per_prompt": pp, "judge_only_slopes_registered_composite": A._judge_only_slopes(cond, [(s, s) for s in sfx])}


# ------------------------------------------------------------------ R2
def r2_interaction(dfs: dict[str, pd.DataFrame]) -> dict:
    pooled = pd.concat(dfs.values(), ignore_index=True)
    pooled["grp"] = 0
    pooled["model"] = pd.Categorical(pooled["model"], categories=["sd35", "sdxl"])
    pooled["prompt_model"] = pooled["prompt"] + ":" + pooled["model"].astype(str)
    pooled["seed_model"] = pooled["seed"] + ":" + pooled["model"].astype(str)
    f = "composite ~ guidance_std * model"
    term = "guidance_std:model[T.sdxl]"
    shared = _fit(f, pooled, VCF, term)
    nested = _fit(f, pooled, {**VCF, "prompt_model": "0 + C(prompt_model)",
                              "seed_model": "0 + C(seed_model)"}, term)
    return {"interaction_sdxl_minus_sd35": {"shared_intercepts_as_published": shared,
                                            "checkpoint_specific_intercepts": nested}}


# ------------------------------------------------------------------ R3
def _partial_multi(y: np.ndarray, x: np.ndarray, Z: np.ndarray) -> float:
    ry, rx = S.ranks(y), S.ranks(x)
    RZ = np.column_stack([np.ones(len(y))] + [S.ranks(z) for z in Z.T])
    ey = ry - RZ @ np.linalg.lstsq(RZ, ry, rcond=None)[0]
    ex = rx - RZ @ np.linalg.lstsq(RZ, rx, rcond=None)[0]
    return float(np.corrcoef(ey, ex)[0, 1])


def r3_two_covariates(cond: list[dict], n_boot: int = 5000, seed: int = 0) -> dict:
    comp = np.array([r["composite"] for r in cond])
    g = np.array([r["guidance"] for r in cond], float)
    Z = np.column_stack([[r["clip_iqa"] for r in cond], [r["aesthetic"] for r in cond]]).astype(float)
    rho = _partial_multi(comp, g, Z)
    rng = np.random.default_rng(seed)
    n = len(cond)
    img = [_partial_multi(comp[i], g[i], Z[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    by_p = defaultdict(list)
    for i, r in enumerate(cond):
        by_p[r["prompt_id"]].append(i)
    pids = list(by_p)
    clus = []
    for _ in range(1000):
        idx = np.concatenate([by_p[p] for p in rng.choice(pids, len(pids))])
        clus.append(_partial_multi(comp[idx], g[idx], Z[idx]))
    single = {}
    for name, z in (("clip_iqa", Z[:, 0]), ("aesthetic", Z[:, 1])):
        lo, hi = S.partial_spearman_ci(comp, g, z)
        single[name] = {"rho": round(S.partial_spearman(comp, g, z), 4), "ci95_image": [round(lo, 4), round(hi, 4)]}
    return {"partial_rho_controlling_both": round(rho, 4),
            "ci95_image_bootstrap": [round(float(np.percentile(img, 2.5)), 4), round(float(np.percentile(img, 97.5)), 4)],
            "ci95_prompt_cluster_bootstrap": [round(float(np.percentile(clus, 2.5)), 4), round(float(np.percentile(clus, 97.5)), 4)],
            "single_covariate": single,
            "registered_Q": {"rho": round(S.partial_spearman(comp, g, np.array([r["Q"] for r in cond])), 4)}}


# ------------------------------------------------------------------ R4
def r4_tally(cond: list[dict]) -> dict:
    pp = A._per_prompt_slopes(cond)["per_prompt_spearman_composite_vs_g"]
    test = {k: v for k, v in pp.items() if k not in CONTROL}
    return {"per_prompt": pp, "control": {k: pp[k] for k in CONTROL},
            "registered_6_prompt_tally": f"{sum(v < 0 for v in pp.values())}/{len(pp)}",
            "five_test_prompt_tally": f"{sum(v < 0 for v in test.values())}/{len(test)}"}


# ------------------------------------------------------------------ R6
def r6_lmm_quality(df: pd.DataFrame) -> dict:
    return {"unadjusted": _fit("composite ~ guidance_std", df, VCF, "guidance_std"),
            "plus_Q": _fit("composite ~ guidance_std + Q", df, VCF, "guidance_std"),
            "plus_both_proxies": _fit("composite ~ guidance_std + z_iqa + z_aes", df, VCF, "guidance_std")}


# ------------------------------------------------------------------ R7
def r7_drop_screening(model: str, cond: list[dict], uncond: list[dict], judges: list[tuple[str, str]]) -> dict:
    subset = json.loads((HERE / "human_subset.json").read_text())["items"]
    drop = {it["filename"] for it in subset if it["model"] == model}
    c = [r for r in cond if r["filename"] not in drop]
    u = [r for r in uncond if r["filename"] not in drop]
    lmm = A._lmm_slope(c)
    comp = np.array([r["composite"] for r in c]); g = np.array([r["guidance"] for r in c], float)
    q = np.array([r["Q"] for r in c])
    lo, hi = S.partial_spearman_ci(comp, g, q)
    rel = A._reliability(c, u, judges)
    return {"n_dropped": len(cond) + len(uncond) - len(c) - len(u), "n_conditioned": len(c),
            "lmm_slope": lmm["slope_standardized"], "ci95": lmm["ci95"], "converged": lmm["converged"],
            "partial_rho_Q": round(S.partial_spearman(comp, g, q), 4), "partial_ci95": [round(lo, 4), round(hi, 4)],
            "tally": r4_tally(c),
            "composite_kappa_full_set": rel["composite_weighted_kappa"],
            "composite_kappa_ci": rel["composite_bootstrap"],
            "composite_kappa_conditioned_only": A._reliability(c, [], judges)["composite_weighted_kappa"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="claude,qwen")
    args = ap.parse_args()
    judges = args.judges.split(",")
    report = {"label": "EXPLORATORY (posthoc_review_xsci_prespec.md, 2026-10-22)", "models": {}}
    dfs = {}
    for m in PREREG["models"]["confirmatory"]:
        cond, uncond, notes = A._load_records(m, judges)
        A._build_metric(cond, uncond, notes)
        sfx = tuple(s for (s, _n) in notes["judges"])[:2]
        dfs[m] = _df(cond, m)
        reg = A._lmm_slope(cond)
        report["models"][m] = {
            "n_conditioned": len(cond),
            "registered_lmm_slope": {"slope": reg["slope_standardized"], "ci95": reg["ci95"]},
            "R1_within_judge_z_composite": r1_within_judge_z(cond, sfx),
            "R3_two_quality_covariates": r3_two_covariates(cond),
            "R4_prompt_tally": r4_tally(cond),
            "R6_lmm_with_quality": r6_lmm_quality(dfs[m]),
            "R7_drop_28_screening_images": r7_drop_screening(m, cond, uncond, notes["judges"]),
        }
    report["R2_interaction"] = r2_interaction(dfs)
    (HERE / "posthoc_review_xsci.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
