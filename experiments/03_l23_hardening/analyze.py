"""
Experiment 03 — analysis (executes analysis_plan.md end to end).

Builds the pre-registered composite metric, fits the primary LMM dose-response,
runs the guidance-matching de-confound (partial correlation + matched-quality
two-arm contrast), computes inter-rater reliability (Claude vs Qwen; human vs
each), applies Benjamini-Hochberg to the secondary per-field family, and emits a
verdict against the decision table. Every number is regenerated from the judge /
quality / human JSON in results-local (Methodology Section 6: no trust-me plots).

Usage:
    python analyze.py --model sdxl --plot
    python analyze.py --model sd35 --plot
    python analyze.py --both --plot        # both models + writes combined verdict
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import rubric as R
import statlib as S

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results-local"
INT = list(R.INT_FIELDS)          # reduplication, fragmentation, condensation, distortion
ALL = list(R.ALL_FIELDS)          # + tiling


# ------------------------------- data loading ----------------------------------


def _load_judge(model_dir: Path, fname: str) -> dict:
    p = model_dir / fname
    if not p.exists():
        raise SystemExit(f"missing {p} — run the judge first")
    return json.loads(p.read_text()).get("images", {})


def _load_records(model: str) -> tuple[list[dict], list[dict], dict]:
    """Return (conditioned records, uncond records, notes). Each record has
    per-field judge-mean scores + quality components, for images scored by BOTH
    judges without error."""
    d = RESULTS / model
    A = _load_judge(d, "judgements_claude.json")
    B = _load_judge(d, "judgements_qwen.json")
    qpath = d / "quality.json"
    Q = json.loads(qpath.read_text()).get("per_image", {}) if qpath.exists() else {}

    notes = {"quality_present": bool(Q)}
    cond, uncond = [], []
    dropped = 0
    for fn, ra in A.items():
        rb = B.get(fn)
        meta = R.parse_filename(fn)
        if meta is None or rb is None or "error" in ra or "error" in rb:
            dropped += 1
            continue
        rec = {"filename": fn, **{k: meta[k] for k in ("prompt_id", "guidance", "seed", "kind")}}
        for f in ALL:
            rec[f + "_a"], rec[f + "_b"] = float(ra[f]), float(rb[f])
            rec[f] = (float(ra[f]) + float(rb[f])) / 2.0  # judge-mean
        q = Q.get(fn, {})
        rec["clip_iqa"] = q.get("clip_iqa")
        rec["aesthetic"] = q.get("aesthetic")
        (uncond if meta["kind"] == "unconditional" else cond).append(rec)
    notes["dropped_missing_or_error"] = dropped
    notes["n_conditioned"] = len(cond)
    notes["n_uncond"] = len(uncond)
    return cond, uncond, notes


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s else x * 0.0


def _build_metric(cond: list[dict], uncond: list[dict], notes: dict):
    """z-score fields within model; composite = mean of 4 z-fields. Quality Q =
    mean of standardized components (aesthetic optional). Uncond projected onto
    the conditioned z-scaling."""
    for f in INT:
        col = np.array([r[f] for r in cond], float)
        mu, sd = col.mean(), (col.std() or 1.0)
        for r in cond:
            r["z_" + f] = (r[f] - mu) / sd
        for r in uncond:
            r["z_" + f] = (r[f] - mu) / sd
    for r in cond + uncond:
        r["composite"] = float(np.mean([r["z_" + f] for f in INT]))

    # Quality Q (conditioned only; baselines not part of the curve)
    iqa = np.array([r["clip_iqa"] for r in cond if r["clip_iqa"] is not None], float)
    aes = [r["aesthetic"] for r in cond if r["aesthetic"] is not None]
    has_aes = len(aes) == len(cond) and len(cond) > 0
    notes["quality_uses_aesthetic"] = has_aes
    if len(iqa) == len(cond) and len(cond):
        ziqa = {r["filename"]: v for r, v in zip(cond, _z([r["clip_iqa"] for r in cond]))}
        if has_aes:
            zaes = {r["filename"]: v for r, v in zip(cond, _z([r["aesthetic"] for r in cond]))}
            for r in cond:
                r["Q"] = float((ziqa[r["filename"]] + zaes[r["filename"]]) / 2)
        else:
            for r in cond:
                r["Q"] = float(ziqa[r["filename"]])
        notes["quality_available"] = True
    else:
        for r in cond:
            r["Q"] = None
        notes["quality_available"] = False


# --------------------------------- LMM -----------------------------------------


def _lmm_slope(cond: list[dict]) -> dict:
    """Primary endpoint: composite ~ guidance_std + (1|prompt) + (1|seed)."""
    g = np.array([r["guidance"] for r in cond], float)
    g_std = _z(g)
    try:
        import pandas as pd
        import statsmodels.formula.api as smf

        df = pd.DataFrame({
            "composite": [r["composite"] for r in cond],
            "guidance_std": g_std,
            "prompt": [r["prompt_id"] for r in cond],
            "seed": [str(r["seed"]) for r in cond],
            "grp": 0,
        })
        vcf = {"prompt": "0 + C(prompt)", "seed": "0 + C(seed)"}
        md = smf.mixedlm("composite ~ guidance_std", df, groups="grp",
                         vc_formula=vcf, re_formula="0")
        mdf = md.fit(reml=True, method="lbfgs", maxiter=200)
        ci = mdf.conf_int().loc["guidance_std"]
        return {"method": "statsmodels MixedLM (crossed prompt+seed RE)",
                "slope_standardized": float(mdf.fe_params["guidance_std"]),
                "ci95": [float(ci[0]), float(ci[1])],
                "p": float(mdf.pvalues["guidance_std"])}
    except Exception as e:  # noqa: BLE001
        # Fallback: within-prompt-centered OLS slope (removes prompt baselines by
        # hand). Reported as a fallback if statsmodels is unavailable.
        comp = np.array([r["composite"] for r in cond], float)
        by_p = defaultdict(list)
        for i, r in enumerate(cond):
            by_p[r["prompt_id"]].append(i)
        centered = comp.copy()
        for idxs in by_p.values():
            centered[idxs] -= centered[idxs].mean()
        x = g_std - g_std.mean()
        slope = float((x * centered).sum() / (x**2).sum()) if (x**2).sum() else 0.0
        return {"method": f"fallback within-prompt OLS ({type(e).__name__}: {e})",
                "slope_standardized": slope, "ci95": None,
                "p": float(S.spearman_perm_p(g, centered))}


# ------------------------------ guidance matching ------------------------------


def _deconfound(cond: list[dict], notes: dict) -> dict:
    if not notes.get("quality_available"):
        return {"computable": False, "reason": "quality.json missing/incomplete"}
    comp = np.array([r["composite"] for r in cond], float)
    g = np.array([r["guidance"] for r in cond], float)
    q = np.array([r["Q"] for r in cond], float)
    rho = S.partial_spearman(comp, g, q)
    lo, hi = S.partial_spearman_ci(comp, g, q)
    raw = S.spearman(comp, g)
    return {"computable": True, "raw_spearman_composite_vs_g": round(raw, 4),
            "partial_spearman_controlling_Q": round(rho, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "ci_excludes_zero": (lo < 0 and hi < 0) or (lo > 0 and hi > 0)}


def _matched_quality_arms(cond: list[dict], notes: dict) -> dict:
    if not notes.get("quality_available"):
        return {"computable": False, "reason": "quality.json missing/incomplete"}
    low = [r for r in cond if r["guidance"] <= 3]
    high = [r for r in cond if r["guidance"] > 8]
    if not low or not high:
        return {"computable": False, "reason": "one arm empty"}
    ql, qh = [r["Q"] for r in low], [r["Q"] for r in high]
    band = (max(min(ql), min(qh)), min(max(ql), max(qh)))
    if band[0] >= band[1]:
        return {"computable": False, "reason": "quality ranges do not overlap",
                "low_arm_Q_range": [min(ql), max(ql)], "high_arm_Q_range": [min(qh), max(qh)]}
    lo_c = np.array([r["composite"] for r in low if band[0] <= r["Q"] <= band[1]])
    hi_c = np.array([r["composite"] for r in high if band[0] <= r["Q"] <= band[1]])
    if len(lo_c) < 3 or len(hi_c) < 3:
        return {"computable": False, "reason": "too few images in the overlap band",
                "n_low": len(lo_c), "n_high": len(hi_c)}
    delta = S.cliffs_delta(lo_c, hi_c)
    ci = S.bootstrap_delta_ci(lo_c, hi_c)
    return {"computable": True, "overlap_Q_band": [round(band[0], 3), round(band[1], 3)],
            "n_low": len(lo_c), "n_high": len(hi_c),
            "cliffs_delta_low_vs_high": round(delta, 4),
            "ci95": [round(ci[0], 4), round(ci[1], 4)],
            "interpretation": "positive = more L2/3 at the low-g arm at matched quality "
                              "(effect specific to under-conditioning, not generic quality loss)"}


# ------------------------------ inter-rater ------------------------------------


def _reliability(cond: list[dict], uncond: list[dict]) -> dict:
    imgs = cond + uncond
    out = {"n_images": len(imgs), "per_field": {}}
    field_kappas = []
    for f in ALL:
        q = 2 if f == "tiling" else 4
        a = [int(round(r[f + "_a"])) for r in imgs]
        b = [int(round(r[f + "_b"])) for r in imgs]
        k = S.weighted_cohens_kappa(a, b, q)
        out["per_field"][f] = {
            "weighted_kappa": round(k, 4),
            "gwet_ac2": round(S.gwet_ac2(a, b, q), 4),
            "percent_agreement": round(S.percent_agreement(a, b), 4),
        }
        if f in INT:
            field_kappas.append(k)
    out["composite_weighted_kappa"] = round(float(np.nanmean(field_kappas)), 4)
    return out


def _human_reliability(model_records: dict[str, list[dict]]) -> dict:
    sub_p, rat_p = HERE / "human_subset.json", HERE / "human_ratings.json"
    if not (sub_p.exists() and rat_p.exists()):
        return {"available": False}
    subset = {s["blind_id"]: s for s in json.loads(sub_p.read_text())["items"]}
    ratings = json.loads(rat_p.read_text())
    # index VLM judge-mean scores by (model, filename)
    idx = {(m, r["filename"]): r for m, recs in model_records.items() for r in recs}
    pairs_h_a = {f: ([], []) for f in ALL}  # human vs claude
    pairs_h_b = {f: ([], []) for f in ALL}  # human vs qwen
    used = 0
    for bid, hrec in ratings.items():
        s = subset.get(bid)
        if not s:
            continue
        vrec = idx.get((s["model"], s["filename"]))
        if not vrec:
            continue
        used += 1
        for f in ALL:
            pairs_h_a[f][0].append(int(hrec[f]))
            pairs_h_a[f][1].append(int(round(vrec[f + "_a"])))
            pairs_h_b[f][0].append(int(hrec[f]))
            pairs_h_b[f][1].append(int(round(vrec[f + "_b"])))
    out = {"available": True, "n_rated_used": used, "human_vs_claude": {}, "human_vs_qwen": {}}
    for f in ALL:
        q = 2 if f == "tiling" else 4
        out["human_vs_claude"][f] = round(S.weighted_cohens_kappa(*pairs_h_a[f], q), 4)
        out["human_vs_qwen"][f] = round(S.weighted_cohens_kappa(*pairs_h_b[f], q), 4)
    return out


# ------------------------- per-prompt / per-field ------------------------------


def _per_prompt_slopes(cond: list[dict]) -> dict:
    by_p = defaultdict(list)
    for r in cond:
        by_p[r["prompt_id"]].append(r)
    slopes = {}
    for pid, recs in by_p.items():
        g = np.array([r["guidance"] for r in recs], float)
        c = np.array([r["composite"] for r in recs], float)
        slopes[pid] = round(S.spearman(c, g), 4)
    n_neg = sum(1 for v in slopes.values() if v < 0)
    return {"per_prompt_spearman_composite_vs_g": slopes,
            "n_negative": n_neg, "n_prompts": len(slopes)}


def _per_field_dose(cond: list[dict]) -> dict:
    g = np.array([r["guidance"] for r in cond], float)
    fields = {}
    pvals = []
    for f in INT:
        x = np.array([r[f] for r in cond], float)
        rho = S.spearman(x, g)
        p = S.spearman_perm_p(g, x)
        fields[f] = {"spearman_rho_vs_g": round(rho, 4), "perm_p": round(p, 4)}
        pvals.append(p)
    return fields, pvals


def _baseline_composite(uncond: list[dict]) -> float | None:
    if not uncond:
        return None
    return round(float(np.mean([r["composite"] for r in uncond])), 4)


# ---------------------------------- verdict ------------------------------------


def _verdict(model_out: dict, prereg: dict) -> str:
    cf = prereg["confirm"]
    lmm = model_out["primary_lmm"]
    dec = model_out["deconfound"]
    rel = model_out["reliability"]
    slope = lmm["slope_standardized"]
    ci = lmm.get("ci95")
    kappa = rel["composite_weighted_kappa"]

    if kappa < cf["inter_judge_composite_weighted_kappa_min"]:
        return "inconclusive-null (judges disagree: composite weighted kappa < 0.4)"
    if ci is not None and ci[0] <= 0 <= ci[1]:
        return "null (LMM composite slope CI includes 0)"
    if slope > cf["lmm_composite_slope_standardized_max"]:
        return "null (LMM composite slope not sufficiently negative)"
    if dec.get("computable"):
        if abs(dec["partial_spearman_controlling_Q"]) < prereg["null"]["quality_controlled_partial_abs_rho_below"]:
            return "null (effect vanishes controlling for quality — it was under-conditioning)"
        if dec["partial_spearman_controlling_Q"] > cf["quality_controlled_partial_rho_max"] or not dec["ci_excludes_zero"]:
            return "null (quality-controlled partial correlation fails confirm threshold)"
    else:
        return "provisional-confirm-pending-quality (LMM ok; run quality.py for the de-confound gate)"
    return "confirm (per-model): negative LMM slope survives quality control at adequate kappa"


# ----------------------------------- run ---------------------------------------


def analyze_model(model: str, prereg: dict) -> dict:
    cond, uncond, notes = _load_records(model)
    if not cond:
        raise SystemExit(f"no dual-judged conditioned images for {model}")
    _build_metric(cond, uncond, notes)

    fields, pvals = _per_field_dose(cond)
    out = {
        "model": model, "rubric_version": prereg["rubric_version"], "notes": notes,
        "primary_lmm": _lmm_slope(cond),
        "deconfound": _deconfound(cond, notes),
        "matched_quality_arms": _matched_quality_arms(cond, notes),
        "reliability": _reliability(cond, uncond),
        "per_prompt": _per_prompt_slopes(cond),
        "per_field_dose_response": fields,
        "_field_pvals": pvals,
        "baseline_composite_uncond": _baseline_composite(uncond),
        "composite_mean_by_guidance": {
            str(g): round(float(np.mean([r["composite"] for r in cond if r["guidance"] == g])), 4)
            for g in sorted({r["guidance"] for r in cond})},
        "quality_mean_by_guidance": ({
            str(g): round(float(np.mean([r["Q"] for r in cond if r["guidance"] == g])), 4)
            for g in sorted({r["guidance"] for r in cond})} if notes.get("quality_available") else None),
    }
    out["_records"] = cond  # kept in-memory for plotting; stripped before JSON write
    return out


def _apply_bh_and_verdict(outs: dict[str, dict], prereg: dict) -> None:
    # BH across the secondary family: 4 fields x len(models)
    pvals, keys = [], []
    for m, o in outs.items():
        for f, p in zip(INT, o["_field_pvals"]):
            pvals.append(p)
            keys.append((m, f))
    q = S.benjamini_hochberg(pvals) if pvals else []
    for (m, f), qv in zip(keys, q):
        outs[m]["per_field_dose_response"][f]["bh_q"] = round(qv, 4)
    for m, o in outs.items():
        o.pop("_field_pvals", None)
        o["verdict"] = _verdict(o, prereg)


def _plots(model: str, out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = out["_records"]
    fig_dir = RESULTS / model / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    gs = sorted({r["guidance"] for r in recs})

    # (1) composite dose-response with faint per-prompt lines + quality curve twin
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    by_p = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by_p[r["prompt_id"]][r["guidance"]].append(r["composite"])
    for pid, gm in by_p.items():
        ys = [np.mean(gm[g]) for g in gs if g in gm]
        ax.plot([g for g in gs if g in gm], ys, "-", color="#94a3b8", lw=1, alpha=0.6)
    comp = [out["composite_mean_by_guidance"][str(g)] for g in gs]
    ax.plot(gs, comp, "o-", color="#0d9488", lw=2.5, ms=6, label="composite (all prompts)")
    ax.set_xlabel("guidance"); ax.set_ylabel("composite L2/3 (z)")
    slope = out["primary_lmm"]["slope_standardized"]
    ax.set_title(f"Exp 03 — {model}: composite dose-response (LMM β={slope:+.2f})")
    if out.get("quality_mean_by_guidance"):
        ax2 = ax.twinx()
        qy = [out["quality_mean_by_guidance"][str(g)] for g in gs]
        ax2.plot(gs, qy, "s--", color="#ef4444", lw=1.6, label="quality Q")
        ax2.set_ylabel("quality Q (z)", color="#ef4444")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(fig_dir / "composite_dose_response.png", dpi=150,
                                    bbox_inches="tight"); plt.close(fig)

    # (2) per-field curves
    fig, axes = plt.subplots(1, len(INT), figsize=(4 * len(INT), 3.4), sharex=True)
    for ax, f in zip(axes, INT):
        ys = [np.mean([r[f] for r in recs if r["guidance"] == g]) for g in gs]
        rho = out["per_field_dose_response"][f]["spearman_rho_vs_g"]
        qv = out["per_field_dose_response"][f].get("bh_q")
        ax.plot(gs, ys, "o-", color="#0d9488", lw=2)
        ax.set_title(f"{f}\nρ={rho:+.2f}" + (f" q={qv:.3f}" if qv is not None else ""))
        ax.set_xlabel("guidance"); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Exp 03 — {model}: Klüver L2/3 fields (judge-mean)", y=1.05)
    fig.tight_layout(); fig.savefig(fig_dir / "per_field_curves.png", dpi=150,
                                    bbox_inches="tight"); plt.close(fig)

    # (3) reliability bars
    fig, ax = plt.subplots(figsize=(6, 3.6))
    rel = out["reliability"]["per_field"]
    x = np.arange(len(ALL))
    ax.bar(x - 0.2, [rel[f]["weighted_kappa"] for f in ALL], 0.4, label="weighted κ", color="#0d9488")
    ax.bar(x + 0.2, [rel[f]["gwet_ac2"] for f in ALL], 0.4, label="Gwet AC2", color="#f59e0b")
    ax.axhline(0.4, color="#64748b", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(ALL, rotation=20, ha="right")
    ax.set_title(f"Exp 03 — {model}: Claude vs Qwen agreement"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "reliability.png", dpi=150,
                                    bbox_inches="tight"); plt.close(fig)
    print(f"[analyze] figures -> {fig_dir}")


def main(args: argparse.Namespace) -> None:
    prereg = R.load_prereg()
    models = ["sdxl", "sd35"] if args.both else [args.model]
    outs = {m: analyze_model(m, prereg) for m in models}
    # human reliability uses records from all analyzed models
    human = _human_reliability({m: o["_records"] for m, o in outs.items()})
    _apply_bh_and_verdict(outs, prereg)

    for m, o in outs.items():
        if args.plot:
            _plots(m, o)
        o.pop("_records", None)
        o["human_reliability"] = human if human.get("available") else {"available": False}
        (RESULTS / m / "l23_report.json").write_text(json.dumps(o, indent=2))
        print(f"\n=== {m} ===")
        print(json.dumps({"verdict": o["verdict"],
                          "lmm": o["primary_lmm"],
                          "deconfound": o["deconfound"],
                          "composite_kappa": o["reliability"]["composite_weighted_kappa"],
                          "per_prompt_negative": f"{o['per_prompt']['n_negative']}/{o['per_prompt']['n_prompts']}"},
                         indent=2))
        print(f"[analyze] wrote {RESULTS / m / 'l23_report.json'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 03 analysis (LMM · de-confound · κ · BH).")
    p.add_argument("--model", default="sdxl", choices=["sdxl", "sd35"])
    p.add_argument("--both", action="store_true", help="analyze both models")
    p.add_argument("--plot", action="store_true")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
