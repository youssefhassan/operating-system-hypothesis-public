"""
Experiment 03b — confirmatory analysis of the Suzuki three-axis scale.

Pre-registration: `preregistration_axes.json`. Deliberately a SEPARATE script from
`analyze.py`: that file implements the Klüver pre-registration and its confirmatory
path must not change while a second scale is added beside it.

Endpoints, all requiring BOTH models:

  P1 replication   LMM standardized slope of VERIDICALITY on guidance >= +0.20,
                   CI excluding 0. Direction inherited from Exp 01 (rho +0.67).

  P2 dissociation  partial Spearman(veridicality, guidance | Klüver composite)
                   >= +0.20, bootstrap CI excluding 0.
                   THE novel endpoint. If the global scale merely restates the
                   local one, partialling out the Klüver composite removes the
                   guidance signal and this goes to zero. A surviving association
                   is the quantitative form of "neither scale contains the other".

  P3 reliability   inter-judge quadratic-weighted kappa on the three graded axes
                   >= 0.40.

Secondary (BH-corrected): spontaneity and complexity slopes, coherent_scene,
per-prompt generality of the veridicality slope.

Pre-specified exploratory (NOT gates): E1 overshoot, E2 scale correlation,
E4 shape on SD 3.5.

The uncond arm is EXCLUDED from every spontaneity analysis: with no prompt the
rubric scores spontaneity 3 by construction, so it carries no information.

    python analyze_axes.py --judges claude,qwen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze as A
import rubric_axes as RA
import statlib as S

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results-local"
AXES = list(RA.INT_FIELDS)              # veridicality, spontaneity, complexity
PREREG = json.loads((HERE / "preregistration_axes.json").read_text())


def _load_axes(model: str, judges: list[str]) -> dict[str, dict]:
    """filename -> {field: judge-mean, field_<judge>: raw}. Images scored without
    error by EVERY present judge, matching analyze.py's completeness rule."""
    per_judge = {}
    for j in judges:
        p = RESULTS / model / f"judgements_{j}_axes.json"
        if not p.exists():
            raise SystemExit(f"missing {p} — run the axes judges first (RUN_AXES.md)")
        per_judge[j] = json.loads(p.read_text())["images"]

    out: dict[str, dict] = {}
    for fn in per_judge[judges[0]]:
        recs = {j: per_judge[j].get(fn) for j in judges}
        if any(r is None or "error" in r for r in recs.values()):
            continue
        rec = {}
        for f in AXES + list(RA.BIN_FIELDS):
            for j in judges:
                rec[f"{f}_{j}"] = float(recs[j][f])
            rec[f] = sum(float(recs[j][f]) for j in judges) / len(judges)
        out[fn] = rec
    return out


def _merge(model: str, judges: list[str]):
    """Klüver records (for the composite P2 controls for) joined to axes scores."""
    cond, uncond, notes = A._load_records(model, judges)
    A._build_metric(cond, uncond, notes)          # adds r['composite'] = Klüver
    axes = _load_axes(model, judges)

    merged = []
    for r in cond:
        a = axes.get(r["filename"])
        if a is None:
            continue
        rec = dict(r)
        rec["kluver_composite"] = r["composite"]
        rec.update(a)
        merged.append(rec)
    return merged, notes


def _z(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s else v * 0.0


def _field_lmm(recs: list[dict], field: str) -> dict:
    """Fit the registered LMM after standardising one axes field."""
    vals = _z([r[field] for r in recs])
    rows = [dict(r, composite=float(v)) for r, v in zip(recs, vals)]
    return A._lmm_slope(rows)


def analyze_model(model: str, judges: list[str]) -> dict:
    recs, notes = _merge(model, judges)
    g = np.array([r["guidance"] for r in recs], float)

    # ---- P1: veridicality dose-response -------------------------------------
    p1 = _field_lmm(recs, "veridicality")

    # ---- P2: dissociation ---------------------------------------------------
    ver = np.array([r["veridicality"] for r in recs], float)
    kluver = np.array([r["kluver_composite"] for r in recs], float)
    p2_rho = S.partial_spearman(ver, g, kluver)
    p2_ci = S.partial_spearman_ci(ver, g, kluver)
    raw_rho = S.spearman(ver, g)

    # ---- P3: inter-judge reliability on the three graded axes ---------------
    per_field_kappa = {}
    for f in AXES:
        a = [int(round(r[f"{f}_{judges[0]}"])) for r in recs]
        b = [int(round(r[f"{f}_{judges[1]}"])) for r in recs]
        per_field_kappa[f] = round(S.weighted_cohens_kappa(a, b, 4), 4)
    p3 = round(float(np.mean(list(per_field_kappa.values()))), 4)

    # ---- secondary ----------------------------------------------------------
    secondary = {}
    for f in AXES[1:] + ["coherent_scene"]:        # spontaneity, complexity, coherent
        sub = recs if f != "spontaneity" else [r for r in recs if r["kind"] == "conditioned"]
        gg = np.array([r["guidance"] for r in sub], float)
        vv = np.array([r[f] for r in sub], float)
        secondary[f] = {
            "lmm": _field_lmm(sub, f),
            "spearman_vs_g": round(float(S.spearman(gg, vv)), 4),
            "n": len(sub),
        }

    # per-prompt generality of the veridicality slope
    per_prompt = {}
    for p in sorted({r["prompt_id"] for r in recs}):
        sub = [r for r in recs if r["prompt_id"] == p]
        per_prompt[p] = round(float(S.spearman(
            np.array([r["guidance"] for r in sub], float),
            np.array([r["veridicality"] for r in sub], float))), 4)

    # ---- pre-specified exploratory -----------------------------------------
    by_g = {}
    for gv in sorted({r["guidance"] for r in recs}):
        sub = [r["veridicality"] for r in recs if r["guidance"] == gv]
        by_g[str(gv)] = round(float(np.mean(sub)), 4)
    gs = sorted(by_g, key=float)
    peak = max(by_g, key=lambda k: by_g[k])
    e1 = {
        "veridicality_by_guidance": by_g,
        "peak_at_g": peak,
        "top_of_dial": gs[-1],
        "overshoot": by_g[gs[-1]] < by_g[peak],
        "drop_from_peak": round(by_g[peak] - by_g[gs[-1]], 4),
    }
    e2 = {"spearman_axes_vs_kluver_composite": round(float(S.spearman(ver, kluver)), 4)}

    # E3: the painterly confound. Exp 03's biggest construct-validity threat is
    # that low-g "distortion" is partly a change in RENDERING STYLE rather than
    # objecthood coming apart. Veridicality measures rendering style directly, so
    # if it accounts for the distortion signal, partialling it out should shrink
    # distortion's association with guidance toward zero.
    dist = np.array([r["distortion"] for r in recs], float)
    raw_dg = S.spearman(dist, g)
    part_dg = S.partial_spearman(dist, g, ver)
    part_ci = S.partial_spearman_ci(dist, g, ver)
    e3 = {
        "raw_spearman_distortion_vs_g": round(float(raw_dg), 4),
        "partial_controlling_veridicality": round(float(part_dg), 4),
        "ci95": [round(float(c), 4) for c in part_ci],
        "shrinkage": round(float(abs(raw_dg) - abs(part_dg)), 4),
        # Only meaningful when the partial is SMALLER than the raw. If controlling
        # for veridicality makes the association larger, that is suppression, not
        # confounding, and a "share explained" figure would be actively misleading.
        "proportional_attenuation_in_absolute_association": (
            round(float(1 - abs(part_dg) / abs(raw_dg)), 4)
            if raw_dg and abs(part_dg) < abs(raw_dg) else None),
        "pattern": ("attenuation: distortion and veridicality share rank association "
                    "with guidance; this is not a causal decomposition"
                    if abs(part_dg) < abs(raw_dg) else
                    "suppression: the conditional association is larger after adjustment "
                    "for veridicality; this is not a causal decomposition"),
    }

    # E4: does the global scale show a gradient where the local one showed a cliff?
    sub = [r for r in recs if r["guidance"] > 1.0]
    e4 = {
        "excluding_g1_spearman_veridicality_vs_g": round(float(S.spearman(
            np.array([r["guidance"] for r in sub], float),
            np.array([r["veridicality"] for r in sub], float))), 4),
        "n": len(sub),
    }

    return {
        "model": model, "n": len(recs), "judges": judges,
        "P1_veridicality_lmm": p1,
        "P2_dissociation": {
            "raw_spearman_veridicality_vs_g": round(float(raw_rho), 4),
            "partial_spearman_controlling_kluver": round(float(p2_rho), 4),
            "ci95": [round(float(c), 4) for c in p2_ci],
        },
        "P3_reliability": {"per_field": per_field_kappa, "composite_weighted_kappa": p3},
        "secondary": secondary,
        "per_prompt_veridicality_spearman": per_prompt,
        "n_prompts_positive": sum(v > 0 for v in per_prompt.values()),
        "exploratory": {"E1_overshoot": e1, "E2_scale_correlation": e2,
                        "E3_painterly_confound": e3, "E4_shape": e4},
    }


def _apply_bh_and_verdict(models: dict[str, dict]) -> None:
    model_names = list(models)

    # Registered primary family: one veridicality LMM per checkpoint.
    p1_q = S.benjamini_hochberg(
        [models[m]["P1_veridicality_lmm"]["p"] for m in model_names])
    for m, q in zip(model_names, p1_q):
        p1 = models[m]["P1_veridicality_lmm"]
        p1["bh_q"] = float(q)
        p1["bh_family_n"] = len(model_names)

    # Registered secondary family: three LMMs per checkpoint.
    keys = [(m, f) for m in model_names for f in models[m]["secondary"]]
    secondary_q = S.benjamini_hochberg(
        [models[m]["secondary"][f]["lmm"]["p"] for m, f in keys])
    for (m, f), q in zip(keys, secondary_q):
        models[m]["secondary"][f]["lmm"]["bh_q"] = float(q)
        models[m]["secondary"][f]["lmm"]["bh_family_n"] = len(keys)

    for m in model_names:
        r = models[m]
        p1 = r["P1_veridicality_lmm"]
        p2 = r["P2_dissociation"]
        p3 = r["P3_reliability"]["composite_weighted_kappa"]
        verdict = []
        verdict.append(
            "P1 pass" if (
                p1.get("converged", False)
                and p1["slope_standardized"] >= 0.20
                and p1["ci95"][0] > 0
                and p1["bh_q"] <= 0.05
            ) else "P1 FAIL")
        verdict.append(
            "P2 pass" if (
                p2["partial_spearman_controlling_kluver"] >= 0.20
                and p2["ci95"][0] > 0
            ) else "P2 FAIL")
        verdict.append("P3 pass" if p3 >= 0.40 else "P3 FAIL")
        r["verdict"] = verdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judges", default="claude,qwen")
    ap.add_argument("--models", default="sdxl,sd35")
    ap.add_argument("--out", default="axes_report.json")
    args = ap.parse_args()

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    report = {"prereg": PREREG["rubric_version"], "judges": judges, "models": {}}
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        report["models"][m] = analyze_model(m, judges)
    _apply_bh_and_verdict(report["models"])

    (HERE / args.out).write_text(json.dumps(report, indent=2))

    print(f"\n{'':14s} {'SDXL':>26s} {'SD 3.5':>26s}")
    def row(label, fn):
        vals = [fn(report["models"][m]) for m in ("sdxl", "sd35") if m in report["models"]]
        print(f"{label:14s} " + " ".join(f"{v:>26s}" for v in vals))

    row("P1 slope", lambda r: f"{r['P1_veridicality_lmm']['slope_standardized']:+.3f} "
                              f"[{r['P1_veridicality_lmm']['ci95'][0]:+.3f},{r['P1_veridicality_lmm']['ci95'][1]:+.3f}]")
    row("P2 partial", lambda r: f"{r['P2_dissociation']['partial_spearman_controlling_kluver']:+.3f} "
                                f"[{r['P2_dissociation']['ci95'][0]:+.3f},{r['P2_dissociation']['ci95'][1]:+.3f}]")
    row("  (raw rho)", lambda r: f"{r['P2_dissociation']['raw_spearman_veridicality_vs_g']:+.3f}")
    row("P3 kappa", lambda r: f"{r['P3_reliability']['composite_weighted_kappa']:.3f}")
    row("verdict", lambda r: ", ".join(r["verdict"]))
    print()
    for m, r in report["models"].items():
        e1 = r["exploratory"]["E1_overshoot"]
        print(f"{m}: veridicality by g {e1['veridicality_by_guidance']}")
        print(f"{m}: peak at g={e1['peak_at_g']}, top of dial g={e1['top_of_dial']}, "
              f"overshoot={e1['overshoot']} (drop {e1['drop_from_peak']})")
    print(f"\nwrote {HERE / args.out}")


if __name__ == "__main__":
    main()
