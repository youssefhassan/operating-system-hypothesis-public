"""
Experiment 02 analysis: the pre-registered gates (score-vs-mu curve, blank
baseline, four classes, planform -> class confusion matrix) and the positive
control for the Exp 01 Level-1 judge added by the 2026-09-03 amendment.

Reads results-local/{metadata,judge_manifest,judgements}.json, writes
results-local/report.json and prints the tables.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OUT = HERE / "results-local"
PRE = json.loads((HERE / "preregistration.json").read_text())
CLASSES = ("lattice", "cobweb", "tunnel", "spiral")
POS_MU = {1.05, 1.1, 1.25, 1.5}
BLANK_MU = {0.8, 0.9}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (round(c - h, 3), round(c + h, 3))


def main() -> None:
    manifest = json.loads((OUT / "judge_manifest.json").read_text())
    J = json.loads((OUT / "judgements.json").read_text())
    model, rubric = J["claude_model"], J["rubric_version"]
    rows = []
    for it in manifest:
        j = J["images"].get(it["blind_id"])
        if not j or "error" in j:
            continue
        rows.append({**it, **j, "M": j["geometric_intensity"] / 3.0,
                     "structured": int(any(j[c] for c in CLASSES))})
    renders = [r for r in rows if r["kind"] == "render"]
    photos = [r for r in rows if r["kind"] == "photo"]
    n_err = len(manifest) - len(rows)

    # ---------------- curve: M and structured rate by regime x mu ----------------
    curve: dict = defaultdict(dict)
    for reg in sorted({r["regime"] for r in renders}):
        for m in sorted({r["mu_over_mu_c"] for r in renders}):
            sel = [r for r in renders if r["regime"] == reg and r["mu_over_mu_c"] == m]
            if sel:
                curve[reg][m] = {"n": len(sel), "M_mean": round(float(np.mean([r["M"] for r in sel])), 3),
                                 "structured_rate": round(float(np.mean([r["structured"] for r in sel])), 3),
                                 "classes": dict(Counter(c for r in sel for c in CLASSES if r[c]))}
    trend = {}
    rng = np.random.default_rng(0)
    for reg in curve:
        sel = [r for r in renders if r["regime"] == reg]
        mu = np.array([r["mu_over_mu_c"] for r in sel]); M = np.array([r["M"] for r in sel])
        rho = spearmanr(mu, M).correlation
        null = [spearmanr(mu, rng.permutation(M)).correlation for _ in range(5000)]
        p = float(np.mean([abs(x) >= abs(rho) for x in null]))
        trend[reg] = {"spearman_rho": round(float(rho), 3), "perm_p": round(p, 4), "n": len(sel)}

    # ---------------- pre-registered gates ----------------
    cf = PRE["confirm"]
    supra_M = max(v["M_mean"] for reg in curve for m, v in curve[reg].items() if m in POS_MU)
    blank_M = float(np.mean([r["M"] for r in renders if r["mu_over_mu_c"] == 0.8]))
    realized = {c: int(any(r[c] for r in renders if r["mu_over_mu_c"] in POS_MU)) for c in CLASSES}
    # confusion: predicted class (planform rule) x judged classes, supra-threshold only
    conf: dict = defaultdict(Counter)
    for r in renders:
        if r["mu_over_mu_c"] in POS_MU and r.get("predicted_class"):
            for c in CLASSES:
                if r[c]:
                    conf[r["predicted_class"]][c] += 1
            if not r["structured"]:
                conf[r["predicted_class"]]["none"] += 1
    def diag_ok(pred: str, cnt: Counter) -> bool:
        match = {"lattice": ("lattice", "cobweb"), "tunnel": ("tunnel",), "spiral": ("spiral",)}[pred]
        if not cnt:
            return False
        top = max(cnt.values())
        return max(cnt[c] for c in match) == top
    diag = {p: diag_ok(p, c) for p, c in conf.items()}
    gates = {
        "above_threshold_score": {"value": round(supra_M, 3), "min": cf["above_threshold_score_min"], "pass": supra_M >= cf["above_threshold_score_min"]},
        "blank_baseline_score": {"value": round(blank_M, 3), "max": cf["blank_baseline_score_max"], "pass": blank_M <= cf["blank_baseline_score_max"]},
        "all_four_classes_realized": {"value": realized, "pass": all(realized.values())},
        "confusion_diagonal_dominant": {"value": diag, "pass": bool(diag) and all(diag.values())},
    }
    gates["verdict"] = "confirm" if all(g["pass"] for k, g in gates.items() if k != "verdict") else "partial / null (see gates)"

    # ---------------- positive control for the Exp 01 judge ----------------
    pos = [r for r in renders if r["mu_over_mu_c"] in POS_MU]
    blank = [r for r in renders if r["mu_over_mu_c"] in BLANK_MU]
    def rate(sel):
        k = sum(r["structured"] for r in sel); n = len(sel)
        return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "ci95": wilson(k, n)}
    pc = {"positives": rate(pos), "blank_negatives": rate(blank), "photo_negatives": rate(photos),
          "positives_by_regime": {reg: rate([r for r in pos if r["regime"] == reg]) for reg in curve},
          "photo_negatives_by_arch": {a: rate([r for r in photos if r["arch"] == a]) for a in ("sdxl", "sd35")},
          "photo_false_positive_classes": dict(Counter(c for r in photos for c in CLASSES if r[c])),
          "photo_M_mean": round(float(np.mean([r["M"] for r in photos])), 3) if photos else None}
    pr, bn, pn = pc["positives"]["rate"], pc["blank_negatives"]["rate"], pc["photo_negatives"]["rate"]
    if pr >= 0.8 and bn <= 0.2 and pn <= 0.2:
        pc["verdict"] = "validated"
    elif pr >= 0.5 and bn <= 0.2 and pn <= 0.2:
        pc["verdict"] = "partially_sensitive"
    else:
        pc["verdict"] = "failed"

    report = {"judge": {"model": model, "rubric_version": rubric, "n_scored": len(rows), "n_errors": n_err},
              "curve": curve, "trend": trend, "gates": gates,
              "confusion_matrix_supra_threshold": {p: dict(c) for p, c in conf.items()},
              "positive_control": pc}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"judge {model} / {rubric}; scored {len(rows)}, errors {n_err}\n")
    for reg in curve:
        print(f"[{reg}]  mu/mu_c   n   M_mean  structured  classes")
        for m, v in curve[reg].items():
            print(f"         {m:5.2f}  {v['n']:3d}   {v['M_mean']:.3f}     {v['structured_rate']:.2f}     {v['classes']}")
        print(f"         trend: rho={trend[reg]['spearman_rho']}  perm p={trend[reg]['perm_p']}\n")
    print("gates:", json.dumps(gates, indent=1, default=str))
    print("confusion (predicted -> judged):", json.dumps({p: dict(c) for p, c in conf.items()}, indent=1))
    print("positive control:", json.dumps(pc, indent=1, default=str))


if __name__ == "__main__":
    main()
