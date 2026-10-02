#!/usr/bin/env python3
"""Human-human agreement on the Exp 03 blind subset.

    python human_kappa2.py --rater NAME      # compares human_ratings.json (author) with
                                             # human_ratings_NAME.json (second rater)

Writes results-local/human_human_kappa.json. Same estimators as analyze.py's
human-versus-judge block (statlib.weighted_cohens_kappa, gwet_ac2, percent_agreement,
composite_kappa_ci), so the two numbers are comparable. Sensitivity analysis only: it was
not pre-registered and carries no gate.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import statlib as S

HERE = Path(__file__).resolve().parent
INT = ["reduplication", "fragmentation", "condensation", "distortion"]
ALL = INT + ["tiling"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rater", required=True, help="suffix of human_ratings_<rater>.json")
    ap.add_argument("--out", default=str(HERE / "results-local" / "human_human_kappa.json"))
    a = ap.parse_args()
    r1 = json.loads((HERE / "human_ratings.json").read_text())
    r2 = json.loads((HERE / f"human_ratings_{a.rater}.json").read_text())
    subset = {s["blind_id"]: s for s in json.loads((HERE / "human_subset.json").read_text())["items"]}
    common = [b for b in subset if b in r1 and b in r2]
    pairs = {f: ([int(r1[b][f]) for b in common if f in r1[b] and f in r2[b]],
                 [int(r2[b][f]) for b in common if f in r1[b] and f in r2[b]]) for f in ALL}
    out = {"label": "EXPLORATORY: human-human agreement, not pre-registered, no gate",
           "rater_1": "author (knows the hypothesis)", "rater_2": a.rater,
           "n_subset": len(subset), "n_common": len(common),
           "degenerate_on_subset": {"rater_1": [f for f in ALL if len(set(pairs[f][0])) < 2],
                                    "rater_2": [f for f in ALL if len(set(pairs[f][1])) < 2]}}
    q = lambda f: 2 if f == "tiling" else 4
    out["weighted_kappa"] = {f: round(S.weighted_cohens_kappa(*pairs[f], q(f)), 4) for f in ALL}
    out["gwet_ac2"] = {f: round(S.gwet_ac2(*pairs[f], q(f)), 4) for f in ALL}
    out["percent_agreement"] = {f: round(S.percent_agreement(*pairs[f]), 4) for f in ALL}
    out["composite_weighted_kappa"] = round(float(np.nanmean([out["weighted_kappa"][f] for f in INT])), 4)
    out["composite_bootstrap"] = S.composite_kappa_ci([pairs[f] for f in INT], 4)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("n_common", "weighted_kappa", "composite_weighted_kappa",
                                           "composite_bootstrap")}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
