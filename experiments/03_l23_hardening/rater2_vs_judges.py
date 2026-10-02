#!/usr/bin/env python3
"""Second human rater against each judge on the Exp 03 blind subset. EXPLORATORY, no gate.

    python rater2_vs_judges.py --rater rater2 --judges claude,qwen

Uses analyze._human_reliability unchanged, pointed at human_ratings_<rater>.json instead of
the author's file, so the numbers are computed exactly as the author-versus-judge numbers in
the paper. The author's block is recomputed alongside as a check that it reproduces the
published values. Writes results-local/rater2_vs_judges.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze as A

HERE = Path(__file__).resolve().parent


def _block(model_records, judges, ratings_file: Path) -> dict:
    original = HERE / "human_ratings.json"
    saved = original.read_text()
    try:
        if ratings_file != original:
            original.write_text(ratings_file.read_text())
        return A._human_reliability(model_records, judges)
    finally:
        original.write_text(saved)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rater", required=True)
    ap.add_argument("--judges", default="claude,qwen")
    a = ap.parse_args()
    only = [j.strip() for j in a.judges.split(",")]
    recs, judges = {}, None
    for m in ["sdxl", "sd35"]:
        cond, uncond, notes = A._load_records(m, only)
        A._build_metric(cond, uncond, notes)
        recs[m] = cond
        judges = notes["judges"]
    out = {"label": "EXPLORATORY: second human rater versus each judge, not pre-registered, no gate",
           "rater": a.rater,
           "rater_note": "naive second rater; plain-language rubric (rater2_message.txt), same images and order",
           "author": _block(recs, judges, HERE / "human_ratings.json"),
           "rater2": _block(recs, judges, HERE / f"human_ratings_{a.rater}.json")}
    path = HERE / "results-local" / "rater2_vs_judges.json"
    path.write_text(json.dumps(out, indent=2))
    for who in ("author", "rater2"):
        print(who, {k: out[who][k]["composite_weighted_kappa"] for k in out[who] if k.startswith("human_vs_")},
              {k: out[who][k]["composite_bootstrap"]["ci95"] for k in out[who] if k.startswith("human_vs_")})
    print("wrote", path)


if __name__ == "__main__":
    main()
