"""
Experiment 03 — blind human rating tool for the inter-rater subset.

Draws a stratified sample of ~25-30 conditioned images across
(model x guidance-bin x prompt), assigns blind ids, and walks you through
scoring each on the SAME Klüver L2/3 rubric the VLM judges use — with the
guidance value hidden (the intended scene is shown, as it is to the judges).
Results feed the human-vs-Claude and human-vs-Qwen Cohen's kappa in analyze.py.

Single rater (author) — flagged as a limitation in analysis.md; no human-human
reliability is available.

Two files:
  - human_subset.json   : the sampled worklist + blind_id -> (model, filename)
                          un-blinding key (built once, then reused).
  - human_ratings.json  : your scores, keyed by blind_id (resumable).

Usage:
    python human_rate.py --sample          # build the subset once (deterministic)
    python human_rate.py                    # rate remaining items (resumable)
    python human_rate.py --show-progress
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import rubric as R

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results-local"
SUBSET = HERE / "human_subset.json"
RATINGS = HERE / "human_ratings.json"
MODELS = ("sdxl", "sd35")


def _gbin(g: float) -> str:
    return "low" if g <= 3 else ("mid" if g <= 8 else "high")


def _all_conditioned() -> list[dict]:
    items = []
    for model in MODELS:
        d = RESULTS / model
        if not d.exists():
            continue
        for p in sorted(d.glob("*.png")):
            meta = R.parse_filename(p.name)
            if meta and meta["kind"] == "conditioned":
                items.append({"model": model, "filename": p.name,
                              "prompt_id": meta["prompt_id"], "guidance": meta["guidance"],
                              "gbin": _gbin(meta["guidance"])})
    return items


def build_sample(n: int, seed: int) -> None:
    items = _all_conditioned()
    if not items:
        raise SystemExit(f"no conditioned images under {RESULTS} — generate first")
    rng = random.Random(seed)

    # Stratify by (model, gbin); spread prompts within each stratum. Round-robin
    # draw across strata until we hit n, so coverage is balanced across the range.
    strata: dict[tuple, list[dict]] = {}
    for it in items:
        strata.setdefault((it["model"], it["gbin"]), []).append(it)
    for lst in strata.values():
        rng.shuffle(lst)
    keys = sorted(strata)
    chosen: list[dict] = []
    while len(chosen) < min(n, len(items)):
        progressed = False
        for k in keys:
            if strata[k]:
                chosen.append(strata[k].pop())
                progressed = True
                if len(chosen) >= min(n, len(items)):
                    break
        if not progressed:
            break

    rng.shuffle(chosen)  # present in random order; blind_id hides model+guidance
    subset = [{"blind_id": f"h{i:02d}", **it} for i, it in enumerate(chosen)]
    SUBSET.write_text(json.dumps({"n": len(subset), "seed": seed, "items": subset}, indent=2))
    print(f"[human] built subset: {len(subset)} images across "
          f"{len({(s['model'], s['gbin']) for s in subset})} (model x gbin) strata")
    print(f"[human] wrote {SUBSET}. Now run:  python human_rate.py")


def _open_image(path: Path) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(path)], check=False)
        else:
            print(f"[human] open this image manually: {path}")
    except Exception:
        print(f"[human] open this image manually: {path}")


def _ask_int(field: str, hi: int) -> int | None:
    while True:
        raw = input(f"    {field} [0-{hi}] (s=skip image, q=quit): ").strip().lower()
        if raw in ("s", "skip"):
            return None
        if raw in ("q", "quit"):
            raise KeyboardInterrupt
        if raw.isdigit() and 0 <= int(raw) <= hi:
            return int(raw)
        print(f"    ! enter an integer 0-{hi}")


def rate() -> None:
    if not SUBSET.exists():
        raise SystemExit("no human_subset.json — run `python human_rate.py --sample` first")
    subset = json.loads(SUBSET.read_text())["items"]
    prereg = R.load_prereg()
    ratings = json.loads(RATINGS.read_text()) if RATINGS.exists() else {}

    todo = [s for s in subset if s["blind_id"] not in ratings]
    print(f"[human] {len(subset)} in subset, {len(todo)} left to rate. "
          "Score only what is visible; guidance is hidden by design.\n")

    for s in todo:
        path = RESULTS / s["model"] / s["filename"]
        _open_image(path)
        print(f"--- {s['blind_id']} ---")
        # Show the intended-scene context only (same as the judges get); no guidance.
        print(R.build_rubric(s["prompt_id"], prereg).split("Return STRICT JSON")[0].strip())
        try:
            rec: dict = {}
            skipped = False
            for f in R.INT_FIELDS:
                v = _ask_int(f, 3)
                if v is None:
                    skipped = True
                    break
                rec[f] = v
            if not skipped:
                v = _ask_int("tiling", 1)
                if v is None:
                    skipped = True
                else:
                    rec["tiling"] = v
            if skipped:
                print("    (skipped)\n")
                continue
        except KeyboardInterrupt:
            print("\n[human] quit — progress saved.")
            break
        ratings[s["blind_id"]] = rec
        RATINGS.write_text(json.dumps(ratings, indent=2))
        print(f"    saved {rec}\n")

    rated = sum(1 for s in subset if s["blind_id"] in ratings)
    print(f"[human] {rated}/{len(subset)} rated -> {RATINGS}")


def show_progress() -> None:
    if not SUBSET.exists():
        raise SystemExit("no subset yet")
    subset = json.loads(SUBSET.read_text())["items"]
    ratings = json.loads(RATINGS.read_text()) if RATINGS.exists() else {}
    print(f"[human] {sum(1 for s in subset if s['blind_id'] in ratings)}/{len(subset)} rated")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 03 blind human rating tool.")
    p.add_argument("--sample", action="store_true", help="build the stratified subset (once)")
    p.add_argument("--n", type=int, default=28, help="target subset size (25-30)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show-progress", action="store_true")
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    if a.sample:
        build_sample(a.n, a.seed)
    elif a.show_progress:
        show_progress()
    else:
        rate()
