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
JUDGE_FILES = ("judgements_claude.json", "judgements_qwen.json", "judgements_llama.json")


def _gbin(g: float) -> str:
    return "low" if g <= 3 else ("mid" if g <= 8 else "high")


def _sampling_frame() -> list[dict]:
    """Conditioned images scored without error by every judge present.

    Matching analyze.py's listwise-complete set matters: an image no judge pair
    covers contributes nothing to human-vs-judge kappa, so spending a human
    rating on it wastes a slot out of ~28.
    """
    items = []
    for model in MODELS:
        d = RESULTS / model
        if not d.exists():
            continue
        judged: list[dict] = []
        for fn in JUDGE_FILES:
            p = d / fn
            if p.exists():
                judged.append(json.loads(p.read_text()).get("images", {}))
        for p in sorted(d.glob("*.png")):
            meta = R.parse_filename(p.name)
            if not meta or meta["kind"] != "conditioned":
                continue
            recs = [j.get(p.name) for j in judged]
            if any(r is None or "error" in r for r in recs):
                continue
            items.append({"model": model, "filename": p.name,
                          "prompt_id": meta["prompt_id"], "guidance": meta["guidance"],
                          "gbin": _gbin(meta["guidance"])})
    return items


def build_sample(n: int, seed: int) -> None:
    items = _sampling_frame()
    if not items:
        raise SystemExit(f"no fully-judged conditioned images under {RESULTS} — "
                         "generate and judge first")
    rng = random.Random(seed)

    # Stratify by (model x gbin), and inside each stratum round-robin over
    # prompt_id so all 6 prompts are represented rather than drawn by luck.
    strata: dict[tuple, dict[str, list[dict]]] = {}
    for it in items:
        strata.setdefault((it["model"], it["gbin"]), {}).setdefault(it["prompt_id"], []).append(it)
    for by_prompt in strata.values():
        for lst in by_prompt.values():
            rng.shuffle(lst)

    # Each stratum only gets ~4-5 of the 28 slots, so a cursor starting at 0
    # everywhere would never reach the last prompts — p6_forest, the
    # low-objecthood control, would be sampled zero times. Stagger the starting
    # prompt per stratum so the 6 prompts are covered across the whole subset.
    keys = sorted(strata)
    cursors = {k: i for i, k in enumerate(keys)}
    g_used: dict[float, int] = {}
    chosen: list[dict] = []
    target = min(n, len(items))
    while len(chosen) < target:
        progressed = False
        for k in keys:
            prompts = sorted(strata[k])
            for _ in range(len(prompts)):
                pid = prompts[cursors[k] % len(prompts)]
                cursors[k] += 1
                bucket = strata[k][pid]
                if bucket:
                    # within the bin, favour the least-used guidance value so the
                    # subset spans the grid rather than piling onto one g
                    pick = min(range(len(bucket)), key=lambda j: g_used.get(bucket[j]["guidance"], 0))
                    it = bucket.pop(pick)
                    g_used[it["guidance"]] = g_used.get(it["guidance"], 0) + 1
                    chosen.append(it)
                    progressed = True
                    break
            if len(chosen) >= target:
                break
        if not progressed:
            break

    rng.shuffle(chosen)  # present in random order; blind_id hides model+guidance
    subset = [{"blind_id": f"h{i:02d}", **it} for i, it in enumerate(chosen)]
    SUBSET.write_text(json.dumps(
        {"n": len(subset), "seed": seed, "frame": len(items),
         "frame_note": "conditioned images scored without error by all present judges",
         "items": subset}, indent=2))
    cells = {(s["model"], s["gbin"]) for s in subset}
    print(f"[human] built subset: {len(subset)} images from a frame of {len(items)}, "
          f"across {len(cells)} (model x gbin) strata")
    for m in MODELS:
        for b in ("low", "mid", "high"):
            got = [s for s in subset if s["model"] == m and s["gbin"] == b]
            if got:
                print(f"          {m:5s} {b:4s}: {len(got):2d} images, "
                      f"{len({s['prompt_id'] for s in got})} distinct prompts")
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


REOPEN = object()


def _ask_int(field: str, hi: int) -> int | None:
    while True:
        raw = input(f"    {field} [0-{hi}] (r=reopen image, s=skip image, q=quit): ").strip().lower()
        if raw in ("r", "reopen"):
            return REOPEN  # type: ignore[return-value]
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

    done = len(subset) - len(todo)
    for i, s in enumerate(todo, start=1):
        path = RESULTS / s["model"] / s["filename"]
        _open_image(path)
        print(f"--- {s['blind_id']}  ({done + i}/{len(subset)}) ---")
        # Show the intended-scene context only (same as the judges get); no guidance.
        print(R.build_rubric(s["prompt_id"], prereg).split("Return STRICT JSON")[0].strip())
        try:
            rec: dict = {}
            skipped = False
            for f, hi in [(f, 3) for f in R.INT_FIELDS] + [("tiling", 1)]:
                while True:
                    v = _ask_int(f, hi)
                    if v is REOPEN:
                        _open_image(path)
                        continue
                    break
                if v is None:
                    skipped = True
                    break
                rec[f] = v
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
