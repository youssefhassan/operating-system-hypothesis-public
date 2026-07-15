"""
Experiment 03 — Judge A: Claude Sonnet 5, blind Klüver L2/3 scorer.

Ported from Exp 01b's judge_kluver2.py, using the shared per-prompt rubric
(rubric.py). Blinding is identical: the model sees only image pixels + the fixed
rubric (which names the intended scene — public — but never the guidance value),
scored in shuffled order. Un-blinding (filename -> guidance) happens only at
analysis. Writes `judgements_claude.json`.

Model is pinned via EXP03_CLAUDE_MODEL (default: claude-sonnet-5). Sonnet 5 is
newer than Exp 01's Sonnet 4.6, so analyze.py runs a calibration cross-check
against the archived Exp 01 scores (see analysis_plan.md §5).

Usage:
    EXP03_CLAUDE_MODEL=claude-sonnet-5 python judge.py --dir results-local/sdxl
    python judge.py --dir results-local/sd35 --workers 8
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

import rubric as R

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

CLAUDE_MODEL = os.environ.get("EXP03_CLAUDE_MODEL", "claude-sonnet-5")
JUDGE_NAME = "claude"
OUT_FILE = "judgements_claude.json"


def _b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def judge_one(client, path: Path, prereg: dict) -> dict:
    pid = R.prompt_id_of(path.name)
    if pid is None:
        raise ValueError(f"unrecognized filename (cannot pick rubric): {path.name}")
    prompt_text = R.build_rubric(pid, prereg)
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/png", "data": _b64(path)}},
                {"type": "text", "text": prompt_text},
            ],
        }],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return R.coerce(R.extract_json(text))


def score(args: argparse.Namespace) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(f"Missing ANTHROPIC_API_KEY in {_PROJECT_ROOT / '.env'}")
    import anthropic

    model_dir = Path(args.dir)
    if not model_dir.is_absolute():
        model_dir = Path(__file__).resolve().parent / model_dir
    images = sorted(p for p in model_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"no PNGs in {model_dir}")

    prereg = R.load_prereg()
    out_path = model_dir / OUT_FILE
    existing = {}
    if out_path.exists() and not args.overwrite:
        existing = json.loads(out_path.read_text()).get("images", {})

    order = list(images)
    random.Random(args.shuffle_seed).shuffle(order)  # blind: shuffled order

    results: dict[str, dict] = dict(existing)
    lock = threading.Lock()
    client = anthropic.Anthropic()

    def _save() -> None:
        out_path.write_text(json.dumps({
            "judge": JUDGE_NAME, "model": CLAUDE_MODEL,
            "rubric_version": R.rubric_version(),
            "fields": list(R.ALL_FIELDS), "images": results,
        }, indent=2))

    todo = [p for p in order
            if args.overwrite or "error" in results.get(p.name, {"error": 1})
            or p.name not in results]
    print(f"[j-claude] {CLAUDE_MODEL}: {len(order)} images, {len(todo)} to score, "
          f"{args.workers} workers", flush=True)

    def _work(path: Path):
        try:
            return path.name, judge_one(client, path, prereg)
        except Exception as e:  # noqa: BLE001
            return path.name, {"error": str(e)}

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fn, rec in ex.map(_work, todo):
            with lock:
                results[fn] = rec
                done += 1
                shown = {k: rec.get(k) for k in R.ALL_FIELDS} if "error" not in rec else rec
                print(f"[j-claude] ({done}/{len(todo)}) {fn} {shown}", flush=True)
                if done % 10 == 0:
                    _save()
    _save()
    print(f"[j-claude] wrote {out_path} ({len(results)} images)", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 03 Judge A (Claude Sonnet 5) L2/3 scorer.")
    p.add_argument("--dir", required=True, help="results-local/<model> directory")
    p.add_argument("--overwrite", action="store_true", help="re-score already-scored images")
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8, help="concurrent API calls")
    return p


if __name__ == "__main__":
    score(build_parser().parse_args())
