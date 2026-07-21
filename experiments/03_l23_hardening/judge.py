"""
Experiment 03 — Judge A: Claude Sonnet 5, blind Klüver L2/3 scorer.

Ported from Exp 01b's judge_kluver2.py, using the shared per-prompt rubric
(rubric.py). Blinding is identical: the model sees only image pixels + the fixed
rubric (which names the intended scene — public — but never the guidance value).
Un-blinding (filename -> guidance) happens only at analysis. Writes
`judgements_claude.json`.

Runs via the **Message Batches API** by default — 50% cheaper than per-request
calls, and this is a non-latency-sensitive batch job (the ideal fit). The batch
id is cached in `.batch_claude.json` so an interrupted poll resumes instead of
re-submitting. `--sync` forces the old per-request path (use only for a tiny
smoke subset where you don't want to wait on batch turnaround).

Model is pinned via EXP03_CLAUDE_MODEL (default: claude-sonnet-5). Sonnet 5 is
newer than Exp 01's Sonnet 4.6, so analyze.py runs a calibration cross-check
against the archived Exp 01 scores (see analysis_plan.md §5).

Usage:
    EXP03_CLAUDE_MODEL=claude-sonnet-5 python judge.py --dir results-local/sdxl
    python judge.py --dir results-local/sdxl --sync --workers 8   # smoke only
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

import rubric as R

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

CLAUDE_MODEL = os.environ.get("EXP03_CLAUDE_MODEL", "claude-sonnet-5")
JUDGE_NAME = "claude"
OUT_FILE = "judgements_claude.json"
BATCH_STATE = ".batch_claude.json"


def _b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def _message_params(path: Path, prereg: dict) -> dict:
    pid = R.prompt_id_of(path.name)
    if pid is None:
        raise ValueError(f"unrecognized filename (cannot pick rubric): {path.name}")
    return {
        "model": CLAUDE_MODEL,
        "max_tokens": 400,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/png", "data": _b64(path)}},
                {"type": "text", "text": R.build_rubric(pid, prereg)},
            ],
        }],
    }


def _save(out_path: Path, results: dict) -> None:
    out_path.write_text(json.dumps({
        "judge": JUDGE_NAME, "model": CLAUDE_MODEL,
        "rubric_version": R.rubric_version(),
        "fields": list(R.ALL_FIELDS), "images": results,
        "via": "batches",
    }, indent=2))


def _text_of(message) -> str:
    return next((b.text for b in message.content if b.type == "text"), "")


# --------------------------------- batch path ----------------------------------


def _run_batch(client, model_dir, todo, prereg, results, out_path) -> None:
    """Submit in CHUNKS. Each base64 PNG is ~1-2 MB, and the Batches API caps a
    request at 256 MB — all ~430 images in one batch is ~640 MB and 413s. Chunk
    to stay well under the cap; results save after every chunk, so a re-run
    resumes via the todo/skip-existing logic (no explicit batch-id state needed)."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    CHUNK = 60  # ~60 base64 PNGs ≈ 90-120 MB, comfortably under the 256 MB cap
    chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
    print(f"[j-claude] {len(todo)} images in {len(chunks)} batch(es) of <= {CHUNK} "
          f"(50% batch pricing)", flush=True)

    # Submit ALL chunk-batches first — the Batches API processes them in parallel
    # server-side, so total wall-time ≈ one batch's latency instead of the sum.
    submitted = []  # list of (batch_id, cid_to_fn)
    for ci, chunk in enumerate(chunks, 1):
        cid_to_fn = {f"b{ci}i{j}": p.name for j, p in enumerate(chunk)}
        fn_to_path = {p.name: p for p in chunk}
        requests = [
            Request(custom_id=cid,
                    params=MessageCreateParamsNonStreaming(**_message_params(fn_to_path[fn], prereg)))
            for cid, fn in cid_to_fn.items()
        ]
        batch = client.messages.batches.create(requests=requests)
        submitted.append((batch.id, cid_to_fn))
        print(f"[j-claude] batch {ci}/{len(chunks)} submitted {batch.id} ({len(requests)} reqs)",
              flush=True)

    # Now poll + collect each (later ones are usually already ended by the time we reach them).
    ok = err = 0
    for bi, (batch_id, cid_to_fn) in enumerate(submitted, 1):
        while True:
            b = client.messages.batches.retrieve(batch_id)
            if b.processing_status == "ended":
                break
            time.sleep(20)
        for result in client.messages.batches.results(batch_id):
            fn = cid_to_fn.get(result.custom_id)
            if fn is None:
                continue
            if result.result.type == "succeeded":
                # A per-image refusal / empty reply must NOT crash the whole run:
                # mark it an error (dropped at analysis; re-judged on a later run).
                try:
                    results[fn] = R.coerce(R.extract_json(_text_of(result.result.message)))
                    ok += 1
                except Exception as e:  # noqa: BLE001
                    stop = getattr(result.result.message, "stop_reason", None)
                    results[fn] = {"error": f"parse:{stop or e}"}
                    err += 1
            else:
                results[fn] = {"error": f"batch:{result.result.type}"}
                err += 1
        _save(out_path, results)  # persist after each collected batch (resumable)
        print(f"[j-claude] batch {bi}/{len(submitted)} collected ({ok} ok, {err} err cumulative)",
              flush=True)
    print(f"[j-claude] all batches done: {ok} scored, {err} errored -> {out_path}", flush=True)


# ---------------------------------- sync path ----------------------------------


def _run_sync(client, todo, prereg, results, out_path, workers) -> None:
    lock = threading.Lock()

    def _work(path: Path):
        try:
            resp = client.messages.create(**_message_params(path, prereg))
            return path.name, R.coerce(R.extract_json(_text_of(resp)))
        except Exception as e:  # noqa: BLE001
            return path.name, {"error": str(e)}

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fn, rec in ex.map(_work, todo):
            with lock:
                results[fn] = rec
                done += 1
                shown = {k: rec.get(k) for k in R.ALL_FIELDS} if "error" not in rec else rec
                print(f"[j-claude] ({done}/{len(todo)}) {fn} {shown}", flush=True)
                if done % 10 == 0:
                    _save(out_path, results)
    _save(out_path, results)
    print(f"[j-claude] sync done -> {out_path} ({len(results)} images)", flush=True)


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
    results: dict[str, dict] = {}
    if out_path.exists() and not args.overwrite:
        results = json.loads(out_path.read_text()).get("images", {})

    order = list(images)
    random.Random(args.shuffle_seed).shuffle(order)  # parity with judge_qwen (independent per-image)
    todo = [p for p in order
            if args.overwrite or "error" in results.get(p.name, {"error": 1})
            or p.name not in results]

    mode = "sync" if args.sync else "batch"
    print(f"[j-claude] {CLAUDE_MODEL}: {len(order)} images, {len(todo)} to score ({mode})", flush=True)
    if not todo:
        return

    client = anthropic.Anthropic()
    if args.sync:
        _run_sync(client, todo, prereg, results, out_path, args.workers)
    else:
        _run_batch(client, model_dir, todo, prereg, results, out_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 03 Judge A (Claude Sonnet 5) L2/3 scorer — batches by default.")
    p.add_argument("--dir", required=True, help="results-local/<model> directory")
    p.add_argument("--overwrite", action="store_true", help="re-score already-scored images")
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--sync", action="store_true",
                   help="per-request instead of Batches API (smoke only; forfeits 50%% discount)")
    p.add_argument("--workers", type=int, default=8, help="concurrent API calls (sync mode only)")
    return p


if __name__ == "__main__":
    score(build_parser().parse_args())
