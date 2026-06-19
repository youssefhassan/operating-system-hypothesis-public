"""
Experiment 01 — Vision guidance sweep on FLUX.2-flex (BFL API).

Holds prompt + seed fixed and sweeps `guidance` (classifier-free guidance scale)
across its full allowed range. Guidance is the model's "how strongly do I commit
to the conditioning prior" knob — the closest API-exposed analogue to top-down
predictive suppression.

This module exposes `iter_sweep()` as a generator so both the CLI here and the
FastAPI server (`serve.py`) can drive the same sweep — one source of truth.

NOTE: This is the *API baseline* only. BFL's flux-2-flex guidance floor is 1.5,
which can't reach the low / base-layer regime the experiment cares about, and
FLUX guidance is distilled (not classic CFG). The plan-aligned primary run is
`sweep_local.py` (open-weights diffusers, guidance down to 1.0 + an
unconditional baseline, multi-model). See README.md.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://api.bfl.ai/v1"
MODEL_ENDPOINT = "flux-2-flex"

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "site" / "public" / "results" / "01"
PUBLIC_URL_PREFIX = "/results/01"

PROMPT = "a watermelon, a glass half-filled with water, and a set of keys on a wooden table"
GUIDANCE_SWEEP = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.0, 10.0]
SEEDS = [42, 43, 44]
STEPS = 50
WIDTH = 1024
HEIGHT = 1024
OUTPUT_FORMAT = "png"

POLL_INTERVAL_SEC = 1.0
POLL_TIMEOUT_SEC = 180.0


class BFLError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("BFL_API_KEY")
    if not key:
        raise BFLError("BFL_API_KEY not set; copy .env.example to .env and fill it in.")
    return key


def _submit(prompt: str, guidance: float, seed: int) -> dict:
    resp = requests.post(
        f"{API_BASE}/{MODEL_ENDPOINT}",
        headers={"x-key": _api_key(), "Content-Type": "application/json"},
        json={
            "prompt": prompt,
            "prompt_upsampling": False,  # critical: otherwise BFL silently rewrites the prompt
            "seed": seed,
            "width": WIDTH,
            "height": HEIGHT,
            "steps": STEPS,
            "guidance": guidance,
            "output_format": OUTPUT_FORMAT,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _poll(polling_url: str) -> dict:
    """Non-streaming poll, used only by direct callers (not the sweep)."""
    deadline = time.monotonic() + POLL_TIMEOUT_SEC
    while time.monotonic() < deadline:
        resp = requests.get(polling_url, headers={"x-key": _api_key()}, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        status = body.get("status")
        if status == "Ready":
            return body
        if status in {"Error", "Failed", "Content Moderated", "Request Moderated"}:
            raise BFLError(f"BFL job failed: status={status} body={body}")
        time.sleep(POLL_INTERVAL_SEC)
    raise BFLError(f"BFL job timed out after {POLL_TIMEOUT_SEC}s: {polling_url}")


def _download(url: str, dest: Path) -> None:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def clear_results() -> None:
    if not RESULTS_DIR.exists():
        return
    for path in RESULTS_DIR.iterdir():
        if path.is_file():
            path.unlink()


def existing_metadata() -> dict | None:
    meta = RESULTS_DIR / "metadata.json"
    if not meta.exists():
        return None
    try:
        return json.loads(meta.read_text())
    except json.JSONDecodeError:
        return None


def iter_sweep() -> Iterator[dict]:
    """Run the sweep and yield events as it progresses.

    Event types (each carries index, total, guidance, seed):
      "submitting"  — about to call BFL submit endpoint
      "submitted"   — BFL accepted the job (carries bfl_id)
      "polling"     — one poll round (carries attempt, elapsed_sec, bfl_status)
      "downloading" — BFL Ready, about to download the image
      "completed"   — image saved (carries filename, url)
      "done"        — all images saved (carries metadata_url, total)
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    runs: list[dict] = []
    pairs = [(g, s) for g in GUIDANCE_SWEEP for s in SEEDS]
    total = len(pairs)

    for i, (guidance, seed) in enumerate(pairs):
        filename = f"g{guidance:.1f}_s{seed}.{OUTPUT_FORMAT}"
        url = f"{PUBLIC_URL_PREFIX}/{filename}"
        image_path = RESULTS_DIR / filename
        base = {"index": i, "total": total, "guidance": guidance, "seed": seed}

        yield {"type": "submitting", **base}

        submission = _submit(PROMPT, guidance, seed)
        polling_url = submission["polling_url"]

        yield {"type": "submitted", **base, "bfl_id": submission.get("id")}

        poll_started = time.monotonic()
        deadline = poll_started + POLL_TIMEOUT_SEC
        attempt = 0
        result: dict | None = None
        while time.monotonic() < deadline:
            attempt += 1
            resp = requests.get(polling_url, headers={"x-key": _api_key()}, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            status = body.get("status")
            if status == "Ready":
                result = body
                break
            if status in {"Error", "Failed", "Content Moderated", "Request Moderated"}:
                raise BFLError(f"BFL job failed: status={status} body={body}")
            yield {
                "type": "polling",
                **base,
                "attempt": attempt,
                "elapsed_sec": round(time.monotonic() - poll_started, 1),
                "bfl_status": status or "Pending",
            }
            time.sleep(POLL_INTERVAL_SEC)
        if result is None:
            raise BFLError(f"BFL job timed out after {POLL_TIMEOUT_SEC}s: {polling_url}")

        yield {"type": "downloading", **base}

        sample_url = result["result"]["sample"]
        _download(sample_url, image_path)

        runs.append(
            {
                "guidance": guidance,
                "seed": seed,
                "filename": filename,
                "url": url,
                "submission_id": submission.get("id"),
            }
        )

        yield {"type": "completed", **base, "filename": filename, "url": url}

    metadata = {
        "experiment": "01_image_test",
        "model": MODEL_ENDPOINT,
        "api_base": API_BASE,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "prompt": PROMPT,
        "fixed": {
            "steps": STEPS,
            "width": WIDTH,
            "height": HEIGHT,
            "output_format": OUTPUT_FORMAT,
            "prompt_upsampling": False,
        },
        "sweep": {"guidance": GUIDANCE_SWEEP, "seeds": SEEDS},
        "runs": runs,
    }
    metadata_path = RESULTS_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    yield {
        "type": "done",
        "total": total,
        "metadata_url": f"{PUBLIC_URL_PREFIX}/metadata.json",
    }


def run_sweep() -> None:
    """CLI driver: run iter_sweep() and print progress to stdout."""
    for event in iter_sweep():
        t = event["type"]
        if t == "submitting":
            print(
                f"[{event['index'] + 1}/{event['total']}] "
                f"g={event['guidance']} s={event['seed']} submitting…",
                flush=True,
            )
        elif t == "polling":
            print(
                f"  polling {event['bfl_status']} "
                f"(attempt {event['attempt']}, {event['elapsed_sec']}s)",
                flush=True,
            )
        elif t == "downloading":
            print("  downloading…", flush=True)
        elif t == "completed":
            print(f"  -> saved {event['filename']}", flush=True)
        elif t == "done":
            print(f"\nDone. {event['total']} images. Metadata: {RESULTS_DIR / 'metadata.json'}")


if __name__ == "__main__":
    run_sweep()
