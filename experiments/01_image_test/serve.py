"""
HTTP layer for Experiment 01.

Reuses the same `iter_sweep()` generator that the CLI uses, so there's a single
source of truth for the experiment. Run with:

    uvicorn experiments.01_image_test.serve:app --reload --port 8000

...except that "01_image_test" isn't a valid Python module name (starts with a
digit), so the more practical invocation is:

    cd experiments/01_image_test
    uvicorn serve:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from run import (  # type: ignore[import-not-found]
    GUIDANCE_SWEEP,
    PROMPT,
    PUBLIC_URL_PREFIX,
    SEEDS,
    clear_results,
    existing_metadata,
    iter_sweep,
)

app = FastAPI(title="Operating System Hypothesis — Experiment 01")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4321",
        "http://127.0.0.1:4321",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/experiments/01/results")
def get_results() -> JSONResponse:
    """Return what's already on disk so the page can render existing results."""
    meta = existing_metadata()
    if meta is None:
        return JSONResponse(
            {
                "exists": False,
                "prompt": PROMPT,
                "sweep": {"guidance": GUIDANCE_SWEEP, "seeds": SEEDS},
                "expected_total": len(GUIDANCE_SWEEP) * len(SEEDS),
            }
        )
    return JSONResponse({"exists": True, "metadata": meta})


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


# Comment-only SSE line. Browsers buffer streaming responses until ~1–2KB of
# bytes have arrived; emitting this once at the start of the stream forces the
# flush so subsequent events become visible immediately.
_STARTUP_PAD = b":" + (b" " * 2048) + b"\n\n"
_HEARTBEAT = b": keepalive\n\n"
_HEARTBEAT_INTERVAL = 1.5


async def _sweep_stream() -> AsyncIterator[bytes]:
    """Drive iter_sweep on a worker thread; relay events to the SSE stream and
    emit heartbeat comments so the connection never goes silent.
    """
    yield _STARTUP_PAD
    yield _sse("start", {"total": len(GUIDANCE_SWEEP) * len(SEEDS), "ts": time.time()})

    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def worker() -> None:
        try:
            clear_results()
            for event in iter_sweep():
                loop.call_soon_threadsafe(queue.put_nowait, _sse(event["type"], event))
        except Exception as e:  # noqa: BLE001
            loop.call_soon_threadsafe(
                queue.put_nowait,
                _sse("error", {"message": str(e), "traceback": traceback.format_exc()}),
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    task = asyncio.create_task(asyncio.to_thread(worker))
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_INTERVAL)
            except asyncio.TimeoutError:
                yield _HEARTBEAT
                continue
            if chunk is None:
                break
            yield chunk
    finally:
        await task


@app.post("/api/experiments/01/sweep")
async def post_sweep() -> StreamingResponse:
    """Clear existing results and run a fresh sweep, streaming progress as SSE."""
    return StreamingResponse(
        _sweep_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
