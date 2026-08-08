"""
Experiment 03 — Judge C: Llama-3.2-11B-Vision (local, MLX), blind Klüver L2/3 scorer.

The *third* independent judge (added 2026-07-21). Three model families now score
every image — Anthropic (Claude, judge A), Alibaba/Qwen (judge B), Meta/Llama
(this, judge C) — so inter-rater reliability triangulates across three lineages
instead of two, and no single family's quirks drive the composite kappa. Runs
locally on Apple Silicon via mlx-vlm — $0 per call. Uses the *identical* shared
rubric (rubric.py) as the other judges. Writes `judgements_llama.json`.

Blinding: same as A/B — image pixels + fixed rubric, shuffled order, guidance
never shown, un-blinded only at analysis. Deterministic decoding (temp=0).

MODEL — safe dense default (chosen 2026-07-21):
  Default is **Llama-3.2-11B-Vision-Instruct 4-bit** (~7 GB) — small, robust, and
  entirely adequate for a fixed-rubric ordinal judge (the point of judge C is a
  *third model family*, not size). It leaves ample headroom on a 64 GB M5.
  OPTIONAL "latest" upgrade — Llama 4 Scout (17B active / 109B total MoE), the
  newest Meta vision model, needs mlx-vlm>=0.1.21 and its 4-bit weights are
  ~55 GB (borderline on 64 GB — smoke-test first, may OOM):
      EXP03_LLAMA_MODEL=mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit python judge_llama.py ...

Setup (on the M5 Pro 64GB host):
    pip install -U mlx-vlm       # >= 0.1.21 if you opt into Llama 4 Scout
    # first run downloads the weights (11B default ~7 GB; Scout ~55 GB)
Model is pinned via EXP03_LLAMA_MODEL.

Usage:
    python judge_llama.py --dir results-local/sdxl
    python judge_llama.py --dir results-local/sd35

NOTE: mlx-vlm generate/apply_chat_template signatures have shifted across
releases; _to_text() and the call below handle the common variants (identical to
judge_qwen.py). If a call fails on your installed version, pin mlx-vlm.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import rubric as R

LLAMA_MODEL = os.environ.get(
    "EXP03_LLAMA_MODEL", "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit"
)
# 400 truncated ~90 replies mid-JSON in the 2026-07-22 run, and those images were
# then dropped listwise, taking their Claude and Qwen scores with them.
MAX_TOKENS = int(os.environ.get("EXP03_JUDGE_MAX_TOKENS", "700"))
JUDGE_NAME = "llama"
OUT_FILE = "judgements_llama.json"
RAW_FILE = "judgements_llama_raw.json"


def _to_text(result) -> str:
    """mlx-vlm generate() has returned str | (text, usage) | GenerationResult."""
    if isinstance(result, str):
        return result
    if hasattr(result, "text"):
        return result.text
    if isinstance(result, (tuple, list)) and result:
        return _to_text(result[0])
    return str(result)


class LlamaJudge:
    def __init__(self, model_path: str):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        print(f"[j-llama] loading {model_path} (first run downloads weights)…", flush=True)
        self.model, self.processor = load(model_path)
        try:
            self.config = load_config(model_path)
        except Exception:
            self.config = getattr(self.model, "config", None)

    def score_image(self, path: Path, prompt_text: str) -> tuple[dict, str]:
        """-> (coerced record, raw reply). The raw reply is kept as an audit
        trail: without it, a field defaulted to 0 by `coerce` is indistinguishable
        from a judge that genuinely scored 0."""
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        formatted = apply_chat_template(self.processor, self.config, prompt_text, num_images=1)
        result = generate(
            self.model, self.processor, formatted, [str(path)],
            max_tokens=MAX_TOKENS, temperature=0.0, verbose=False,
        )
        raw = _to_text(result)
        return R.coerce(R.extract_json(raw)), raw


def score(args: argparse.Namespace) -> None:
    model_dir = Path(args.dir)
    if not model_dir.is_absolute():
        model_dir = Path(__file__).resolve().parent / model_dir
    images = sorted(p for p in model_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"no PNGs in {model_dir}")

    prereg = R.load_prereg()
    out_path, raw_path = model_dir / OUT_FILE, model_dir / RAW_FILE
    results: dict[str, dict] = {}
    if out_path.exists() and not args.overwrite:
        results = json.loads(out_path.read_text()).get("images", {})
    raws: dict[str, str] = {}
    if raw_path.exists() and not args.overwrite:
        raws = json.loads(raw_path.read_text()).get("images", {})

    order = list(images)
    random.Random(args.shuffle_seed).shuffle(order)  # blind: shuffled order

    def _save() -> None:
        out_path.write_text(json.dumps({
            "judge": JUDGE_NAME, "model": LLAMA_MODEL,
            "rubric_version": R.rubric_version(),
            "fields": list(R.ALL_FIELDS), "images": results,
        }, indent=2))
        raw_path.write_text(json.dumps({
            "judge": JUDGE_NAME, "model": LLAMA_MODEL,
            "max_tokens": MAX_TOKENS, "images": raws,
        }, indent=2))

    todo = [p for p in order
            if args.overwrite or "error" in results.get(p.name, {"error": 1})
            or p.name not in results]
    print(f"[j-llama] {LLAMA_MODEL}: {len(order)} images, {len(todo)} to score", flush=True)
    if not todo:
        return

    judge = LlamaJudge(LLAMA_MODEL)  # single GPU → serial
    for i, path in enumerate(todo, start=1):
        pid = R.prompt_id_of(path.name)
        if pid is None:
            results[path.name] = {"error": f"unrecognized filename: {path.name}"}
            continue
        raw = ""
        try:
            rec, raw = judge.score_image(path, R.build_rubric(pid, prereg))
        except Exception as e:  # noqa: BLE001
            rec = {"error": str(e)}
        results[path.name] = rec
        raws[path.name] = raw[:600]
        shown = {k: rec.get(k) for k in R.ALL_FIELDS} if "error" not in rec else rec
        print(f"[j-llama] ({i}/{len(todo)}) {path.name} {shown}", flush=True)
        # Save every image: a hang mid-batch otherwise loses up to 9 scores
        # (2026-08-08: the 32B re-judge stalled at 70/430 with 9 unsaved).
        _save()
    _save()
    print(f"[j-llama] wrote {out_path} ({len(results)} images)", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 03 Judge C (Llama-3.2-11B-Vision, MLX) L2/3 scorer.")
    p.add_argument("--dir", required=True, help="results-local/<model> directory")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--shuffle-seed", type=int, default=0)
    return p


if __name__ == "__main__":
    score(build_parser().parse_args())
