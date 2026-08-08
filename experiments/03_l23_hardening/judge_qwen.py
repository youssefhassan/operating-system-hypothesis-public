"""
Experiment 03 — Judge B: Qwen2.5-VL-7B (local, MLX), blind Klüver L2/3 scorer.

The open-weight second judge that breaks judge-circularity (NOT another Claude).
Runs locally on Apple Silicon via mlx-vlm — $0 per call. Uses the *identical*
shared rubric (rubric.py) as Judge A, so the cross-judge Cohen's kappa reflects
model disagreement only. Writes `judgements_qwen.json`.

Blinding: same as Judge A — image pixels + fixed rubric, shuffled order, guidance
never shown, un-blinded only at analysis. Deterministic decoding (temp=0).

Setup (on the M5 Pro 64GB host):
    pip install mlx-vlm
    # first run downloads the weights (~5 GB for the 4-bit 7B)
Model is pinned via EXP03_QWEN_MODEL (default: the 4-bit 7B). On the 64GB host
the 32B (EXP03_QWEN_MODEL=mlx-community/Qwen2.5-VL-32B-Instruct-4bit) is the
drop-in fallback if 7B's human-subset kappa is poor (preregistration.json).

Usage:
    python judge_qwen.py --dir results-local/sdxl
    python judge_qwen.py --dir results-local/sd35

NOTE: the mlx-vlm generate/apply_chat_template signatures have shifted across
releases; _to_text() and the call below handle the common variants. If a call
fails on your installed version, pin mlx-vlm and adjust the two marked lines.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import rubric as R

QWEN_MODEL = __import__("os").environ.get(
    "EXP03_QWEN_MODEL", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
)
# 400 truncated ~90 replies mid-JSON in the 2026-07-22 run, and those images were
# then dropped listwise, taking their Claude and Qwen scores with them.
MAX_TOKENS = int(__import__("os").environ.get("EXP03_JUDGE_MAX_TOKENS", "700"))
JUDGE_NAME = "qwen"
OUT_FILE = "judgements_qwen.json"
RAW_FILE = "judgements_qwen_raw.json"


def _to_text(result) -> str:
    """mlx-vlm generate() has returned str | (text, usage) | GenerationResult."""
    if isinstance(result, str):
        return result
    if hasattr(result, "text"):
        return result.text
    if isinstance(result, (tuple, list)) and result:
        return _to_text(result[0])
    return str(result)


class QwenJudge:
    def __init__(self, model_path: str):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        print(f"[j-qwen] loading {model_path} (first run downloads weights)…", flush=True)
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

        # (1) format prompt for a single image
        formatted = apply_chat_template(self.processor, self.config, prompt_text, num_images=1)
        # (2) generate — deterministic
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
            "judge": JUDGE_NAME, "model": QWEN_MODEL,
            "rubric_version": R.rubric_version(),
            "fields": list(R.ALL_FIELDS), "images": results,
        }, indent=2))
        raw_path.write_text(json.dumps({
            "judge": JUDGE_NAME, "model": QWEN_MODEL,
            "max_tokens": MAX_TOKENS, "images": raws,
        }, indent=2))

    todo = [p for p in order
            if args.overwrite or "error" in results.get(p.name, {"error": 1})
            or p.name not in results]
    print(f"[j-qwen] {QWEN_MODEL}: {len(order)} images, {len(todo)} to score", flush=True)
    if not todo:
        return

    judge = QwenJudge(QWEN_MODEL)  # single GPU → serial
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
        print(f"[j-qwen] ({i}/{len(todo)}) {path.name} {shown}", flush=True)
        if i % 10 == 0:
            _save()
    _save()
    print(f"[j-qwen] wrote {out_path} ({len(results)} images)", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 03 Judge B (Qwen2.5-VL-7B, MLX) L2/3 scorer.")
    p.add_argument("--dir", required=True, help="results-local/<model> directory")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--shuffle-seed", type=int, default=0)
    return p


if __name__ == "__main__":
    score(build_parser().parse_args())
