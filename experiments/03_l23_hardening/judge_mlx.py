"""
Experiment 03 — generic local MLX judge (any mlx-vlm model), blind Klüver L2/3 scorer.

`judge_qwen.py` and `judge_llama.py` were near-identical copies differing only in
a model string and an output filename. Judge C needed replacing after the
2026-08-08 probes showed Llama-3.2-11B emits no variance at all, and a third copy
was not the answer — this takes the model and the judge name as arguments and
writes `judgements_<name>.json`.

Uses the *identical* pinned rubric (rubric.py) as every other judge, so the
cross-judge kappa reflects model disagreement only. Blinding matches judges A/B:
image pixels plus the fixed rubric, shuffled order, guidance never shown,
un-blinded only at analysis. Deterministic decoding (temp=0).

Carries the 2026-08-08 harness repairs: `EXP03_JUDGE_MAX_TOKENS` (700, up from
the 400 that truncated ~90 replies mid-JSON), and a `judgements_<name>_raw.json`
sidecar holding the raw reply per image — without it, a field defaulted to 0 by
`rubric.coerce` is indistinguishable from a judge that genuinely scored 0.

A new judge name must also be registered in `analyze.py`'s JUDGES table to be
picked up by the analysis.

Usage:
    python judge_mlx.py --name gemma --model mlx-community/gemma-3-27b-it-qat-4bit \
        --dir results-local/sdxl
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import rubric as R

MAX_TOKENS = int(os.environ.get("EXP03_JUDGE_MAX_TOKENS", "700"))


def _to_text(result) -> str:
    """mlx-vlm generate() has returned str | (text, usage) | GenerationResult."""
    if isinstance(result, str):
        return result
    if hasattr(result, "text"):
        return result.text
    if isinstance(result, (tuple, list)) and result:
        return _to_text(result[0])
    return str(result)


class MlxJudge:
    def __init__(self, model_path: str, tag: str):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.tag = tag
        print(f"[{tag}] loading {model_path} (first run downloads weights)…", flush=True)
        self.model, self.processor = load(model_path)
        try:
            self.config = load_config(model_path)
        except Exception:
            self.config = getattr(self.model, "config", None)

    def score_image(self, path: Path, prompt_text: str) -> tuple[dict, str]:
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
    tag = f"j-{args.name}"
    model_dir = Path(args.dir)
    if not model_dir.is_absolute():
        model_dir = Path(__file__).resolve().parent / model_dir
    images = sorted(p for p in model_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"no PNGs in {model_dir}")

    prereg = R.load_prereg()
    out_path = model_dir / f"judgements_{args.name}.json"
    raw_path = model_dir / f"judgements_{args.name}_raw.json"
    results: dict[str, dict] = {}
    raws: dict[str, str] = {}
    if not args.overwrite:
        if out_path.exists():
            results = json.loads(out_path.read_text()).get("images", {})
        if raw_path.exists():
            raws = json.loads(raw_path.read_text()).get("images", {})

    order = list(images)
    random.Random(args.shuffle_seed).shuffle(order)  # blind: shuffled order

    def _save() -> None:
        out_path.write_text(json.dumps({
            "judge": args.name, "model": args.model,
            "rubric_version": R.rubric_version(),
            "fields": list(R.ALL_FIELDS), "images": results,
        }, indent=2))
        raw_path.write_text(json.dumps({
            "judge": args.name, "model": args.model,
            "max_tokens": MAX_TOKENS, "images": raws,
        }, indent=2))

    todo = [p for p in order
            if args.overwrite or "error" in results.get(p.name, {"error": 1})
            or p.name not in results]
    print(f"[{tag}] {args.model}: {len(order)} images, {len(todo)} to score", flush=True)
    if not todo:
        return

    judge = MlxJudge(args.model, tag)  # single GPU → serial
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
        print(f"[{tag}] ({i}/{len(todo)}) {path.name} {shown}", flush=True)
        # Save every image: a hang mid-batch otherwise loses up to 9 scores
        # (2026-08-08: the 32B re-judge stalled at 70/430 with 9 unsaved).
        _save()
    _save()
    print(f"[{tag}] wrote {out_path} ({len(results)} images)", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 03 generic MLX L2/3 judge.")
    p.add_argument("--name", required=True, help="judge name -> judgements_<name>.json")
    p.add_argument("--model", required=True, help="mlx-vlm model path")
    p.add_argument("--dir", required=True, help="results-local/<model> directory")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--shuffle-seed", type=int, default=0)
    return p


if __name__ == "__main__":
    score(build_parser().parse_args())
