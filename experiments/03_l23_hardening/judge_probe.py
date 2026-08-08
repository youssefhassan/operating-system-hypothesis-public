"""
Experiment 03 — cheap screening probe for candidate MLX judge configurations.

The 2026-07-22 run failed its inter-judge reliability gate, and the 2026-08-08
human subset showed why: Qwen2.5-VL-7B scored 0 on all four fields on all 28
rated images and Llama-3.2-11B was flat on three of four. Neither disagreed with
the human — neither produced any variance at all. Before paying for a full
860-image re-judge this probe asks the only question that matters first: does a
candidate configuration produce *graded* output on images we already know span
the range?

It scores the committed 28-image human subset (all 6 prompts, all 7 guidance
values, both models) with the identical pinned rubric, and reports:

  - the score distribution per field, and whether the judge ever uses 1 or 2;
  - `gradedness`, the share of scored images landing on an interior level, which
    is what both failing judges scored ~0 on;
  - Spearman rank agreement with Claude's committed scores;
  - parse failures and fields silently defaulted to 0 by `rubric.coerce`.

SELECTION CRITERION — deliberately not human kappa. The human ratings are the
independent validation of whichever judge is chosen; selecting a judge by its
agreement with them and then reporting that same agreement would be circular.
The screen here uses only gradedness and Claude rank agreement. Human kappa is
printed *after* the choice is locked, via `analyze.py`, not used to make it.

Nothing here changes the rubric text or any pre-registered field. Judge model
and quantization are listed under `swappable_without_reclassifying` in
preregistration.json, which is the whole reason this is a legal move.

Usage:
    python judge_probe.py --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit
    python judge_probe.py --model mlx-community/Qwen2.5-VL-32B-Instruct-4bit
    python judge_probe.py --compare          # table over every probe run so far
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import rubric as R

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results-local"
SUBSET = HERE / "human_subset.json"
PROBES = HERE / "probes"
INT = R.INT_FIELDS

RUBRICS = {"kluver": "rubric", "axes": "rubric_axes"}


def _use_rubric(name: str) -> None:
    """Swap the active scale. Probe files record which one they used, so
    --compare can read mixed-scale runs without this being set."""
    global R, INT
    R = importlib.import_module(RUBRICS[name])
    INT = R.INT_FIELDS


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
    def __init__(self, model_path: str, max_tokens: int):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        print(f"[probe] loading {model_path} (first run downloads weights)…", flush=True)
        t0 = time.time()
        self.model, self.processor = load(model_path)
        try:
            self.config = load_config(model_path)
        except Exception:
            self.config = getattr(self.model, "config", None)
        self.max_tokens = max_tokens
        print(f"[probe] loaded in {time.time() - t0:.0f}s", flush=True)

    def raw_reply(self, path: Path, prompt_text: str) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        formatted = apply_chat_template(self.processor, self.config, prompt_text, num_images=1)
        return _to_text(generate(
            self.model, self.processor, formatted, [str(path)],
            max_tokens=self.max_tokens, temperature=0.0, verbose=False,
        ))


def _slug(model_path: str) -> str:
    return model_path.rstrip("/").split("/")[-1]


def run(model_path: str, max_tokens: int, rubric_name: str = "kluver") -> Path:
    if not SUBSET.exists():
        raise SystemExit("no human_subset.json — run `python human_rate.py --sample` first")
    items = json.loads(SUBSET.read_text())["items"]
    prereg = R.load_prereg()
    judge = MlxJudge(model_path, max_tokens)

    PROBES.mkdir(exist_ok=True)
    suffix = "" if rubric_name == "kluver" else f"_{rubric_name}"
    out_path = PROBES / f"probe_{_slug(model_path)}{suffix}.json"
    records: dict[str, dict] = {}
    t0 = time.time()
    for i, s in enumerate(items, start=1):
        path = RESULTS / s["model"] / s["filename"]
        try:
            raw_text = judge.raw_reply(path, R.build_rubric(s["prompt_id"], prereg))
            rec = R.coerce(R.extract_json(raw_text))
        except Exception as e:  # noqa: BLE001
            raw_text, rec = locals().get("raw_text", ""), {"error": str(e)[:300]}
        rec["_raw"] = str(raw_text)[:600]  # audit trail the original judges lacked
        records[s["blind_id"]] = rec
        shown = rec.get("error") or {k: rec[k] for k in R.ALL_FIELDS}
        print(f"[probe] ({i}/{len(items)}) {s['blind_id']} {shown}", flush=True)

    out_path.write_text(json.dumps({
        "model": model_path, "max_tokens": max_tokens,
        "rubric": rubric_name, "rubric_version": R.rubric_version(),
        "fields": list(R.INT_FIELDS),
        "n": len(items), "seconds": round(time.time() - t0, 1),
        "records": records,
    }, indent=2))
    print(f"\n[probe] wrote {out_path} in {time.time() - t0:.0f}s")
    return out_path


# ------------------------------- reporting ------------------------------------


def _claude_scores(rubric_name: str = "kluver") -> dict[str, dict]:
    """Claude's committed scores for the subset images, keyed by blind_id.

    Returns {} when Claude has not scored this scale yet, which is the normal
    state the first time a new rubric is probed. Gradedness is the screening
    criterion; rank agreement with Claude is a bonus when it happens to exist.
    """
    items = json.loads(SUBSET.read_text())["items"]
    suffix = "" if rubric_name == "kluver" else f"_{rubric_name}"
    by_model = {}
    for m in {s["model"] for s in items}:
        path = RESULTS / m / f"judgements_claude{suffix}.json"
        if not path.exists():
            return {}
        by_model[m] = json.loads(path.read_text())["images"]
    out = {}
    for s in items:
        rec = by_model[s["model"]].get(s["filename"])
        if rec and "error" not in rec:
            out[s["blind_id"]] = rec
    return out


def summarize(path: Path) -> dict:
    import numpy as np
    import statlib as S

    blob = json.loads(path.read_text())
    recs = blob["records"]
    ok = {b: r for b, r in recs.items() if "error" not in r}
    rubric_name = blob.get("rubric", "kluver")
    # Fields come from the probe file, so --compare reads mixed-scale runs.
    fields = tuple(blob.get("fields") or INT)
    claude = _claude_scores(rubric_name)

    dist = {f: {v: sum(1 for r in ok.values() if r[f] == v) for v in (0, 1, 2, 3)} for f in fields}
    interior = sum(dist[f][1] + dist[f][2] for f in fields)
    total = sum(sum(dist[f].values()) for f in fields)
    shared = [b for b in ok if b in claude]
    rho = {}
    for f in fields:
        a = np.array([ok[b][f] for b in shared], float)
        c = np.array([claude[b][f] for b in shared], float)
        rho[f] = round(S.spearman(a, c), 3) if len(set(a)) > 1 and len(set(c)) > 1 else float("nan")
    # All-nan when every field is flat (rho undefined), which numpy warns about.
    finite = [v for v in rho.values() if v == v]
    mean_rho = float(np.mean(finite)) if finite else float("nan")

    return {
        "model": blob["model"],
        "rubric": rubric_name,
        "n_ok": len(ok),
        "n_error": len(recs) - len(ok),
        "n_coerce_defaulted": sum(1 for r in ok.values() if r.get("missing_fields")),
        "distribution": dist,
        "gradedness": round(interior / total, 4) if total else 0.0,
        "flat_fields": [f for f in fields if len({r[f] for r in ok.values()}) < 2],
        "spearman_vs_claude": rho,
        "mean_spearman_vs_claude": round(mean_rho, 3) if mean_rho == mean_rho else None,
        "seconds": blob.get("seconds"),
    }


def compare() -> None:
    paths = sorted(PROBES.glob("probe_*.json")) if PROBES.exists() else []
    if not paths:
        raise SystemExit(f"no probe runs in {PROBES} — run one first")
    print(f"{'model':40s} {'scale':>7s} {'ok':>4s} {'err':>4s} {'graded':>7s} "
          f"{'rho_claude':>11s}  flat fields")
    for p in paths:
        s = summarize(p)
        mr = s["mean_spearman_vs_claude"]
        print(f"{_slug(s['model']):40s} {s['rubric']:>7s} {s['n_ok']:4d} {s['n_error']:4d} "
              f"{s['gradedness']:7.3f} {(f'{mr:11.3f}' if mr is not None else f'{'n/a':>11s}')}  "
              f"{','.join(s['flat_fields']) or '-'}")
    print("\ngradedness = share of field scores on an interior level (1 or 2). "
          "The two failed judges scored ~0.00.\nFlat fields never varied at all, "
          "which forces kappa to 0 by construction rather than by disagreement.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 03 candidate-judge screening probe.")
    p.add_argument("--model", help="MLX model path, e.g. mlx-community/Qwen2.5-VL-32B-Instruct-4bit")
    p.add_argument("--max-tokens", type=int, default=700,
                   help="700 (default) vs the original 400 that truncated ~90 replies")
    p.add_argument("--compare", action="store_true", help="table over all probe runs so far")
    p.add_argument("--rubric", choices=sorted(RUBRICS), default="kluver",
                   help="which scale to screen on. A judge that cleared one "
                        "scale has NOT thereby cleared another: re-probe.")
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    if a.compare:
        compare()
    elif a.model:
        _use_rubric(a.rubric)
        summary = summarize(run(a.model, a.max_tokens, a.rubric))
        print(json.dumps(summary, indent=2))
    else:
        build_parser().error("pass --model or --compare")
