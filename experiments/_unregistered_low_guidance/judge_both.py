"""
Run the Qwen3-VL judge over the sub-CFG sweep, into a tagged result file.

Single judge (Qwen3-VL-32B) by decision: the 30B-A3B second rater was abandoned
after its download stalled, so there is NO cross-judge agreement for this run and
its scores rest on one scorer's idiosyncrasies. Exp 03's judge_qwen.py
is reused *unmodified* (it is committed evidence for Exp 03); this only rebinds its
module-level output filenames and model id per pass, so each judge writes
`judgements_qwen_<tag>.json` instead of both fighting over one file.

Sequential by construction: the 30B-A3B and the 32B are ~17-19 GB each in 4-bit and
loading both at once would thrash a 64 GB host.

Usage:
    python judge_both.py                       # both judges, both models
    python judge_both.py --judges 30b          # one judge
    python judge_both.py --models sdxl
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP03 = HERE.parent / "03_l23_hardening"
sys.path.insert(0, str(EXP03))

# Only the 32B is used. The 30B-A3B was tried and abandoned: it was never in the
# local cache (contrary to expectation) and its download stalled at 10 GB of ~17 GB.
# It is left in the map so re-adding it is a one-word change, not a rewrite.
JUDGES = {
    "32b": "mlx-community/Qwen3-VL-32B-Instruct-4bit",      # dense — the judge in use
    "30b": "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit",  # MoE, not downloaded
}
DEFAULT_JUDGES = ["32b"]


def out_names(tag: str) -> tuple[str, str]:
    return f"judgements_qwen_{tag}.json", f"judgements_qwen_{tag}_raw.json"


def adopt_unsuffixed(model_dir: Path, tag: str) -> None:
    """run.sh's first pass wrote the plain `judgements_qwen.json`. If that file is
    from this judge, move it into the suffixed name rather than re-scoring 110
    images for nothing."""
    plain = model_dir / "judgements_qwen.json"
    if not plain.exists():
        return
    dest = model_dir / out_names(tag)[0]
    if dest.exists():
        return
    import json
    try:
        if json.loads(plain.read_text()).get("model") != JUDGES[tag]:
            return
    except Exception:
        return
    shutil.move(str(plain), str(dest))
    raw = model_dir / "judgements_qwen_raw.json"
    if raw.exists():
        shutil.move(str(raw), str(model_dir / out_names(tag)[1]))
    print(f"[both] adopted existing {plain.name} -> {dest.name}")


def run(tags: list[str], models: list[str], overwrite: bool) -> None:
    import judge_qwen as J

    for tag in tags:
        for model in models:
            model_dir = (HERE / "results-local" / model).resolve()
            if not model_dir.is_dir():
                print(f"[both] no results-local/{model} yet — skipping")
                continue
            adopt_unsuffixed(model_dir, tag)
            J.QWEN_MODEL = JUDGES[tag]
            J.OUT_FILE, J.RAW_FILE = out_names(tag)
            print(f"\n[both] === judge {tag} ({JUDGES[tag]}) on {model} ===", flush=True)
            args = argparse.Namespace(dir=str(model_dir), overwrite=overwrite,
                                      shuffle_seed=0)
            J.score(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES, choices=sorted(JUDGES))
    ap.add_argument("--models", nargs="+", default=["sdxl", "sd35"])
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    run(a.judges, a.models, a.overwrite)
