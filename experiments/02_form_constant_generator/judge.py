"""
Experiment 02: score the Stage A renders and the negative sets with the archived
Exp 01 judge (experiments/01_image_test/judge.py, rubric exp01-formconstant-v1,
claude-sonnet-4-6), unchanged, blind and shuffled.

Builds results-local/judge_set/ with blind ids -> {renders, blank renders,
photographic negatives from Exp 03}, then writes results-local/judgements.json.

Usage:  python judge.py --build      # draw negatives, assemble the blind set
        python judge.py              # score (resumable)
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01_image_test"))
import judge as exp01  # noqa: E402  (the archived instrument)

P = json.loads((HERE / "params.json").read_text())
OUT = HERE / "results-local"
SET = OUT / "judge_set"
ROOT = HERE.parents[1]


def _rel(p: Path) -> str:
    """Repo-relative path, so the manifest carries no home-directory path."""
    return str(Path(p).resolve().relative_to(ROOT))


def build() -> None:
    meta = json.loads((OUT / "metadata.json").read_text())
    items = []
    for r in meta["runs"]:
        items.append({"kind": "render", "src": _rel(OUT / "renders" / f"{r['name']}.png"), **r})
    # photographic negatives from Exp 03, drawn once with a fixed seed
    ng = P["negatives"]
    rng = np.random.default_rng(ng["draw_seed"])
    prompts = ["p1_stilllife", "p2_portrait", "p3_bicycle", "p4_oranges", "p5_livingroom", "p6_forest"]
    for arch in ("sdxl", "sd35"):
        d = HERE.parent / "03_l23_hardening" / "results-local" / arch
        per_prompt = [ng["per_arch"] // 6 + (1 if i < ng["per_arch"] % 6 else 0) for i in range(6)]
        for prompt, k in zip(prompts, per_prompt):
            pool = sorted(p for g in ng["guidance"] for p in d.glob(f"{prompt}_g{g}_s*.png"))
            for p in rng.choice(pool, size=k, replace=False):
                items.append({"kind": "photo", "src": _rel(Path(p)), "arch": arch, "prompt": prompt, "file": Path(p).name})
    random.Random(0).shuffle(items)
    SET.mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(items):
        it["blind_id"] = f"b{i:03d}"
        dst = SET / f"{it['blind_id']}.png"
        if not dst.exists():
            # normalise everything to PNG at the file level; renders already are
            from PIL import Image
            Image.open(ROOT / it["src"]).convert("RGB").save(dst)
    (OUT / "negatives.json").write_text(json.dumps([i for i in items if i["kind"] == "photo"], indent=2))
    (OUT / "judge_manifest.json").write_text(json.dumps(items, indent=2))
    print(f"[build] {len(items)} images: {sum(i['kind']=='render' for i in items)} renders, "
          f"{sum(i['kind']=='photo' for i in items)} photographic negatives")


def score() -> None:
    exp01._require_keys(["claude"])
    manifest = json.loads((OUT / "judge_manifest.json").read_text())
    out_path = OUT / "judgements.json"
    res = json.loads(out_path.read_text())["images"] if out_path.exists() else {}
    from concurrent.futures import ThreadPoolExecutor, as_completed
    todo = [it["blind_id"] for it in manifest if not (it["blind_id"] in res and "error" not in res[it["blind_id"]])]
    def one(bid):
        try:
            return bid, exp01.judge_claude(SET / f"{bid}.png")
        except Exception as e:  # noqa: BLE001
            return bid, {"error": str(e)}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, fut in enumerate(as_completed([ex.submit(one, b) for b in todo]), 1):
            bid, r = fut.result(); res[bid] = r
            print(f"[judge] ({i}/{len(todo)}) {bid} -> {r}")
            out_path.write_text(json.dumps({"rubric_version": exp01.RUBRIC_VERSION,
                                            "claude_model": exp01.CLAUDE_MODEL,
                                            "images": res}, indent=2))
    print(f"[judge] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    build() if a.build else score()
