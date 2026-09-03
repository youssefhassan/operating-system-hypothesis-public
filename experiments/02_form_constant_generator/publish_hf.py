"""
Publish Exp 02 Stage A artifacts to Hugging Face Hub as a dataset.

Same split as 03_l23_hardening/publish_hf.py, for the same reason:

  * Renders, fields and the blind judge set -> the PRIVATE results-local tree
    (gitignored; they exist nowhere else).
  * Text -> the PUBLIC repo copy, which sync_public.sh has already redacted.
    Re-deriving it from the private tree would silently undo that.

Usage:
    python publish_hf.py --dry-run
    python publish_hf.py --repo-id youssefhassan13/exp02-form-constant-generator

Requires HF_TOKEN (write-scoped) in the project-root .env.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULTS = HERE / "results-local"
PUBLIC_ROOT = Path(os.environ.get("PUBLIC_REPO", Path.home() / "Development" / "operating-system-hypothesis-public"))
PUBLIC_EXP = PUBLIC_ROOT / "experiments" / "02_form_constant_generator"
GITHUB = "https://github.com/youssefhassan/operating-system-hypothesis-public"
EXP_URL = f"{GITHUB}/tree/main/experiments/02_form_constant_generator"

DENY_RE = re.compile(r"(^|/)(log\.md|HANDOFF\.md|[^/]*\.log)$")
PUBLIC_TEXT = ("preregistration.json", "params.json", "README.md", "analysis.md", "run.py", "judge.py", "analyze.py")
PUBLIC_RESULT_TEXT = ("metadata.json", "judge_manifest.json", "negatives.json", "judgements.json", "report.json")
AUDIT_PATTERNS = REPO_ROOT / "scripts" / "audit_patterns.txt"
AUDIT_SUFFIXES = {".md", ".json", ".txt", ".py", ".sh", ".csv"}


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_private_binaries(dst: Path) -> dict[str, int]:
    counts = {"render_png": 0, "field_npy": 0, "judge_set_png": 0, "figure_png": 0}
    for sub, key, suf in (("renders", "render_png", ".png"), ("fields", "field_npy", ".npy"),
                          ("judge_set", "judge_set_png", ".png"), ("figures", "figure_png", ".png")):
        d = RESULTS / sub
        if not d.is_dir():
            raise SystemExit(f"missing {d}; run run.py / judge.py --build first")
        for p in sorted(d.glob(f"*{suf}")):
            _copy(p, dst / sub / p.name)
            counts[key] += 1
    return counts


def _copy_public_text(dst: Path) -> int:
    if not PUBLIC_EXP.is_dir():
        raise SystemExit(f"no public export at {PUBLIC_EXP}; run scripts/sync_public.sh first")
    n = 0
    for name in PUBLIC_TEXT:
        src = PUBLIC_EXP / name
        if not src.exists():
            raise SystemExit(f"expected {src} in the public repo; sync before publishing")
        _copy(src, dst / name); n += 1
    for name in PUBLIC_RESULT_TEXT:
        src = PUBLIC_EXP / "results-local" / name
        if not src.exists():
            raise SystemExit(f"expected {src} in the public repo; sync before publishing")
        _copy(src, dst / name); n += 1
    return n


def audit(export_dir: Path) -> list[str]:
    if not AUDIT_PATTERNS.exists():
        print("!! audit SKIPPED: scripts/audit_patterns.txt not found")
        return []
    rxs = [re.compile(l, re.IGNORECASE) for l in AUDIT_PATTERNS.read_text().splitlines() if l.strip()]
    hits = []
    for p in sorted(export_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in AUDIT_SUFFIXES:
            continue
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            for rx in rxs:
                m = rx.search(line)
                if m:
                    hits.append(f"{p.relative_to(export_dir)}:{i}: …{line[max(0, m.start()-40):m.end()+40]}…")
                    break
    return hits


def _readme(manifest: dict) -> str:
    c = manifest["counts"]
    return f"""---
license: mit
tags:
  - neural-field
  - ermentrout-cowan
  - bressloff
  - form-constants
  - hallucination
  - vlm-as-judge
  - positive-control
  - pre-registered
  - operating-system-hypothesis
language:
  - en
size_categories:
  - n<1K
---

# Exp 02 — Ermentrout–Cowan form-constant generator (Stage A)

Pre-registered experiment from the [Operating System Hypothesis]({GITHUB})
project. The only experiment in the program that runs the biological mechanism
as code: a scalar neural field on a periodic cortical sheet, pushed through its
Turing bifurcation by the gain μ and rendered to the visual field through the
inverse complex-log map. Then the archived Exp 01 judge (Claude Sonnet 4.6,
rubric unchanged) scored every render blind, mixed with blank renders and
ordinary generated scenes, as a **positive control** for the Exp 01 null.

Pre-registered 2026-06-29; run 2026-09-03 under two dated amendments committed
before any scoring. The commit dates in the public repo are the proof.

## What is here that is not on GitHub

- `renders/` — {c['render_png']} visual-field renders, 512 × 512, `{{regime}}_mu{{μ/μ_c}}_s{{seed}}.png`
- `fields/` — {c['field_npy']} steady cortical fields, 256 × 256 float32 `.npy`
- `judge_set/` — the {c['judge_set_png']} blind images exactly as the judge saw them (`b000`…), mapped by `judge_manifest.json`
- `figures/` — contact sheet and the curve / control figure

Every text artifact is also in the [public repo]({EXP_URL}).

## Design

- Model: τ ∂a/∂t = −a + f(μ (w ∗ a)); difference-of-Gaussians kernel σ_e = 1, σ_i = 2; sheet 256², side 12 λ_c
- Regimes: **hex** (asymmetric sigmoid, pure-noise start → hexagonal lattices) and **stripe** (odd sigmoid, oriented seed at noise amplitude → rolls whose orientation gives tunnels, funnels, spirals)
- Sweep: μ ∈ {{0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5}} μ_c × 10 seeds × 2 regimes
- Negatives: 40 sub-threshold blanks; 40 images from the Exp 03 corpus at guidance 7 and 11

## Headline, including the miss

Replication: blank below μ_c, all four Klüver classes above it, monotone in μ.
Three of four gates pass; the planform → class specificity gate fails (the
judge is multi-label under the log-polar map).

Positive control for the Exp 01 judge:

| Set | n | judged structured |
|---|---|---|
| rendered form constants | 80 | 80 |
| blank renders | 40 | 0 |
| ordinary generated scenes | 40 | 18 |

The pre-specified control **failed on specificity**: the binary class flags
fire on literal grids (brick walls, window mullions, tiles). On the score Exp 01
actually used, no ordinary scene reaches the top level and every form constant
does. Both readings are in `analysis.md`; the second is labelled exploratory.

## Provenance

- Code, pre-registration, write-up: [{EXP_URL}]({EXP_URL})
- Git commit: `{manifest.get('git_commit')}`
- Exported: {manifest.get('exported_at')}
- Files: {json.dumps(c)}

## License

MIT, as the parent repository.
"""


def build(export_dir: Path) -> dict:
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": "02_form_constant_generator", "exported_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": _git_sha(), "counts": _copy_private_binaries(export_dir)}
    manifest["counts"]["text"] = _copy_public_text(export_dir)
    (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run(args: argparse.Namespace) -> None:
    load_dotenv(REPO_ROOT / ".env")
    with tempfile.TemporaryDirectory(prefix="exp02_hf_") as tmp:
        export_dir = Path(tmp) / "dataset"
        manifest = build(export_dir)
        (export_dir / "README.md").write_text(_readme(manifest))
        print(json.dumps(manifest, indent=2))
        if manifest["counts"]["render_png"] != 160 or manifest["counts"]["judge_set_png"] != 200:
            raise SystemExit("export incomplete: expected 160 renders and 200 judge-set images")
        print("\n=== AUDIT ===")
        hits = audit(export_dir)
        if hits:
            for h in hits[:40]:
                print(" ", h)
            raise SystemExit("!! AUDIT FAILED; nothing uploaded")
        print("clean")
        n = sum(1 for p in export_dir.rglob("*") if p.is_file())
        mb = sum(p.stat().st_size for p in export_dir.rglob("*") if p.is_file()) / 1e6
        print(f"{n} files, {mb:.0f} MB")
        if args.dry_run:
            print(f"[dry-run] would upload -> {args.repo_id}")
            return
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN missing from .env")
        from huggingface_hub import HfApi, create_repo
        create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True, token=token)
        HfApi(token=token).upload_folder(folder_path=str(export_dir), repo_id=args.repo_id, repo_type="dataset",
                                         commit_message=f"Exp 02 Stage A export {manifest['exported_at']} ({(manifest.get('git_commit') or '')[:7]})")
        print(f"uploaded -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="youssefhassan13/exp02-form-constant-generator")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    run(p.parse_args())
