"""
Publish Exp 03 / 03b artifacts to Hugging Face Hub as a dataset.

The 860 generated PNGs are the reason this dataset exists: every text artifact
(pre-registrations, judgements, reports, analyses) is already public on GitHub,
but the images are gitignored and live only on this machine.

WHERE EACH FILE COMES FROM, and why it matters:

  * Images  -> the PRIVATE results-local tree. They exist nowhere else.
  * Text    -> the PUBLIC repo, not the private one. The public copies were
               content-scrubbed by git-filter-repo (see scripts/sync_public.sh),
               and re-deriving them here would silently undo that scrubbing.
               Copying the already-public file is the only way to guarantee the
               HF text matches the GitHub text.

That split is the whole safety argument. Do not "simplify" it by reading text
from the private tree.

Denylist and audit mirror scripts/sync_public.sh, because a public dataset is
exactly as public as a public repo and the 2026-08-09 near-miss (working logs
full of unpublished ideas) applies here too.

Usage:
    python publish_hf.py --dry-run          # build, audit, print manifest. No upload.
    python publish_hf.py --repo-id youssefhassan13/exp03-l23-hardening
    python publish_hf.py --repo-id youssefhassan13/exp03-l23-hardening --private

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
RESULTS_ROOT = HERE / "results-local"
PUBLIC_ROOT = Path(
    os.environ.get("PUBLIC_REPO", Path.home() / "Development" / "operating-system-hypothesis-public")
)
PUBLIC_EXP = PUBLIC_ROOT / "experiments" / "03_l23_hardening"

GITHUB = "https://github.com/youssefhassan/operating-system-hypothesis-public"
EXP_URL = f"{GITHUB}/tree/main/experiments/03_l23_hardening"

MODELS = ("sdxl", "sd35")
SKIP_NAMES = {".DS_Store", "Thumbs.db"}

# Never cross, even from the public tree. log.md and HANDOFF.md mirror
# sync_public.sh. The *.log files are a gap in that denylist: they are download
# progress noise and they embed absolute /Users/<name>/ paths.
DENY_RE = re.compile(r"(^|/)(log\.md|HANDOFF\.md|[^/]*\.log)$")

# Text artifacts to lift from the PUBLIC repo. Everything here is already
# published on GitHub; the dataset carries a copy so it stands alone.
PUBLIC_TEXT = (
    "preregistration.json",
    "preregistration_axes.json",
    "analysis.md",
    "analysis_axes.md",
    "analysis_plan.md",
    "axes_report.json",
    "posthoc_report.json",
    "human_subset.json",
    "human_ratings.json",
    "README.md",
    "RUN_AXES.md",
)
PUBLIC_TEXT_DIRS = ("probes", "archive")

# Per-model text that lives under results-local in the public repo.
MODEL_TEXT = (
    "judgements_claude.json",
    "judgements_qwen.json",
    "judgements_qwen_raw.json",
    "judgements_claude_axes.json",
    "judgements_qwen_axes.json",
    "judgements_qwen_axes_raw.json",
    "judgements_llama.json",
    "l23_report.json",
    "l23_report_claude-qwen.json",
    "quality.json",
    "metadata.json",
)

# Anything matching these in the built export fails the audit. Mirrors
# sync_public.sh's audit grep.
AUDIT_RE = re.compile(
    r"BRAIN\.md|BRAIN §|TIMELINE\.md|PLAN_II|IDEAS\.md|handover_fellows|substack/"
    r"|LinkedIn|outreach|visa|Exeter|research role|/Users/",
    re.IGNORECASE,
)
AUDIT_SUFFIXES = {".md", ".json", ".txt", ".py", ".sh", ".csv"}


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_images(model: str, dst_dir: Path) -> dict[str, int]:
    """Images come from the private tree; they are gitignored and exist nowhere else."""
    src = RESULTS_ROOT / model
    if not src.is_dir():
        raise SystemExit(f"missing {src} — the images are not on this machine")

    counts = {"png": 0, "figure": 0}
    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.name in SKIP_NAMES:
            continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif"}:
            continue
        rel = p.relative_to(src)
        if DENY_RE.search(str(rel)):
            continue
        _copy(p, dst_dir / rel)
        counts["figure" if rel.parts[0].startswith("figures") else "png"] += 1
    return counts


def _copy_public_text(dst_root: Path) -> int:
    """Text comes from the PUBLIC repo so the HF copy cannot diverge from GitHub."""
    if not PUBLIC_EXP.is_dir():
        raise SystemExit(
            f"no public export at {PUBLIC_EXP}. "
            "Run scripts/sync_public.sh first; text must come from the scrubbed copy."
        )

    n = 0
    for name in PUBLIC_TEXT:
        src = PUBLIC_EXP / name
        if not src.exists():
            raise SystemExit(f"expected {src} in the public repo — sync it before publishing")
        _copy(src, dst_root / name)
        n += 1

    for d in PUBLIC_TEXT_DIRS:
        src_dir = PUBLIC_EXP / d
        if not src_dir.is_dir():
            continue
        for p in sorted(src_dir.rglob("*")):
            if p.is_file() and p.name not in SKIP_NAMES and not DENY_RE.search(str(p)):
                _copy(p, dst_root / d / p.relative_to(src_dir))
                n += 1

    for model in MODELS:
        for name in MODEL_TEXT:
            src = PUBLIC_EXP / "results-local" / model / name
            if src.exists():
                _copy(src, dst_root / model / name)
                n += 1
    return n


def audit(export_dir: Path) -> list[str]:
    """Grep the built export for private references. Same rules as sync_public.sh."""
    hits: list[str] = []
    for p in sorted(export_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in AUDIT_SUFFIXES:
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            m = AUDIT_RE.search(line)
            if m:
                hits.append(f"{p.relative_to(export_dir)}:{i}: …{line[max(0, m.start() - 40):m.end() + 40]}…")
    return hits


def _dataset_readme(manifest: dict) -> str:
    prereg = json.loads((HERE / "preregistration.json").read_text())
    grid = ", ".join(str(g) for g in prereg.get("guidance_grid", []))
    counts = manifest.get("counts", {})

    return f"""---
license: mit
task_categories:
  - text-to-image
  - image-classification
tags:
  - diffusion
  - classifier-free-guidance
  - stable-diffusion-xl
  - stable-diffusion-3
  - guidance-scale
  - hallucination
  - vlm-as-judge
  - pre-registered
  - operating-system-hypothesis
language:
  - en
size_categories:
  - n<1K
---

# Exp 03 / 03b — Klüver L2/3 hardening study (SDXL + SD 3.5)

Pre-registered study from the [Operating System Hypothesis]({GITHUB}) project.
Sweep classifier-free guidance across two architectures and score every output
blind, on two independent rubrics, for how far object structure has come apart.

The prediction was written down and committed **before** the run. The commit
dates in the GitHub repo are the proof.

## What is here that is not on GitHub

The **{counts.get('png', 0)} generated PNGs**. Every text artifact in this dataset
is also in the [public repo]({EXP_URL}); the images are gitignored there because
of their size, so this dataset is the reproduction path for anything that needs
to look at the actual pictures.

## Design

- **Guidance grid:** {grid}
- **Prompts:** 6, including a designed control
- **Seeds:** 10 per cell, held fixed across guidance and model
- **Unconditional baseline:** empty prompt, same seeds
- **Models:** SDXL (convolutional UNet) and Stable Diffusion 3.5 (MMDiT)

## Two rubrics on the same images

- **Exp 03, local scale.** Klüver Level 2/3: reduplication, fragmentation,
  condensation, distortion, each 0–3, plus a binary tiling flag.
- **Exp 03b, global scale.** Suzuki-style continuous axes: veridicality,
  spontaneity, complexity. Pre-registered separately, before scoring.

## Headline result, including the miss

The overall pre-registered claim required **both** models to clear every gate,
and **it did not confirm**.

| | SDXL | SD 3.5 |
|---|---|---|
| Dose-response slope (needed ≤ −0.20) | −0.34 ✓ | −0.18 ✗ |
| Partial ρ, quality controlled | −0.43 ✓ | −0.25 ✓ |
| Prompts with a negative slope | 6 of 6 ✓ | 5 of 6 ✓ |
| Inter-judge agreement (needed ≥ 0.40) | 0.56 ✓ | 0.44 ✓ |

Post-hoc, the two models differ in **shape** rather than magnitude: SDXL is a
gradient that survives dropping the lowest guidance setting, SD 3.5 is a cliff
that does not.

## The judges are part of the data

Two of the three originally budgeted judges were **inert** and failed silently:
well-formed JSON, fluent captions, and zero variance across the whole scale.
`judgements_llama.json` and `archive/judgements_qwen_Qwen2.5-VL-7B_*.json` are
kept deliberately so that failure is inspectable rather than described. The
screening probes in `probes/` show the capacity threshold between 8B and 32B.

If you are building a VLM-as-judge pipeline, that is the most transferable
thing in this dataset.

## Layout

```
sdxl/
  p{{prompt}}_g{{guidance}}_s{{seed}}.png   # conditioned outputs
  uncond_s{{seed}}.png                    # empty-prompt baseline
  judgements_claude.json                  # judge A, blind
  judgements_qwen.json                    # judge B, blind (Qwen3-VL-32B)
  judgements_llama.json                   # dead rater, kept as evidence
  judgements_*_axes.json                  # Exp 03b, Suzuki axes
  l23_report_claude-qwen.json             # amended confirmatory panel
  l23_report.json                         # pre-amendment 3-judge record
  quality.json, metadata.json
  figures/, figures_claude-qwen/
sd35/   (same layout)
preregistration.json, preregistration_axes.json
analysis.md, analysis_axes.md, analysis_plan.md
posthoc_report.json, axes_report.json
human_subset.json, human_ratings.json     # blind human validation subset
probes/, archive/
manifest.json
```

## Limitations, stated up front

- One human rater, who is the author, and who knows the hypothesis. Guidance and
  model were hidden and order was shuffled.
- Human–judge agreement is **0.34** (Claude) and **0.12** (Qwen3-VL-32B). The two
  models agree with each other more than either agrees with the human. Clearing
  an inter-judge threshold is not the same as measuring what a person sees.
- The distortion field partly tracks **rendering style** rather than object
  structure: holding photographic realism fixed shrinks it from −0.49 to −0.18.
- Two lineages of judge, not three.

## Provenance

- **Code, pre-registration, full write-up:** [{EXP_URL}]({EXP_URL})
- **Git commit:** `{manifest.get("git_commit", "unknown")}`
- **Exported:** {manifest.get("exported_at", "unknown")}
- **Files:** {json.dumps(counts)}

## License

MIT, same as the parent repository. Generated images are research artifacts;
the underlying Stable Diffusion weights remain under their own model licenses.
"""


def build_export(export_dir: Path) -> dict:
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "experiment": "03_l23_hardening",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_sha(),
        "models": {},
        "counts": {"png": 0, "figure": 0, "text": 0},
    }

    for model in MODELS:
        counts = _copy_images(model, export_dir / model)
        manifest["models"][model] = counts
        manifest["counts"]["png"] += counts["png"]
        manifest["counts"]["figure"] += counts["figure"]

    manifest["counts"]["text"] = _copy_public_text(export_dir)
    (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run(args: argparse.Namespace) -> None:
    load_dotenv(REPO_ROOT / ".env")

    with tempfile.TemporaryDirectory(prefix="exp03_hf_") as tmp:
        export_dir = Path(tmp) / "dataset"
        manifest = build_export(export_dir)
        (export_dir / "README.md").write_text(_dataset_readme(manifest))

        print(json.dumps(manifest, indent=2))

        total_png = manifest["counts"]["png"]
        if total_png < 800:
            raise SystemExit(f"expected ~860 PNGs, found {total_png} — export incomplete")

        print("\n=== AUDIT: private references in the export ===")
        hits = audit(export_dir)
        if hits:
            for h in hits[:40]:
                print(f"  {h}")
            if len(hits) > 40:
                print(f"  ... and {len(hits) - 40} more")
            raise SystemExit(
                "!! AUDIT FAILED — nothing was uploaded. "
                "Fix the source file in the public repo, re-run sync_public.sh, then retry."
            )
        print("clean")

        size_mb = sum(p.stat().st_size for p in export_dir.rglob("*") if p.is_file()) / 1e6
        n_files = sum(1 for p in export_dir.rglob("*") if p.is_file())
        print(f"\n{n_files} files, {size_mb:.0f} MB")

        if args.dry_run:
            print(f"\n[publish_hf] dry-run: would upload -> {args.repo_id}")
            print("  first 15 files:")
            for p in sorted(export_dir.rglob("*"))[:15]:
                if p.is_file():
                    print(f"    {p.relative_to(export_dir)}")
            return

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN not set — add it to the project-root .env")

        from huggingface_hub import HfApi, create_repo

        api = HfApi(token=token)
        who = api.whoami()
        user = who.get("name", "")
        expected_ns = args.repo_id.split("/")[0]
        if user and user != expected_ns:
            raise SystemExit(
                f"HF_TOKEN is for user {user!r} but --repo-id namespace is {expected_ns!r}."
            )
        scoped = who.get("auth", {}).get("accessToken", {}).get("fineGrained", {}).get("scoped", [])
        perms: set[str] = set()
        for s in scoped:
            perms.update(s.get("permissions", []))
        if perms and not any("write" in p for p in perms):
            raise SystemExit(
                "HF_TOKEN is read-only. Create a write-scoped token at "
                "https://huggingface.co/settings/tokens and update .env."
            )

        create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        print(f"\n[publish_hf] uploading to https://huggingface.co/datasets/{args.repo_id} ...")
        api.upload_folder(
            folder_path=str(export_dir),
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=args.message or f"Exp 03 export {manifest['exported_at'][:10]}",
        )
        print(f"[publish_hf] done -> https://huggingface.co/datasets/{args.repo_id}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Publish Exp 03 results to Hugging Face Hub.")
    p.add_argument(
        "--repo-id",
        default="youssefhassan13/exp03-l23-hardening",
        help="HF dataset repo (default: youssefhassan13/exp03-l23-hardening)",
    )
    p.add_argument("--private", action="store_true", help="create/upload as private dataset")
    p.add_argument("--dry-run", action="store_true", help="build and audit only; no upload")
    p.add_argument("--message", default="", help="optional commit message on HF")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
