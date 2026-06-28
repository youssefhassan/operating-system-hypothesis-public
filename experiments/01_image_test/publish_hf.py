"""
Publish Exp 01 results-local artifacts to Hugging Face Hub as a dataset.

Uploads generated images, blind judgements, analysis reports, figures, and
dose-sweep GIFs. Skips logs, macOS junk, and symlink browse folders.

Usage:
    python publish_hf.py --dry-run
    python publish_hf.py --repo-id youssefhassan13/exp01-guidance-sweep
    python publish_hf.py --repo-id youssefhassan13/exp01-guidance-sweep --private

Requires HF_TOKEN in the project-root .env (same token used for gated models).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULTS_ROOT = HERE / "results-local"
PREREG = HERE / "preregistration.json"

MODELS = ("sdxl", "sd35")
SKIP_NAMES = {".DS_Store", "Thumbs.db"}
SKIP_SUFFIXES = (".log",)
# Root-level symlink browse trees only (not figures/by_seed/gifs).
SKIP_ROOT_DIRS = {"by_seed", "by_guidance", "_hf_export"}


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


def _should_include(rel: Path) -> bool:
    if rel.name in SKIP_NAMES:
        return False
    if rel.name.endswith(SKIP_SUFFIXES):
        return False
    if rel.parts and rel.parts[0] in SKIP_ROOT_DIRS:
        return False
    return True


def _copy_model_tree(src: Path, dst: Path) -> dict[str, int]:
    counts = {"png": 0, "gif": 0, "json": 0, "other": 0}
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        if not _should_include(rel):
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        ext = p.suffix.lower()
        if ext == ".png":
            counts["png"] += 1
        elif ext == ".gif":
            counts["gif"] += 1
        elif ext == ".json":
            counts["json"] += 1
        else:
            counts["other"] += 1
    return counts


def _dataset_readme(repo_id: str, manifest: dict) -> str:
    prereg = json.loads(PREREG.read_text())
    prompt = prereg["prompt"]
    grid = ", ".join(str(g) for g in prereg["guidance_grid"])
    github = "https://github.com/youssefhassan/operating-system-hypothesis"
    exp_path = f"{github}/tree/main/experiments/01_image_test"

    return f"""---
license: mit
task_categories:
  - text-to-image
tags:
  - diffusion
  - classifier-free-guidance
  - stable-diffusion-xl
  - stable-diffusion-3
  - guidance-scale
  - hallucination
  - research
  - operating-system-hypothesis
language:
  - en
size_categories:
  - 100<n<1K
---

# Exp 01 — CFG guidance sweep (SDXL + SD 3.5)

Pre-registered confirmatory experiment from the [Operating System Hypothesis]({github})
project: sweep classifier-free guidance on fixed prompt and seeds, compare
convolutional UNet (SDXL) vs MMDiT (Stable Diffusion 3.5).

## Prompt (fixed)

> {prompt}

## Grid

- **Guidance:** {grid}
- **Seeds:** 42–51 (10 per setting)
- **Unconditional baseline:** empty prompt, same seeds
- **Images per model:** 100 PNGs (90 conditioned + 10 uncond)

## Confirmatory result

**Null on geometric structure M(g)** on both architectures (|Spearman ρ| ≪ 0.3).
Exploratory Suzuki-style rubric curves (veridicality, spontaneity, complexity)
show a strong guidance dose–response on SDXL and a compressed profile on SD 3.5.

## Layout

```
sdxl/
  g{{guidance}}_s{{seed}}.png    # conditioned outputs
  uncond_s{{seed}}.png           # empty-prompt baseline
  judgements.json                # blind VLM scores
  analysis_report.json           # confirmatory + exploratory stats
  figures/                       # M curve, rubric curves, contact sheets, GIFs
sd35/
  (same layout)
preregistration.json
manifest.json
```

## Provenance

- **Code / prereg:** [{exp_path}]({exp_path})
- **Git commit:** `{manifest.get("git_commit", "unknown")}`
- **Exported:** {manifest.get("exported_at", "unknown")}
- **Judge:** Claude Sonnet (blind, shuffled order)
- **PNG count:** {manifest.get("counts", {})}

## Citation

If you use this dataset, please link the GitHub repo and note the pre-registration
in `preregistration.json`. A formal paper citation will be added when available.

## License

MIT — same as the parent repository. Generated images are research artifacts;
underlying Stable Diffusion weights remain subject to their respective model licenses.
"""


def build_export(export_dir: Path) -> dict:
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "experiment": "01_image_test",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_sha(),
        "models": {},
        "counts": {"png": 0, "gif": 0, "json": 0, "other": 0},
    }

    shutil.copy2(PREREG, export_dir / "preregistration.json")

    for model in MODELS:
        src = RESULTS_ROOT / model
        if not src.is_dir():
            raise SystemExit(f"missing {src} — run the sweep and analyze first")
        dst = export_dir / model
        counts = _copy_model_tree(src, dst)
        manifest["models"][model] = counts
        for k, v in counts.items():
            manifest["counts"][k] += v

    audit = RESULTS_ROOT / "iterations.jsonl"
    if audit.exists():
        shutil.copy2(audit, export_dir / "iterations.jsonl")
        manifest["counts"]["other"] += 1

    (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run(args: argparse.Namespace) -> None:
    load_dotenv(REPO_ROOT / ".env")

    with tempfile.TemporaryDirectory(prefix="exp01_hf_") as tmp:
        export_dir = Path(tmp) / "dataset"
        manifest = build_export(export_dir)
        readme = _dataset_readme(args.repo_id, manifest)
        (export_dir / "README.md").write_text(readme)

        print(json.dumps(manifest, indent=2))
        total_png = manifest["counts"]["png"]
        if total_png < 100:
            raise SystemExit(f"expected ~200 PNGs, found {total_png} — export incomplete")

        if args.dry_run:
            print(f"\n[publish_hf] dry-run: would upload {export_dir} -> {args.repo_id}")
            for p in sorted(export_dir.rglob("*"))[:20]:
                if p.is_file():
                    print(f"  {p.relative_to(export_dir)}")
            print(f"  ... ({sum(1 for _ in export_dir.rglob('*') if _.is_file())} files total)")
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
                f"HF_TOKEN is for user {user!r} but --repo-id namespace is {expected_ns!r}. "
                f"Use --repo-id {user}/exp01-guidance-sweep"
            )
        scoped = who.get("auth", {}).get("accessToken", {}).get("fineGrained", {}).get("scoped", [])
        perms: set[str] = set()
        for s in scoped:
            perms.update(s.get("permissions", []))
        if not any("write" in p for p in perms):
            raise SystemExit(
                "HF_TOKEN is read-only (repo.content.read only). "
                "Create a new token at https://huggingface.co/settings/tokens with "
                "'Write access to contents/settings of all repos under your personal namespace', "
                "update HF_TOKEN in .env, then re-run."
            )

        create_repo(
            args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        print(f"\n[publish_hf] uploading to https://huggingface.co/datasets/{args.repo_id} ...")
        api.upload_folder(
            folder_path=str(export_dir),
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=args.message or f"Exp 01 export {manifest['exported_at'][:10]}",
        )
        print(f"[publish_hf] done -> https://huggingface.co/datasets/{args.repo_id}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Publish Exp 01 results to Hugging Face Hub.")
    p.add_argument(
        "--repo-id",
        default="youssefhassan13/exp01-guidance-sweep",
        help="HF dataset repo (default: youssefhassan13/exp01-guidance-sweep)",
    )
    p.add_argument("--private", action="store_true", help="create/upload as private dataset")
    p.add_argument("--dry-run", action="store_true", help="build export and print manifest only")
    p.add_argument("--message", default="", help="optional commit message on HF")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
