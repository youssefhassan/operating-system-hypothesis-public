"""
Publish the sub-CFG guidance sweep to Hugging Face Hub as a dataset.

Same structure and same safety argument as Exp 03's publish_hf.py:

  * Images -> the PRIVATE results-local tree. They are gitignored and exist
              nowhere else.
  * Text   -> the PUBLIC repo, never the private one. The public copies were
              content-scrubbed by scripts/sync_public.sh; re-deriving them here
              would silently undo that scrubbing. Copying the already-public
              file is the only way to guarantee the HF text matches the GitHub
              text.

Do not "simplify" that split. A public dataset is exactly as public as a public
repo, and the same audit applies.

ONE DIFFERENCE FROM EXP 03 WORTH STATING LOUDLY: that dataset is pre-registered
and this one is not. There is no preregistration.json here, the card carries no
`pre-registered` tag, and the exploratory status is the first thing a reader
sees. Do not copy Exp 03's tag list onto this dataset.

Usage:
    python publish_hf.py --dry-run          # build, audit, print manifest. No upload.
    python publish_hf.py --repo-id youssefhassan13/low-guidance-cfg-sweep

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
PUBLIC_EXP = PUBLIC_ROOT / "experiments" / "_unregistered_low_guidance"

GITHUB = "https://github.com/youssefhassan/operating-system-hypothesis-public"
EXP_URL = f"{GITHUB}/tree/main/experiments/_unregistered_low_guidance"

MODELS = ("sdxl", "sd35")
SKIP_NAMES = {".DS_Store", "Thumbs.db"}
EXPECTED_PNG = 220  # 110 per model: 108 conditioned + 2 empty-prompt baselines

# Mirrors scripts/sync_public.sh. The *.log rule matters: those files embed
# absolute home-directory paths. The pattern is not spelled out in prose here
# because the audit greps this file too.
DENY_RE = re.compile(r"(^|/)(log\.md|HANDOFF\.md|[^/]*\.log)$")

# Text lifted from the PUBLIC repo — already on GitHub; the dataset carries a
# copy so it stands alone.
PUBLIC_TEXT = ("README.md", "analysis.md", "report.json")

# Per-model text, from the public repo's results-local tree.
MODEL_TEXT = (
    "judgements_claude.json",
    "judgements_qwen_32b.json",
    "judgements_qwen_32b_raw.json",
    "metadata.json",
)

# Patterns live OUTSIDE this file, in scripts/audit_patterns.txt, which sits
# outside the public-export allowlist — this script itself ships publicly, and a
# literal pattern list here would publish the meta-information the audit exists
# to catch.
AUDIT_PATTERNS = REPO_ROOT / "scripts" / "audit_patterns.txt"
AUDIT_SUFFIXES = {".md", ".json", ".txt", ".py", ".sh", ".csv"}


def _audit_re() -> "re.Pattern | None":
    if AUDIT_PATTERNS.exists():
        return re.compile(AUDIT_PATTERNS.read_text().strip(), re.IGNORECASE)
    return None


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_images(model: str, dst_dir: Path) -> dict[str, int]:
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
    if not PUBLIC_EXP.is_dir():
        raise SystemExit(
            f"no public export at {PUBLIC_EXP}. Run scripts/sync_public.sh first; "
            "text must come from the scrubbed copy."
        )
    n = 0
    for name in PUBLIC_TEXT:
        src = PUBLIC_EXP / name
        if not src.exists():
            raise SystemExit(f"expected {src} in the public repo — sync it before publishing")
        _copy(src, dst_root / name)
        n += 1
    for model in MODELS:
        for name in MODEL_TEXT:
            src = PUBLIC_EXP / "results-local" / model / name
            if src.exists():
                _copy(src, dst_root / model / name)
                n += 1
    return n


def audit(export_dir: Path) -> list[str]:
    rx = _audit_re()
    if rx is None:
        print("!! private-reference audit SKIPPED: scripts/audit_patterns.txt not found. "
              "Fine on a public checkout; on the private machine this is a red flag.")
        return []
    hits: list[str] = []
    for p in sorted(export_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in AUDIT_SUFFIXES:
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            m = rx.search(line)
            if m:
                hits.append(f"{p.relative_to(export_dir)}:{i}: "
                            f"…{line[max(0, m.start() - 40):m.end() + 40]}…")
    return hits


def _dataset_readme(manifest: dict) -> str:
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
  - vlm-as-judge
  - exploratory
  - operating-system-hypothesis
language:
  - en
size_categories:
  - n<1K
---

# Sub-CFG guidance sweep (g = 0 → 2), SDXL + SD 3.5

> **Exploratory. Not pre-registered. Not a result.**
> No hypothesis was committed before these runs, there is no pre-specified
> statistical model, and no p-values are reported anywhere in this dataset.
> The sibling [Exp 03 dataset](https://huggingface.co/datasets/youssefhassan13/exp03-l23-hardening)
> *is* pre-registered, with commit dates as proof. **This one is not.** Treat it
> as a reason to design an experiment, not as evidence for a claim.

From the [Operating System Hypothesis]({GITHUB}) project. Exp 01 and Exp 03 both
swept classifier-free guidance from 1.0 upward, because that is where diffusers
turns CFG on. This looks at the region *below* that boundary.

## What is here that is not on GitHub

The **{counts.get('png', 0)} generated PNGs**. Every text artifact is also in the
[public repo]({EXP_URL}); the images are gitignored there because of their size.

## The library behaviour this had to work around

Both pipelines gate CFG on `guidance_scale > 1`:

- `StableDiffusionXLPipeline.do_classifier_free_guidance` → `self._guidance_scale > 1 and ...`
- `StableDiffusion3Pipeline.do_classifier_free_guidance` → `self._guidance_scale > 1`

So a naive sweep of 0 → 2 returns **five byte-identical images** for g ≤ 1 and
only four distinct points above it. The flat floor would be an artifact of the
library, not of the model. `sweep_low_g.py` forces the CFG branch on at every
value, so `pred = pred_uncond + g * (pred_cond - pred_uncond)` runs throughout
and g = 0 is the pure negative-branch render.

`check_g0.py` is the validity check: if the patch had not taken, g = 0 would
still show the prompt. It asserts all six prompts collapse to one identical image
per seed at g = 0, and it passes on all 220 images.

**"Unconditional" is not one thing across architectures.** On SDXL the g = 0
image is *not* the empty-prompt baseline — `model_index.json` sets
`force_zeros_for_empty_prompt: true`, so the negative branch is a literal zero
vector, while `uncond_s{{seed}}.png` is the *encoding* of the empty string. On
SD 3.5, which passes an explicit `negative_prompt=""`, the two coincide.

## Design

- **Guidance:** 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0 (CFG forced on throughout)
- **Prompts:** the 6 from Exp 03's pre-registration, verbatim
- **Seeds:** 42, 43 — only two, which is the binding limit on everything below
- **Fixed:** 25 steps, 1024×1024
- **Models:** SDXL base 1.0 (UNet) and SD 3.5 medium (MMDiT)
- **Judges:** Qwen3-VL-32B (local) and Claude Sonnet 5, on Exp 03's rubric, blind and shuffled

## What it looks like

Every Klüver field declines as guidance rises, on both models, under both judges —
**all 20 field × model × judge correlations are negative**. Same sign as Exp 03,
roughly double the magnitude over this window, consistent with the structure being
a property of the prior that guidance suppresses. There is no U-shape.

## The judge disagreement is the interesting part

Composite quadratic-weighted κ is **+0.710** (SDXL) and **+0.661** (SD 3.5), across
two different model families. But that headline averages two regimes that fail in
opposite directions on the two models:

| | g < 1 | g ≥ 1 |
|---|---|---|
| SDXL | +0.36 … +0.64 | +0.48 … +0.86 |
| SD 3.5 | +0.47 … +0.82 | +0.21 … +0.40 |

Both are range restriction, mirrored. On SDXL the low-g cells are ceiling-pinned
**for one judge only**: Qwen scored 3-on-all-four-ordinal-fields for 26 of 48
low-g images, Claude for 0 of 48 — so the flat maximum is a property of that
judge, not of the images. On SD 3.5 the high-g cells sit near zero for both, and
`tiling` at g ≥ 1 is undefined outright because both judges scored every image 0.

Agreement is strongest exactly where the scores move. If you are building a
VLM-as-judge pipeline, that is the most transferable thing here.

## Layout

```
sdxl/
  p{{prompt}}_g{{guidance}}_s{{seed}}.png   # conditioned outputs, 9 guidance values
  uncond_s{{seed}}.png                    # empty-prompt baseline
  judgements_claude.json                  # Claude Sonnet 5, blind
  judgements_qwen_32b.json                # Qwen3-VL-32B, blind
  judgements_qwen_32b_raw.json            # raw replies, kept as audit trail
  metadata.json
  figures/                                # contact sheets, one row per prompt
sd35/   (same layout)
README.md, analysis.md, report.json
manifest.json
```

## Limitations, stated up front

- **Unregistered and exploratory.** No pre-specified hypothesis, model, or correction.
- **Two seeds.** Too thin for any claim about the character of the prior, and the
  two disagree sharply — at seed 42 SDXL's prior is a dense floral tiling and
  SD 3.5's is a readable scene; at seed 43 that reverses.
- **n = 2 at g = 0.** All six prompts render one identical image per seed, so that
  column holds two distinct images, not twelve. The analysis deduplicates by SHA
  before computing means, CIs and the pooled ρ.
- **Rubric ceiling.** The scale stops at 3 and one judge reaches it often at low g.
- **Not poolable with Exp 03.** Its g = 1.0 point was CFG-*off*; this one's is CFG-on.

## Provenance

- **Code and full write-up:** [{EXP_URL}]({EXP_URL})
- **Git commit:** `{manifest.get("git_commit", "unknown")}`
- **Exported:** {manifest.get("exported_at", "unknown")}
- **Files:** {json.dumps(counts)}

## License

MIT, same as the parent repository. Generated images are research artifacts; the
underlying Stable Diffusion weights remain under their own model licenses.
"""


def build_export(export_dir: Path) -> dict:
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "experiment": "_unregistered_low_guidance",
        "status": "exploratory, not pre-registered",
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

    with tempfile.TemporaryDirectory(prefix="lowg_hf_") as tmp:
        export_dir = Path(tmp) / "dataset"
        manifest = build_export(export_dir)
        (export_dir / "README.md").write_text(_dataset_readme(manifest))
        print(json.dumps(manifest, indent=2))

        total_png = manifest["counts"]["png"]
        if total_png != EXPECTED_PNG:
            raise SystemExit(
                f"expected {EXPECTED_PNG} PNGs, found {total_png} — export incomplete")

        print("\n=== AUDIT: private references in the export ===")
        hits = audit(export_dir)
        if hits:
            for h in hits[:40]:
                print(f"  {h}")
            if len(hits) > 40:
                print(f"  ... and {len(hits) - 40} more")
            raise SystemExit(
                "!! AUDIT FAILED — nothing was uploaded. Fix the source file in the "
                "public repo, re-run sync_public.sh, then retry.")
        print("clean")

        size_mb = sum(p.stat().st_size for p in export_dir.rglob("*") if p.is_file()) / 1e6
        n_files = sum(1 for p in export_dir.rglob("*") if p.is_file())
        print(f"\n{n_files} files, {size_mb:.0f} MB")

        if args.dry_run:
            print(f"\n[publish_hf] dry-run: would upload -> {args.repo_id}")
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
                f"HF_TOKEN is for user {user!r} but --repo-id namespace is {expected_ns!r}.")
        scoped = who.get("auth", {}).get("accessToken", {}).get("fineGrained", {}).get("scoped", [])
        perms: set[str] = set()
        for s in scoped:
            perms.update(s.get("permissions", []))
        if perms and not any("write" in p for p in perms):
            raise SystemExit(
                "HF_TOKEN is read-only. Create a write-scoped token at "
                "https://huggingface.co/settings/tokens and update .env.")

        create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        print(f"\n[publish_hf] uploading to https://huggingface.co/datasets/{args.repo_id} ...")
        api.upload_folder(
            folder_path=str(export_dir), repo_id=args.repo_id, repo_type="dataset",
            commit_message=args.message or f"Low-guidance sweep export {manifest['exported_at'][:10]}",
        )
        print(f"[publish_hf] done -> https://huggingface.co/datasets/{args.repo_id}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Publish the sub-CFG sweep to Hugging Face Hub.")
    p.add_argument("--repo-id", default="youssefhassan13/low-guidance-cfg-sweep")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--message", default=None)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
