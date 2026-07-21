# Experiment 03 — Hardening the Klüver Level-2/3 finding

**Status:** **pre-registered + smoke-validated; ready for the full run.**
Pre-registration drafted 2026-07-15; the whole pipeline was validated
end-to-end on the M5 on 2026-07-21 (a 2-image smoke run: generate → Claude +
Qwen judge → quality → analyze, all green). The only issues found were
environment/download plumbing, now fixed and documented under
[Environment notes](#environment-notes-from-the-2026-07-21-smoke-run). The
science code is unchanged. Next action is the full sweep below.

**Derives from** [Exp 01](../01_image_test/). Exp 01's *confirmatory* metric
(Klüver level-1 geometry) was a clean null. Its *exploratory, post-hoc* Klüver
**Level-2/3** re-judge (reduplication, fragmentation, condensation, distortion)
showed a dose-response as guidance falls — "the knob dissolves objecthood, not
V1 geometry" (SDXL distortion ρ = −0.64). That result is currently fragile:
one prompt, one VLM judge, no inter-rater reliability, low-g confounded with
under-conditioning, no multiple-comparison correction. **Exp 03 hardens it into
a citable, pre-registered result** (target: NeurIPS 2026 workshop, ~late Aug).

## The four hardening axes

1. **Multiple prompts.** A designed 6-prompt set spanning object count (1 → few
   → many) and objecthood (rigid object → organic → scene → texture), including a
   low-objecthood control (pine forest) that should *dissociate* the fields if
   the effect is object-bound. See `preregistration.json` → `prompts`, each with
   a pre-registered *intended inventory* so reduplication is well-defined.
2. **Independent judges + human subset.** **Three model families** (2026-07-21
   amendment), deliberately *not* three Claudes, to break judge-circularity:
   Judge A = **Claude Sonnet 5** (`claude-sonnet-5`, pinned via
   `EXP03_CLAUDE_MODEL`, run through the **Message Batches API** — 50% cheaper);
   Judge B = **Qwen2.5-VL-7B** (MLX, local, $0); Judge C = **Llama 4 Scout Vision**
   (MLX, local, $0; `EXP03_LLAMA_MODEL`, fallback Llama-3.2-11B-Vision if it OOMs
   on 64 GB). Plus a 25–30 image human-rated subset. Reliability = **mean of the
   pairwise** quadratic-weighted **Cohen's κ** across the three judges (with Gwet
   AC2 + percent agreement alongside, because the fields are 0-heavy). The
   composite-κ ≥ 0.4 gate is unchanged.
3. **Guidance-matching.** The low-g breakdown is confounded with
   under-conditioning (SDXL's comfort zone is g ≈ 5–8; g = 1.0 is out of
   distribution). Build a per-model **no-reference quality curve** (CLIP-IQA +
   LAION-aesthetic — both local, stable at small N, and the right inverted-U
   shape; FID is noisy at our per-bin N and CLIP-score just measures adherence).
   Then (a) partial-correlate the L2/3 composite against g **controlling for
   quality** — the primary de-confound — and (b) compare the low-g arm against the
   quality-matched high-g arm. If the effect survives, it is not just "low g
   looks bad."
4. **Multiple-comparison correction.** Primary endpoint is a **single composite**
   per model (2 tests total); the 4 individual fields are secondary, corrected by
   **Benjamini–Hochberg FDR** with effect sizes.

## Design at a glance

| | |
|---|---|
| Models | SDXL (conv UNet), SD 3.5 (MMDiT) — same as Exp 01 |
| Prompts | 6 (still life anchor + portrait + bicycle + oranges + living room + forest control) |
| Guidance | `{1, 2, 3, 5, 7, 11, 15}` — spans both arms of the inverted-U quality curve |
| Seeds | 10 (42–51) |
| Images | 430/model (+ empty-prompt baselines) = **860** |
| Host | Apple **M5 Pro 64GB** (single box; no sharding, SD 3.5 runs without cpu-offload) |
| Judges | **3 families** (2026-07-21): Claude Sonnet 5 (API via **Batches**, ~$3.50) + Qwen2.5-VL-7B (local, $0) + **Llama 4 Scout Vision** (local, $0; fallback Llama-3.2-11B-Vision) |
| Primary stat | Linear mixed model: `composite ~ guidance_std + (1\|prompt) + (1\|seed)`, per model |
| De-confound | partial Spearman controlling for quality; matched-quality two-arm contrast |
| Human subset | 25–30 stratified images, author-rated, blind |

## Hypothesis (confirmatory)

Lowering guidance g on SDXL and SD 3.5 (prompt/seed/steps fixed) raises a
composite objecthood-dissolution score (z-scored mean of the four L2/3 fields),
and this rise **(a)** generalizes across the 6 prompts, **(b)** survives
controlling for image quality (is not under-conditioning artifact), and **(c)**
is agreed on by both judges at adequate reliability.

**Null** (any of): the pooled slope is flat on either model; or it vanishes after
partialling out quality; or the two judges disagree (composite weighted κ < 0.4).
Each is a first-class reportable outcome — see `analysis_plan.md` §8.

## What is (deliberately) *not* here

- **No Opus/Fable judge.** A fixed-rubric ordinal scoring task doesn't need a
  frontier tier, and API cost scales with every image × every pass — Sonnet 5 is
  the capable-but-cheap choice; the open-weight Qwen judge is what breaks
  circularity. (See the pre-registration for the cost line.)
- **No FID / CLIP-score comfort-zone metric** — wrong shape or too noisy at our N
  (see axis 3).
- **No new theory.** This is a rigor upgrade of an existing exploratory signal,
  not a new mechanism claim.

## Files (planned)

| File | Purpose | Status |
|---|---|---|
| `preregistration.json` | Machine-readable confirmatory pre-registration | ✅ committed |
| `analysis_plan.md` | Pre-committed statistics plan (LMM, κ, partial-corr, BH) | ✅ committed |
| `README.md` | This human-readable twin | ✅ |
| `sweep_local.py` | Generation, generalized to the 6-prompt set (adapted from Exp 01) | ✅ written |
| `rubric.py` | Shared per-prompt L2/3 rubric (both judges use it) | ✅ written |
| `judge.py` | Claude Sonnet 5 L2/3 judge — **Batches API** (`--sync` escape hatch) | ✅ written |
| `judge_qwen.py` | Qwen2.5-VL-7B MLX judge, identical rubric | ✅ written |
| `judge_llama.py` | **Llama 4 Scout Vision** MLX judge (judge C), identical rubric | ✅ written (2026-07-21) |
| `quality.py` | CLIP-IQA + LAION-aesthetic, per image | ✅ written |
| `human_rate.py` | Blind local rating tool for the 25–30 subset | ✅ written |
| `statlib.py` | numpy stats (Spearman/partial, Cliff's δ, BH, κ, Gwet AC2) | ✅ written |
| `analyze.py` | LMM, κ/AC2, partial-corr, matched-quality, BH, figures | ✅ written |
| `log.md` / `analysis.md` | Daily log / written up after the run | log ✅ / analysis ⬜ |

> **Status of the code (updated 2026-07-21):** all eight scripts now run green
> end-to-end on the M5. The 2-image smoke run validated: SDXL fp16 generation on
> MPS, the Claude judge, the Qwen2.5-VL judge, CLIP-IQA + LAION-aesthetic
> quality, and `analyze.py`'s full LMM / κ / partial-correlation / report path
> (deconfound computable). N=2 makes the *statistics* degenerate (NaN CIs) but
> proves the *wiring*. Every failure hit was environment/download, not logic —
> see [Environment notes](#environment-notes-from-the-2026-07-21-smoke-run).

## Environment notes (from the 2026-07-21 smoke run)

Prerequisites the smoke run surfaced — apply these before the full sweep so it
runs unattended:

1. **Disable Hugging Face Xet.** `huggingface_hub` 1.23 defaults to the Xet CAS
   backend, which 401s on public repos (SDXL, Qwen). Export
   **`HF_HUB_DISABLE_XET=1`** for every download/run to use the classic path.
2. **Extra quality deps.** `torchmetrics` CLIP-IQA needs **`piq`** (+
   `torchvision`), now in `requirements.txt`. Without it `quality.py` raises a
   `ValueError` at init.
3. **piq's CLIP snapshot can stall.** `quality.py`'s aesthetic head pulls
   `RN50.pt` (~244 MB) from a GitHub release via piq's no-retry downloader,
   which can hang/corrupt. If `quality.py` stalls or throws a
   `PytorchStreamReader` error, fetch it directly and re-run:
   `curl -L --retry 5 -C - -o ~/.cache/clip/RN50.pt https://github.com/photosynthesis-team/piq/releases/download/v0.7.1/RN50.pt`
4. **SDXL runs off local weights.** The fp16 snapshot lives in `sdxl_local/`
   (gitignored, ~6.8 GB); pass `--model-path sdxl_local --variant fp16` to
   avoid re-downloading. **SD 3.5 is not downloaded yet** and may be HF-gated
   (needs `HF_TOKEN` + license acceptance) — resolve before its generation leg.
5. `quality.py` was patched for **transformers ≥ 5** (`get_image_features` now
   returns an output object; we unwrap `pooler_output`).

## Running it

```bash
pip install -r ../../requirements.txt          # statsmodels, torchmetrics, piq, mlx-vlm
export HF_HUB_DISABLE_XET=1                     # required — see Environment notes

# 0) SMOKE TEST — DONE 2026-07-21 (all green). To repeat on a tiny subset:
#   python sweep_local.py --model sdxl --model-path sdxl_local --variant fp16 \
#       --prompts p2_portrait --guidance 1 15 --seeds 42 --skip-existing
#   EXP03_CLAUDE_MODEL=claude-sonnet-5 python judge.py --dir results-local/sdxl
#   python judge_qwen.py --dir results-local/sdxl && python quality.py --dir results-local/sdxl
#   python analyze.py --model sdxl

# 1) generation (single M5 Pro 64GB host; --skip-existing makes it resumable)
#    SDXL runs off the local fp16 weights (no re-download):
python sweep_local.py --model sdxl --model-path sdxl_local --variant fp16 --prompts all --unconditional --skip-existing
#    SD 3.5: download first (may need HF_TOKEN + license); then:
python sweep_local.py --model sd35 --prompts all --unconditional --skip-existing
# judging (blind, shuffled) + quality — run per model dir
# Judge A (Claude) uses the Batches API by default (~50% cheaper); resumes from
# .batch_claude.json if a poll is interrupted. Judges B/C are local MLX ($0).
for m in sdxl sd35; do
  EXP03_CLAUDE_MODEL=claude-sonnet-5 python judge.py --dir results-local/$m   # batches
  python judge_qwen.py  --dir results-local/$m         # Qwen2.5-VL-7B (local)
  python judge_llama.py --dir results-local/$m         # Llama 4 Scout (local; see EXP03_LLAMA_MODEL)
  python quality.py     --dir results-local/$m
done

# human subset (25–30 blind ratings for inter-rater κ)
python human_rate.py --sample        # build the stratified subset once
python human_rate.py                 # rate (resumable; opens each image, guidance hidden)

# analysis: LMM · quality de-confound · κ/AC2 · matched-quality · BH · figures + verdict
python analyze.py --both --plot
```

Outputs per model: `results-local/<model>/l23_report.json` (all statistics +
verdict) and `figures/` (dose-response, per-field, reliability). Publish the
bundle to Hugging Face after the run, as in Exp 01.

## Pre-registration discipline

`preregistration.json` and `analysis_plan.md` are committed **before** any
analyzed generation run (git-history pre-registration, Methodology §4). Changing
any `forbidden_to_automate` field after seeing data reclassifies the run as
exploratory and requires a separate dated commit.
