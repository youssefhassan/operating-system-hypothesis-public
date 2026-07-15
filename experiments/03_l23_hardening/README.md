# Experiment 03 — Hardening the Klüver Level-2/3 finding

**Status:** **pre-registered, not yet run** (drafted 2026-07-15). Generation
begins only after the pre-registration + analysis plan are reviewed and the
commit lands.

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
2. **Second judge + human subset.** Judge A = **Claude Sonnet 5**
   (`claude-sonnet-5`, pinned via `EXP03_CLAUDE_MODEL`); Judge B = **Qwen2.5-VL-7B**
   via MLX (open-weight, local, $0) — deliberately *not* a second Claude, to
   break judge-circularity. Plus a 25–30 image human-rated subset. Report
   quadratic-weighted **Cohen's κ** (with Gwet AC2 + percent agreement alongside,
   because the fields are 0-heavy) across judges.
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
| Judges | Claude Sonnet 5 (API, ~$7) + Qwen2.5-VL-7B (local, $0; 32B available on 64GB as fallback) |
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
| `preregistration.json` | Machine-readable confirmatory pre-registration | ✅ written |
| `analysis_plan.md` | Pre-committed statistics plan (LMM, κ, partial-corr, BH) | ✅ written |
| `README.md` | This human-readable twin | ✅ written |
| `sweep_local.py` | Generation, generalized to the 6-prompt set (adapted from Exp 01) | ⬜ after approval |
| `judge.py` | Claude Sonnet 5 L2/3 judge (per-prompt intended inventories) | ⬜ |
| `judge_qwen.py` | Qwen2.5-VL-7B MLX judge, identical rubric | ⬜ |
| `quality.py` | CLIP-IQA + LAION-aesthetic, per image | ⬜ |
| `human_rate.py` | Blind local rating tool for the 25–30 subset | ⬜ |
| `analyze.py` | LMM, κ/AC2, partial-corr, matched-quality, BH, figures | ⬜ |
| `log.md` / `analysis.md` | Daily log / written up after the run | ⬜ |

## Running it (after code is written)

```bash
# generation (single M5 Pro 64GB host — no sharding needed; --skip-existing makes it resumable)
python sweep_local.py --model sdxl --prompts all --unconditional --skip-existing
python sweep_local.py --model sd35 --prompts all --unconditional --skip-existing
# judging (blind, shuffled)
EXP03_CLAUDE_MODEL=claude-sonnet-5 python judge.py --dir results-local/sdxl
python judge_qwen.py --dir results-local/sdxl        # local MLX
# quality + analysis
python quality.py --dir results-local/sdxl
python analyze.py --model sdxl --plot
```

## Pre-registration discipline

`preregistration.json` and `analysis_plan.md` are committed **before** any
analyzed generation run (git-history pre-registration, Methodology §4). Changing
any `forbidden_to_automate` field after seeing data reclassifies the run as
exploratory and requires a separate dated commit.
