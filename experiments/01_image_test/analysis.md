# Experiment 01 — Analysis

**Completed 2026-06-28.** Full write-up: [Substack post #3](https://youssefhassan13.substack.com/p/the-base-layer-and-a-first-experiment). Artifacts: [Hugging Face dataset](https://huggingface.co/datasets/youssefhassan13/exp01-guidance-sweep).

## Confirmatory (pre-registered)

**Hypothesis:** M (Klüver level-1 geometry) rises as guidance falls (negative ρ).

| Model | ρ (M vs g) | p | Verdict |
|---|---|---|---|
| SDXL | +0.01 | 0.94 | null-eligible |
| SD 3.5 | +0.10 | 0.37 | null-eligible |

M barely lifts off the floor at any guidance value. **Clean null on form constants.**

## Exploratory (post-hoc, `judge_kluver2.py`)

Klüver level-2/3 phenomena vs guidance (negative ρ = stronger at low g):

| Phenomenon | SDXL | SD 3.5 |
|---|---|---|
| distortion | −0.64 | −0.20 |
| fragmentation | −0.59 | −0.39 |
| condensation | −0.57 | −0.47 |
| reduplication | −0.36 | −0.24 |

**Interpretation:** the knob dissolves objecthood (fragmentation, fusion, multiplication), not V1-shaped geometry. SDXL stronger than SD 3.5.

## Exploratory rubric (Suzuki-style, same blind judge pass)

SDXL dose–response on veridicality (+0.67), coherent scene (+0.41), complexity (−0.54), spontaneity (−0.40). Guidance = perfection-vs-surprise dial.

## Failed metric (`texture_metrics.py`)

FFT periodicity looked architecture-specific; VLM cross-check refuted it as a tiling/reduplication proxy (ρ ≈ −0.3 wrong-way on SDXL, ~0 on SD 3.5). **Not reported as a finding.**

## Key qualitative calls

1. **Low-guidance ≠ empty-prompt baseline.** De-amplifying the prompt and removing it are different operations.
2. **SDXL vs SD 3.5 temperament.** SDXL breaks scenes apart at low g; SD 3.5 holds the still life together.
3. **Under-conditioning caveat.** SDXL at g = 1.0 is outside its comfort zone (5–8); some breakdown may be artifact.

## Pending follow-ups (not yet run)

- Phase 1: prompt-specificity gradient (concrete → abstract → empty) × guidance
- Phase 2a: sub-1 / true precision-relaxation guidance hook in `sweep_local.py`
- Phase 2b: timestep-targeted guidance schedule
- Phase 3: architecture breadth (SD 1.5 / VQ / GAN)

Each needs new preregistration before generation.
