# Exp 03 — sensitivity analyses, pre-specified 2026-09-03

**Exploratory.** Written before `posthoc_sensitivity.py` was run, after the
confirmatory verdict and after `posthoc.py` §9. Same data, same judge panel
(Claude Sonnet 5 + Qwen3-VL-32B), same rubric. No gate, no threshold, no change
to the pre-registered endpoint. Motivation: `docs/REVIEW_program_I_2026-09-02.md`
items A3 and A8.

## S1. Guidance scale (review A3)

The pre-registered covariate is z-scored raw guidance on the grid
{1, 2, 3, 5, 7, 11, 15}, which is close to geometric. Under CFG, g = 1 is the
conditional model with no extrapolation and g = 2 is 2× extrapolation, so the
1→2 step is one fourteenth of the linear range and the largest mechanistic
change on the dial. Refit the same mixed model with the covariate replaced by:

- **z(log2 g)**, and
- **the rank of g** (an ordinal trend).

Report the standardized slope with its 95% CI for each model, next to the
pre-registered linear slope. Prediction stated in advance: on the log scale the
SD 3.5 slope moves toward SDXL's, and the gap between the two models narrows,
because the cliff at g = 1 is spread over a larger share of the covariate's
range. No pass/fail: the pre-registered gate stays the linear slope.

## S2. Composite without distortion (review A8)

`analysis.md` §9.3 and `analysis_axes.md` §4 established that distortion is
bidirectional (melt at low g, waxiness at high g). Rebuild the composite as the
mean of the three remaining z-scored fields (reduplication, fragmentation,
condensation), then rerun on it: the linear LMM slope, the quality-controlled
partial Spearman, and the per-prompt sign count. Prediction stated in advance:
slopes stay negative on both models; SD 3.5's portrait prompt loses its positive
sign. Reported next to the four-field numbers.

## What is fixed

Judges, rubric, records, z-scoring within model, the LMM specification
`composite ~ covariate + (1|prompt) + (1|seed)`, the quality covariate Q. Nothing
else is computed. Results go to `posthoc_sensitivity.json` and a short
`analysis_sensitivity.md`, both labelled exploratory.
