# Exp 03 sensitivity analyses, round 3 (pre-specified 2026-10-22, before running)

Prompted by the seven-point review of Preprint I v1 on xsci (2026-10-22). Same conditioned
images (418 SDXL, 417 SD 3.5), same amended panel (Claude Sonnet 5 + Qwen3-VL-32B), same
rubric. **Exploratory. No gate. The pre-registered verdict is unchanged whatever these
return.** Script: `posthoc_review_xsci.py`. Output: `posthoc_review_xsci.json`.

Each item is numbered after the review comment it answers.

## R1. Within-judge z-scoring before averaging (review 1)

The registered composite averages the two judges' raw 0 to 3 scores per field, then
z-scores across conditioned images. Because Qwen's scores have less variance, the composite
leans toward Claude. Recompute: z each field within judge (across the model's conditioned
images), average the two judges' z per field, composite = mean of the four. Refit the
registered LMM. Also report the two single-judge composite slopes (already in
`analysis.md`) next to it.

Expectation: the within-judge composite slope sits between the two single-judge slopes and
keeps its sign and its CI away from zero on SDXL; on SD 3.5 it stays above -0.2.

## R2. Interaction model with checkpoint-specific random intercepts (review 2)

S3 pooled both checkpoints with `(1|prompt) + (1|seed)` shared across checkpoints. Refit
with `(1|prompt) + (1|seed) + (1|prompt:model) + (1|seed:model)` so each checkpoint keeps
its own prompt and seed baselines. Report both specifications side by side.

Expectation: the interaction estimate moves little; its CI widens somewhat.

## R3. Partial correlation controlling both quality proxies at once (review 3)

Report the partial Spearman of composite vs guidance controlling CLIP-IQA and the aesthetic
score as two separate covariates (rank residualisation on both), with image-level and
prompt-cluster bootstrap CIs, next to the three already computed (Q, CLIP-IQA alone,
aesthetic alone).

Expectation: within 0.05 of the single-covariate values.

## R4. Prompt-generality tally on the five test prompts (review 4)

The registered gate counts 5 of 6 prompts including the forest control. Report the tally
restricted to the five test prompts alongside it.

Expectation from the published per-prompt numbers: SDXL 5/5, SD 3.5 4/5.

## R6. Quality-adjusted slope inside the mixed model (review 6)

Fit `composite ~ guidance_std + Q + (1|prompt) + (1|seed)` and
`composite ~ guidance_std + z(CLIP-IQA) + z(aesthetic) + (1|prompt) + (1|seed)` per model.
Report the guidance fixed effect with CI.

Expectation: the guidance slope keeps its sign and stays close to the unadjusted LMM slope,
as the pooled partial correlations did.

## R7. Excluding the 28 screening images (review 7)

The 28 screening images are the human subset (`human_subset.json`, 14 per checkpoint).
Drop them and recompute the registered LMM slope, the partial correlation controlling Q,
the per-prompt tally and the full-set inter-judge composite kappa.

Expectation: no change to the second decimal.

---

# Addendum, 2026-10-22 (later the same day, before running): third report (XSci 9bcf95ad)

Script: `posthoc_review_xsci2.py`. Output: `posthoc_review_xsci2.json`. Exploratory; no gate.

## X1. Prompt adherence as a covariate (major concern 1)

CLIP image-text similarity: cosine between CLIP ViT-L/14 (`openai/clip-vit-large-patch14`,
the backbone the aesthetic predictor already uses) image and text embeddings, the text being
the registered prompt verbatim. Per model: Spearman of S against guidance; partial Spearman
of composite against guidance controlling S, and controlling S plus CLIP-IQA plus aesthetic;
LMM guidance slope with z(S) as a covariate, and with z(S) + z(CLIP-IQA) + z(aesthetic).

Expectation: S rises with guidance. If the composite is prompt adherence under another name,
the guidance slope collapses toward zero once S is entered. If the SDXL LMM slope with all
three covariates stays below -0.2 with the interval away from zero, the concern is answered
for SDXL. I have no confident expectation for SD 3.5.

## X2. Inter-judge kappa within guidance bins (major concern 2)

Composite weighted kappa (Claude vs Qwen3-VL-32B), conditioned images only, computed inside
each guidance value separately and inside three bins: low (g = 1, 2), mid (g = 3, 5),
high (g = 7, 11, 15). Expectation, following the reviewer: within-bin kappa is well below
the pooled 0.542 / 0.394, because between-level contrast inflates the pooled value.

## X3. Seed random effect dropped (minor 7)

Registered LMM refitted with `(1|prompt)` only, both models. Expectation: SD 3.5 slope and
interval unchanged to the third decimal, since its seed variance is estimated at zero.
