# Exp 03 sensitivity analyses, round 2 (pre-specified 2026-09-17, before running)

Prompted by the external review of the preprint draft (GPT, 2026-09-17). Same 835
conditioned images, same amended panel (Claude Sonnet 5 + Qwen3-VL-32B), same rubric.
**Exploratory. No gate. The pre-registered verdict is unchanged whatever these return.**
Script: `posthoc_sensitivity2.py`. Output: `posthoc_sensitivity2.json`.

## S3. Model x guidance interaction

The draft compares the two architectures' slopes side by side without testing the
difference. Fit, on the pooled data, `composite ~ guidance_std * model + (1|prompt) +
(1|seed)` with crossed random intercepts, twice: (a) on the pre-registered composite,
which is z-scored within model (so the interaction is on the standardised scale each model
was judged on); (b) on the raw judge-mean composite (mean of the four 0 to 3 fields, not
z-scored), which keeps level differences between models.

Expectation written in advance: the interaction term is negative for SDXL relative to
SD 3.5 (SDXL steeper) on both scales, with a CI excluding zero on (a). If the CI includes
zero, the paper must say the model difference in slope is not established.

## S4. Random slopes by prompt

The pre-registered model has random intercepts only, which understates prompt
heterogeneity. Fit `composite ~ guidance_std + (1 + guidance_std | prompt)` per model
(groups = prompt, random intercept and slope; the seed intercept is dropped because
statsmodels cannot cross a random slope with a second grouping factor). Report the fixed
slope with CI and the SD of the random slope.

Expectation: the fixed slope changes little; its CI widens; the random-slope SD is
substantial on both models (the per-prompt Spearman range already spans 0.46 on SDXL).

## S5. Cluster bootstrap

The reported CIs treat images as independent. Resample (a) seeds with replacement,
keeping all prompts, and (b) prompts with replacement, keeping all seeds; recompute the
pooled Spearman composite-vs-g and the LMM slope on each resample (500 and 200 resamples
respectively). Report percentile CIs.

Expectation: seed-cluster CIs are close to the image-level CIs; prompt-cluster CIs are
wider, and on SD 3.5 the prompt-cluster CI for the slope may include zero.

## S6. Model versus model agreement on the human subset

The draft compares full-corpus model-model kappa (n = 835) with subset human-model kappa
(n = 28). Compute Claude versus Qwen3-VL-32B quadratic-weighted kappa per field and the
mean per-field kappa, with the bootstrap CI, on exactly the 28 blind-subset images, so
the three quantities (model-model, human-Claude, human-Qwen) are on the same images.

Expectation: model-model kappa on the 28 is lower than the full-corpus value (range
restriction on a small subset) but still above human-Qwen.

## S7, S8 (added 2026-09-25, before running, after a second external review)

S7. Residual quality imbalance in the pre-registered "matched-quality" arms. The arms are
common-support trimmed, not balanced, so report the standardised mean difference in Q
between them. Expectation: positive and not small on both models (the high-guidance arm
stays higher in quality), which means the arm comparison is an adjustment, not a match.

S8. Prompt-cluster bootstrap of the quality-controlled partial Spearman (composite vs g
given Q). Expectation: SDXL's interval excludes zero; SD 3.5's may not.
