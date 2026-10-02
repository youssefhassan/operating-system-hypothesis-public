# Exp 03 sensitivity analyses, round 2 (EXPLORATORY)

**Run 2026-09-17**, pre-specified the same day in
[`posthoc_sensitivity2_prespec.md`](posthoc_sensitivity2_prespec.md) before
`posthoc_sensitivity2.py` executed. Same 835 conditioned images, same amended panel
(Claude Sonnet 5 + Qwen3-VL-32B), same rubric. **No gate.** The pre-registered verdict
in `analysis.md` section 1 is unchanged. Output: `posthoc_sensitivity2.json`. Prompted by
the external review of the preprint draft.

## S3. Model x guidance interaction (pooled LMM, crossed prompt and seed intercepts)

| outcome | SD 3.5 slope | interaction (SDXL minus SD 3.5) | 95% CI | p |
|---|---|---|---|---|
| z-composite (pre-registered scale) | -0.182 | **-0.158** | [-0.248, -0.067] | 0.0006 |
| raw judge-mean composite (0 to 3) | -0.091 | **-0.164** | [-0.222, -0.105] | < 0.0001 |

Prediction held on both scales: SDXL's slope is steeper and the difference excludes zero.
The paper may say the slope difference between the two checkpoints is established;
it may not attribute it to architecture, since each architecture is one checkpoint and
is confounded with training data, objective, text encoder and sampler.

## S4. Random slopes by prompt (groups = prompt, random intercept and slope, seed RE dropped)

| | fixed slope | 95% CI | random-slope SD | random-intercept SD |
|---|---|---|---|---|
| SDXL | -0.340 | [-0.448, -0.231] | 0.112 | 0.315 |
| SD 3.5 | -0.182 | [-0.337, -0.027] | 0.177 | 0.292 |

Prediction held: point estimates unchanged to three decimals, intervals wider (SD 3.5's
upper bound moves from -0.116 to -0.027), prompt heterogeneity in the slope is
substantial and larger on SD 3.5.

## S5. Cluster bootstrap (percentile 95% CI; 500 resamples for Spearman, 200 for the LMM)

| | cluster | Spearman CI | LMM slope CI |
|---|---|---|---|
| SDXL (rho -0.463, slope -0.340) | seed (10) | [-0.527, -0.402] | [-0.422, -0.261] |
| | prompt (6) | [-0.592, -0.370] | [-0.435, -0.262] |
| SD 3.5 (rho -0.216, slope -0.182) | seed (10) | [-0.290, -0.142] | [-0.238, -0.125] |
| | prompt (6) | [-0.407, -0.001] | [-0.322, -0.050] |

Prediction held for seed clustering (close to the image-level CIs) and for widening
under prompt clustering. The written expectation that SD 3.5's prompt-cluster interval
"may include zero" did not materialise: both intervals exclude zero, the Spearman one by
0.001. With six prompt clusters these intervals are themselves imprecise.

## S6. Claude vs Qwen3-VL-32B on the 28 human-subset images

| pair (same 28 images) | mean per-field kappa | 95% CI |
|---|---|---|
| Claude vs Qwen3-VL-32B | **0.271** | [0.056, 0.448] |
| human (author) vs Claude | 0.337 | [0.106, 0.495] |
| human (author) vs Qwen3-VL-32B | 0.124 | [-0.059, 0.313] |

Per field, Claude vs Qwen on the 28: reduplication 0.387, fragmentation 0.462,
condensation 0.167, distortion 0.069; Gwet AC2 0.75 to 0.98; percent agreement 0.29
(distortion) to 0.86 (fragmentation).

Prediction held in part: model-model agreement on the subset (0.27) is well below the
full-corpus values (0.56 / 0.44) and above human-Qwen (0.12). But it is also **below
human-Claude (0.34)** on the same images. The sentence in the draft that "the two models
agree with each other more than either agrees with the human" was a comparison of
full-corpus kappa (n = 835) with subset kappa (n = 28) and does not hold when all three
are computed on the same images. The paper must replace it with the same-image
comparison. Distortion is the field where the two judges agree least (0.07), which is
also the field where human-Claude agreement is weakest (0.21).

## What follows for the preprint

- 3.2: report S3 as the test behind any "the slopes differ" sentence; report S4 and S5
  in one sentence each as sensitivity on the SD 3.5 interval.
- 3.4: replace the models-agree-more-than-human sentence with the S6 row set; keep the
  full-corpus kappa as the panel's reliability number and say the two are on different
  image sets.
- "Architecture-dependent" becomes "differs between the two checkpoints tested".
