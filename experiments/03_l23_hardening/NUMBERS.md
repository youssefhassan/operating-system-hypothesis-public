# Exp 03 / 03b — the numbers, verified

Pulled straight from the report JSONs on 2026-08-26. Use this when you need to
quote a figure and do not want to re-derive it. Source of truth is still
`analysis.md`; this is the short version.

## The pre-registered gates

Panel: Claude Sonnet 5 + Qwen3-VL-32B. 418 conditioned images on SDXL, 417 on
SD 3.5, 860 including the empty-prompt baselines.

| Gate | Threshold | SDXL | SD 3.5 |
|---|---|---|---|
| Dose-response slope | ≤ −0.20 | **−0.340** CI [−0.400, −0.279] ✓ | **−0.182** CI [−0.248, −0.116] ✗ |
| Partial ρ, quality controlled | ≤ −0.20 | −0.433 ✓ | −0.245 ✓ |
| Prompts with a negative slope | ≥ 5 of 6 | 6/6 ✓ | 5/6 ✓ |
| Inter-judge agreement | ≥ 0.40 | 0.562 ✓ | 0.440 ✓ |
| **Verdict** | | **confirms** | **null on slope** |

**The overall claim required both models. It did not confirm.** SD 3.5 missed
one gate by 0.018.

## Post-hoc: the two models differ in shape, not strength

Exploratory. Looked at after seeing the data, so it is a hypothesis for the next
experiment, not a result of this one.

| | SDXL | SD 3.5 |
|---|---|---|
| Share of total drop in the g1 → g2 step | 60% | **91%** |
| Refit with guidance 1 removed | ρ −0.244, p = 0.0002 (**survives**) | ρ +0.048, p = 0.38 (**gone**) |

SDXL is a gradient across the usable range. SD 3.5 is a cliff at the bottom of
the dial. A linear slope reports that as a difference in magnitude; it is a
difference in kind.

Note this cuts *against* the SD 3.5 case, not for it: guidance 1 is the most
under-conditioned setting and the hardest to separate from plain artifact, and
it is now the only place SD 3.5's effect lives.

## The judge failure

| Judge | Uses the scale? | Flat on |
|---|---|---|
| Qwen2.5-VL-7B | no | all four fields |
| Qwen3-VL-8B | no | all four fields |
| **Qwen3-VL-32B** | **yes** | none |
| Gemma-3-27B | partly | fragmentation, condensation |
| Llama-3.2-11B-Vision | no | 3 of 4 fields |

Composite κ before → after swapping the 7B for the 32B, **no rubric change**:

- SDXL **0.29 → 0.56**
- SD 3.5 **0.16 → 0.44**
- Three-judge composite including Llama was **0.13**

Same family, same release, same quantisation. The failure was silent: no errors,
no refusals, well-formed JSON, fluent confident captions.

## Human validation

28-image blind subset, guidance and model hidden, order shuffled. One rater.

- Human vs Claude: **0.337** (bootstrap CI [0.106, 0.495])
- Human vs Qwen3-VL-32B: **0.124** (CI includes zero)
- Weakest field, human vs Claude: distortion at **0.212**

The two models agree with each other more than either agrees with the human.

## Exp 03b: the style confound

Separately pre-registered before scoring. Holding photographic realism fixed:

- SDXL distortion vs guidance: **−0.486 → −0.175**. About **64%** of the
  association was rendering style, not object structure.
- Not evenly spread. Oranges −0.70 → −0.08; living room −0.53 → −0.39.

This is the finding that most damages the earlier work, and it only exists
because the second scale was run.

## Scale

6 prompts, 7 guidance values (1, 2, 3, 5, 7, 11, 15), 10 seeds, 2 architectures,
plus empty-prompt baselines. 860 images. Two pre-registrations, one amendment
log, one blind human subset.
