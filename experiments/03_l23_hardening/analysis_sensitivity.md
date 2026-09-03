# Exp 03 — sensitivity analyses (EXPLORATORY)

**Run 2026-09-03**, pre-specified the same day in
[`posthoc_sensitivity_prespec.md`](posthoc_sensitivity_prespec.md) before
`posthoc_sensitivity.py` executed. Same 835 conditioned images, same amended
panel (Claude Sonnet 5 + Qwen3-VL-32B), same rubric, same mixed model. **No
gate.** The pre-registered verdict in `analysis.md` §1 is unchanged: SDXL
confirms, SD 3.5 misses the linear slope bar by 0.018. Everything below is
about how much that miss depends on two choices made in the pre-registration.
Output: `posthoc_sensitivity.json`.

## S1. The covariate scale

Standardized LMM slope of the four-field composite, by how guidance enters the
model:

| covariate | SDXL | SD 3.5 |
|---|---|---|
| **linear g** (pre-registered) | **−0.340** [−0.400, −0.279] | **−0.182** [−0.248, −0.116] |
| log2 g | −0.440 [−0.494, −0.386] | −0.301 [−0.362, −0.239] |
| rank of g | −0.415 [−0.471, −0.359] | −0.265 [−0.328, −0.202] |

Both predictions written in advance held: on the log scale SD 3.5's slope moves
toward SDXL's, and the gap between the models narrows (0.158 linear, 0.139 log,
0.150 rank). On either alternative scale the SD 3.5 estimate and its whole
confidence interval sit below the −0.20 bar the pre-registration set.

This does not convert the miss into a confirm. It locates it: the pre-registered
endpoint, linear in raw guidance on a near-geometric grid, is the scale on which
SD 3.5's bottom-of-dial threshold counts least, because the g = 1 → 2 step is one
fourteenth of the range. It is also the scale that was committed before the
data, so it is the one that stands. The honest sentence for the preprint is that
SD 3.5's effect is smaller than SDXL's on every scale, and below the
pre-specified size of interest only on the pre-specified scale.

## S2. The composite without distortion

`analysis.md` §9.3 showed the distortion field absorbs two opposite failure
modes. Rebuilding the composite from reduplication, fragmentation and
condensation only:

| | SDXL | SD 3.5 |
|---|---|---|
| LMM slope, linear g | −0.308 [−0.365, −0.251] | −0.206 [−0.269, −0.144] |
| LMM slope, log2 g | −0.395 [−0.447, −0.344] | −0.307 [−0.365, −0.248] |
| partial ρ controlling quality | −0.383 [−0.468, −0.288] | −0.299 [−0.388, −0.204] |
| prompts with a negative slope | 6 / 6 | **6 / 6** (portrait −0.42) |

Both advance predictions held. Slopes stay negative on both models, and SD 3.5's
portrait prompt, the one that cost it the 6/6 generality criterion, is now
negative (−0.42 against +0.30 with distortion included). Dropping distortion
costs SDXL a little (−0.340 → −0.308) and helps SD 3.5 a little (−0.182 →
−0.206), which is what §9.3's diagnosis implies: on SD 3.5 the distortion field
was scoring high-guidance waxiness against the low-guidance signal.

The three-field SD 3.5 slope sits 0.006 past the pre-registered bar. That is
noise-level, and it is exactly why this analysis was pre-specified with no gate:
a composite chosen after seeing which field misbehaved cannot be used to pass a
threshold set for a different composite.

## What follows for the preprint

- Report the pre-registered numbers as the result and these two tables as
  sensitivity analyses, labelled as such, in the same section.
- The miss is real on the committed endpoint and fragile to two defensible
  design choices. Say both. Do not say SD 3.5 "would have confirmed".
- The next pre-registration in this program should choose a log or ordinal
  covariate and a three-field composite (or a directional distortion field)
  *in advance*, per `RESEARCH_METHODOLOGY.md` §5.1.
