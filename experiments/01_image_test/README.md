# Experiment 01 — Image Test (Vision Temperature Gradient)

**Status:** in progress

## Hypothesis

If predictive suppression in a vision model behaves like top-down priors in the brain, then weakening the model's confident "application-layer" inferences (object identity, scene gist) should reveal lower-layer representational primitives — textures, contours, repeating lattice structure — analogous to the geometric perceptual content reported under sub-anesthetic ketamine.

## Method

1. Pick a fixed prompt (or fixed input image, depending on model).
2. Generate samples across a graded sweep of the relevant "dose" parameter — sampling temperature, classifier-free guidance scale, or noise schedule, depending on the model used.
3. Hold seed and all other variables fixed across the sweep so that only the suppression-analogue parameter varies.
4. For each generated output, log: parameter value, raw output, and a brief structured description (object-level vs. texture-level vs. geometric).

## Expected outcome

At the low end of the sweep (high suppression / low temperature), outputs should be coherent objects and scenes. As the parameter is raised, content should not simply become "noisier" in a uniform way — the prediction is a structured shift toward repeating geometric / lattice / texture-dominant content before fully decohering. A null result would be uniform noise with no intermediate geometric regime.

## Files

- `run.py` — sweep driver (placeholder).
- `results/` — generated outputs (gitignored).
- `analysis.md` — written up after the sweep completes.
