# Experiment 01 — Image Test (Vision Temperature Gradient)

**Status:** in progress

## Hypothesis

If predictive suppression in a vision model behaves like top-down priors in the brain, then weakening the model's confident "application-layer" inferences (object identity, scene gist) should reveal lower-layer representational primitives — textures, contours, repeating lattice structure — analogous to the geometric perceptual content reported under sub-anesthetic ketamine.

## Method

**Model:** FLUX.2-flex via the Black Forest Labs API (`/v1/flux-2-flex`). Chosen because it exposes `guidance` (classifier-free guidance scale) across the widest range available on BFL (1.5–10.0), with `seed` and `steps` also under explicit control. Guidance is the model's "how strongly do I commit to the conditioning prior" knob — the closest API-exposed analogue to top-down predictive suppression.

**Fixed across the sweep:** prompt (`"a watermelon, a glass half-filled with water, and a set of keys on a wooden table"` — mixed materials and colors so any decomposition into lower-layer primitives is visually distinguishable rather than hidden inside a single-color object), steps (50), width/height (1024×1024), `prompt_upsampling=false` (otherwise BFL silently rewrites the prompt and reproducibility is gone).

**Variable:** `guidance ∈ {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.0, 10.0}`, run with `seed ∈ {42, 43, 44}` per guidance value (24 images total) so one weird sample doesn't drive the read.

**Judgement (v1):** by eye, sorted by guidance ascending. We are looking specifically for a structured intermediate regime — textures, contours, lattice / repeating geometry — rather than a smooth blur from "good apple" to "unrecognizable noise." A vision-LM classifier (object / part / texture / geometric / noise) can be wired in later if the by-eye signal looks real.

**Cross-model replication (planned, separate run):** the same sweep on SDXL via `diffusers`, where CFG can range much wider than 1.5–10. If the predicted regime appears in both models, it's evidence for an architecture-general property; if only in one, it's a model quirk.

## Expected outcome

At the low end of the sweep (high suppression / low temperature), outputs should be coherent objects and scenes. As the parameter is raised, content should not simply become "noisier" in a uniform way — the prediction is a structured shift toward repeating geometric / lattice / texture-dominant content before fully decohering. A null result would be uniform noise with no intermediate geometric regime.

## Files

- `run.py` — sweep driver. Requires `BFL_API_KEY` in `.env`.
- `results/` — generated images + `metadata.json` (gitignored).
- `analysis.md` — written up after the sweep completes.
