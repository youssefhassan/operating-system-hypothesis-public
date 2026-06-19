# Experiment 01 — Analysis

*(To be written after the sweeps complete.)*

## What to fill in per run

For each model (`sdxl`, `sd35`, `flux`, `sd15`):

1. **Contact sheet.** Images tiled by guidance ascending (columns) × seed (rows),
   with the unconditional baseline column at the far low end. Eyeball pass first.
2. **Regime call.** Does lowering guidance produce (a) a structured intermediate
   regime (textures / contours / lattice / repeating geometry) before
   decoherence, or (b) a smooth blur to uniform noise with no structure? This is
   the core yes/no.
3. **Drift-toward-baseline.** Do the low-guidance conditioned images visibly
   approach the unconditional (empty-prompt) baseline? Does the baseline itself
   carry geometry or just noise?
4. **Form-constant tally (v1, by eye).** Rough counts of the four Klüver classes
   (lattice, cobweb, tunnel, spiral) per guidance bin. Replace with the Exp 14
   vision-LM judge once it's calibrated on the Bertolero corpus.

## Cross-model synthesis

- Does the structured regime appear across **both UNet (sdxl/sd15) and MMDiT
  (sd35)**? If yes → architecture-general signal. If only one → model-specific.
- Treat `flux` separately: guidance is distilled, so a different curve there is
  expected and is not evidence against the general claim.

## Decision (per the project plan, after Phase I)

- Signal present → proceed to Exp 14 (form-constant detector) and Exp 26
  (Suzuki-SDXL replication).
- No signal → run the detector on these outputs anyway to check for structure
  the eye missed; if still nothing, document the negative result honestly (the
  analogy fails for diffusion specifically) before any further generation runs.
