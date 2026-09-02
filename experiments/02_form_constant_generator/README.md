# Experiment 02 — Form-constant generator (V1 mechanistic anchor)

**Status:** **Stage A run 2026-09-03** under two dated pre-scoring amendments (see
`preregistration.json` → `amendments`, and the note below). Write-up:
[`analysis.md`](analysis.md). Pre-registered 2026-06-29 (renumbered on
2026-06-30); untouched for 65 days before the run.

> **Amendments 2026-09-03, recorded before any image was scored.** (1) The
> "planned scorer" this file names below was never built. The instrument is the
> archived **Exp 01 judge** (`../01_image_test/judge.py`, rubric
> `exp01-formconstant-v1`, `claude-sonnet-4-6`), unchanged, which is the only
> detector that ever scored the AI outputs and therefore the faithful reading of
> "the SAME detector". That makes deliverable (i) do double duty as a **positive
> control for Exp 01's Level-1 null**, with thresholds fixed in advance. (2) After
> a two-seed smoke test and before scoring: two nonlinearity regimes (asymmetric
> sigmoid → hexagons; odd sigmoid with an oriented seed → rolls), a denser sheet,
> bilinear rendering, and a refined planform-labelling rule. Parameters are in
> `params.json`. The design text below is left as pre-registered; the amendments
> are the record of what changed and why.

> **Note on the renumber.** Originally carried a higher number in the legacy phased plan.
> Renumbered under the new rule: sequential IDs assigned when an experiment
> actually starts (here, at pre-registration). The legacy "scorer / replication / paper"
> forward references below point to the form-constant scorer, Suzuki-style
> replication, and framing paper — those are plain-named work items now
> (see [`the project plan`](the project plan) and [`the project plan`](the project plan)).

**Anchor papers.**
- Bressloff, Cowan, Golubitsky, Thomas & Wiener (2001), *Geometric visual
  hallucinations, Euclidean symmetry and the functional architecture of
  striate cortex*, Phil. Trans. R. Soc. B 356:299–330 — the result we
  replicate.
- Ermentrout & Cowan (1979), *A mathematical theory of visual hallucination
  patterns* — the minimal scalar precursor we implement first.
- Klüver (1966) — the form-constant taxonomy the output is scored against.
- Bertolero et al. (2026) — the empirical Dreamachine drawing corpus this
  experiment's synthetic corpus complements for planned scorer calibration.

## Why this experiment exists (the gap it fills)

The program **cites** Bressloff three times — the V1-shaped-output
prediction, the form-constant rubric, and auditory form constants by
analogy, but **never implements it**. Every experiment in `the project plan` is an
*AI-side* perturbation (CFG sweep, dropout, SAE intervention). The only place
the program runs a *biological* mechanism as code was a planned fork of Bredenberg,
for the predictive-coding / audio side. There is no vision-side mechanistic
model.

This experiment is that model. It is to the vision thread what that fork would be to the
predictive-coding thread: a faithful replication of a published generative
account, run as code, that the AI-side experiments can be measured *against*.

It also fixes a calibration gap. the form-constant experiment's form-constant detector is currently
calibrated only on empirical human drawings (Bertolero) and Suzuki's synthetic
DeepDream set. Neither is a *parametric, labeled, ground-truth* generator of the
four Klüver classes. This experiment produces exactly that — an infinite, knob-
controlled corpus of canonical form constants with known generative labels.

## Hypothesis (single mechanistic question, per Methodology §3.1)

When the cortical excitability / gain parameter **μ** is raised through the
Turing-bifurcation threshold **μ_c** in an Ermentrout–Cowan / Bressloff
neural-field model on V1 — holding the lateral-connectivity kernel and the
retino-cortical map fixed — the visual-field rendering transitions from a blank
field to the **four Klüver form-constant classes** (lattices/honeycombs,
cobwebs, tunnels/funnels, spirals), **because** reduced inhibition destabilizes
the homogeneous cortical state into doubly-periodic planforms that the inverse
log-polar map renders as those specific geometries.

This maps onto all three subquestions in `RESEARCH_METHODOLOGY.md` §1:

- **Q1 (existence).** Does the field model produce the four Klüver classes
  *qualitatively*, rather than degrading to noise?
- **Q2 (specificity).** Does *which* cortical planform appear (stripe
  orientation, lattice symmetry) predict *which* form constant the map
  renders — the Bressloff correspondence (horizontal cortical stripes →
  concentric tunnels; vertical → radial funnels; oblique → logarithmic
  spirals; hexagonal lattice → honeycomb/cobweb)?
- **Q3 (dose-response).** Does the μ-sweep produce a graded blank → faint
  geometry → full form-constant curve whose *shape* can be compared against
  the AI-side CFG/temperature dose-response curves (Exp 01 and its planned follow-ups)?

## Method

**Stage A — minimal scalar field (Ermentrout–Cowan), confirmatory.**
A Wilson–Cowan scalar neural field `a(x, t)` on a periodic 2-D cortical sheet:

```
τ ∂a/∂t = −a + f( μ · (w ∗ a) )
```

with `f` a sigmoid and `w` a difference-of-Gaussians ("Mexican-hat") lateral
kernel (short-range excitation, longer-range inhibition). The homogeneous rest
state loses stability via a Turing bifurcation at μ_c set by the kernel; the
most-unstable wavenumber k_c fixes the pattern scale. A quadratic term in the
nonlinearity makes hexagons (lattices) competitive with stripes near onset.

Render to the visual field via the inverse retino-cortical (complex-log) map:
cortical `(x, y)` ↔ visual-field polar `(r, θ)` with `x ∝ ln r`, `y ∝ θ`. Under
that map, cortical stripes of different orientation become tunnels, funnels, and
logarithmic spirals; cortical lattices become honeycomb/cobweb — the four
Klüver classes.

**Stage B — orientation / shift-twist field (full Bressloff), exploratory
extension.** Add an orientation coordinate φ (state `a(x, φ)` on cortex × S¹),
anisotropic lateral connectivity that couples iso-orientation cells along their
preferred axis, and shift-twist E(2) symmetry. Equivariant bifurcation theory
selects the stable *contoured* planforms (rolls / squares / hexagons made of
oriented line elements) that the scalar model cannot produce — the contoured
form constants. Heavier; staged only if Stage A replicates.

**Scoring.** Run the *same* blind form-constant detector built as the planned scorer (the
four-class Klüver rubric) over the rendered visual-field images. Reusing the form-constant experiment's
detector is deliberate: it ties this experiment's ground truth to the same
instrument that scores the AI outputs, so the comparison in Q3 is
apples-to-apples.

## What we expect — and what would count against it

This is a **replication**, so the honest prior is that Stage A *succeeds*: the
Ermentrout–Cowan model is a 45-year-old result and reproducing its form
constants is a verification of our pipeline, not a discovery. The scientific
yield is therefore in the two derived deliverables, not in the existence claim
alone:

1. A **labeled synthetic form-constant corpus** (parametric, knob-controlled,
   class-labeled) → feeds planned scorer calibration alongside Bertolero.
2. A **biological-side dose-response curve** (μ-sweep) → the reference curve the
   AI-side CFG/temperature curves are compared to. This is the only curve in the
   program generated by the actual V1 mechanism, so it is what "do the AI curves
   *rhyme* with the brain's math?" is measured against.

**Honest failure to replicate is a result** (Methodology §2.3). If the μ-sweep
never yields detector scores above the blank/noise baseline for any class, or
produces only one degenerate pattern with no planform → class correspondence,
then our implementation fails to reproduce Bressloff — and that gets written
into the book as a methods failure, not silently dropped.

**Important scope limit.** This model is the *reference*, not a claim that any AI
system implements it. A diffusion UNet has no retinotopic map and no orientation
columns (see Exp 01 README, "What we do and don't expect"). Q3 compares the
*shape* of two dose-response curves; it does not assert shared mechanism.

## Cost and tractability (Methodology §3.4)

Pure local CPU / numpy PDE integration. No API spend, no GPU required. Stage A
is days, not weeks. Clears the resource budget (§9) trivially — this is exactly
the kind of cheap, high-yield analysis the methodology says to prioritize.

## Sample size and statistics (Methodology §5)

The PDE is deterministic given (μ, kernel, initial condition), but planform
selection is multistable near onset, so the random initial-condition **seed** is
the unit of replication: N = 10 seeds per μ per kernel regime. μ grid:
`{0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5} · μ_c`. Each rendered field scored
by the planned scorer. Primary product is the **curve** (detector score vs μ),
per §5's "effect estimation over significance." Trend tested by permutation;
Q2 specificity reported as a planform → detected-class **confusion matrix**
(expected near-diagonal per the Bressloff correspondence).

See `preregistration.json` for the exact committed parameters and thresholds.
