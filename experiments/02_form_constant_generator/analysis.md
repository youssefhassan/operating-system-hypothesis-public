# Experiment 02 — Analysis (Stage A)

**Run 2026-09-03.** Pre-registration 2026-06-29 (`preregistration.json`), two dated
amendments on 2026-09-03, both before any image was scored (see the README status
block). Code: `run.py` (field model + render + planform label), `judge.py` (the
archived Exp 01 judge, unchanged), `analyze.py` → `results-local/report.json`.
Figures: `results-local/figures/contact_sheet.png`, `curve_and_control.png`.

---

## 0. What this experiment is

The only experiment in Program I that runs the biological mechanism as code: an
Ermentrout–Cowan scalar neural field on a periodic cortical sheet, pushed through
its Turing bifurcation by the gain μ, rendered to the visual field through the
inverse complex-log map. Two jobs. (i) Replicate the 1979 result that this
produces Klüver's four classes. (ii) Use the labelled renders as ground truth for
the one form-constant detector this program ever had, the Exp 01 judge, whose
clean null on diffusion outputs had never been checked against a case where a
form constant is actually present.

| | |
|---|---|
| Model | τ ∂a/∂t = −a + f(μ (w ∗ a)); DoG kernel σ_e = 1, σ_i = 2; k_c = 0.961; sheet 256², side 12 λ_c |
| Regimes | **hex**: β 4, θ 0.5 (quadratic term, pure-noise start), μ_c = 5.04. **stripe**: β 4, θ 0 (odd sigmoid, oriented seed at noise amplitude, random orientation per seed), μ_c = 2.12 |
| Sweep | μ ∈ {0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5} μ_c × 10 seeds × 2 regimes = 160 renders |
| Judge | Exp 01's `judge.py`, rubric `exp01-formconstant-v1`, `claude-sonnet-4-6`, blind, shuffled, single call per image |
| Negatives | 40 sub-threshold blank renders; 40 photographs from Exp 03 at guidance 7 and 11, 20 per architecture, stratified over the six prompts, drawn with seed 0 (`results-local/negatives.json`) |
| Scored | 200 images, 0 errors, about $3 |

---

## 1. Results at a glance

**Replication (pre-registered gates): three of four pass.**

| Gate | Threshold | Result | |
|---|---|---|---|
| Score above threshold | mean M ≥ 0.50 at some supra-threshold μ | **1.00** (every μ ≥ 1.05 μ_c, both regimes) | ✓ |
| Blank baseline | mean M ≤ 0.15 at 0.8 μ_c | **0.00** | ✓ |
| All four classes realized | each class flagged on ≥ 1 supra-threshold render | lattice, cobweb, tunnel, spiral all flagged | ✓ |
| Planform → class confusion diagonal-dominant | | **fails** for tunnel and lattice rows; passes for spiral | ✗ |

Verdict under the pre-registered rule: **partial**. The mechanism reproduces
(blank below μ_c, structured above, all four classes present, monotone in μ with
Spearman ρ = 0.87 hex / 0.91 stripe, permutation p < 0.001). What fails is the
specificity mapping from cortical planform to judged class, §3.

**Positive control (amendment 1 thresholds): pre-specified verdict "failed", on
the negatives.**

| Set | n | judged "structured" (any class flag) | 95% CI |
|---|---|---|---|
| Form-constant renders (μ ≥ 1.05 μ_c) | 80 | **1.00** | [0.95, 1.00] |
| Blank renders (μ ≤ 0.9 μ_c) | 40 | **0.00** | [0.00, 0.09] |
| Photographs (Exp 03, g ∈ {7, 11}) | 40 | **0.45** | [0.31, 0.60] |

Sensitivity is perfect and the blank rate is zero; the photograph rate is more
than double the 0.20 ceiling fixed in advance, so the control fails as specified.
§2 says what the failure is made of, because it changes what it means for Exp 01.

---

## 2. What the judge does on photographs

The 18 flagged photographs are not hallucinations of geometry. Every note names a
real object: the brick wall behind the bicycle (6 of 6 bicycle images flagged
lattice, 5 of them cobweb for the wheel spokes), window mullions in the living
room (4 of 6), a subway-tile backsplash and orange segments (2 of 6), conifer
crowns in rows (3 of 6), cut-glass facets (3 of 8). The portrait, which contains
no grid, was flagged 0 of 8.

The rubric says *"lattice: grids, honeycombs, checkerboards, tilings, regularly
repeated cells"*. The judge applied it literally. This is a construct problem in
the rubric, not a hallucinating judge: the binary class fields detect
*scene-level periodicity*, whether hallucinatory or architectural.

On the metric Exp 01 actually pre-registered, M = geometric_intensity / 3, the
separation is clean and was not part of the pre-specified control criterion:

| Set | GI = 0 | 1 | 2 | 3 | mean M |
|---|---|---|---|---|---|
| Form-constant renders (80) | 0 | 0 | 0 | **80** | 1.00 |
| Blank renders (40) | 40 | 0 | 0 | 0 | 0.00 |
| Photographs (40) | 22 | 14 | 4 | **0** | 0.18 |

*Exploratory, stated after seeing the table:* a threshold of GI ≥ 2 gives 100%
sensitivity and 10% false positives; GI = 3 gives 100% and 0%. The pre-specified
"any class flag" criterion is the wrong summary of this judge and the amendment
should have used M, which is what Exp 01 used. That is recorded here rather than
repaired.

**What this means for Exp 01's null.** Exp 01 reported M flat near the floor at
every guidance value on both models. The judge that produced that number scores
M = 1.00 on every one of 80 rendered form constants, including faint ones just
past threshold (stripe regime at exactly μ_c: GI 1–2 on all ten seeds, which the
simulation's own amplitude rule had labelled blank). Had the diffusion outputs
contained anything like a Klüver form constant, this judge would have seen it.
The null is a null on a sensitive instrument. The caveat that now attaches to it
is specificity in the other direction: some of Exp 01's floor may be literal
scene lattices (a tiled table, a glass), which can only make the floor higher,
not hide a rise.

---

## 3. Why the specificity gate fails

Confusion of predicted class (planform rule) against judged classes, supra-threshold
renders, multiple flags per image allowed:

| predicted → judged | lattice | cobweb | tunnel | spiral |
|---|---|---|---|---|
| lattice (hex, n = 21) | 16 | 9 | 21 | 21 |
| tunnel (rolls, n = 8) | 0 | 5 | 8 | 12 |
| spiral (rolls, n = 24) | 0 | 0 | 15 | 24 |

Two things are happening. First, the judge flags *tunnel* and *spiral* on
essentially every render, including every honeycomb. Look at the contact sheet:
under the complex-log map a hexagonal cortical lattice becomes blobs arranged
along logarithmic spirals converging on the fovea. The renders genuinely carry
both a local class (lattice) and a global one (spiral/tunnel), and a
multi-label judge reports both. Second, the labelling rule's "concentric"
category (wavevector within 22.5° of the ln r axis) renders as a tight spiral
unless the angle is very close to zero, so "tunnel" was predicted for images
that are, to the eye, spirals. The judge was right and the rule was coarse.

So the gate fails partly on the instrument (it does not pick one class) and
partly on the pre-registered rule (a 22.5° band is too wide for concentric). It
does not fail on the mechanism: within the rolls, orientation does move the
judged class (the spiral row is clean; the tunnel row is the one the rule
mislabels).

---

## 4. The biological reference curve

Both regimes give the pre-registered shape: nothing below μ_c, an abrupt rise at
μ_c, saturation immediately above it. The stripe regime shows the graded onset
(M = 0.47 at exactly μ_c, 1.00 from 1.05 μ_c); the hex regime is a step. This is
the Q3 reference the pre-registration asked for. It is a **threshold**, not a
gradient, which is worth stating plainly next to Exp 03: the diffusion side
showed a gradient on SDXL and a threshold at g = 1 on SD 3.5, and the biological
mechanism, at least in this minimal scalar form, is threshold-shaped. Whether
"the curves rhyme" is therefore not a yes/no; it depends on which architecture
and on the scale of the knob, and no rhyme test was pre-registered, so none is
claimed here.

---

## 5. Limitations

- **Single judge, single call, no human check.** The same limitation as Exp 01.
  The point of this run was to test that judge, not to replace it.
- **The pre-specified control criterion was the wrong summary statistic** (any
  binary flag rather than M). Reported as failed; the M reading is exploratory.
- **Photographic negatives are diffusion outputs**, not photographs. They are
  the images Exp 01's judge actually faced, which is the right negative set for
  the control, but they contain generated periodic textures (bricks, tiles).
- **Stage A only.** The scalar model gives non-contoured planforms; Bressloff's
  contoured forms (Stage B) were gated on Stage A and are not run.
- **The oriented seed in the stripe regime selects orientation.** It does not
  set wavelength or amplitude, but the class of a stripe render is not
  spontaneous the way the hexagons are. The hex regime is the cleaner
  replication; the stripe regime is a controlled sampling of the roll family.
- **The planform rule mislabels near-concentric rolls** (§3). Fixing it after
  seeing the confusion matrix would be post-hoc and is left for a follow-up.
- Amendment 2 was written after a two-seed smoke test of the simulation and
  before any scoring, on the Exp 03 precedent. A stricter reading would call the
  regime change exploratory; the record shows what was seen and when.

## 6. What this does *not* claim

- Does not claim the Exp 01 judge is class-specific. It is not.
- Does not claim any diffusion model implements this mechanism (pre-registered
  scope limit).
- Does not claim the biological and diffusion curves rhyme; no rhyme test was
  pre-registered.
- Does not convert the failed control into a pass. The pre-specified verdict
  stands; the M analysis is offered as the reason to re-specify, next time, in
  advance.

## 7. Artifacts

`results-local/metadata.json` (all parameters, μ_c, per-run planform labels),
`judge_manifest.json`, `negatives.json`, `judgements.json` (raw judge output,
200 records), `report.json`, `figures/`. Fields and renders are regenerable from
`run.py` and `params.json` in about eight minutes on the M5.
