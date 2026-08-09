# Experiment 03b — Analysis (Suzuki three-axis scale)

**Completed 2026-08-09.** Pre-registration:
[`preregistration_axes.json`](preregistration_axes.json), committed `bd4a4a1`
before any axes scoring run. Analysis: [`analyze_axes.py`](analyze_axes.py) →
[`axes_report.json`](axes_report.json). Corpus: the same 860 Exp 03 images, no new
generation.

---

## 0. What this was for

Exp 03 scored the corpus on a **local** scale: Klüver's Level-2/3 inventory of
nameable events (what broke). Exp 03b scores the same images on a **global** one:
Suzuki's continuous axes (what the whole image is like). Two questions:

1. **Replication.** Do Exp 01's exploratory directions survive on a hardened,
   multi-prompt, two-architecture corpus?
2. **Dissociation.** Does the global scale carry information about guidance that
   the local scale does not, or is it a restatement of it?

The second is the novel endpoint, and the pre-registration committed in advance to
what its failure would mean: that the two scales are redundant and the
"complementary instruments" argument is wrong.

Field definitions are **verbatim** from Exp 01, which is what makes this a
replication rather than a new instrument.

---

## 1. Results at a glance

**Overall confirmatory claim: not confirmed** (`both_models` required).

| | SDXL (n=418) | SD 3.5 (n=417) |
|---|---|---|
| **P1** veridicality LMM slope (≥ +0.20) | **+0.463** [+0.382, +0.543] ✓ | −0.261 ✗ ⚠ |
| raw ρ (veridicality vs g) | +0.576 | −0.071 |
| **P2** partial ρ, Klüver held fixed (≥ +0.20) | **+0.417** [+0.322, +0.506] ✓ | −0.161 ✗ |
| **P3** inter-judge κ (≥ 0.40) | **0.529** ✓ | **0.485** ✓ |
| Verdict | **all three pass** | P1, P2 fail |

⚠ **The SD 3.5 mixed model did not converge** (MLE failure; Hessian not positive
definite). SDXL converged with no warnings. **−0.261 is not a usable estimate**
and must not be quoted as an effect size.

---

## 2. The result that mattered: P2 passes on SDXL

Veridicality still tracks guidance at **+0.417** after partialling out the Klüver
composite. The two composites correlate at only **−0.62** (E2), sharing under 40%
of variance.

So on the convolutional model, **the global scale is not a restatement of the
local one**. An image's overall character and the inventory of what broke inside
it are related but separable measurements. That is the quantitative form of the
"neither scale contains the other" claim, which until now rested on two
hand-picked images.

P1 also replicates Exp 01 cleanly on SDXL: **+0.463** standardized against Exp 01's
exploratory ρ of +0.67, and **6/6 prompts positive** (bicycle 0.83, oranges 0.78,
still life 0.58, forest 0.53, portrait 0.53, living room 0.40).

---

## 3. Why SD 3.5 fails, and why the failure is uninformative

Not because the effect is absent. Because **both endpoints are monotonic tests
applied to a non-monotonic curve** (E1, pre-specified):

| mean veridicality by g | 1 | 2 | 3 | 5 | 7 | 11 | 15 |
|---|---|---|---|---|---|---|---|
| SDXL | 1.26 | 2.13 | 2.58 | 2.81 | 2.81 | **2.83** | 2.67 |
| SD 3.5 | 2.00 | 2.77 | 2.84 | 2.91 | **2.92** | 2.47 | **1.85** |

SD 3.5 peaks at g=7 and collapses to **1.85 at g=15, below where it started**, a
drop of 1.07. Raw ρ is −0.07 because the two limbs cancel. Spearman and a linear
slope cannot see an inverted U; per-prompt, only 1/6 prompts is positive, which is
the same cancellation showing up prompt by prompt.

**This is the second time the same mistake has cost an endpoint.** Exp 03's
post-hoc §9.1 found that a linear slope could not distinguish SDXL's gradient from
SD 3.5's cliff. Exp 03b's pre-registered endpoints cannot see an inverted U. Both
times the *instrument* was fine and the *assumed shape of the response* was wrong.
Any future pre-registration in this program should choose a shape-agnostic
endpoint by default and justify a linear one, rather than the reverse.

---

## 4. E1: over-conditioning degrades realism too

The overshoot is confirmed on **both** models, gently on SDXL (peak g=11, drop
0.16) and severely on SD 3.5 (peak g=7, drop 1.07).

This unifies three observations previously filed as separate anomalies:

- **SD 3.5's portrait sign flip** in Exp 03 (distortion +0.30 composite, +0.33 on
  the distortion field alone)
- **The distortion field firing at both ends of the dial** (`analysis.md` §9.3)
- **SD 3.5's inverted-U quality curve**, peaking at g=11 and falling at g=15
  (`analysis.md` §8.4)

One phenomenon: **turning guidance up too far degrades realism as surely as
turning it down.** The high-guidance failure is waxy, over-saturated and
over-detailed rather than melted, but it is a failure of veridicality either way.
Pre-specified as E1, so this is exploratory-but-predicted, not a story fitted
afterwards.

---

## 5. E3: most of the distortion signal was rendering style

The largest construct-validity threat in Exp 03 was that low-guidance
"distortion" partly measures **how painterly an image looks** rather than
objecthood coming apart (`analysis.md` §6). Veridicality measures rendering style
directly, so E3 partials it out.

| | raw ρ(distortion, g) | veridicality held fixed | pattern |
|---|---|---|---|
| SDXL | −0.486 | **−0.175** [−0.268, −0.075] | confounding, **64% explained** |
| SD 3.5 | −0.114 | −0.179 [−0.280, −0.078] | **suppression** |

**On SDXL, roughly two thirds of the distortion-guidance association is accounted
for by veridicality.** The caveat is not a hedge; it is a measured effect of
substantial size. Exp 01's headline field was in large part tracking how painterly
the picture had become. Distortion retains a real independent association
(−0.175, CI excludes 0), so the field is not empty, but it is much smaller than
the −0.486 that Exp 01 and Exp 03 both leaned on.

On SD 3.5 the pattern **inverts**: the partial is *larger* than the raw, which is
suppression rather than confounding. Consistent with E1, since both variables are
non-monotonic there and veridicality was masking the distortion signal rather than
producing it. A "share explained" figure is undefined in this case and the report
returns null rather than a negative percentage.

**This strengthens the paired-instrument argument rather than damaging Exp 03.**
The local scale was contaminated in a way that only a global axis could reveal,
which is precisely the case for running both.

---

## 6. Secondary endpoints (BH-corrected, all q ≤ .0004)

| | SDXL | SD 3.5 | Exp 01 direction |
|---|---|---|---|
| spontaneity vs g | −0.27 | −0.22 | −0.40 ✓ |
| complexity vs g | −0.17 | −0.24 | −0.54 ✓ |
| coherent_scene vs g | +0.36 | +0.20 | +0.41 ✓ |

All three replicate Exp 01's directions **on both models**, which is a stronger
generality result than the primary endpoint achieved. Note the uncond arm is
excluded from every spontaneity analysis: with no prompt the rubric scores
spontaneity 3 by construction.

---

## 7. Reliability

| field | SDXL | SD 3.5 |
|---|---|---|
| veridicality | 0.685 | 0.723 |
| spontaneity | 0.625 | 0.348 |
| **complexity** | **0.276** | 0.383 |

⚠ **Complexity is the weak field.** The composite clears 0.40 on both models, but
complexity alone would not on SDXL. Two capable judges do not agree on what
"elaborate" means, which is unsurprising for the least concrete of the three axes
and should be treated as the least trustworthy secondary result.

Veridicality, the primary field, is the **most** reliable on both models. The
endpoint the pre-registration rides on is the one the judges agree about.

---

## 8. Limitations

- The SD 3.5 primary estimate is **not converged** and is reported only to be
  transparent about the failure.
- Both primary endpoints assume monotonicity, which E1 shows is false on SD 3.5
  and mildly false on SDXL.
- Judges are shared with the Klüver scale, so P2's dissociation could reflect
  rater-specific structure rather than a property of the images. The blind human
  subset was not re-rated on these axes; that check remains open.
- Weaker pre-registration standing than Exp 03: the images existed and had been
  seen, and the Klüver results were known, when the predictions were written.
  Only the scores were pre-data. Declared in the pre-registration.
- Complexity's reliability (§7) undercuts its secondary result.
- E3's SD 3.5 suppression pattern is described, not explained.

---

## 9. What this does *not* claim

- Does not claim the two scales are dissociable in general. Demonstrated on SDXL;
  untestable on SD 3.5 with the endpoints chosen.
- Does not claim SD 3.5 lacks a veridicality dose-response. It has a strong
  non-monotonic one that the pre-registered tests cannot express.
- Does not claim the distortion field is worthless. It retains an independent
  association after veridicality is held fixed; it is simply much smaller than
  previously reported.
- Does not convert Exp 03's null into a confirm, or vice versa.

---

## 10. What follows

1. **A shape-agnostic re-test of SD 3.5.** New analysis on seen data, so it must
   be pre-specified and labelled exploratory before it runs, exactly as §9 of
   `analysis.md` was.
2. **Re-rate the human subset on these axes**, closing the independent-validation
   gap and testing whether human–model agreement is better on continuous axes than
   it was on the ordinals (human–Claude was κ 0.337 there).
3. **The paired profile** the two scales now justify: a local inventory plus
   global axes, reported as a curve rather than a scalar.

---

## 11. Artifacts

- Pre-registration: `preregistration_axes.json` (committed before scoring)
- Rubric: `rubric_axes.py` (definitions verbatim from Exp 01)
- Judgements: `results-local/{sdxl,sd35}/judgements_{claude,qwen}_axes.json`,
  1,720 scores, 0 errors, 0 coerce-defaults
- Screening probe: `probes/probe_Qwen3-VL-32B-Instruct-4bit_axes.json`
  (gradedness 0.476, no flat field)
- Analysis: `analyze_axes.py` → `axes_report.json`
