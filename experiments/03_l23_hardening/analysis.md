# Experiment 03 — Analysis

**Completed 2026-08-08** (amended judge panel; post-hoc shape/control analyses
added the same day). Pre-registration: [`preregistration.json`](preregistration.json) +
[`analysis_plan.md`](analysis_plan.md). Daily record: kept privately.
Handoff for follow-on work: [`HANDOFF.md`](HANDOFF.md).

---

## 0. What this experiment is

Exp 03 is not primarily a yes/no on one slope. It is a **controllable neuroAI
assay** for objecthood dissolution under precision relaxation:

| Role | Choice |
|---|---|
| Independent variable | Classifier-free guidance *g* (the "knob") |
| Dependent measure | Klüver Level-2/3 fields — reduplication, fragmentation, condensation, distortion (+ tiling) |
| Neuroscience import | Klüver's hallucinatory constants; REBUS-style precision relaxation as the *analogy* for lowering *g* (computational, not identity) |
| Substrate factor | SDXL (conv UNet) vs SD 3.5 (MMDiT) |
| Generality factor | 6 prompts spanning object count and objecthood, plus forest control |
| Confound check | No-reference quality (CLIP-IQA + LAION-aesthetic); partial correlation + matched-quality arms |
| Reliability | Independent VLM judges + blind human subset |

The win is the scaffold: a doseable generative system, a phenomenology neuroscience
already has language for, and pre-registered hygiene (multi-prompt, multi-judge,
quality de-confound, FDR). The gate table below is what that scaffold produced —
including an honest overall non-confirm under `both_models_required`.

Derives from Exp 01: Level-1 form constants were a clean null; Level-2/3
objecthood dissolution was the exploratory signal this run hardens.

---

## 1. Results at a glance

**Overall confirmatory claim: not confirmed** (`both_models_required`; SD 3.5
slope −0.182 vs threshold ≤ −0.2). That is scientific honesty inside a working
method, not a failed setup.

| | SDXL (Claude + Qwen3-VL-32B) | SD 3.5 (Claude + Qwen3-VL-32B) |
|---|---|---|
| LMM slope (≤ −0.2) | **−0.340** ✓ | −0.182 ✗ |
| Partial ρ (≤ −0.2) | **−0.433** ✓ | **−0.246** ✓ |
| Prompts (≥5/6) | **6/6** ✓ | **5/6** ✓ |
| Composite κ (≥0.4) | **0.562** ✓ | **0.440** ✓ |
| Per-model verdict | **confirm** | **null** (slope) |

**Interpretation (revised 2026-08-08 after §9).** The two models do not differ by
*degree*, they differ by *shape*. On SDXL objecthood dissolution is a **graded
dose-response**: it survives dropping the bottom of the dial (ρ = −0.244 with
g = 1 excluded) and survives the quality de-confound with g = 1 excluded
(Cliff's δ 0.355). On SD 3.5 the entire effect is a **single threshold event at
g = 1**: with g = 1 removed the dose-response is gone (ρ = +0.048, *p* = .38)
and the matched-quality contrast collapses to zero (δ = −0.043, CI spans 0).

So the honest reading is not "SD 3.5 shows the same effect, weaker." It is
"SD 3.5 shows a cliff where SDXL shows a gradient," and the pre-registered
*linear* endpoint could not tell those apart. Substrate matters, and it matters
in a sharper way than the gate table alone implies. This is post-hoc; see §9 for
the discipline note.

Secondary products of the same scaffold:

1. **Silent VLM-judge failure below ~30B** — open-weight judges return
   well-formed JSON and "no anomaly" captions for visibly ghosted images.
2. **Panel repair works** — Qwen3-VL-32B raises Claude–Qwen κ from 0.29→0.56
   (SDXL) and 0.16→0.44 (SD 3.5) without changing the rubric.
3. **Human–model gap** — human–Claude κ ≈ 0.34; human–Qwen32 κ ≈ 0.12 even when
   the two VLMs agree with each other.
4. **The composite hides field-level structure** (§9.2–9.3): the forest control
   dissociated on the field it was designed to dissociate on and on tiling, but
   its *composite* still tracked g on SDXL; and SD 3.5's portrait reversal is a
   sign flip in one field, not noise.

The original 2026-07-22 3-judge run was an inconclusive-null driven by dead
open-weight judges; superseded by the amended panel, not quietly replaced.

---

## 2. What was run

| Leg | Status |
|---|---|
| Generation (SDXL + SD 3.5, 430 images each) | ✅ 2026-07-22 |
| Judge A — Claude Sonnet 5 (Batches) | ✅ |
| Judge B — Qwen2.5-VL-7B → **Qwen3-VL-32B** (dated amendment) | ✅ 2026-08-08 |
| Judge C — Llama-3.2-11B | ⚠ archived (dead); Gemma-3-27B probed, not promoted |
| Quality (CLIP-IQA + LAION-aesthetic) | ✅ |
| Blind human subset (28 images, author) | ✅ 2026-08-08 |
| Analysis (`analyze.py --both --judges claude,qwen --plot`) | ✅ |
| Post-hoc shape / control analyses (`posthoc.py`) | ✅ 2026-08-08 |
| Sonnet 5 vs 4.6 calibration (§5 of plan) | ❌ not run (declared) |

Reports: `results-local/<model>/l23_report_claude-qwen.json` (amended
confirmatory panel), `posthoc_report.json` (exploratory). Archived 7B judgements
under `archive/`. Original 3-judge `l23_report.json` kept as the pre-amendment
record.

---

## 3. Pre-registered confirm criteria

Confirm requires **both** models, all of:

1. LMM standardized composite slope ≤ −0.2, BH *p* ≤ .05, CI excludes 0
2. Quality-controlled partial Spearman ≤ −0.2, CI excludes 0
3. Prompt generality: negative slope in ≥ 5/6 prompts
4. Inter-judge composite weighted κ ≥ 0.4

---

## 4. Original 3-judge panel (2026-07-22) — instrument failure

| Criterion | SDXL 3-judge | SDXL Claude+Qwen7B | SD 3.5 3-judge | SD 3.5 Claude+Qwen7B |
|---|---|---|---|---|
| LMM slope | −0.297 ✓ | −0.318 ✓ | −0.135 ✗ | −0.167 ✗ |
| Partial ρ | −0.433 ✓ | −0.420 ✓ | −0.188 ✗ | −0.245 ✓ |
| Prompts | 6/6 ✓ | 6/6 ✓ | 4/6 ✗ | 5/6 ✓ |
| Composite κ | 0.126 ✗ | 0.291 ✗ | 0.135 ✗ | 0.158 ✗ |

**Cause (not soft disagreement):** Llama-3.2-11B scored fragmentation =
condensation = distortion = 0 on every completed image (κ vs it is 0 by
arithmetic; ~90 "errors" were truncated JSON at `max_tokens=400`).
Qwen2.5-VL-7B is a near-binary detector; on the human subset it scored **0 on
all four fields on all 28 images**.

---

## 5. Human subset — rubric is scoreable

28 images, stratified, author-rated blind, guidance hidden, identical rubric.
Single rater — stated limitation.

| vs. human (n=28) | composite weighted κ | 95% bootstrap CI |
|---|---|---|
| Claude | **0.337** | [0.106, 0.495] |
| Qwen3-VL-32B (after amendment) | 0.124 | [−0.059, 0.313] |
| Qwen2.5-VL-7B (pre-amendment) | 0.0 | [0, 0] |
| Llama-3.2-11B | −0.027 | [−0.052, 0] |

Human–Claude fair-to-moderate. Human–Qwen32 weak even though Claude–Qwen32
agree at κ 0.56 / 0.44 — the VLMs share a signal that only partially overlaps
the human.

Per-field human vs Claude: fragmentation 0.417, condensation 0.382,
reduplication 0.338, **distortion 0.212** (Exp 01's headline field; hardest for
human–model agreement). Tiling untestable (0/28). §9.3 gives a probable
mechanism for why distortion is the weak field.

---

## 6. Screening probes — size threshold; Gemma partial fail

| probe (same 28 images) | gradedness | mean ρ vs Claude | flat fields |
|---|---|---|---|
| Qwen2.5-VL-7B-4bit | 0.000 | n/a | all four |
| Qwen3-VL-8B-4bit | 0.000 | n/a | all four |
| **Qwen3-VL-32B-4bit** | **0.134** | **0.355** | **none** |
| Gemma-3-27B-it-qat-4bit | 0.134 | 0.225 | fragmentation, condensation |

Images reach the models; stripping rubric priming does not help. Sub-~30B VLMs
**fail silently**. Same family 8B vs 32B ⇒ capacity threshold, not model age.
Gemma matches Qwen32 gradedness but is dead on half the composite — not promoted
to judge C (would reintroduce the Llama κ failure mode).

**Construct-validity caveat:** on a painterly still life the author scored
condensation/distortion 3; Qwen3-VL-32B scored near 0. Part of the low-g signal
may be painterly looseness vs true objecthood dissolution. Ghosted living-room
cases are unambiguous.

---

## 7. Dated amendments

`judge_models` ∈ `swappable_without_reclassifying`. Rubric text unchanged.

| Date | Amendment |
|---|---|
| 2026-07-21 | Third judge (Llama) added; Claude Batches API |
| 2026-08-08 | Qwen2.5-VL-7B → **Qwen3-VL-32B** after human-subset κ = 0 and probes. Deviation from the *named* Qwen2.5-VL-32B fallback (one generation newer; cleared the screen). |
| 2026-08-08 | Llama archived; Claude+Qwen32 = amended confirmatory panel. Gemma-3-27B probed, not promoted. |

Harness repairs: truncated-JSON repair, `max_tokens=700`, `missing_fields` on
coerce, raw-reply sidecars, save-every-image.

**Not an amendment:** §9 is exploratory analysis of the same data, computed after
the verdict. It adds no gate and changes no threshold.

---

## 8. Amended panel detail (Claude + Qwen3-VL-32B)

### 8.1 Primary gates

| Criterion | SDXL | SD 3.5 |
|---|---|---|
| LMM slope (95% CI) | −0.340 [−0.400, −0.279] | −0.182 [−0.248, −0.116] |
| Partial ρ controlling Q | −0.433 [−0.508, −0.347] | −0.246 [−0.338, −0.146] |
| Prompt generality | 6/6 negative | 5/6 (portrait +0.30) |
| Composite weighted κ | 0.562 | 0.440 |
| Matched-quality Cliff's δ | 0.489 [0.362, 0.606] | 0.188 [0.050, 0.320] |

### 8.2 Per-field inter-judge κ (Claude vs Qwen32)

| field | SDXL | SD 3.5 |
|---|---|---|
| fragmentation | 0.706 | 0.513 |
| condensation | 0.594 | 0.391 |
| tiling | 0.583 | 0.456 |
| distortion | 0.515 | 0.418 |
| reduplication | 0.432 | 0.439 |

### 8.3 Per-prompt Spearman (composite vs g)

**SDXL:** bicycle −0.74, forest −0.62, living room −0.57, still life −0.57,
oranges −0.49, portrait −0.28.

**SD 3.5:** living room −0.49, bicycle −0.46, oranges −0.40, still life −0.24,
forest −0.02, **portrait +0.30** (generality holdout).

### 8.4 Composite and quality by guidance

| g | SDXL composite | SDXL Q | SD 3.5 composite | SD 3.5 Q |
|---|---|---|---|---|
| 1 | 1.108 | −0.691 | 1.094 | −0.591 |
| 2 | 0.224 | −0.127 | −0.151 | −0.153 |
| 3 | −0.051 | −0.112 | −0.166 | −0.008 |
| 5 | −0.242 | 0.049 | −0.260 | 0.118 |
| 7 | −0.360 | 0.156 | −0.280 | 0.312 |
| 11 | −0.340 | 0.291 | −0.165 | 0.364 |
| 15 | −0.350 | 0.438 | −0.082 | −0.043 |
| **uncond** | **1.510** | n/a | **2.173** | n/a |

The empty-prompt baseline is the extreme on this measure on both models, above
even g = 1. (Contrast Exp 01, where the uncond arm was *lateral* to the sweep on
the Level-1 metric rather than at its end.) SDXL quality rises monotonically;
SD 3.5 quality is an inverted U that turns over after g = 11.

---

## 9. Post-hoc analyses (`posthoc.py`) — EXPLORATORY

Computed **after** the confirmatory verdict, on the same data, with no change to
the rubric, thresholds, or primary endpoint. These are hypotheses for the next
experiment, not a rescue of this one. Full output: `posthoc_report.json`.

### 9.1 Shape: SDXL is a gradient, SD 3.5 is a cliff

The pre-registered endpoint is a **linear** slope in guidance. Neither curve is
linear (§8.4). Refitting with the bottom of the dial removed separates "graded
dose-response" from "single threshold event."

| | SDXL | SD 3.5 |
|---|---|---|
| Share of total composite drop in the g=1→2 step | 60% | **91%** |
| Spearman vs g, **excluding g = 1** | **−0.244** (*p* = .0002) | +0.048 (*p* = .38) |
| LMM slope, excluding g = 1 | **−0.162** [−0.209, −0.115] | +0.029 [−0.007, 0.065] |
| Spearman vs g, excluding g ≤ 2 | −0.120 (*p* = .034) | +0.102 (*p* = .080) |
| LMM slope, excluding g ≤ 2 | −0.088 [−0.127, −0.049] | +0.047 [0.010, 0.084] |
| Matched-quality δ, low arm g ≤ 3 (pre-reg) | 0.489 [0.362, 0.606] | 0.188 [0.050, 0.320] |
| Matched-quality δ, low arm g ∈ {2,3} | **0.355** [0.199, 0.500] | **−0.043** [−0.199, 0.113] |

**SDXL survives every removal.** The dose-response is still negative and still
quality-de-confounded with g = 1 dropped, and remains (weakly) negative with
g ≤ 2 dropped. There is a real gradient across the usable range.

**SD 3.5's effect is entirely carried by g = 1.** Remove that one dose level and
the dose-response vanishes (*p* = .38), the matched-quality contrast collapses to
zero with a CI spanning 0, and above g ≤ 2 the sign flips slightly *positive*.

Consequence for §1: the SD 3.5 result should be described as a **threshold at the
bottom of the dial**, not as an attenuated gradient. Two caveats that cut against
over-reading it:

- g = 1 is the most confounded dose level (fully un-amplified conditioning, well
  outside either model's operating range), so an effect that lives only there is
  the hardest kind to separate from under-conditioning artifact. The pre-registered
  matched-quality result (δ 0.188) *does* control quality, but it pools g ≤ 3 and
  is therefore also carried by g = 1.
- SDXL's gradient is the stronger claim precisely because it does not depend on
  the single most degenerate setting.

**Design lesson for the next experiment:** a linear endpoint cannot distinguish a
gradient from a cliff, and the two are different scientific claims. Denser
sampling at the bottom (g ∈ {1, 1.25, 1.5, 2}) plus a shape-agnostic endpoint
(monotone trend test, or an explicit threshold model) is the fix.

### 9.2 The forest control: it dissociated on the fields it was designed for

`preregistration.json` designates `p6_forest` a low-objecthood control that
"should show tiling but LOW reduplication/condensation if the effect is
object-bound." Its *composite* ρ of −0.62 on SDXL reads as a failed control. The
per-field breakdown says something more specific.

Per-field Spearman vs g, forest prompt:

| | reduplication | fragmentation | condensation | distortion | tiling |
|---|---|---|---|---|---|
| SDXL | **+0.05** | −0.61 | −0.58 | −0.50 | −0.26 |
| SD 3.5 | **+0.52** | −0.38 | −0.23 | +0.10 | +0.17 |

Tiling rate at the lowest guidance level, by prompt (SDXL): **forest 0.40**,
oranges 0.20, still life 0.10, portrait 0.10, bicycle 0.00, living room 0.00. On
SD 3.5 no prompt tiles at all (0.00 everywhere).

So on SDXL the control did **two of the three things it was designed to do**:
reduplication is flat (+0.05, the only prompt where it does not track g), and the
forest tiles at twice the rate of the next prompt and 4× the median. What it did
*not* do is keep the composite flat, because fragmentation, condensation and
distortion of *tree forms* all track g strongly.

Two readings, not yet separable:

- **(a) The composite is the wrong instrument for this prompt.** It averages
  fields the control was designed to dissociate on with fields it was not.
  Reduplication and tiling behaved exactly as pre-registered.
- **(b) The SDXL effect is more general than object binding.** Something happens
  to *form* at low g whether or not there are discrete objects to unbind.

On SD 3.5 the flat forest composite (−0.02) is **cancellation, not absence**:
reduplication runs strongly *positive* (+0.52) against fragmentation at −0.38.
Reporting the composite alone would have called this a clean control result. It
is not one.

### 9.3 Distortion is bidirectional, which explains the portrait holdout

SD 3.5's portrait is the prompt that costs it 6/6 generality (composite +0.30).
The per-field breakdown localises the reversal to one field:

| p2_portrait | reduplication | fragmentation | condensation | distortion | composite |
|---|---|---|---|---|---|
| SDXL | −0.25 | −0.54 | −0.55 | −0.23 | −0.28 |
| SD 3.5 | +0.00 | −0.32 | −0.42 | **+0.33** | **+0.30** |

On SD 3.5, fragmentation and condensation run the expected direction; only
**distortion inverts**. The judge notes say why: at g = 15 SD 3.5 portraits are
scored for *over*-conditioning artifacts ("exaggerated wrinkle patterns,
oversaturated skin tone, slightly warped"; "warped nose/mouth proportions, waxy
uneven skin texture"), i.e. the same 0–3 field is absorbing two opposite
failure modes:

- **low g**: melted, smeared, under-resolved anatomy;
- **high g**: over-baked, waxy, exaggerated anatomy.

This is a **construct problem in the rubric**, not noise in the data, and it is
consistent with distortion being simultaneously (i) the weakest human–Claude
field (κ 0.212, §5), (ii) the weakest per-field dose-response on SD 3.5 (ρ
−0.114), and (iii) Exp 01's headline field. A single ordinal that fires at both
ends of the dial cannot be a clean dose measure.

**Fix for the next experiment:** split distortion into signed sub-scales, or move
to Suzuki's continuous *realism* axis, which has a defined direction rather than
a magnitude-only ordinal.

---

## 10. Methodological finding (VLM-as-judge)

Open-weight VLMs **below ~30B** do not detect object-level generative artifacts
under any framing tested here; a frontier judge does and tracks a human at
κ ≈ 0.34. Failure mode is silent. This is reportable independently of the
dose-response and is a timely warning for VLM-as-judge pipelines.

---

## 11. Limitations

- Single human rater; author knew the hypothesis (model/guidance blinding intact).
- Human–Qwen32 κ weak (0.12); Claude–Qwen agreement ≠ human agreement.
- **Distortion mixes two opposite failure modes** (§9.3): under-conditioned melt
  and over-conditioned waxiness score on the same ordinal. This also means part
  of the low-g signal may be painterly looseness rather than objecthood loss.
- **SD 3.5's effect rests entirely on g = 1** (§9.1), the single most
  under-conditioned and therefore most confound-prone dose level.
- **The pre-registered linear endpoint cannot distinguish a gradient from a
  cliff** (§9.1); the two models differ in shape and the primary metric reports
  that difference only as a difference in magnitude.
- The forest control's composite did not stay flat on SDXL (§9.2), so the
  object-bound interpretation is not established; and its flat composite on
  SD 3.5 is field-level cancellation rather than a genuine null.
- Sonnet 5 vs Exp 01 Sonnet 4.6 calibration pre-registered as reported; **not run**.
- No third model family after Llama/Gemma; triangulation is two lineages.
- §9 is post-hoc on data already seen. Its tests were not pre-specified and carry
  no multiplicity correction.

---

## 12. What this does *not* claim

- Does not revive Exp 01's Level-1 form-constant hypothesis (still a clean null).
- Does not equate guidance with REBUS precision-relaxation (analogy only).
- Does not claim a cross-architecture confirm — SD 3.5 is threshold-shaped, not a
  weaker version of SDXL's gradient.
- Does not claim the effect is established as object-*bound*; §9.2 leaves that
  open on SDXL.
- Clearing inter-judge κ does not mean the metric matches humans closely.
- §9 is exploratory and does not convert the SD 3.5 null into a confirm; if
  anything it makes the SD 3.5 evidence weaker, not stronger.

---

## 13. Artifacts

- Reports: `results-local/{sdxl,sd35}/l23_report_claude-qwen.json` (confirmatory),
  `posthoc_report.json` (exploratory, §9)
- Figures: `results-local/{sdxl,sd35}/figures_claude-qwen/`
- Judgements: `judgements_claude.json`, `judgements_qwen.json` (Qwen3-VL-32B),
  `judgements_qwen_raw.json`
- Archive: `archive/judgements_qwen_Qwen2.5-VL-7B_{sdxl,sd35}.json`
- Human: `human_subset.json`, `human_ratings.json`
- Probes: `probes/probe_Qwen*.json`, `probes/probe_gemma-3-27b-it-qat-4bit.json`
- Code: `analyze.py` (confirmatory), `posthoc.py` (§9)

