# Experiment 03 — Analysis

**Completed 2026-08-08** (amended judge panel). Pre-registration:
[`preregistration.json`](preregistration.json) +
[`analysis_plan.md`](analysis_plan.md). Daily record: [`log.md`](log.md).
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

**Interpretation:** objecthood dissolution tracks guidance on the convolutional
UNet (SDXL) after quality control and at adequate inter-judge reliability. On
the MMDiT (SD 3.5) the same direction is present (CI excludes 0) but attenuated
below the pre-registered effect-size gate. Substrate matters — consistent with
the series' depth-and-substrate framing.

Secondary products of the same scaffold:

1. **Silent VLM-judge failure below ~30B** — open-weight judges return
   well-formed JSON and "no anomaly" captions for visibly ghosted images.
2. **Panel repair works** — Qwen3-VL-32B raises Claude–Qwen κ from 0.29→0.56
   (SDXL) and 0.16→0.44 (SD 3.5) without changing the rubric.
3. **Human–model gap** — human–Claude κ ≈ 0.34; human–Qwen32 κ ≈ 0.12 even when
   the two VLMs agree with each other.

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
| Sonnet 5 vs 4.6 calibration (§5 of plan) | ❌ not run (declared) |

Reports: `results-local/<model>/l23_report_claude-qwen.json` (amended
confirmatory panel). Archived 7B judgements under `archive/`. Original 3-judge
`l23_report.json` kept as the pre-amendment record.

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
human–model agreement). Tiling untestable (0/28).

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

---

## 9. Methodological finding (VLM-as-judge)

Open-weight VLMs **below ~30B** do not detect object-level generative artifacts
under any framing tested here; a frontier judge does and tracks a human at
κ ≈ 0.34. Failure mode is silent. This is reportable independently of the
dose-response and is a timely warning for VLM-as-judge pipelines.

---

## 10. Limitations

- Single human rater; author knew the hypothesis (model/guidance blinding intact).
- Human–Qwen32 κ weak (0.12); Claude–Qwen agreement ≠ human agreement.
- Distortion may mix painterly looseness with objecthood dissolution.
- Sonnet 5 vs Exp 01 Sonnet 4.6 calibration pre-registered as reported; **not run**.
- No third model family after Llama/Gemma; triangulation is two lineages.
- SD 3.5 portrait reverses expected sign.

---

## 11. What this does *not* claim

- Does not revive Exp 01's Level-1 form-constant hypothesis (still a clean null).
- Does not equate guidance with REBUS precision-relaxation (analogy only).
- Does not claim a cross-architecture confirm — SD 3.5 is attenuated.
- Clearing inter-judge κ does not mean the metric matches humans closely.

---

## 12. Artifacts

- Reports: `results-local/{sdxl,sd35}/l23_report_claude-qwen.json`
- Figures: `results-local/{sdxl,sd35}/figures_claude-qwen/`
- Judgements: `judgements_claude.json`, `judgements_qwen.json` (Qwen3-VL-32B),
  `judgements_qwen_raw.json`
- Archive: `archive/judgements_qwen_Qwen2.5-VL-7B_{sdxl,sd35}.json`
- Human: `human_subset.json`, `human_ratings.json`
- Probes: `probes/probe_Qwen*.json`, `probes/probe_gemma-3-27b-it-qat-4bit.json`
