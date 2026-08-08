# Experiment 03 — Analysis

**Completed 2026-08-08** (amended judge panel). Generation, quality, Claude
judging, blind human subset, open-weight judge diagnosis/repair, and
confirmatory analysis are done. Pre-registration:
[`preregistration.json`](preregistration.json) +
[`analysis_plan.md`](analysis_plan.md). Daily record: [`log.md`](log.md).

This experiment hardens Exp 01's *exploratory* Klüver Level-2/3 dose-response
into a confirmatory test along four axes: multi-prompt generality, inter-rater
reliability (including a human subset), quality-controlled de-confound, and
FDR-corrected secondary fields.

---

## Verdict (headline)

**Overall confirmatory claim: not confirmed** — `both_models_required` is true,
and SD 3.5 misses the pre-registered slope threshold (−0.182 vs ≤ −0.2).

| | SDXL (Claude + Qwen3-VL-32B) | SD 3.5 (Claude + Qwen3-VL-32B) |
|---|---|---|
| LMM slope (≤ −0.2) | **−0.340** ✓ | −0.182 ✗ |
| Partial ρ (≤ −0.2) | **−0.433** ✓ | **−0.246** ✓ |
| Prompts (≥5/6) | **6/6** ✓ | **5/6** ✓ |
| Composite κ (≥0.4) | **0.562** ✓ | **0.440** ✓ |
| Per-model verdict | **confirm** | **null** (slope) |

SDXL is a clean confirm on every gate after repairing judge B. SD 3.5 clears
reliability and the quality de-confound, and its slope CI excludes 0
([−0.248, −0.116]), but the point estimate sits just short of the pre-registered
−0.2 threshold. Honest line: *the objecthood-dissolution dose-response hardens
on the convolutional UNet (SDXL) and is attenuated on the MMDiT (SD 3.5).*

The original 2026-07-22 3-judge run was an inconclusive-null driven by dead
open-weight judges; that is superseded by the amended panel below, not quietly
replaced.

---

## 1. What was run

| Leg | Status |
|---|---|
| Generation (SDXL + SD 3.5, 430 images each) | ✅ 2026-07-22 |
| Judge A — Claude Sonnet 5 (Batches) | ✅ |
| Judge B — Qwen2.5-VL-7B → **Qwen3-VL-32B** (dated amendment) | ✅ 2026-08-08 |
| Judge C — Llama-3.2-11B | ⚠ archived (dead rater; no replacement landed) |
| Quality (CLIP-IQA + LAION-aesthetic) | ✅ |
| Blind human subset (28 images, author) | ✅ 2026-08-08 |
| Analysis (`analyze.py --both --judges claude,qwen --plot`) | ✅ |
| Sonnet 5 vs 4.6 calibration (§5 of plan) | ❌ not run (declared) |

Reports: `results-local/<model>/l23_report_claude-qwen.json` (amended
confirmatory panel). The archived 7B judgements live under `archive/`. The
original 3-judge `l23_report.json` files remain as the pre-amendment record.

---

## 2. Pre-registered confirm criteria (reminder)

Confirm requires **both** models, all of:

1. LMM standardized composite slope ≤ −0.2, BH *p* ≤ .05, CI excludes 0
2. Quality-controlled partial Spearman ≤ −0.2, CI excludes 0
3. Prompt generality: negative slope in ≥ 5/6 prompts
4. Inter-judge composite weighted κ ≥ 0.4

---

## 3. Original 3-judge panel (2026-07-22) — why it failed

| Criterion | SDXL 3-judge | SDXL Claude+Qwen7B | SD 3.5 3-judge | SD 3.5 Claude+Qwen7B |
|---|---|---|---|---|
| LMM slope | −0.297 ✓ | −0.318 ✓ | −0.135 ✗ | −0.167 ✗ |
| Partial ρ | −0.433 ✓ | −0.420 ✓ | −0.188 ✗ | −0.245 ✓ |
| Prompts | 6/6 ✓ | 6/6 ✓ | 4/6 ✗ | 5/6 ✓ |
| Composite κ | 0.126 ✗ | 0.291 ✗ | 0.135 ✗ | 0.158 ✗ |

**Cause of the κ failure (not soft disagreement):**

- **Llama-3.2-11B** scored fragmentation = condensation = distortion = 0 on every
  completed image. Pairwise κ against it is exactly 0 by arithmetic. ~90
  "errors" were truncated JSON at `max_tokens=400`.
- **Qwen2.5-VL-7B** is a near-binary detector (~96% zeros). On the human subset
  it scored **0 on all four fields on all 28 images**.

Axis 2 of the hardening did not succeed as first executed.

---

## 4. Human subset — rubric is scoreable

28 images, stratified, author-rated blind, guidance hidden, identical rubric.
Single rater — stated limitation.

| vs. human (n=28) | composite weighted κ | 95% bootstrap CI |
|---|---|---|
| Claude | **0.337** | [0.106, 0.495] |
| Qwen3-VL-32B (after amendment) | 0.124 | [−0.059, 0.313] |
| Qwen2.5-VL-7B (pre-amendment) | 0.0 | [0, 0] |
| Llama-3.2-11B | −0.027 | [−0.052, 0] |

Human–Claude is fair-to-moderate. Human–Qwen32 is weak (CI includes 0) even
though Claude–Qwen32 agree with each other at κ 0.56 / 0.44. The two capable
VLMs share a signal that only partially overlaps the human — important for
interpretation, and consistent with the construct-validity caveat in §5.

Per-field human vs Claude: fragmentation 0.417, condensation 0.382,
reduplication 0.338, **distortion 0.212**. Distortion was Exp 01's headline and
remains the hardest field for human–model agreement. Tiling untestable (0/28).

---

## 5. Screening probes — size threshold between 8B and 32B

| probe (same 28 images) | gradedness | mean ρ vs Claude | flat fields |
|---|---|---|---|
| Qwen2.5-VL-7B-4bit | 0.000 | n/a | all four |
| Qwen3-VL-8B-4bit | 0.000 | n/a | all four |
| **Qwen3-VL-32B-4bit** | **0.134** | **0.355** | **none** |

Images reach the models; stripping rubric priming does not help. Sub-~30B VLMs
**fail silently** — well-formed JSON, confident "no anomaly" captions for
visibly ghosted images. Same family/release for 8B vs 32B ⇒ capacity threshold,
not model age.

**Construct-validity caveat:** on a painterly still life the author scored
condensation/distortion 3; Qwen3-VL-32B scored near 0. Part of the low-g signal
may be reading painterly looseness as objecthood dissolution. The ghosted
living-room case is unambiguous.

---

## 6. Dated amendments

`judge_models` ∈ `swappable_without_reclassifying`. Rubric text unchanged.

| Date | Amendment |
|---|---|
| 2026-07-21 | Third judge (Llama) added; Claude Batches API |
| 2026-08-08 | Qwen2.5-VL-7B → **Qwen3-VL-32B** after human-subset κ = 0 and probe evidence. Deviation from the *named* Qwen2.5-VL-32B fallback (one generation newer; cleared the screen). |
| 2026-08-08 | Llama retained in archive only; 2-judge Claude+Qwen32 is the amended confirmatory panel. Gemma-3-27B candidate for a new judge C was not probed in this write-up. |

Harness repairs: truncated-JSON repair, `max_tokens=700`, `missing_fields` on
coerce, raw-reply sidecars, save-every-image.

---

## 7. Amended panel results (Claude + Qwen3-VL-32B)

### 7.1 Primary gates

| Criterion | SDXL | SD 3.5 |
|---|---|---|
| LMM slope (95% CI) | −0.340 [−0.400, −0.279] | −0.182 [−0.248, −0.116] |
| Partial ρ controlling Q | −0.433 [−0.508, −0.347] | −0.246 [−0.338, −0.146] |
| Prompt generality | 6/6 negative | 5/6 (portrait +0.30) |
| Composite weighted κ | 0.562 | 0.440 |
| Matched-quality Cliff's δ | 0.489 [0.362, 0.606] | 0.188 [0.050, 0.320] |

### 7.2 Per-field inter-judge κ (Claude vs Qwen32)

| field | SDXL | SD 3.5 |
|---|---|---|
| fragmentation | 0.706 | 0.513 |
| condensation | 0.594 | 0.391 |
| tiling | 0.583 | 0.456 |
| distortion | 0.515 | 0.418 |
| reduplication | 0.432 | 0.439 |

Distortion — Exp 01's headline field — clears 0.4 on both models after the
judge repair (it was 0.27 / 0.20 under Qwen-7B).

### 7.3 Per-prompt Spearman (composite vs g)

**SDXL:** bicycle −0.74, forest −0.62, living room −0.57, still life −0.57,
oranges −0.49, portrait −0.28.

**SD 3.5:** living room −0.49, bicycle −0.46, oranges −0.40, still life −0.24,
forest −0.02, **portrait +0.30** (the generality miss).

---

## 8. Methodological finding (independent of the dose-response)

Open-weight VLMs **below ~30B** do not detect object-level generative artifacts
under any framing tested here, while a frontier judge does and tracks a human
at κ ≈ 0.34. The failure mode is silent. Repairing the panel (Qwen3-VL-32B)
raises inter-judge κ from 0.29 → 0.56 on SDXL and 0.16 → 0.44 on SD 3.5 — enough
to clear the reliability gate — without changing the rubric.

---

## 9. Limitations

- Single human rater; no human–human reliability. Author knew the hypothesis
  (model/guidance blinding intact).
- Human–Qwen32 κ remains weak (0.12); Claude–Qwen agreement ≠ human agreement.
- Distortion construct may mix painterly looseness with true objecthood
  dissolution.
- Sonnet 5 vs Exp 01 Sonnet 4.6 calibration was pre-registered as reported and
  was **not run**.
- No third model family after Llama's failure; triangulation is two lineages.
- SD 3.5 portrait prompt reverses the expected sign.

---

## 10. What this does *not* claim

- Does not revive Exp 01's Level-1 form-constant hypothesis (still a clean null).
- Does not equate guidance with REBUS precision-relaxation.
- Does not claim a cross-architecture confirm — SD 3.5 is attenuated.
- Clearing inter-judge κ does not mean the metric matches humans closely.

---

## 11. Artifacts

- Reports: `results-local/{sdxl,sd35}/l23_report_claude-qwen.json`
- Figures: `results-local/{sdxl,sd35}/figures_claude-qwen/`
- Judgements: `judgements_claude.json`, `judgements_qwen.json` (now Qwen3-VL-32B),
  raw sidecars `judgements_qwen_raw.json`
- Archive: `archive/judgements_qwen_Qwen2.5-VL-7B_{sdxl,sd35}.json`
- Human: `human_subset.json`, `human_ratings.json`
- Probes: `probes/probe_Qwen*.json`
