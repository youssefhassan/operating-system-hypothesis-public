# Experiment 03 — Analysis

**Status:** in progress (2026-08-08). Generation, quality, Claude judging, and the
blind human subset are complete. The open-weight judge panel was diagnosed and is
being re-run; numbers below marked **TBD** will be filled when that lands.
Pre-registration: [`preregistration.json`](preregistration.json) +
[`analysis_plan.md`](analysis_plan.md). Daily record: [`log.md`](log.md).

This experiment hardens Exp 01's *exploratory* Klüver Level-2/3 dose-response
into a confirmatory test along four axes: multi-prompt generality, inter-rater
reliability (including a human subset), quality-controlled de-confound, and
FDR-corrected secondary fields.

---

## 1. What was run

| Leg | Status |
|---|---|
| Generation (SDXL + SD 3.5, 430 images each) | ✅ 2026-07-22 |
| Judge A — Claude Sonnet 5 (Batches) | ✅ |
| Judge B — Qwen2.5-VL-7B → **amended** to Qwen3-VL-32B | 🔄 re-judge in progress |
| Judge C — Llama-3.2-11B (dead rater; replacement pending) | ⚠ archived |
| Quality (CLIP-IQA + LAION-aesthetic) | ✅ |
| Blind human subset (28 images, author) | ✅ 2026-08-08 |
| Analysis (`analyze.py`, figures) | ⬜ after re-judge |
| Sonnet 5 vs 4.6 calibration (§5 of plan) | ⬜ not run (declared) |

---

## 2. Pre-registered confirm criteria (reminder)

Confirm requires **both** models, all of:

1. LMM standardized composite slope ≤ −0.2, BH *p* ≤ .05, CI excludes 0
2. Quality-controlled partial Spearman ≤ −0.2, CI excludes 0
3. Prompt generality: negative slope in ≥ 5/6 prompts
4. Inter-judge composite weighted κ ≥ 0.4

Null on any of: slope CI includes 0 on either model; |partial ρ| < 0.1 after
quality control; inter-judge κ < 0.4.

---

## 3. Original 3-judge panel (2026-07-22) — inconclusive-null

| Criterion | SDXL 3-judge | SDXL Claude+Qwen | SD 3.5 3-judge | SD 3.5 Claude+Qwen |
|---|---|---|---|---|
| LMM slope (≤ −0.2) | −0.297 ✓ | −0.318 ✓ | −0.135 ✗ | −0.167 ✗ |
| Partial ρ (≤ −0.2) | −0.433 ✓ | −0.420 ✓ | −0.188 ✗ | −0.245 ✓ |
| Prompts (≥5/6) | 6/6 ✓ | 6/6 ✓ | 4/6 ✗ | 5/6 ✓ |
| Composite κ (≥0.4) | 0.126 ✗ | 0.291 ✗ | 0.135 ✗ | 0.158 ✗ |

Printed verdict on both models: *inconclusive-null (judges disagree)*. That
string short-circuits on the κ gate and hides SD 3.5's independent misses on
slope / generality — which must be reported separately.

**Cause of the κ failure (diagnosed 2026-08-08, not a soft disagreement):**

- **Llama-3.2-11B** scored `fragmentation = condensation = distortion = 0` on
  every completed image (zero variance). Pairwise κ against it is exactly 0 by
  arithmetic. ~90 "errors" were truncated JSON at `max_tokens=400`, not model
  refusals.
- **Qwen2.5-VL-7B** is a binary detector: ~96% zeros, jumps to 3, almost never
  uses 1 or 2. On the human subset it scored **0 on all four fields on all 28
  images**.

The 3-judge composite κ of ~0.13 is therefore largely a mechanical artifact of
averaging in dead instruments. Axis 2 of the hardening (independent judges) did
not succeed as executed.

---

## 4. Human subset — rubric is scoreable; open-weight judges were not

28 images, stratified across (model × guidance-bin × prompt), author-rated
blind, guidance hidden, identical rubric. Single rater — stated limitation; no
human–human κ.

| vs. human | composite weighted κ | 95% bootstrap CI |
|---|---|---|
| Claude | **0.337** | [0.106, 0.495] |
| Qwen2.5-VL-7B | 0.0 | [0, 0] |
| Llama-3.2-11B | −0.027 | [−0.052, 0] |

Qwen and Llama were **degenerate on the subset** (flatlined), so their κ of 0 is
not disagreement. Human–Claude is fair-to-moderate: the CI excludes 0 but also
includes the 0.4 gate. Human κ was never a pre-registered confirm threshold
(that gate is inter-judge).

Per-field human vs Claude (κ / Spearman / means):

| field | κ | Spearman | human mean / Claude mean |
|---|---|---|---|
| fragmentation | 0.417 | 0.509 | 0.43 / 0.14 |
| condensation | 0.382 | 0.437 | 0.68 / 0.50 |
| reduplication | 0.338 | 0.433 | 0.18 / 0.68 |
| **distortion** | **0.212** | **0.306** | 1.04 / 0.89 |

**Distortion is the weakest field**, and it is Exp 01's headline (ρ = −0.64).
Rank agreement is no better than level agreement (composite Spearman 0.324), so
this is not a tidy calibration offset. Claude over-reports reduplication and
under-reports fragmentation relative to the human; the offsets roughly cancel in
the composite (0.58 vs 0.55).

Tiling is untestable from this draw (human and Claude both 0/28) — sampling
limit, not a result.

---

## 5. Screening probes — a size threshold, not a generation wall

`judge_probe.py` scores the same 28 images under candidate MLX configs.
Selection uses **gradedness** (share of scores on interior levels 1–2) and
Spearman vs Claude — deliberately **not** human κ, so the ratings stay an
independent validation of whichever judge is chosen.

| probe | ok | err | gradedness | mean ρ vs Claude | flat fields |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B-4bit | 28 | 0 | 0.000 | n/a | all four |
| Qwen3-VL-8B-4bit | 28 | 0 | 0.000 | n/a | all four |
| **Qwen3-VL-32B-4bit** | 28 | 0 | **0.134** | **0.355** | **none** |

**Hypothesis elimination on Qwen3-VL-8B raw replies:**

1. Images never reach the model — **false** (accurate free-form captions).
2. Rubric priming suppresses anomalies — **false** (neutral and
   "AI-artifact" framings also return "no anomalies").
3. Caption normalization — **supported**. On a visibly ghosted living room
   (`sd35/p5_livingroom_g1_s47.png`) the 8B reported a "realistic, well-lit
   living room … without any signs of melting, warping, or impossibility."

Qwen3-VL-8B and Qwen3-VL-32B are the same family, release, quantization, rubric,
and harness. The 8B is blind; the 32B is not. So this is a **capacity threshold
between 8B and 32B**, not model age. Sub-~30B VLMs fail *silently* — well-formed
JSON, zero parse errors, confident plausible captions — which is the dangerous
failure mode for VLM-as-judge pipelines.

**Construct-validity caveat (cuts against the author):** on the painterly still
life the author scored condensation 3 / distortion 3, the 32B scored 0/0/0/1
("slight warping"), and free-form description is basically a normal still life.
Part of the low-g human/Claude signal may be reading painterly looseness as
distortion. The ghosted living room is unambiguous either way.

---

## 6. Dated amendments (judge models)

`judge_models` is listed under `swappable_without_reclassifying` in the
pre-registration; changing the judge does not change `rubric_version`.

| Date | Amendment |
|---|---|
| 2026-07-21 | Third judge (Llama) added; Claude Batches API |
| 2026-08-08 | Qwen2.5-VL-7B → **Qwen3-VL-32B** after human-subset κ = 0 and probe evidence. Deviation from the *named* fallback (prereg said Qwen2.5-VL-32B; Qwen3-VL is one generation newer and cleared the screen). Logged, not quietly adopted. |
| 2026-08-08 | Llama-3.2-11B retained in the archive only; judge C replacement (Gemma-3-27B candidate) pending probe. |

Harness repairs that do not change the instrument: truncated-JSON repair,
`max_tokens=700`, `missing_fields` on coerce, raw-reply sidecars, save-every-image.

---

## 7. Re-judged panel results — TBD

*Fill after `judgements_qwen.json` is complete for both models and
`analyze.py --both --plot` (and `--judges claude,qwen`) has been re-run.*

### 7.1 Primary panel (as amended)

| Criterion | SDXL | SD 3.5 |
|---|---|---|
| LMM slope | TBD | TBD |
| Partial ρ | TBD | TBD |
| Prompts | TBD | TBD |
| Composite κ | TBD | TBD |
| Human vs Claude / vs Qwen32 | TBD | — |

### 7.2 Verdict

TBD against the decision table in `analysis_plan.md` §8. Report SD 3.5's
independent gate failures even if κ short-circuits the printed string.

---

## 8. Methodological finding (independent of the dose-response)

Current open-weight vision-language models **below ~30B parameters** do not
detect object-level generative artifacts under any prompt framing tested here,
while a frontier judge (Claude Sonnet 5) does and tracks a human at composite
κ ≈ 0.34. The failure mode is silent. This is reportable on its own given how
fast VLM-as-judge pipelines are spreading, and it reframes Exp 03's original
inconclusive-null from "the metric failed" to "here is why this class of
pipeline fails below a size threshold."

---

## 9. Limitations

- Single human rater; no human–human reliability.
- Author knew the hypothesis while rating (stated; blinding of model/guidance
  intact).
- Distortion construct may mix "painterly looseness" with true objecthood
  dissolution — needs sharper operationalization if the field stays primary.
- Sonnet 5 vs Exp 01's Sonnet 4.6 calibration cross-check was pre-registered as
  reported-in-analysis and was **not run**; declare rather than imply.
- SD 3.5 may remain below the confirm slope threshold even with a repaired
  panel (`both_models_required` is true).
- Tiling untested on the human subset.

---

## 10. What this does *not* claim

- It does not revive Exp 01's form-constant (Level-1) hypothesis — that remains
  a clean null.
- It does not claim that guidance *is* REBUS precision-relaxation; the analogy
  stays computational and suggestive.
- A working open-weight judge is not automatically an *agreeing* one — clearing
  gradedness does not guarantee clearing κ ≥ 0.4.
