# Experiment 03 — Analysis plan (pre-committed)

**Committed before any analyzed run**, in the same commit as
[`preregistration.json`](preregistration.json). This file spells out every
statistic that will be computed and how it is interpreted, so that no analytic
choice is made after seeing results (Methodology §4; user pre-registration
discipline). Deviations after data lands are dated amendments and reclassify the
affected analysis as exploratory.

The finding being hardened is Exp 01's post-hoc Klüver Level-2/3 dose-response.
Exp 03 tests whether it is real along four axes: **generality** (6 prompts),
**reliability** (2 judges + human subset), **not-a-confound** (quality-matched),
and **not-multiplicity** (FDR-corrected).

---

## 0. Inputs

- `results-local/<model>/*.png` — 430 images/model (6 prompts × 7 g × 10 seeds
  + 10 empty-prompt baselines), filenames encode `p{prompt}_g{guidance}_s{seed}`
  and `uncond_s{seed}` (guidance blinded at judging, un-blinded only here).
- `judgements_claude.json`, `judgements_qwen.json` — per-image L2/3 scores from
  the two judges (fields: reduplication, fragmentation, condensation, distortion,
  tiling), same fixed rubric text (`rubric_version = exp03-kluver-l2l3-v1`).
- `quality.json` — per-image CLIP-IQA + LAION-aesthetic scores.
- `human_ratings.json` — 25–30 stratified images scored by the author, blind.

Every figure is regenerated from these by `analyze.py` (Methodology §6: no
trust-me plots).

---

## 1. Metric construction (fixed)

1. Per image *i*, per field *f* ∈ {reduplication, fragmentation, condensation,
   distortion}: `score[f,i] = mean(claude[f,i], qwen[f,i])`.
2. Within each model, z-score each field across all **conditioned** images
   (baselines excluded from the z-scoring fit, then projected onto it).
3. `composite[i] = mean over the 4 z-scored fields`. This is the **primary
   endpoint** — one number per model.
4. `tiling` (binary) is analysed separately as a secondary field.

Rationale for averaging the two judges into the metric (rather than picking one):
it is more robust and mirrors Exp 01's original "mean over judges" definition of
M. Averaging is only valid if the judges agree, which §4's κ gate enforces.

---

## 2. Primary confirmatory test — LMM dose-response

Per model, fit:

```
composite ~ guidance_std + (1 | prompt) + (1 | seed)
```

(`statsmodels` MixedLM; `guidance_std` = z-scored guidance value.) The
random **prompt** intercept is load-bearing: it removes each prompt's baseline
L2/3 level, so the fixed-effect slope is the *within-prompt* guidance effect —
which is what makes the high-baseline-multiplicity prompts (p4 oranges, p6
forest) contribute correctly instead of dominating with their high absolute
reduplication.

- **Effect of interest:** the `guidance_std` slope β. Negative β ⇒ composite
  rises as g falls (the hardened direction).
- **Reported:** standardized β, its 95% CI, and p — **effect size first**
  (Methodology §5).
- **Confirm** (both models): standardized β ≤ −0.2, BH-corrected p ≤ .05, CI
  excludes 0.
- **Null:** β CI includes 0 on **either** model.
- **Robustness:** refit on Judge-A-only and Judge-B-only composites; the sign
  must agree with the pooled fit. Reported, not gated.

---

## 3. Guidance-matching — the de-confound

### 3.1 Quality curve
Compute per-image no-reference quality `Q` = mean of standardized CLIP-IQA and
LAION-aesthetic. Aggregate `Q(model, g)` over prompts × seeds; the peak
`g* = argmax_g Q` is each model's comfort zone. Plot the inverted-U.

### 3.2 Primary de-confound — partial correlation *(always computable)*
Partial Spearman ρ of `composite` vs `guidance`, controlling for `Q`, pooled
across prompts/seeds, per model.

- **Confirm:** partial ρ ≤ −0.2 with bootstrap CI (image-resampled, 10 000
  draws) excluding 0 — the g→breakdown relationship survives removing quality,
  so it is **not** "low g just looks bad."
- **Null:** |partial ρ| < 0.1 — the effect *was* under-conditioning; Exp 01's
  signal does not survive. Reported plainly.

### 3.3 Secondary — matched-quality two-arm contrast
Because `Q` is inverted-U, each sub-peak quality level occurs at both a low-g and
a high-g point. For overlapping-quality strata, compare `composite` at the
**low-g arm** vs the **quality-matched high-g arm** (Cliff's δ, bootstrap CI). If
low-g breakdown were generic quality loss, the oversaturated high-g point at
matched Q would show equal composite; more L2/3 at the low arm ⇒ the effect is
specific to the under-conditioning direction. If the two arms' quality ranges do
not overlap, this is reported as *not computable* and §3.2 stands as the primary
de-confound.

---

## 4. Inter-rater reliability

### 4.1 Full set (Claude vs Qwen)
Per field and for the composite: **quadratic-weighted Cohen's κ** (ordinal 0–3),
**Gwet's AC2**, and **percent agreement**. Tiling: unweighted κ. Because several
fields are 0-heavy (skewed marginals → the "κ paradox" deflates κ), all three are
reported and interpreted together; κ is never read in isolation.

- **Confirm gate:** composite weighted κ ≥ 0.4. Below that, the metric is
  declared unreliable and the confirmatory claim is **withheld**
  (inconclusive-null) regardless of the LMM result.

### 4.2 Human subset (25–30 images)
Stratified across (model × guidance-bin × prompt). Author scores blind, shuffled,
guidance hidden, identical rubric, via `human_rate.py`. Report human-vs-Claude
and human-vs-Qwen weighted κ per field. **Single rater** — flagged as a
limitation in `analysis.md`; no human-human κ is available.

---

## 5. Calibration cross-check (Sonnet 5 vs Sonnet 4.6)

Re-judge a fixed sample of archived Exp 01 images with Sonnet 5 on the
generalized rubric; Spearman the Sonnet-5 field scores against the archived
Sonnet-4.6 numbers. High correlation ⇒ Exp 03's metric sits on the same
instrument as Exp 01's. Reported in `analysis.md`; **not** a confirm/null gate.

---

## 6. Multiplicity

- **Primary family = 2 tests** (composite LMM slope per model). Deliberately
  tiny so the headline claim is not multiplicity-inflated.
- **Secondary family = 8 tests** (4 fields × 2 models, per-field dose-response).
  Benjamini–Hochberg FDR; reported as **q-values with standardized effect
  sizes**, never bare p.
- Per-prompt slopes (§7) are descriptive, not added to a test family.

---

## 7. Descriptive / secondary outputs (not confirm/null gates)

- Per-prompt guidance slope of the composite (6 × 2 table) — feeds the
  "generality" confirm sub-criterion (sign negative in ≥5/6 prompts per model).
- Per-field dose-response curves with FDR q-values (the Exp 01-style table,
  now multi-prompt and two-judge).
- The control test: p6 (forest) should show high `tiling` but low
  `reduplication`/`condensation` dose-response if the effect is object-bound —
  a dissociation that, if present, strengthens the "dissolves objecthood"
  reading; if absent, noted as a caveat.
- Empty-prompt baseline composite per model (the Exp 01 "de-amplify ≠ remove"
  anchor).

---

## 8. Decision table

| Observed | Verdict |
|---|---|
| LMM β ≤ −0.2 (both models, BH p ≤ .05) **and** partial ρ ≤ −0.2 (CI excl. 0) **and** κ ≥ 0.4 **and** sign consistent ≥5/6 prompts | **Confirmed** — hardened, workshop-submittable |
| LMM β CI includes 0 on either model | **Null** — not generalizable / not robust |
| Partial ρ vanishes (\|ρ\| < 0.1) after controlling Q | **Null** — it was under-conditioning |
| Composite κ < 0.4 | **Inconclusive-null** — metric unreliable; report and revisit rubric/judge |

All four outcomes are reportable. A null or inconclusive result is written up as
the finding (Methodology §2.3), and the workshop framing narrows accordingly.
