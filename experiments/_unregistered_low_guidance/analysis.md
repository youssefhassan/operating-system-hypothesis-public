# Analysis — the sub-CFG guidance regime (g = 0 → 2)

**Exploratory. Not pre-registered. Not a result.** No hypothesis was committed before
these runs, there is no pre-specified statistical model, and no p-values are reported
anywhere in this directory — quoting one would dress a look around up as a test. What
follows is a description of what two judges saw. Read it as a reason to design a proper
experiment, not as evidence for a claim.

Numbers below are reproduced by `python analyze_low_g.py` (writes `report.json`).

## 1. What was run

| | |
|---|---|
| Models | SDXL base 1.0, SD 3.5 medium |
| Prompts | the 6 from `../03_l23_hardening/preregistration.json`, verbatim |
| Guidance | 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0 |
| Seeds | 42, 43 |
| Fixed | 25 steps, 1024x1024 |
| Images | 110 per model (108 conditioned + 2 empty-prompt), 220 total |
| Judges | Qwen3-VL-32B-Instruct-4bit (local MLX) and Claude Sonnet 5 (Batch API) |
| Rubric | Exp 03's `rubric.py`, unchanged, blind and shuffled as in Exp 03 |

Exp 01 and Exp 03 both swept guidance from 1.0 upward, because that is where diffusers
turns CFG on. This run asks what the region *below* that boundary looks like.

## 2. The library behaviour this run had to work around

Both pipelines gate CFG on `guidance_scale > 1`:

- `StableDiffusionXLPipeline.do_classifier_free_guidance` → `self._guidance_scale > 1 and ...`
- `StableDiffusion3Pipeline.do_classifier_free_guidance` → `self._guidance_scale > 1`

A naive sweep of 0 → 2 therefore returns **five byte-identical images** (g = 0, 0.25, 0.5,
0.75, 1.0 all take the CFG-off branch) and only four distinct points above 1. The apparent
flat floor would be an artifact of the library, not a property of the model.

`sweep_low_g.py` forces the CFG branch on at every g, so the ordinary combination

    pred = pred_uncond + g * (pred_cond - pred_uncond)

runs at all nine values. g = 0 is then the pure negative-branch render.

**Validity check.** If the patch had not taken, g = 0 would still show the prompt.
`check_g0.py` asserts that all six prompts collapse to one identical image per seed at
g = 0, and exits non-zero otherwise. It passes on all 220 images:

| model | seed 42 | seed 43 |
|---|---|---|
| SDXL | `363ee66758904128` | `219d17040d93dbd4` |
| SD 3.5 | `2e80eaa706d584d3` | `f274b3db1e430c71` |

**Two different unconditional anchors.** On SDXL the g = 0 image is *not* the empty-prompt
baseline; on SD 3.5 it is. SDXL's `model_index.json` sets `force_zeros_for_empty_prompt:
true`, so its negative branch is a literal zero vector, while `uncond_s{seed}.png` passes
`prompt=""` and gets the *encoding* of the empty string. Exp 03's SD 3.5 embed builder
passes an explicit `negative_prompt=""`, so there the two coincide. "Unconditional" is not
one thing across architectures.

## 3. The direction, which both judges agree on

Every Klüver field declines as guidance rises, on both models, under both judges.
**All 20 field x model x judge correlations are negative.**

Descriptive Spearman rho over g = 0..2 (not comparable to Exp 03's rho, which was fitted
over g = 1..15 with 10 seeds and a pre-registered model):

| field | SDXL Qwen | SDXL Claude | SD 3.5 Qwen | SD 3.5 Claude |
|---|---|---|---|---|
| reduplication | −0.398 | −0.462 | −0.430 | −0.211 |
| fragmentation | −0.555 | −0.459 | −0.539 | −0.520 |
| condensation | −0.710 | −0.570 | −0.596 | −0.521 |
| distortion | −0.658 | −0.521 | −0.616 | −0.592 |
| tiling | −0.493 | −0.280 | −0.514 | −0.330 |

All 60 per-prompt correlations under Qwen are negative (−0.93 to −0.19); no prompt
reverses the direction.

There is **no U-shape**. An earlier version of the shape detector reported one, but it
was flagging cases where the final point sat a hundredth above the minimum. With a
quarter-rubric-point threshold, every field on both models is monotone: the largest
rebound anywhere is +0.08.

Sign agrees with Exp 03's negative correlation, at roughly double the magnitude over this
window. The reading that fits both: the structure is a property of the prior, which
guidance suppresses — and Exp 03's grid started at g = 1.0, where it has largely decayed.

## 4. A judge-free companion measurement

`divergence.py` measures mean absolute pixel distance from each seed's g = 0 render,
which needs no rubric and no judge.

| g | SDXL | SD 3.5 |
|---|---|---|
| 0.25 | 6.77 | 11.63 |
| 0.50 | 11.21 | 17.37 |
| 1.00 | 18.42 | 28.64 |
| 1.50 | 23.99 | 38.12 |
| 2.00 | 28.26 | 44.63 |

The movement is smooth and *decelerating* (SDXL increments +6.8, +4.4, +3.8, +3.4, +3.0,
+2.6, +2.2, +2.1). There is no threshold. Scenes appear to "snap" into legibility by eye
somewhere around g = 1, but nothing discontinuous happens in pixel space — the semantic
transition sits on a smooth trajectory. SD 3.5 travels ~35% further at matched g.

Per-prompt, the images that travel least (`oranges` 23.1, `forest` 22.7) are exactly the
ones still rendering as repeating fields at g = 2 in the contact sheets, while `bicycle`
travels furthest (38.5) and is coherent by g = 1.25. Prompt-dependence, consistent with
Exp 03 §5.1.

## 5. Where the judges disagree, and why it matters more than the agreement

Composite quadratic-weighted kappa over the four ordinal fields:

- SDXL **+0.710** [+0.658, +0.750]
- SD 3.5 **+0.661** [+0.581, +0.725]

Both judges come from different families (open-weight Qwen vs Claude), so this is the same
circularity break Exp 03 used, and the agreement is about the images rather than about one
lineage's blind spots.

But the headline number averages two regimes that behave differently, in **opposite
directions on the two models**:

| | g < 1 | g >= 1 |
|---|---|---|
| SDXL | +0.36 … +0.64 | +0.48 … +0.86 |
| SD 3.5 | +0.47 … +0.82 | +0.21 … +0.40 |

Both are range restriction, mirrored:

- **SDXL's low-g cells are ceiling-pinned, and only for one judge.** Qwen scored
  3-on-all-four-ordinal-fields for **26 of 48** low-g images; Claude did so for **0 of 48**.
  The flat 3.000 that Qwen reports at g = 0 and g = 0.25 is a property of that judge, not
  of the images. This corrects the natural reading of the per-guidance tables.
- **SD 3.5's high-g cells are floor-pinned for both judges.** Scores sit near zero, so
  there is almost no variance left to agree about; `tiling` at g >= 1 is undefined
  outright, because both judges scored every image 0.

Agreement is strongest exactly where the scores move. That is the honest summary, and it
is why `analyze_low_g.py` prints the regime split and the per-judge ceiling counts rather
than leaving the composite to speak for itself.

## 6. Limits

- **Exploratory and unregistered.** No pre-specified hypothesis, model, or correction.
- **Two seeds.** Too thin for anything about the character of the prior — and the two
  seeds disagree sharply. At seed 42 SDXL's prior is a dense floral tiling and SD 3.5's is
  a readable scene; at seed 43 that reverses. No architecture-level claim about what the
  prior "looks like" is supportable here.
- **n = 2 at g = 0.** All six prompts render one image per seed, so that column holds two
  distinct images, not twelve. `analyze_low_g.py` deduplicates by SHA before computing
  means, CIs, and the pooled rho; counting them as six would shrink the interval by ~sqrt(6).
- **Rubric ceiling.** The scale stops at 3 and one judge reaches it often at low g.
- **Not poolable with Exp 03.** Its g = 1.0 point was CFG-*off*; this run's is CFG-on.

## 7. What a pre-registered follow-up would need

1. More seeds — 10, as Exp 03 used — before any claim about the prior.
2. A rubric that extends past 3, or an explicit saturation policy, so the low-g region is
   measurable rather than censored.
3. Both judges retained; the disagreement structure here is the most informative output of
   the run and would be lost with a single rater.
4. A pre-specified model and hypothesis, committed before generation.
