# Unregistered scratch: the sub-CFG guidance regime (g = 0 → 2)

**Status: exploratory, NOT pre-registered.** No hypothesis was committed before running,
no ID was assigned under the project's sequential numbering rule, and nothing here may
be reported as a confirmatory result. It is a look, not a test. If it shows something, the
next step is to pre-register it properly as a numbered experiment — not to write it up.

## The question

Exp 01 and Exp 03 both swept guidance from 1.0 upward, because that is where diffusers
turns CFG on. That leaves the whole *sub-conditional* regime unlooked-at. This run walks
g from **0 to 2 in steps of 0.25** (9 points) and asks whether the Klüver L2/L3 phenomena
appear as the image falls apart *downward*, toward the prior, rather than upward toward
over-guidance.

At these values the three landmarks are:

| g | what the model is doing |
|---|---|
| 0.00 | pure unconditional prediction — the prompt is arithmetically cancelled |
| 1.00 | the plain conditional prediction, no CFG amplification |
| 2.00 | mild CFG, the bottom of Exp 03's grid |

## The library gotcha this run exists to work around

Both pipelines gate CFG on `guidance_scale > 1`:

- `StableDiffusionXLPipeline.do_classifier_free_guidance` → `self._guidance_scale > 1 and ...`
- `StableDiffusion3Pipeline.do_classifier_free_guidance` → `self._guidance_scale > 1`

So a naive sweep of 0 → 2 by 0.25 returns **five byte-identical images** (g = 0, 0.25, 0.5,
0.75, 1.0 all take the CFG-off branch and produce the conditional-only image) and only four
distinct points above 1. The apparent "flat floor" would be an artifact of diffusers, not a
property of the model.

`sweep_low_g.py` forces `do_classifier_free_guidance = True` for every g, so the standard
combination runs unmodified at all nine values:

    pred = pred_uncond + g * (pred_cond - pred_uncond)

which is well defined below 1 and gives the pure prior at g = 0.

**Consequence:** these g values are *not* on the same footing as Exp 01/03's grid, whose
g = 1.0 point was CFG-off. Do not pool them into an Exp 03 regression.

## Built-in validity check

With CFG forced on, g = 0 cancels the prompt entirely, so **every prompt's g = 0 image
must be byte-identical to every other prompt's g = 0 image at the same seed**.
`check_g0.py` asserts this and exits non-zero if it fails. Verified on a 2-prompt smoke
run before the full sweep: `p1` and `p3` both rendered `dae5ef0b891e7de4` at g = 0 while
their g = 1 images differed.

### Two different "unconditional" anchors

`uncond_s{seed}.png` (the empty-prompt baseline, inherited from Exp 03) is **not** the
same image as g = 0 on SDXL, and the check does not require it to be:

- SDXL's `model_index.json` sets `force_zeros_for_empty_prompt: true`. With no negative
  prompt passed, the CFG negative branch is a literal **zero vector**, so the g = 0 image
  is the zero-embedding render.
- `uncond_s{seed}.png` passes `prompt=""`, which is the *encoding of the empty string* —
  a different vector, and a different image.
- SD 3.5 goes the other way: Exp 03's `_build_sd3_embed` passes an explicit
  `negative_prompt=""`, so its negative branch really is encode(""), and its g = 0 image
  should match its `uncond` baseline.

That asymmetry is worth keeping rather than papering over: "zero embedding" and "empty
string" are two distinct notions of the prior, and this run gets both for free.

## Design

- **Models:** SDXL base 1.0, SD 3.5 medium (same two as Exp 03)
- **Prompts:** the 6 from `../03_l23_hardening/preregistration.json` (reused verbatim, so the
  rubric's per-prompt "intended content" clause still applies)
- **Guidance:** 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
- **Seeds:** 42, 43 (two only — this is a cheap look)
- **Fixed:** 25 steps, 1024x1024, as Exp 03
- **Count:** 6 x 9 x 2 = 108 conditioned + 2 uncond = 110 per model, 220 total
- **Judge:** Qwen3-VL-32B-Instruct-4bit (MLX, local, $0), Exp 03's `rubric.py` unchanged,
  blind and shuffled exactly as Exp 03. **One judge only** — no cross-judge kappa, which
  is on its own sufficient reason this cannot be confirmatory. A Qwen3-VL-30B-A3B second
  rater was attempted and dropped: it was not in the local cache and its download stalled
  at 10 GB of ~17 GB. Two other cached Qwen models are unusable as they stand —
  Qwen2.5-VL-32B is a broken snapshot (3.9 GB of ~18 GB, 0 shards, 8 `.incomplete` files,
  which is the likely cause of the Exp 03 note about the 32B re-judge stalling at 70/430)
  and Qwen3-VL-8B has `…of-00002` weight files against an index expecting `…of-00004`.
  Adding a second rater is the first thing a pre-registered follow-up should fix.

## Run

    ./run.sh                       # everything, resumable
    # or step by step:
    python sweep_low_g.py --model sdxl      # SDXL auto-uses Exp 03's sdxl_local/ fp16 snapshot
    python sweep_low_g.py --model sd35      # --unconditional and --skip-existing are on by default
    python check_g0.py
    EXP03_QWEN_MODEL=mlx-community/Qwen3-VL-32B-Instruct-4bit \
      python ../03_l23_hardening/judge_qwen.py --dir results-local/sdxl
    python analyze_low_g.py
