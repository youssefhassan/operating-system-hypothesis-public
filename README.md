# Operating System Hypothesis — experiments

Pre-registered experiments on what happens to a diffusion model's output when you
withdraw its conditioning, scored against a taxonomy borrowed from hallucination
research.

The method is the same every time:

> **Take a generative system that works. Loosen something. Read the structure of what breaks.**

You don't learn what a system is made of by watching it succeed, because
succeeding is exactly when the machinery is hidden. Neuroscience has run on that
logic for a century under names like lesion studies and TMS. This is a gentler
version, run on image models, using classifier-free guidance as the knob.

Two rules keep it from being a metaphor exercise. **The prediction is committed to
git before the analysed run**, so the goalposts can't move afterwards. And
**resemblance is not mechanism**: nothing here claims a diffusion model and a
brain share machinery.

---

## What's here

### `experiments/01_image_test` — form constants (a clean null)

Pre-registered prediction: lowering guidance would raise a score for Klüver's
Level-1 geometric *form constants* (lattice, cobweb, tunnel, spiral).

**It didn't.** The score was flat and near the floor across the whole dial, on
both a convolution-dominant model (SDXL) and a transformer (SD 3.5). 200 images.

A separate, explicitly post-hoc re-score against Klüver's Level-2/3 *transforms*
did move: objects fragmented, fused and multiplied as guidance fell. That weaker
finding is what Exp 03 was built to attack.

Images and blind judgements: [Hugging Face](https://huggingface.co/datasets/youssefhassan13/exp01-guidance-sweep).

### `experiments/03_l23_hardening` — hardening that signal

860 images. Six prompts spanning object count and objecthood, including a
low-objecthood control designed to dissociate. Seven guidance values, ten seeds,
two architectures, a no-reference image-quality de-confound, two independent
judges from different model families, and a blind human-rated subset.

**The overall pre-registered claim did not confirm.** Confirmation required both
models; SDXL cleared every gate and SD 3.5 missed the slope threshold
(−0.182 against ≤ −0.20).

| | SDXL | SD 3.5 |
|---|---|---|
| Dose-response slope (needed ≤ −0.20) | **−0.340** | −0.182 |
| Correlation with quality controlled | **−0.433** | **−0.246** |
| Prompts with negative slope (needed ≥5/6) | **6/6** | **5/6** |
| Inter-judge agreement (needed ≥0.40) | **0.562** | **0.440** |

Three things the scaffold produced that the gate table doesn't show, written up in
[`analysis.md`](experiments/03_l23_hardening/analysis.md):

**Open-weight VLM judges below roughly 30B fail silently.** Two of three judges
returned well-formed JSON, zero parse errors, and confident captions such as
*"no signs of melting, warping, or impossibility"* for images that were visibly
ghosted. Nothing in the output indicates failure. An 8B and a 32B from the same
family, same release and quantisation, differ completely, which isolates capacity
from model age. [`judge_probe.py`](experiments/03_l23_hardening/judge_probe.py)
is the screening tool that catches it: it asks only whether a candidate judge ever
uses the middle of the scale.

**SDXL is a gradient; SD 3.5 is a cliff.** Post-hoc
([`posthoc.py`](experiments/03_l23_hardening/posthoc.py)): drop the bottom
guidance value and SDXL's effect survives (ρ −0.244, quality-matched δ 0.355)
while SD 3.5's vanishes entirely (ρ +0.048, p = .38). The two models differ in
*shape*, not magnitude, and a linear endpoint cannot express that. This makes the
SD 3.5 evidence weaker, not stronger.

**The composite hid field-level structure.** The forest control looked like a
failure on the composite while having behaved as designed on the two fields it was
built to dissociate. And the distortion field fires at *both* ends of the dial,
absorbing under-conditioned melt and over-conditioned waxiness on one scale.

### Exp 03b — a second scale on the same corpus

[`preregistration_axes.json`](experiments/03_l23_hardening/preregistration_axes.json)
adds continuous axes (veridicality, spontaneity, complexity) whose definitions are
reused verbatim from Exp 01, scored on the same 860 images. The novel endpoint is a
dissociation test: does the global scale track guidance *after* partialling out the
local one, or is it a restatement? Write-up:
[`analysis_axes.md`](experiments/03_l23_hardening/analysis_axes.md).

**Also not confirmed** (both models required). SDXL passes all three endpoints;
SD 3.5 fails the two monotonic ones. Two results matter more than the verdict.

**The scales are not redundant, on SDXL.** Veridicality still tracks guidance at
+0.417 after partialling out the Kluver composite, and the two composites correlate
at only -0.62. What an image is *like* overall, and an inventory of what broke
inside it, are separable measurements.

**Two thirds of the distortion signal was rendering style.** Holding veridicality
fixed drops distortion's association with guidance on SDXL from -0.486 to -0.175.
That lands on Exp 01's headline field: much of "the knob dissolves objecthood" was
the picture becoming painterly rather than the objects coming apart. Distortion
survives at a third of its apparent size. The only way to find this was to run the
second scale against the first.

SD 3.5's failure is uninformative rather than negative. Its veridicality is an
inverted U, peaking at g=7 and collapsing below its starting value by g=15, and
both endpoints are monotonic tests. **That is the second time an assumed response
shape has cost an endpoint here**; the first was a linear slope that could not
distinguish a gradient from a cliff.

---

## Reproducing

```bash
pip install -r requirements.txt
cd experiments/03_l23_hardening
python sweep_local.py --model sdxl --prompts all --unconditional --skip-existing
python judge.py     --dir results-local/sdxl        # frontier judge, Batches API
python judge_mlx.py --name qwen --dir results-local/sdxl \
       --model mlx-community/Qwen3-VL-32B-Instruct-4bit   # local, MLX, Apple silicon
python quality.py   --dir results-local/sdxl
python analyze.py --both --judges claude,qwen --plot
```

Generated images are gitignored (860 PNGs). Analysis figures, blind judgements,
statistics reports and probe outputs are committed.

`analyze.py` computes the linear mixed model, quality-controlled partial
correlations, matched-quality arms, weighted κ with Gwet's AC2, and
Benjamini-Hochberg correction. `statlib.py` implements those in numpy with no
heavy dependencies.

---

## On the pre-registration claim

Worth being precise, since it's the load-bearing part.

**Exp 03's ordering is demonstrable in this history**: `preregistration.json`
committed 2026-07-15, results committed 2026-08-07. Check with
`git log -- experiments/03_l23_hardening/preregistration.json`.

**Exp 01's is not, from this repo alone.** Its results were gitignored and
published to Hugging Face rather than committed, so the repo shows the
pre-registration and the write-up landing together on 2026-06-19. The
pre-registration is timestamped; the ordering against the run is not
independently verifiable here. Stated rather than glossed.

---

## What isn't here

This is a filtered export of a private working repo. Reading notes, the research
plan, unrun experiment designs, the writing drafts and the day-by-day working logs
are not included: the logs in particular interleave experiment notes with
unpublished ideas. Papers are cited in each pre-registration rather than
redistributed.

## Related

Write-ups, with the nulls given the same prominence as the results:
[Operating System Hypothesis](https://youssefhassan13.substack.com/p/operating-system-hypothesis-learning).

## Licence

Code is MIT. Prose, figures and pre-registration documents are CC BY 4.0. See
[`LICENSE`](LICENSE).
