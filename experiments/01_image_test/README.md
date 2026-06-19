# Experiment 01 — Image Test (guidance / CFG gradient)

**Status:** scaffold ready, not yet run.

## Hypothesis

If predictive suppression in a vision model behaves like top-down priors in the
brain, then weakening the model's confident "application-layer" inferences
(object identity, scene gist) should let lower-layer representational primitives
(textures, contours, repeating lattice structure) surface, analogous to the
geometric perceptual content reported under sub-anaesthetic ketamine and the
other altered states Klüver catalogued.

## The knob and its direction

The independent variable is **guidance / classifier-free guidance (CFG)**, the
"how strongly do I commit to the conditioning prior" dial. The CFG update is:

```
eps = eps_uncond + g * (eps_cond - eps_uncond)
```

- **g = 1** → pure conditional: the prompt, with no amplification.
- **g → 0** → pure unconditional: the model's own prior, the "base layer."
- **g > 1** → amplified prompt adherence (and, very high, oversaturation/artifacts).

So the base-layer regime we care about is at the **low** end. This corrects the
earlier draft of this experiment, which expected the geometric regime at high
guidance.

**Base-layer caveat (important).** Standard `diffusers` pipelines disable the
unconditional pass when `guidance_scale <= 1`, so you cannot reach `0 < g < 1`
by lowering the dial alone, at `g = 1` you simply get the conditional output. To
anchor the true base layer we therefore also generate an **unconditional
(empty-prompt) baseline** and ask two things: (1) does low-guidance conditioned
output drift *toward* that unconditional output, and (2) does the unconditional
output itself carry structured geometry (form constants) rather than
featureless noise? A custom sub-1 guidance hook that interpolates toward the
unconditional prediction is a noted follow-up, not in this scaffold.

## Why multiple models (and which ones)

Cross-model replication is the single biggest credibility upgrade: a result that
appears across architectures is a property of diffusion guidance, not a quirk of
one model. But you can only sweep models that **expose the guidance knob**.

| Model (`--model`) | Architecture | CFG type | Role |
|---|---|---|---|
| `sdxl` | UNet, latent | true CFG, wide range | **Core** replication |
| `sd35` | MMDiT | true CFG | Different architecture, same knob |
| `flux` | DiT | guidance-**distilled** | Contrast; distillation is a confound to flag |
| `sd15` | older UNet | true CFG | Cheap cross-generation check |

**Deliberately excluded:** Midjourney and Gemini / "Nano Banana" are
prompt-only closed products that do not expose CFG, so they cannot be part of
the controlled sweep (turning a different dial and calling it the same
experiment would not be defensible). At most they could appear later as a
clearly-labelled *qualitative* sidebar (e.g. Midjourney `--chaos`), never as
swept data.

## Method

- **Fixed across the sweep:** prompt (mixed materials/colors so any
  decomposition is visible), steps, size, seed set.
- **Variable:** `guidance ∈ {1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0, 11.0, 15.0}`,
  each at `seed ∈ {42, 43, 44}` so one weird sample doesn't drive the read.
- **Plus:** the unconditional (empty-prompt) baseline per seed.
- **Judgement (v1):** by eye, sorted by guidance ascending, looking for a
  *structured intermediate regime* (textures, contours, lattice / repeating
  geometry) rather than a smooth blur from coherent scene to uniform noise.
  Later: a vision-LM judge scoring the four Klüver classes + Suzuki et al.
  (2024) veridicality / spontaneity / complexity, calibrated on the Bertolero
  (2026) Dreamachine corpus (this becomes the form-constant experiment's analysis layer).

## Expected outcome

As guidance drops toward 1 (and compared against the unconditional baseline),
content should not simply become uniformly noisier, the prediction is a
structured shift toward repeating geometric / lattice / texture-dominant content
before full decoherence. A null result is uniform noise with no intermediate
geometric regime, in which case the analogy fails for diffusion specifically and
the writing pivots to "here is where it breaks."

## Running it

```bash
pip install -r ../../requirements.txt          # includes the diffusers stack
# gated models (sd35, flux) need: huggingface-cli login  + accept the license

python sweep_local.py --model sdxl --unconditional
python sweep_local.py --model sd35 --unconditional      # architecture contrast
```

Outputs go to `results-local/<model>/` (git-ignored) as `g{guidance}_s{seed}.png`,
`uncond_s{seed}.png`, plus `metadata.json`.

Runs on CUDA (Vast.ai), Apple Silicon MPS (the Mac minis), or CPU; device and
dtype auto-detect, override with `--device` / `--dtype`.

## Pre-registration (confirmatory)

The machine-readable pre-registration is in [`preregistration.json`](preregistration.json),
committed *before* any analyzed run (methodology §4 — git-history pre-registration).
Human-readable twin:

- **Hypothesis.** Lowering guidance `g` over `[1.0, 15.0]` (prompt/seed/steps
  fixed) on SDXL, replicated on SD 3.5, raises the blind form-constant score
  `M`, because weakening top-down conditioning lets lower-layer geometry surface
  (diffusion analogue of REBUS). Low-`g` outputs should also move toward the
  unconditional baseline.
- **Metric M.** Per image: mean `geometric_intensity` (0–3) across the two
  judges, ÷3 → [0,1]. Per guidance: `M(g)` = mean over seeds.
- **Confirm.** Spearman `rho(M,g) ≤ −0.6` (p ≤ .05), low-vs-high Cliff's delta
  `≥ 0.33` with bootstrap CI excluding 0, replicated on ≥2 architectures.
- **Null.** `|rho| < 0.3` or the effect CI includes 0 at full N. A null is a
  first-class, publishable outcome.
- **N.** 10 seeds per guidance value (methodology §5 default).

Changing the prompt, metric, rubric, thresholds, or grid bounds after seeing
data reclassifies the run as **exploratory** and requires a separate dated
commit. The loop cannot make those changes.

## The agentic loop

`loop.py` closes the cycle **generate → blind-judge → aggregate → decide → stop**,
under the pre-registration. Its objective is to *characterize the effect and
stop on pre-registered rules*, not to iterate until a finding appears.

- **Judge (`judge.py`)** is blind: it sees only pixels + a fixed rubric, scores
  in shuffled order, with **two** vision models (Claude + GPT) for cross-judge
  agreement. Un-blinding (filename → guidance) happens only at aggregation.
- **Controller** may only take *measurement/coverage* actions: `add_seeds`
  (shrink variance), `refine_grid` (insert a guidance value at a detected
  transition, within bounds), `replicate_model` (next architecture), or `stop`.
  It cannot touch the prompt/metric/rubric/thresholds.
- **Stop** on confirm / null / budget. Every iteration is appended to
  `results-local/iterations.jsonl` for audit.

```bash
python loop.py --dry-run                 # print the next planned step, no runs
python loop.py --init-seeds 3 --max-iters 12   # full loop (generates + judges)
```

Judge models are pinned via `EXP01_CLAUDE_MODEL` / `EXP01_OPENAI_MODEL` (env);
pin exact dated snapshots for confirmatory runs.

## Files

- `sweep_local.py` — **generation.** Multi-model diffusers CFG sweep.
- `judge.py` — **measurement.** Blind dual-VLM form-constant scorer (the form-constant judge).
- `loop.py` — **orchestration.** Methodology-compliant agentic loop.
- `preregistration.json` — confirmatory pre-registration (committed first).
- `run.py` — original BFL FLUX.2-flex API sweep; quick API baseline only
  (guidance floor 1.5, can't reach the low / base-layer regime).
- `serve.py` — FastAPI wrapper around `run.py` for the live site.
- `results-local/` — local sweep outputs, judgements, audit log (git-ignored).
- `analysis.md` — written up after the sweeps complete.
