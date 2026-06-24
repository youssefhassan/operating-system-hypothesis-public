# Experiment 01 — Image Test (guidance / CFG gradient)

**Status:** coarse SDXL sweep in progress (local, MPS float32); confirmatory
analysis pending blind judge + SD 3.5 replication.

## Goal (deny-friendly survey)

This experiment is **not** trying to prove the Operating System / brain↔AI
bridge. The preferred outcome is an **honest null** if the data supports it.

The job is a **survey across models and substrates**: when you relax the top
prior (guidance; later temperature / quantization), what **base vocabulary**
surfaces — hexagons, pixels, watermelon tiles, checkerboard, whatever — and
does any of it **relate to** the geometric families people report in altered
brain states (lattice/hexagon, spiral, tunnel, cobweb)? Or is it always just
that model's own ingredients, unrelated to V1 form constants?

| Outcome | Verdict |
|---|---|
| Low-g = mush / no structure | Deny for diffusion |
| Structure, but only model-specific junk (no cross-architecture rhyme) | Deny the universal bridge |
| Structuredness on conv UNet (SDXL), absent on transformer (SD 3.5) | Narrower "local coupling" thread survives; content match to brains denied |
| Structured low-g + weak form-constant-like scores on ≥2 architectures | Bridge stays testable (not proven) |

See also § "What we do and don't expect" below and the project notes

## Hypothesis

If predictive suppression in a vision model behaves like top-down priors in the
brain, then weakening the model's confident "application-layer" inferences
(object identity, scene gist) should let lower-layer representational primitives
(textures, contours, repeating lattice structure) surface, analogous to the
geometric perceptual content reported under sub-anaesthetic ketamine and the
other altered states Klüver catalogued.

## What we do and don't expect (and why)

A diffusion model is **not** a brain, and the strong reading of this experiment
("AI hallucinates Klüver's shapes") is probably wrong. It's worth saying that
loudly before any data lands, so a likely null isn't dressed up as a surprise.

**Why the naive expectation is *no* form constants.** Form constants in humans
are not a generic "altered state" by-product, they are a fingerprint of one
specific piece of hardware. The Bressloff/Cowan/Ermentrout (2001) account
derives lattice/cobweb/tunnel/spiral from the **symmetry and lateral
connectivity of primary visual cortex (V1)**: the retina→V1 coordinate map and
the orientation-column layout, destabilized. A diffusion model has none of that:
no retinotopic map, no cortical columns, no serotonin. Its unconditional "base
layer" is a draw from its **training distribution of natural images and art**,
so the honest prediction for an empty prompt is an arbitrary, vaguely coherent
scene, not geometry. The pre-registered **null is a live, maybe even likely,
outcome**, and a clean null ("the analogy fails for diffusion") is a real result.

**The weaker claim actually worth testing.** Not "AI is secretly human" but:
*do locally-coupled generative systems, when their prior is relaxed, drift toward
periodic/geometric structure for reasons that rhyme with the brain's math rather
than copy its biology?* V1 form constants are mathematically a **symmetry-breaking
/ pattern-formation** phenomenon (Turing-like instability in a field with local,
translation-invariant coupling). A Stable Diffusion **UNet is convolutional**,
i.e. translation-equivariant with local coupling, the same structural ingredient.
So the interesting hypothesis, if any structure appears, is *convergent pattern
formation from shared symmetry*, not imitation. Two consequences:

- Repeating texture (tiles, brick, foliage, mesh) is statistically cheap and
  common in natural images; low-guidance sampling can collapse toward such
  easy, high-probability modes. Mild structure is plausible even with no brain
  analogy at all, so it must be measured against the unconditional baseline.
- **SDXL (convolutional UNet) vs SD 3.5 (MMDiT transformer, far less
  convolutional)** is itself a test: a geometric effect strong on SDXL and
  absent on SD 3.5 would point at *convolutional locality as mechanism*, a more
  interesting and more falsifiable story than "weird images appear."

**What this means for the metric.** Scoring specifically for the four V1 shapes
is over-specified, a non-V1 system has no reason to honor that taxonomy. So the
load-bearing signal is the generic **`geometric_intensity` / structuredness vs
formless-noise gradient** (this *is* what metric M is built on), with the four
named classes kept only as a secondary "and when there's structure, does it
happen to look form-constant-like?" lens. The design is already hedged this way.

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
the prediction is a shift toward **more structure** (textures, contours,
repetition) rather than a smooth fade into uniform noise, with the four V1
shapes a bonus question, not the expectation (see "What we do and don't expect").
A null result is uniform noise with no intermediate structured regime, or rises
indistinguishable from the unconditional baseline, in which case the analogy
fails for diffusion specifically and the writing pivots to "here is where it
breaks." Both directions are reportable.

## Running it

```bash
pip install -r ../../requirements.txt          # includes the diffusers stack
# gated models (sd35, flux): set HF_TOKEN in the project-root .env (read automatically)
#   + accept the model license once on the Hub. (huggingface-cli login also works.)

python sweep_local.py --model sdxl --unconditional
python sweep_local.py --model sd35 --unconditional      # architecture contrast
```

**MPS stability.** On Apple Silicon, `--dtype auto` uses **float32 for SDXL**
(not bf16): bf16/fp16 works at guidance ≤ 1 but produces **black images at
guidance > 1** once classifier-free guidance activates (two UNet passes).
**SD 3.5 on MPS** uses **float16** automatically (~10 GB vs ~30 GB float32).
The script tries a **full fp16 GPU load** first (DiT stays on Metal; much
faster than shuffling all components via `enable_model_cpu_offload`). If that
OOMs, it falls back to CPU offload. It also **caches prompt embeddings** once
per run (T5-XXL encoding is expensive and your prompt is fixed across the grid).
Attention slicing / VAE tiling are **on by default on MPS** for ≤32 GB machines.
On **48 GB+ unified memory** (e.g. M5 64 GB laptop) the script auto-detects
headroom and skips slicing/tiling and encoder parking for SD 3.5 — use that
machine as the primary sweep host. CUDA `--dtype auto` stays float16.

**Which Mac to use.** Run SD 3.5 on the **fastest / most RAM** box (M5 64 GB
laptop >> M4 24 GB mini). Copy `results-local/sd35/` from the mini if you
already started there, then resume with `--skip-existing`. No sharding needed
on 64 GB unless you want parallel speed across multiple machines.

**Parallelism.** One MPS GPU can't usefully run parallel jobs, but the sweep is
embarrassingly parallel across machines. Split it over the three Mac minis:

```bash
python sweep_local.py --model sdxl --unconditional --num-shards 3 --shard 0   # Mac 1
python sweep_local.py --model sdxl --unconditional --num-shards 3 --shard 1   # Mac 2
python sweep_local.py --model sdxl --unconditional --num-shards 3 --shard 2   # Mac 3
```

Each machine generates a deterministic, disjoint third of the jobs; merge the
three `results-local/sdxl/` folders on the primary, then run `judge.py` +
`loop.py` there. `--skip-existing` makes any run safely resumable.

Outputs go to `results-local/<model>/` (git-ignored) as `g{guidance}_s{seed}.png`,
`uncond_s{seed}.png`, plus `metadata.json`.

Runs on CUDA (Vast.ai), Apple Silicon MPS (the Mac minis), or CPU; device and
dtype auto-detect, override with `--device` / `--dtype`.

## Publish to Hugging Face

Raw PNGs (~400 MB) stay out of git. After sweep + judge + analyze, publish the
full artifact bundle as a public dataset on the Hub:

```bash
pip install huggingface_hub   # or pip install -r ../../requirements.txt
python publish_hf.py --dry-run
python publish_hf.py --repo-id youssefhassan/exp01-guidance-sweep
```

Uses `HF_TOKEN` from the project-root `.env` (write token with dataset create
permission). Upload includes both models' PNGs, `judgements.json`,
`analysis_report.json`, figures, dose GIFs, `preregistration.json`, and a
generated dataset card README.

Add the dataset URL to the Substack post and `analysis.md` once live.

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
  in shuffled order. Default judge: **Claude Sonnet 4.6** (`claude-sonnet-4-6`).
  Un-blinding (filename → guidance) happens only at aggregation.
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

Judge model is pinned via `EXP01_CLAUDE_MODEL` (env); default: `claude-sonnet-4-6`.
Set `preregistration.json` `"judges": ["claude", "gpt"]` and pass `--judges both`
to restore dual-judge mode. Changing judge models does not change `rubric_version`
— that labels the fixed scoring text.

## Files

- `sweep_local.py` — **generation.** Multi-model diffusers CFG sweep.
- `judge.py` — **measurement.** Blind dual-VLM form-constant scorer (the form-constant judge).
- `analyze.py` — **stats + figures.** Confirmatory M(g), exploratory rubric fields, contact sheets.
- `run_remaining.sh` — **batch finish:** SDXL N=10 → SD 3.5 sweep → judge → analyze.
- `loop.py` — **orchestration.** Methodology-compliant agentic loop.
- `preregistration.json` — confirmatory pre-registration (committed first).
- `run.py` — original BFL FLUX.2-flex API sweep; quick API baseline only
  (guidance floor 1.5, can't reach the low / base-layer regime).
- `serve.py` — FastAPI wrapper around `run.py` for the live site.
- `results-local/` — local sweep outputs, judgements, audit log (git-ignored).
- `publish_hf.py` — upload `results-local/` to Hugging Face Hub as a public dataset.
- `analysis.md` — written up after the sweeps complete.
