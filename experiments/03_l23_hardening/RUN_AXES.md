# Exp 03b — running the axes judges

Pre-registration: [`preregistration_axes.json`](preregistration_axes.json).
Rubric: [`rubric_axes.py`](rubric_axes.py) (field definitions verbatim from Exp 01).

**Commit the pre-registration before running any of this.** It is a
git-history pre-registration and it is worthless committed afterwards.

```bash
cd experiments/03_l23_hardening
source ../../.venv/bin/activate
export HF_HUB_DISABLE_XET=1
set -a && . ../../.env && set +a          # HF_TOKEN, ANTHROPIC_API_KEY
```

---

## Step 1 — the screening gate (do this first, ~5 min)

Mandatory per `preregistration_axes.json` → `judges.screening_gate`. Clearing the
Klüver screen does **not** clear a judge on the axes rubric.

```bash
python judge_probe.py --model mlx-community/Qwen3-VL-32B-Instruct-4bit --rubric axes
```

**Pass condition:** `gradedness > 0` and `flat_fields` empty.

`mean_spearman_vs_claude` will print `null`, which is expected and fine: Claude
has not scored this scale yet, so there is no reference. Gradedness is the gate.

**If it fails** (any flat field), stop. Do not run the full pass. Options in
order: raise `--max-tokens`, then probe `Qwen2.5-VL-32B-Instruct-4bit`, then
`Mistral-Small-3.2-24B-Instruct-2506-4bit`. Record whichever you try; a judge
that is graded on one rubric and flat on another is itself a reportable result
and sharpens the Exp 03 silent-failure finding.

Compare across every probe ever run, both scales:

```bash
python judge_probe.py --compare
```

---

## Step 2 — full local pass (~2.3 h for both models, $0)

Only after Step 1 passes.

```bash
for m in sdxl sd35; do
  python judge_mlx.py --name qwen --rubric axes \
    --model mlx-community/Qwen3-VL-32B-Instruct-4bit \
    --dir results-local/$m
done
```

Writes `judgements_qwen_axes.json` + `_raw.json` per model. The Klüver files are
untouched: the `_axes` suffix is applied for every non-default rubric.

Resumable and saves after every image, so a stall costs one call. **If you ever
re-point this at an existing `judgements_*_axes.json` with a different model, you
must pass `--overwrite`** or resume-by-filename will score zero new images. That
cost a full pass on 2026-08-08.

---

## Step 3 — Claude pass (~$3.50, Batches API)

Can run in parallel with Step 2; different device, different cost line.

```bash
for m in sdxl sd35; do
  EXP03_CLAUDE_MODEL=claude-sonnet-5 python judge.py --rubric axes --dir results-local/$m
done
```

Writes `judgements_claude_axes.json`. Batch state is cached separately in
`.batch_claude_axes.json`, so an interrupted poll resumes without re-submitting
and without touching the Klüver batch state.

---

## Step 4 — analysis (not yet written)

`analyze.py` currently knows only the Klüver scale. The axes analysis needs:

1. the axes fields registered alongside the Klüver ones;
2. **P1**: LMM standardized slope of veridicality on `guidance_std`, per model;
3. **P2**: partial Spearman of veridicality against guidance **controlling for
   the Klüver composite** — this is the dissociation test and the novel endpoint.
   `statlib.partial_spearman` and `partial_spearman_ci` already exist and are
   exactly the right tools;
4. **P3**: inter-judge weighted κ on the three graded axes;
5. spontaneity analyses must **exclude the uncond arm** (undefined there, scored
   3 by construction).

---

## What each outcome means

| Result | Reading |
|---|---|
| P1 and P2 both clear on both models | The two scales are complementary and measure different things. This is the claim Substack post #6 makes; it would be earned rather than asserted. |
| P1 clears, P2 fails | Veridicality tracks guidance but adds nothing over the Klüver composite. The scales are redundant and post #6's central argument is **wrong**. Report it that way. |
| P1 fails | Exp 01's exploratory direction does not survive a harder corpus, which is a clean replication failure and worth reporting on its own. |
| E1 (overshoot) confirmed | Veridicality falls at both ends of the dial, which would explain the SD 3.5 portrait sign flip and the distortion bidirectionality as one phenomenon rather than two anomalies. Exploratory, flagged as such. |
