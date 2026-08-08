"""
Experiment 03b — the Suzuki three-axis rubric, as a drop-in sibling of rubric.py.

Exposes the same interface as `rubric.py` (INT_FIELDS / BIN_FIELDS / ALL_FIELDS,
load_prereg, rubric_version, build_rubric, prompt_id_of, parse_filename, coerce,
extract_json) so `judge.py`, `judge_mlx.py` and `judge_probe.py` can score either
scale by swapping the module, with no change to the harness.

WHY THIS IS A REPLICATION, NOT A NEW INSTRUMENT
-----------------------------------------------
The field definitions below are lifted **verbatim** from Exp 01's
`01_image_test/judge.py`, which already scored these three axes (credited there
to Suzuki, Schwartzman & Seth 2024) on the Exp 01 corpus and found, on SDXL:
veridicality rho +0.67, complexity -0.54, spontaneity -0.40 against guidance.
Those are the directional predictions `preregistration_axes.json` commits to.
Re-wording the definitions would forfeit that continuity, so do not touch them:
they sit under `forbidden_to_automate` in the axes pre-registration exactly as
the Klüver text does in the original.

THE UNCONDITIONAL ARM
---------------------
Spontaneity is "content the prompt didn't ask for", which is undefined when there
was no prompt. The empty-prompt variant tells the judge to return 3 (with no
instruction, all content is unrequested) and the pre-registration **excludes the
uncond arm from every spontaneity analysis**. Veridicality and complexity are
well-defined there and are analysed normally.

Shared helpers (filename parsing, truncated-JSON repair) are imported from
`rubric.py` rather than duplicated, so a fix to the repair logic benefits both.
"""

from __future__ import annotations

import json
from pathlib import Path

# Shared, scale-independent machinery. Deliberately reused, not copied.
from rubric import (  # noqa: F401
    BIN_FIELDS as _KLUVER_BIN,
    extract_json,
    parse_filename,
    prompt_id_of,
)

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_axes.json"

INT_FIELDS = ("veridicality", "spontaneity", "complexity")
BIN_FIELDS = ("coherent_scene",)
ALL_FIELDS = INT_FIELDS + BIN_FIELDS

# Verbatim from experiments/01_image_test/judge.py. Do not reword: the whole
# replication claim rests on these being the same words Exp 01 scored with.
_FIELD_DEFS = """- veridicality: 0 abstract/unreal, 3 photoreal coherent scene.
- spontaneity: 0 fully prompt-driven, 3 content the prompt didn't ask for.
- complexity: 0 simple, 3 highly elaborate.
- coherent_scene: 1 if the objects hang together as one believable scene, else 0."""


def load_prereg() -> dict:
    """Axes pre-registration, with prompts INHERITED from the Klüver one.

    Same corpus, same intended content, so restating the prompt list here would
    only create a second copy free to drift. `preregistration_axes.json` records
    the inheritance rather than the prompts.
    """
    own = json.loads(PREREG.read_text())
    own["prompts"] = json.loads((HERE / "preregistration.json").read_text())["prompts"]
    return own


def rubric_version() -> str:
    return load_prereg()["rubric_version"]


def _intended_clause(prompt_id: str, prereg: dict) -> str:
    """The 'this image was meant to depict ...' sentence, needed for spontaneity."""
    if prompt_id == "uncond":
        return (
            "This image was generated from an EMPTY prompt (the model's own prior, "
            "with no intended scene). Because nothing was requested, score "
            "spontaneity = 3 by definition. Score veridicality and complexity "
            "normally on whatever is visibly present."
        )
    entry = next(p for p in prereg["prompts"] if p["id"] == prompt_id)
    inv = entry["intended_inventory"]
    return (
        f"This image was meant to depict: \"{entry['text']}\". "
        f"Intended content: {inv['note']}. "
        "Judge only what is VISIBLY present. Spontaneity means content beyond "
        "that intended request, not content that merely renders it unusually."
    )


_BODY = """Rate this image on three continuous dimensions used to describe \
visual hallucinations, plus one yes/no field. {intended}

Score:
{defs}

These are properties of the image as a whole, not of any single object in it. \
Do not score whether individual objects are broken, duplicated or fused; that is \
a different scale and it is not what is being asked here.

Return STRICT JSON only, no prose, with exactly these keys:
{{
  "veridicality": 0..3,
  "spontaneity": 0..3,
  "complexity": 0..3,
  "coherent_scene": 0 or 1,
  "notes": "one short phrase"
}}"""


def build_rubric(prompt_id: str, prereg: dict | None = None) -> str:
    prereg = prereg or load_prereg()
    return _BODY.format(intended=_intended_clause(prompt_id, prereg), defs=_FIELD_DEFS)


def coerce(raw: dict) -> dict:
    """Clamp a judge's raw JSON into the fixed schema.

    Mirrors `rubric.coerce`, including `missing_fields`: a field defaulted to 0
    is otherwise indistinguishable from a genuine 0, which is the forensics
    problem that cost the 2026-07-22 run its judge diagnosis.
    """
    out: dict = {}
    missing: list[str] = []
    for k in INT_FIELDS:
        try:
            out[k] = max(0, min(3, int(round(float(raw[k])))))
        except (KeyError, TypeError, ValueError):
            out[k], _ = 0, missing.append(k)
    for k in BIN_FIELDS:
        try:
            out[k] = 1 if int(float(raw[k])) else 0
        except (KeyError, TypeError, ValueError):
            out[k], _ = 0, missing.append(k)
    out["notes"] = str(raw.get("notes", ""))[:120]
    if missing:
        out["missing_fields"] = missing
    return out
