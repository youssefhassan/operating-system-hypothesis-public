"""
Experiment 03 — shared Klüver Level-2/3 rubric.

Both judges (Claude Sonnet 5 in judge.py, Qwen2.5-VL-7B in judge_qwen.py) score
with the *identical* rubric text produced here. Keeping the text in one place is
what makes the cross-judge Cohen's kappa meaningful — the judges differ only in
the model, never in the instrument.

Generalized from Exp 01b's judge_kluver2.py rubric: instead of naming one fixed
still life, the intended scene + expected object counts are looked up per prompt
from the committed pre-registration (`intended_inventory`), so reduplication is
well-defined per prompt. The empty-prompt baseline gets a variant that names no
intended scene (there was no prompt).

`rubric_version` is pinned in preregistration.json — changing the scoring text
after data lands reclassifies the run as exploratory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration.json"

INT_FIELDS = ("reduplication", "fragmentation", "condensation", "distortion")
BIN_FIELDS = ("tiling",)
ALL_FIELDS = INT_FIELDS + BIN_FIELDS

# Filenames: conditioned = "{prompt_id}_g{g}_s{s}.png" (prompt_id itself contains
# one underscore, e.g. p1_stilllife); baseline = "uncond_s{s}.png".
FNAME_RE = re.compile(r"^(?P<pid>p\d+_[a-z0-9]+)_g(?P<g>[0-9.]+)_s(?P<s>\d+)\.png$")
UNCOND_RE = re.compile(r"^uncond_s(?P<s>\d+)\.png$")


def load_prereg() -> dict:
    return json.loads(PREREG.read_text())


def rubric_version() -> str:
    return load_prereg()["rubric_version"]


def _intended_clause(prompt_id: str, prereg: dict) -> str:
    """The 'this image was meant to depict ...' sentence for one prompt."""
    if prompt_id == "uncond":
        return (
            "This image was generated from an EMPTY prompt (the model's own prior, "
            "with no intended scene). There is no 'correct' content. Score only the "
            "phenomena listed below on whatever is visibly present."
        )
    entry = next(p for p in prereg["prompts"] if p["id"] == prompt_id)
    inv = entry["intended_inventory"]
    return (
        f"This image was meant to depict: \"{entry['text']}\". "
        f"Intended content: {inv['note']}. "
        "Judge only what is VISIBLY present; do not guess what it was supposed to "
        "look like. Reduplication means repetition BEYOND that intended content."
    )


_BODY = """Score these phenomena from Heinrich Klüver's Level-2 and Level-3 \
hallucinatory constants. {intended}

Score:
- reduplication: objects or motifs repeated far beyond the intended content \
(e.g. many copies where one was intended, rows/fields of an object). 0 none, 3 dominant.
- fragmentation: objects broken into disconnected or shattered pieces. 0 none, 3 dominant.
- condensation: distinct objects fused/merged into hybrid forms (chimeras, one \
object melting into another). 0 none, 3 dominant.
- distortion: melted, warped, or anatomically/physically impossible shapes \
(dysmorphopsia). 0 none, 3 dominant.
- tiling: 1 if the whole image has dissolved into a repeating texture / pattern \
field rather than a single readable scene, else 0.

Return STRICT JSON only, no prose, with exactly these keys:
{{
  "reduplication": 0..3,
  "fragmentation": 0..3,
  "condensation": 0..3,
  "distortion": 0..3,
  "tiling": 0 or 1,
  "notes": "one short phrase"
}}"""


def build_rubric(prompt_id: str, prereg: dict | None = None) -> str:
    prereg = prereg or load_prereg()
    return _BODY.format(intended=_intended_clause(prompt_id, prereg))


def prompt_id_of(filename: str) -> str | None:
    """Return the prompt id for a results filename, or None if unrecognized."""
    m = FNAME_RE.match(filename)
    if m:
        return m["pid"]
    if UNCOND_RE.match(filename):
        return "uncond"
    return None


def parse_filename(filename: str) -> dict | None:
    """-> {'prompt_id','guidance','seed'} for conditioned, or baseline marker."""
    m = FNAME_RE.match(filename)
    if m:
        return {"prompt_id": m["pid"], "guidance": float(m["g"]), "seed": int(m["s"]),
                "kind": "conditioned"}
    m = UNCOND_RE.match(filename)
    if m:
        return {"prompt_id": "uncond", "guidance": None, "seed": int(m["s"]),
                "kind": "unconditional"}
    return None


def coerce(raw: dict) -> dict:
    """Clamp a judge's raw JSON reply into the fixed schema."""
    out: dict = {}
    for k in INT_FIELDS:
        try:
            out[k] = max(0, min(3, int(round(float(raw.get(k, 0))))))
        except (TypeError, ValueError):
            out[k] = 0
    for k in BIN_FIELDS:
        try:
            out[k] = 1 if int(float(raw.get(k, 0))) else 0
        except (TypeError, ValueError):
            out[k] = 0
    out["notes"] = str(raw.get("notes", ""))[:120]
    return out


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in judge reply: {text[:200]!r}")
    return json.loads(text[start:end + 1])
