"""
Experiment 01 — the form-constant judge: blind dual-VLM form-constant judge.

Scores each generated image on the four Klüver form-constant classes (lattice,
cobweb, tunnel, spiral) plus the Suzuki et al. (2024) veridicality / spontaneity
/ complexity triplet, using TWO vision models (Claude + GPT) as independent
judges for cross-judge agreement.

BLINDING: a judge receives only image pixels and a fixed rubric. It never gets
the guidance value or filename semantics, and images are scored in shuffled
order. Un-blinding (filename -> guidance) happens only at aggregation
(loop.py / analysis). The rubric and model versions are pinned constants below;
changing them is a methodology-relevant edit (see README pre-registration).

Output: writes `judgements.json` into the target dir, one record per image with
each judge's raw scores and a derived per-image metric `m_img` in [0, 1].

Usage:
    python judge.py --dir results-local/sdxl                 # claude (default)
    python judge.py --dir results-local/sdxl --judges both     # claude + gpt
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

# Pin judge models for confirmatory runs (override via env). As of 2026-06:
#   Claude Sonnet 4.6 — claude-sonnet-4-6 (dateless ID = pinned snapshot per Anthropic)
#   GPT-5.5 — gpt-5.5-2026-04-23 (dated snapshot; alias gpt-5.5 also works)
CLAUDE_MODEL = os.environ.get("EXP01_CLAUDE_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL = os.environ.get("EXP01_OPENAI_MODEL", "gpt-5.5-2026-04-23")
RUBRIC_VERSION = "exp01-formconstant-v1"

RUBRIC = """You are scoring a SINGLE image for geometric "form constants" — the \
recurring shapes that appear in human visual hallucinations. Judge only what is \
visibly present in this image. Do not guess what it was "supposed" to be.

Form-constant classes:
- lattice: grids, honeycombs, checkerboards, tilings, regularly repeated cells
- cobweb: radial web / net structures
- tunnel: tunnels, funnels, concentric framing pulling toward a center
- spiral: spirals, vortices, logarithmic swirls

Return STRICT JSON only, no prose, with exactly these keys:
{
  "lattice": 0 or 1,
  "cobweb": 0 or 1,
  "tunnel": 0 or 1,
  "spiral": 0 or 1,
  "geometric_intensity": 0..3,   // 0 none, 1 faint, 2 clear, 3 dominant
  "veridicality": 0..3,          // 0 abstract/unreal, 3 photoreal coherent scene
  "spontaneity": 0..3,           // 0 fully prompt-driven, 3 content the prompt didn't ask for
  "complexity": 0..3,            // 0 simple, 3 highly elaborate
  "coherent_scene": 0 or 1,
  "notes": "one short phrase"
}"""

INT_KEYS = ("geometric_intensity", "veridicality", "spontaneity", "complexity")
BIN_KEYS = ("lattice", "cobweb", "tunnel", "spiral", "coherent_scene")


def _b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in judge reply: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def _coerce(raw: dict) -> dict:
    out: dict = {}
    for k in BIN_KEYS:
        out[k] = 1 if int(raw.get(k, 0)) else 0
    for k in INT_KEYS:
        out[k] = max(0, min(3, int(raw.get(k, 0))))
    out["notes"] = str(raw.get("notes", ""))[:120]
    return out


def judge_claude(path: Path) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _b64(path),
                        },
                    },
                    {"type": "text", "text": RUBRIC},
                ],
            }
        ],
    )
    return _coerce(_extract_json(resp.content[0].text))


def judge_gpt(path: Path) -> dict:
    import openai

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": RUBRIC},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_b64(path)}"},
                    },
                ],
            }
        ],
    )
    return _coerce(_extract_json(resp.choices[0].message.content))


JUDGES = {"claude": judge_claude, "gpt": judge_gpt}


def _require_keys(which: list[str]) -> None:
    missing = []
    if "claude" in which and not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if "gpt" in which and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if missing:
        raise SystemExit(
            f"Missing API keys in {_PROJECT_ROOT / '.env'}: {', '.join(missing)}"
        )


def m_img(records: list[dict]) -> float | None:
    """Per-image metric: mean geometric_intensity across judges, normalized 0..1."""
    vals = [r["geometric_intensity"] for r in records if r]
    if not vals:
        return None
    return round(sum(vals) / len(vals) / 3.0, 4)


def run(args: argparse.Namespace) -> None:
    model_dir = Path(args.dir)
    if not model_dir.is_absolute():
        model_dir = Path(__file__).resolve().parent / model_dir
    images = sorted(p for p in model_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"no PNGs in {model_dir}")

    which = list(JUDGES) if args.judges == "both" else [args.judges]
    _require_keys(which)
    out_path = model_dir / "judgements.json"
    existing = {}
    if out_path.exists() and not args.overwrite:
        existing = json.loads(out_path.read_text()).get("images", {})

    order = list(images)
    random.Random(args.shuffle_seed).shuffle(order)  # blind: judge in random order

    results: dict[str, dict] = dict(existing)
    for i, path in enumerate(order, 1):
        fn = path.name
        rec = results.get(fn, {})
        for judge_name in which:
            prev = rec.get(judge_name)
            if (
                prev is not None
                and not args.overwrite
                and "error" not in prev
            ):
                continue
            try:
                rec[judge_name] = JUDGES[judge_name](path)
            except Exception as e:  # noqa: BLE001
                rec[judge_name] = {"error": str(e)}
                print(f"  [{judge_name}] FAILED on {fn}: {e}")
        scored = [rec[j] for j in which if j in rec and "error" not in rec[j]]
        rec["m_img"] = m_img(scored)
        rec["agree_any_formconstant"] = (
            None
            if len(scored) < 2
            else int(
                bool(any(scored[0][k] for k in ("lattice", "cobweb", "tunnel", "spiral")))
                == bool(any(scored[1][k] for k in ("lattice", "cobweb", "tunnel", "spiral")))
            )
        )
        results[fn] = rec
        print(f"[judge] ({i}/{len(order)}) {fn} m_img={rec['m_img']}")

    out_path.write_text(
        json.dumps(
            {
                "rubric_version": RUBRIC_VERSION,
                "claude_model": CLAUDE_MODEL if "claude" in which else None,
                "openai_model": OPENAI_MODEL if "gpt" in which else None,
                "judges": which,
                "images": results,
            },
            indent=2,
        )
    )
    print(f"[judge] wrote {out_path} ({len(results)} images)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 01 blind dual-VLM form-constant judge.")
    p.add_argument("--dir", required=True, help="results-local/<model> directory")
    p.add_argument("--judges", choices=["both", "claude", "gpt"], default="claude")
    p.add_argument("--overwrite", action="store_true", help="re-judge already-scored images")
    p.add_argument("--shuffle-seed", type=int, default=0)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
