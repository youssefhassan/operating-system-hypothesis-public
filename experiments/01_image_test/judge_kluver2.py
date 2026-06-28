"""
Experiment 01b (EXPLORATORY) — Kluver Level-2 / Level-3 judge.

Exp 01's confirmatory judge (judge.py) scores only Kluver's *Level-1* form
constants (lattice, cobweb, tunnel, spiral) and returned a null. But Kluver
names three levels of hallucinatory constant, and the low-guidance behaviour we
actually saw — repeated/tiled motifs, broken and fused objects — is Level-2
(polyopia / dysmegalopsia / dysmorphopsia: reduplication, size change, shape
distortion) and Level-3 (spatio-temporal: fragmentation, condensation,
displacement). This judge scores THOSE.

It is deliberately a SEPARATE module writing a SEPARATE file
(`judgements_l2.json`). It does NOT modify the pre-registered confirmatory
rubric in judge.py. This is a post-hoc exploratory re-score of the existing
Exp 01 images and must be reported as such (methodology Section 4).

Blinding is identical to judge.py: the model sees only image pixels + a fixed
rubric, scored in shuffled order. The guidance value is never shown; un-blinding
(filename -> guidance) happens only at aggregation.

The `tiling` field is a deliberate VLM cross-check on the objective FFT
periodicity from texture_metrics.py: if a vision model's "this dissolved into a
repeating texture field" judgement tracks the autocorrelation periodicity, both
measurements corroborate each other.

Usage:
    python judge_kluver2.py --dir results-local/sdxl            # score
    python judge_kluver2.py --dir results-local/sd35
    python judge_kluver2.py --analyze --model sdxl             # aggregate + plot
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

CLAUDE_MODEL = os.environ.get("EXP01_CLAUDE_MODEL", "claude-sonnet-4-6")
RUBRIC_VERSION = "exp01b-kluver-l2l3-v1"

# The prompt is public (only the guidance value is blinded), so we name the
# intended scene to let the judge detect reduplication beyond the single
# intended instances of each object.
RUBRIC = """You are scoring a SINGLE image for visual phenomena from Heinrich \
Kluver's Level-2 and Level-3 hallucinatory constants. The image was meant to \
depict ONE scene: a watermelon, a glass half-filled with water, and a set of \
keys on a wooden table (one of each object). Judge only what is VISIBLY present. \
Do not guess what it was "supposed" to look like.

Score these phenomena:
- reduplication: objects or motifs repeated far beyond the single intended \
instance (many watermelons, rows of glasses, a field of keys). 0 none, 3 dominant.
- fragmentation: objects broken into disconnected or shattered pieces. 0 none, 3 dominant.
- condensation: distinct objects fused/merged into hybrid forms (a watermelon-glass \
chimera, keys melting into fruit). 0 none, 3 dominant.
- distortion: melted, warped, or anatomically/physically impossible shapes \
(dysmorphopsia). 0 none, 3 dominant.
- tiling: 1 if the whole image has dissolved into a repeating texture / pattern \
field rather than a single readable scene, else 0.

Return STRICT JSON only, no prose, with exactly these keys:
{
  "reduplication": 0..3,
  "fragmentation": 0..3,
  "condensation": 0..3,
  "distortion": 0..3,
  "tiling": 0 or 1,
  "notes": "one short phrase"
}"""

INT_KEYS = ("reduplication", "fragmentation", "condensation", "distortion")
BIN_KEYS = ("tiling",)
ALL_FIELDS = INT_KEYS + BIN_KEYS

FNAME_RE = re.compile(r"^g(?P<g>[0-9.]+)_s(?P<s>\d+)\.png$")
UNCOND_RE = re.compile(r"^uncond_s(?P<s>\d+)\.png$")


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
    for k in INT_KEYS:
        out[k] = max(0, min(3, int(raw.get(k, 0))))
    for k in BIN_KEYS:
        out[k] = 1 if int(raw.get(k, 0)) else 0
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


def _require_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(f"Missing ANTHROPIC_API_KEY in {_PROJECT_ROOT / '.env'}")


def score(args: argparse.Namespace) -> None:
    model_dir = Path(args.dir)
    if not model_dir.is_absolute():
        model_dir = Path(__file__).resolve().parent / model_dir
    images = sorted(p for p in model_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"no PNGs in {model_dir}")

    _require_key()
    out_path = model_dir / "judgements_l2.json"
    existing = {}
    if out_path.exists() and not args.overwrite:
        existing = json.loads(out_path.read_text()).get("images", {})

    order = list(images)
    random.Random(args.shuffle_seed).shuffle(order)  # blind: shuffled order

    results: dict[str, dict] = dict(existing)
    lock = threading.Lock()

    def _save() -> None:
        out_path.write_text(
            json.dumps(
                {
                    "rubric_version": RUBRIC_VERSION,
                    "claude_model": CLAUDE_MODEL,
                    "fields": list(ALL_FIELDS),
                    "images": results,
                },
                indent=2,
            )
        )

    todo = [p for p in order
            if not (results.get(p.name) and "error" not in results[p.name] and not args.overwrite)]
    print(f"[l2] {len(order)} images, {len(todo)} to score, {args.workers} workers", flush=True)

    def _work(path: Path) -> tuple[str, dict]:
        try:
            return path.name, judge_claude(path)
        except Exception as e:  # noqa: BLE001
            return path.name, {"error": str(e)}

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fn, rec in ex.map(_work, todo):
            with lock:
                results[fn] = rec
                done += 1
                shown = {k: rec.get(k) for k in ALL_FIELDS} if "error" not in rec else rec
                print(f"[l2] ({done}/{len(todo)}) {fn} {shown}", flush=True)
                if done % 10 == 0:  # incremental checkpoint: resumable + monitorable
                    _save()

    _save()
    print(f"[l2] wrote {out_path} ({len(results)} images)", flush=True)


# ------------------------------- aggregation -----------------------------------

def analyze(args: argparse.Namespace) -> None:
    import numpy as np

    from loop import RESULTS_ROOT, spearman, spearman_perm_p

    d = RESULTS_ROOT / args.model
    jpath = d / "judgements_l2.json"
    if not jpath.exists():
        raise SystemExit(f"missing {jpath} — run scoring first")
    images = json.loads(jpath.read_text())["images"]

    by_field: dict[str, dict[float, list[float]]] = {f: defaultdict(list) for f in ALL_FIELDS}
    uncond: dict[str, list[float]] = {f: [] for f in ALL_FIELDS}
    # collect (filename -> tiling) for the FFT cross-check
    vlm_tiling: dict[str, float] = {}
    vlm_redup: dict[str, float] = {}

    for fn, rec in images.items():
        if "error" in rec:
            continue
        mc = FNAME_RE.match(fn)
        mu = UNCOND_RE.match(fn)
        if mc:
            g = float(mc["g"])
            for f in ALL_FIELDS:
                by_field[f][g].append(float(rec.get(f, 0)))
        elif mu:
            for f in ALL_FIELDS:
                uncond[f].append(float(rec.get(f, 0)))
        if mc or mu:
            vlm_tiling[fn] = float(rec.get("tiling", 0))
            vlm_redup[fn] = float(rec.get("reduplication", 0))

    out: dict = {"model": args.model, "rubric_version": RUBRIC_VERSION, "fields": {},
                 "uncond_mean": {f: round(float(np.mean(uncond[f])), 3) if uncond[f] else None
                                 for f in ALL_FIELDS}}
    for f in ALL_FIELDS:
        gs, vs = [], []
        for g, lst in by_field[f].items():
            gs += [g] * len(lst)
            vs += lst
        ag, av = np.array(gs), np.array(vs)
        out["fields"][f] = {
            "means_by_guidance": {str(g): round(float(np.mean(by_field[f][g])), 3)
                                  for g in sorted(by_field[f])},
            "spearman_rho_vs_g": round(spearman(ag, av), 4) if len(ag) >= 3 else None,
            "spearman_p": round(spearman_perm_p(ag, av), 4) if len(ag) >= 3 else None,
        }

    # cross-check vs objective FFT periodicity, if available
    tpath = d / "texture_report.json"
    if tpath.exists():
        tex = json.loads(tpath.read_text()).get("per_image", {})
        common = [fn for fn in vlm_tiling if fn in tex]
        if len(common) >= 3:
            per = np.array([tex[fn]["periodicity"] for fn in common])
            til = np.array([vlm_tiling[fn] for fn in common])
            red = np.array([vlm_redup[fn] for fn in common])
            out["cross_check_vs_fft_periodicity"] = {
                "n": len(common),
                "vlm_tiling_vs_fft_periodicity_rho": round(spearman(til, per), 4),
                "vlm_reduplication_vs_fft_periodicity_rho": round(spearman(red, per), 4),
            }

    out_json = d / "l2_report.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps({"model": args.model,
                      "rho_vs_g": {f: out["fields"][f]["spearman_rho_vs_g"] for f in ALL_FIELDS},
                      "uncond": out["uncond_mean"],
                      "cross_check": out.get("cross_check_vs_fft_periodicity")}, indent=2))
    print(f"\n[l2] wrote {out_json}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig_dir = d / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fields = list(ALL_FIELDS)
        fig, axes = plt.subplots(1, len(fields), figsize=(4 * len(fields), 3.6), sharex=True)
        for ax, f in zip(axes, fields):
            fd = out["fields"][f]
            means = fd["means_by_guidance"]
            gs = sorted(float(g) for g in means)
            ys = [means[str(g)] for g in gs]
            rho = fd["spearman_rho_vs_g"]
            ax.plot(gs, ys, "o-", color="#0d9488", linewidth=2, markersize=5)
            ub = out["uncond_mean"].get(f)
            if ub is not None:
                ax.axhline(ub, color="#ef4444", linestyle="--", linewidth=1.2)
            ax.set_title(f"{f}  (rho={rho:+.2f})" if rho is not None else f)
            ax.set_xlabel("guidance")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Exp 01b Kluver L2/L3 rubric — {args.model}", y=1.04)
        fig.tight_layout()
        out_png = fig_dir / "kluver_l2_curves.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[l2] curve -> {out_png}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 01b Kluver Level-2/3 exploratory judge.")
    p.add_argument("--dir", help="results-local/<model> directory (scoring mode)")
    p.add_argument("--analyze", action="store_true", help="aggregate judgements_l2.json instead of scoring")
    p.add_argument("--model", default="sdxl", help="model name for --analyze")
    p.add_argument("--plot", action="store_true", help="write figures/kluver_l2_curves.png (with --analyze)")
    p.add_argument("--overwrite", action="store_true", help="re-score already-scored images")
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8, help="concurrent API calls")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.analyze:
        analyze(args)
    else:
        if not args.dir:
            raise SystemExit("scoring mode requires --dir results-local/<model>")
        score(args)
