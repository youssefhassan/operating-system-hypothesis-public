"""Exp 03 sensitivity analyses, round 3b: the third xsci report (XSci 9bcf95ad).

Pre-specified in posthoc_review_xsci_prespec.md (addendum, 2026-10-22). Exploratory; no gate.
  X1  CLIP image-text similarity as a prompt-adherence covariate
  X2  inter-judge composite kappa within guidance values and bins
  X3  registered LMM with the seed random effect dropped
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze as A
import posthoc_review_xsci as P
import statlib as S

HERE = Path(__file__).resolve().parent
PROMPT_TEXT = {p["id"]: p["text"] for p in P.PREREG["prompts"]}
CACHE = HERE / "clip_similarity.json"
BINS = {"low (g 1-2)": {1.0, 2.0}, "mid (g 3-5)": {3.0, 5.0}, "high (g 7-15)": {7.0, 11.0, 15.0}}


def _emb(out):
    # transformers 5 returns a model output from get_*_features; 4.x returned the tensor
    return out if hasattr(out, "norm") else out.pooler_output


def clip_similarity(model: str, cond: list[dict]) -> dict[str, float]:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    have = cache.get(model, {})
    todo = [r for r in cond if r["filename"] not in have]
    if todo:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        with torch.no_grad():
            texts = {}
            for pid, txt in PROMPT_TEXT.items():
                t = _emb(clip.get_text_features(**proc(text=[txt], return_tensors="pt", padding=True).to(dev)))
                texts[pid] = torch.nn.functional.normalize(t, dim=-1)
            for r in todo:
                img = Image.open(A.RESULTS / model / r["filename"]).convert("RGB")
                v = _emb(clip.get_image_features(**proc(images=img, return_tensors="pt").to(dev)))
                v = torch.nn.functional.normalize(v, dim=-1)
                have[r["filename"]] = float((v @ texts[r["prompt_id"]].T).item())
        cache[model] = have
        CACHE.write_text(json.dumps(cache, indent=1))
    return have


def x1_adherence(cond: list[dict], df: pd.DataFrame, sim: dict[str, float]) -> dict:
    s = np.array([sim[r["filename"]] for r in cond])
    g = np.array([r["guidance"] for r in cond], float)
    comp = np.array([r["composite"] for r in cond])
    iqa = np.array([r["clip_iqa"] for r in cond], float)
    aes = np.array([r["aesthetic"] for r in cond], float)
    df = df.copy()
    df["z_sim"] = A._z(s)
    lo, hi = S.partial_spearman_ci(comp, g, s)
    return {
        "rho_sim_vs_g": round(S.spearman(s, g), 4),
        "rho_composite_vs_sim": round(S.spearman(comp, s), 4),
        "partial_rho_controlling_sim": round(S.partial_spearman(comp, g, s), 4),
        "partial_rho_controlling_sim_ci95": [round(lo, 4), round(hi, 4)],
        "partial_rho_controlling_sim_iqa_aes": round(P._partial_multi(comp, g, np.column_stack([s, iqa, aes])), 4),
        "lmm_plus_sim": P._fit("composite ~ guidance_std + z_sim", df, P.VCF, "guidance_std"),
        "lmm_plus_sim_iqa_aes": P._fit("composite ~ guidance_std + z_sim + z_iqa + z_aes", df, P.VCF, "guidance_std"),
    }


def x2_binned_kappa(cond: list[dict], judges: list[tuple[str, str]]) -> dict:
    out = {"pooled_conditioned": A._reliability(cond, [], judges)["composite_weighted_kappa"],
           "per_guidance": {}, "per_bin": {}}
    for gv in sorted({r["guidance"] for r in cond}):
        sub = [r for r in cond if r["guidance"] == gv]
        out["per_guidance"][str(gv)] = {"n": len(sub), "kappa": A._reliability(sub, [], judges)["composite_weighted_kappa"]}
    for name, gs in BINS.items():
        sub = [r for r in cond if r["guidance"] in gs]
        rel = A._reliability(sub, [], judges)
        out["per_bin"][name] = {"n": len(sub), "kappa": rel["composite_weighted_kappa"],
                                "ci95": rel["composite_bootstrap"].get("ci95"),
                                "per_field": {f: v["weighted_kappa"] for f, v in rel["per_field"].items()}}
    return out


def x3_no_seed(df: pd.DataFrame) -> dict:
    return {"with_seed_as_registered": P._fit("composite ~ guidance_std", df, P.VCF, "guidance_std"),
            "prompt_only": P._fit("composite ~ guidance_std", df, {"prompt": "0 + C(prompt)"}, "guidance_std")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="claude,qwen")
    judges = ap.parse_args().judges.split(",")
    report = {"label": "EXPLORATORY (posthoc_review_xsci_prespec.md addendum, 2026-10-22)",
              "clip_model": "openai/clip-vit-large-patch14", "models": {}}
    for m in P.PREREG["models"]["confirmatory"]:
        cond, uncond, notes = A._load_records(m, judges)
        A._build_metric(cond, uncond, notes)
        df = P._df(cond, m)
        sim = clip_similarity(m, cond)
        report["models"][m] = {"X1_prompt_adherence": x1_adherence(cond, df, sim),
                               "X2_kappa_within_guidance": x2_binned_kappa(cond, notes["judges"]),
                               "X3_seed_effect_dropped": x3_no_seed(df)}
    (HERE / "posthoc_review_xsci2.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
