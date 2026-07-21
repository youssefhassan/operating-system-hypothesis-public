"""
Experiment 03 — no-reference image quality (the guidance-matching curve).

Computes two local, no-reference quality signals per image:
  - CLIP-IQA (torchmetrics) — CLIP-based perceptual quality, in [0,1].
  - LAION aesthetic predictor (improved-aesthetic-predictor, CLIP ViT-L/14 +
    small MLP head) — the classic aesthetic score.

Both run locally on Apple Silicon (no reference distribution, stable at small
per-bin N). Raw scores are written to `quality.json`; analyze.py standardizes
each within a model and averages them into Q, whose per-(model,g) mean is the
inverted-U comfort-zone curve. FID and CLIP-score were rejected in the
pre-registration (noisy at N / wrong shape); see preregistration.json.

Setup:
    pip install torch torchmetrics transformers
    # aesthetic head auto-downloads once (~5 MB); CLIP ViT-L/14 downloads via HF.

Usage:
    python quality.py --dir results-local/sdxl
    python quality.py --dir results-local/sd35
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
AESTHETIC_URL = ("https://github.com/christophschuhmann/improved-aesthetic-predictor/"
                 "raw/main/sac+logos+ava1-l14-linearMSE.pth")
AESTHETIC_CACHE = HERE / ".cache" / "sac+logos+ava1-l14-linearMSE.pth"


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ------------------------------- CLIP-IQA --------------------------------------


def _clip_iqa_metric(device):
    from torchmetrics.multimodal import CLIPImageQualityAssessment

    # single "quality" antonym prompt -> scalar in [0,1], higher = better.
    # data_range=255: we feed [0,255] float tensors (torchmetrics normalizes by this).
    m = CLIPImageQualityAssessment(model_name_or_path="clip_iqa", prompts=("quality",),
                                   data_range=255.0)
    return m.to(device)


def _clip_iqa_score(metric, img_uint8, device) -> float:
    import torch

    t = torch.from_numpy(img_uint8).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = metric(t)  # torchmetrics CLIP-IQA expects [0,255] float
    return float(out.detach().cpu().item())


# --------------------------- LAION aesthetic head ------------------------------


class _AestheticMLP:
    """christophschuhmann improved-aesthetic-predictor head over CLIP ViT-L/14."""

    def __init__(self, device):
        import torch
        import torch.nn as nn
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        class MLP(nn.Module):
            def __init__(self, d=768):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(d, 1024), nn.Dropout(0.2),
                    nn.Linear(1024, 128), nn.Dropout(0.2),
                    nn.Linear(128, 64), nn.Dropout(0.1),
                    nn.Linear(64, 16), nn.Linear(16, 1),
                )

            def forward(self, x):
                return self.layers(x)

        if not AESTHETIC_CACHE.exists():
            AESTHETIC_CACHE.parent.mkdir(parents=True, exist_ok=True)
            print(f"[quality] downloading aesthetic head -> {AESTHETIC_CACHE}", flush=True)
            urllib.request.urlretrieve(AESTHETIC_URL, AESTHETIC_CACHE)
        self.mlp = MLP().to(device)
        state = torch.load(AESTHETIC_CACHE, map_location=device)
        self.mlp.load_state_dict(state)
        self.mlp.eval()

    def score(self, pil_img) -> float:
        import torch
        import torch.nn.functional as F

        inputs = self.proc(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.clip.get_image_features(**inputs)
            # transformers >=5 returns a BaseModelOutputWithPooling whose
            # pooler_output IS the 768-d projected image embedding; <5 returns
            # the tensor directly. Unwrap for both.
            if not isinstance(emb, torch.Tensor):
                emb = emb.pooler_output
            emb = F.normalize(emb, dim=-1)  # repo L2-normalizes before the head
            return float(self.mlp(emb).squeeze().detach().cpu().item())


# ---------------------------------- run ----------------------------------------


def run(args: argparse.Namespace) -> None:
    import numpy as np
    from PIL import Image

    model_dir = Path(args.dir)
    if not model_dir.is_absolute():
        model_dir = HERE / model_dir
    images = sorted(p for p in model_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"no PNGs in {model_dir}")

    out_path = model_dir / "quality.json"
    per_image: dict[str, dict] = {}
    if out_path.exists() and not args.overwrite:
        per_image = json.loads(out_path.read_text()).get("per_image", {})

    device = _device()
    print(f"[quality] device={device}, {len(images)} images", flush=True)
    iqa = _clip_iqa_metric(device)
    try:
        aesthetic = _AestheticMLP(device)
    except Exception as e:  # noqa: BLE001
        print(f"[quality] WARNING: aesthetic head unavailable ({e}); "
              "Q will fall back to CLIP-IQA only (noted in analysis).", flush=True)
        aesthetic = None

    todo = [p for p in images if args.overwrite or p.name not in per_image
            or "error" in per_image.get(p.name, {})]
    for i, path in enumerate(todo, start=1):
        try:
            pil = Image.open(path).convert("RGB")
            rec = {"clip_iqa": _clip_iqa_score(iqa, np.asarray(pil), device)}
            rec["aesthetic"] = aesthetic.score(pil) if aesthetic is not None else None
        except Exception as e:  # noqa: BLE001
            rec = {"error": str(e)}
        per_image[path.name] = rec
        if i % 25 == 0 or i == len(todo):
            out_path.write_text(json.dumps(
                {"components": ["clip_iqa", "aesthetic"], "device": device,
                 "per_image": per_image}, indent=2))
            print(f"[quality] ({i}/{len(todo)}) {path.name} {rec}", flush=True)
    print(f"[quality] wrote {out_path} ({len(per_image)} images)", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 03 no-reference quality (CLIP-IQA + aesthetic).")
    p.add_argument("--dir", required=True, help="results-local/<model> directory")
    p.add_argument("--overwrite", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
