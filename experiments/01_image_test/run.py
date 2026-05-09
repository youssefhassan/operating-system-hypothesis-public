"""
Experiment 01 — Vision temperature gradient.

Sweep a vision model across a "suppression-analogue" parameter
(temperature / guidance / noise) with seed and prompt held fixed,
and log outputs for later structural analysis.

Status: scaffold only. Fill in the sweep once a target model is chosen.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROMPT = "a single red apple on a wooden table"
SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
SEED = 42


def run_sweep() -> None:
    raise NotImplementedError(
        "Pick a vision model (e.g. an image-generation API or a local "
        "diffusion model) and implement the sweep here. Each iteration "
        "should write its output and metadata into RESULTS_DIR."
    )


if __name__ == "__main__":
    run_sweep()
