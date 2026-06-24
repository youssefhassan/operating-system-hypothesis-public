#!/usr/bin/env bash
# Finish Exp 01: SDXL N=10, SD 3.5 replication, judge, analyze.
# Safe to re-run (--skip-existing on generation; judge retries errors).
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(pwd)/.mplcache}"
SEEDS_NEW=(45 46 47 48 49 50 51)
SEEDS_ALL=(42 43 44 45 46 47 48 49 50 51)
GRID=(1.0 1.5 2.0 3.0 4.5 6.0 8.0 11.0 15.0)

echo "=== [1/6] SDXL: missing seeds at g>=1.5 ==="
$PY sweep_local.py --model sdxl --skip-existing \
  --guidance 1.5 2.0 3.0 4.5 6.0 8.0 11.0 15.0 \
  --seeds "${SEEDS_NEW[@]}"

echo "=== [2/6] SDXL: missing unconditional seeds ==="
$PY sweep_local.py --model sdxl --skip-existing --unconditional \
  --guidance 1.0 --seeds "${SEEDS_NEW[@]}"

echo "=== [3/6] SD 3.5: full grid + unconditional (N=10) ==="
if ! $PY sweep_local.py --model sd35 --skip-existing --unconditional \
  --guidance "${GRID[@]}" --seeds "${SEEDS_ALL[@]}"; then
  echo "[warn] SD 3.5 skipped — accept the license at:"
  echo "  https://huggingface.co/stabilityai/stable-diffusion-3.5-medium"
  SD35_OK=0
else
  SD35_OK=1
fi

echo "=== [4/6] Judge SDXL ==="
$PY judge.py --dir results-local/sdxl --judges claude

if [[ "${SD35_OK:-0}" == 1 ]]; then
  echo "=== [5/6] Judge SD 3.5 ==="
  $PY judge.py --dir results-local/sd35 --judges claude
else
  echo "=== [5/6] Judge SD 3.5 — skipped (no images) ==="
fi

echo "=== [6/6] Analyze + figures ==="
$PY analyze.py --model sdxl --contact-sheet --plots
if [[ "${SD35_OK:-0}" == 1 ]]; then
  $PY analyze.py --model sd35 --contact-sheet --plots
fi

echo "Done. SDXL report: results-local/sdxl/analysis_report.json"
