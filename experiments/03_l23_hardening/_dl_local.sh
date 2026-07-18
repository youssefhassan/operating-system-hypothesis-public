#!/usr/bin/env bash
# SDXL fp16 snapshot to a local dir. NO stall-kill (killing fragments the temps
# and interrupts the slow first-byte handshake). One continuous snapshot process
# resumes transient drops internally; only rerun if the process EXITS. Then drive
# the smoke off local fp16 weights. hf_transfer OFF. Scratch file.
set -u
cd /Users/youssefhassan/Development/operating-system-hypothesis/experiments/03_l23_hardening
set -a; . ../../.env 2>/dev/null; set +a
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=20        # stalled stream read raises -> internal retry+resume
VENV=/Users/youssefhassan/Development/operating-system-hypothesis/.venv/bin/python
export LOCAL=$PWD/sdxl_local
LOG=$PWD/dl_local.log; : > "$LOG"
stamp(){ date +%H:%M:%S; }

DL_PY='
import os
from huggingface_hub import snapshot_download
snapshot_download("stabilityai/stable-diffusion-xl-base-1.0",
    local_dir=os.environ["LOCAL"],
    allow_patterns=["*.json","*.txt","*.fp16.safetensors"],
    max_workers=1)          # serial: one file at a time, gentler on a flaky link
print("SNAPSHOT_OK")
'

echo "[$(stamp)] fp16 snapshot (no kill) -> $LOCAL" | tee -a "$LOG"
t=0
while [ $t -lt 400 ]; do
  t=$((t+1))
  echo "[$(stamp)] snapshot attempt $t" | tee -a "$LOG"
  "$VENV" -c "$DL_PY" >> "$LOG" 2>&1 && { echo "[$(stamp)] snapshot DONE" | tee -a "$LOG"; break; }
  echo "[$(stamp)] snapshot exited nonzero -> resume in 5s (dir=$(( $(du -sk "$LOCAL" 2>/dev/null|cut -f1)/1024 ))MB)" | tee -a "$LOG"
  sleep 5
done

resume(){ local l="$1"; shift; local i=0; while [ $i -lt 100 ]; do i=$((i+1)); "$@" >> "$LOG" 2>&1 && { echo "[$(stamp)] [$l] DONE" | tee -a "$LOG"; return 0; }; echo "[$(stamp)] [$l] retry $i" | tee -a "$LOG"; sleep 5; done; }
echo "[$(stamp)] === generation (local fp16 weights) ===" | tee -a "$LOG"
resume gen "$VENV" sweep_local.py --model sdxl --model-path "$LOCAL" --variant fp16 \
       --prompts p2_portrait p3_bicycle --guidance 1 7 15 --seeds 42 43 --skip-existing || exit 1
resume judge-claude env EXP03_CLAUDE_MODEL=claude-sonnet-5 "$VENV" judge.py --dir results-local/sdxl || exit 1
resume judge-qwen "$VENV" judge_qwen.py --dir results-local/sdxl || exit 1
resume quality "$VENV" quality.py --dir results-local/sdxl || exit 1
"$VENV" analyze.py --model sdxl >> "$LOG" 2>&1 && echo "===== SMOKE COMPLETE =====" | tee -a "$LOG"
