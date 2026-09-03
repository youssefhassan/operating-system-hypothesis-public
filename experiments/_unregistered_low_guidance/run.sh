#!/usr/bin/env bash
# Unregistered scratch: full sub-CFG sweep, both judges, and read-out. Resumable —
# re-running skips images that already exist and images already judged.
#
# If the network is down, `HF_HUB_OFFLINE=1 ./run.sh` works: every weight this needs
# (SDXL via Exp 03's sdxl_local/, SD 3.5 and both Qwen judges via the HF cache) is
# already on disk.
set -euo pipefail
cd "$(dirname "$0")"
source ../../.venv/bin/activate

for m in sdxl sd35; do
  echo "=== generating $m ==="
  python sweep_low_g.py --model "$m" 2>&1 | tee "gen_$m.log"
done

echo "=== validity check: g=0 must be the pure prior ==="
python check_g0.py    # exits non-zero if the forced-CFG patch did not take

# One judge (Qwen3-VL-32B), into judgements_qwen_32b.json.
# NOTE: judge_both.py passes absolute --dir paths. judge_qwen.py resolves a
# *relative* --dir against its own directory (experiments/03_l23_hardening), so
# calling it directly with `--dir results-local/sdxl` from here silently looks in the
# wrong place and dies with "no PNGs".
echo "=== judging with Qwen3-VL-32B ==="
python judge_both.py 2>&1 | tee judge_both.log

echo "=== contact sheets ==="
for m in sdxl sd35; do
  for s in 42 43; do python contact_sheet.py --model "$m" --seed "$s" || true; done
done

echo "=== pixel divergence from the prior (judge-free) ==="
for m in sdxl sd35; do python divergence.py --model "$m"; done | tee divergence.log

echo "=== read-out ==="
python analyze_low_g.py 2>&1 | tee analysis.log
