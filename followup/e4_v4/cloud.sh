#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${MODEL_PATH:?Set MODEL_PATH to locally staged Qwen2.5-1.5B-Instruct weights}"
phase="${1:-pilot}"
if [[ "$phase" == pilot ]]; then worlds='961 962'; elif [[ "$phase" == confirm ]]; then
  test -f CONFIRM_LOCK.json
  worlds='971 972 973 974 975 976'
  if [[ ! -d data/confirm ]]; then python generate.py --phase confirm; fi
else echo 'Use pilot or confirm' >&2; exit 2; fi
mkdir -p logs
python -m pip freeze > "logs/environment_${phase}.txt"
nvidia-smi > "logs/gpu_${phase}.txt"
for world in $worlds; do
  python run.py --phase "$phase" --world "$world" --arm edge --model "$MODEL_PATH" --micro 32 --eval-batch 256 --preflight-only
  for arm in edge joint; do
    if [[ -f "runs/$phase/world${world}_${arm}/COMPLETE.json" ]]; then continue; fi
    python run.py --phase "$phase" --world "$world" --arm "$arm" --model "$MODEL_PATH" --micro 32 --eval-batch 256 2>&1 | tee "logs/${phase}_${world}_${arm}.log"
    test -f "runs/$phase/world${world}_${arm}/COMPLETE.json" || exit 3
  done
done
python analyze.py --phase "$phase"
