#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
v=experiments/ablation_v9
a=$v/audits/p2_frozen_classifier_20261125
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
py=${PYTHON:-python}
p2=$v/runs/p2_official_stable_20261121_2320
p1=$v/runs/p1_official_stable_20261117_130401
probe=p2_frozen_classifier_20261125
tune=p1_fullinput_head_20261125
ev=${tune}_real700
trap 'rc=$?; printf "%s exit=%s\n" "$(date -Is)" "$rc" >> "$a/queue.log"' EXIT
CUDA_VISIBLE_DEVICES=0 "$py" "$v/run_experiment.py" --mode probe-p2 --head-plan "$v/configs/step1_heads.json" --config "$p2/config.json" --checkpoint "$p2/checkpoints/best.pth" --parallel none --run-id "$probe" > "$a/p2.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$py" "$v/run_experiment.py" --mode finetune-head --head-plan "$v/configs/step1_heads.json" --config "$p1/config.json" --checkpoint "$p1/checkpoints/best.pth" --parallel none --run-id "$tune" > "$a/p1.log" 2>&1
CUDA_VISIBLE_DEVICES=0,1 "$py" "$v/run_experiment.py" --mode eval --eval-real-only --parallel data_parallel --gpu-ids 0 1 --config "$v/runs/$tune/config.json" --checkpoint "$v/runs/$tune/checkpoints/best.pth" --run-id "$ev" > "$a/real700.log" 2>&1
"$py" "$a/compare.py" "$v/runs/$ev" "$v/configs/step1_heads.json"
