#!/bin/bash
# Multi-GPU ensemble 추론 (accelerate).
# 사용법:
#   bash scripts/run_ensemble_inference.sh \
#       --checkpoint outputs/experiment_1/checkpoint_best.pt
# 또는 env override:
#   CUDA_VISIBLE_DEVICES=1,2 NUM_GPUS=2 bash scripts/run_ensemble_inference.sh ...
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CUDA_VISIBLE_DEVICES:=0,1}"
: "${NUM_GPUS:=2}"
export CUDA_VISIBLE_DEVICES

accelerate launch \
    --num_processes="${NUM_GPUS}" \
    --multi_gpu \
    -m experiments.ensemble_inference \
    --config config/default.yaml \
    --output_dir outputs/ensembles \
    --n_members 30 \
    --past_steps 50 \
    --main_steps 200 \
    "$@"
