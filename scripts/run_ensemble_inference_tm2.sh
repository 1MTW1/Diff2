#!/bin/bash
# Multi-GPU t-2 ensemble 추론 (past 모델만 50-step DDPM).
# 사용법:
#   bash scripts/run_ensemble_inference_tm2.sh --checkpoint outputs/.../checkpoint_best.pt
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CUDA_VISIBLE_DEVICES:=0,1}"
: "${NUM_GPUS:=2}"
export CUDA_VISIBLE_DEVICES

accelerate launch \
    --num_processes="${NUM_GPUS}" \
    --multi_gpu \
    -m experiments2.ensemble_inference \
    --config config/default.yaml \
    --output_dir outputs/ensembles_tm2 \
    --n_members 30 \
    --past_steps 50 \
    "$@"
