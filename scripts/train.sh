#!/bin/bash
# Diffusion² 학습 (단일 GPU: 기본 device 2).
#
# 사용:
#   bash scripts/train.sh                       # fresh start, GPU 2
#   bash scripts/train.sh --resume              # output_dir에서 최신 ckpt 자동 resume
#   bash scripts/train.sh --resume_from path    # 명시적 ckpt
#   GPU_ID=0 bash scripts/train.sh              # 다른 GPU 사용
#
# Multi-GPU로 돌리려면 scripts/train_multi_gpu.sh 참고 (없으면 직접 작성).
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${GPU_ID:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/experiment_1}"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m training.train \
    --config config/default.yaml \
    --output_dir "$OUTPUT_DIR" \
    "$@"
