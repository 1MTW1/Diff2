#!/bin/bash
# Diffusion² 학습 (multi-GPU, accelerate launch).
#
# 사용: GPU 수는 config/accelerate_config.yaml의 num_processes로 조정.
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_DIR="${OUTPUT_DIR:-outputs/experiment_1}"

accelerate launch \
    --config_file config/accelerate_config.yaml \
    -m training.train \
        --config config/default.yaml \
        --output_dir "$OUTPUT_DIR" \
        "$@"
