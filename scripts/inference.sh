#!/bin/bash
# Diffusion² 추론 (ensemble 생성) 스크립트.
set -euo pipefail

cd "$(dirname "$0")/.."

python -m inference.ensemble \
    --config config/default.yaml \
    --checkpoint outputs/experiment_1/checkpoint_best.pt \
    --split test \
    --output_dir outputs/inference_1 \
    --ensemble_size 20 \
    "$@"
