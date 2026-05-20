"""3-stage curriculum (instruction.md §4.10).

Stage 1: DDPM_past만 학습
Stage 2: DDPM_past freeze, DDPM_main 학습
Stage 3: Joint
"""
from __future__ import annotations

import torch.nn as nn


class CurriculumStage:
    PAST_ONLY = 1
    MAIN_ONLY = 2
    JOINT = 3


def get_stage(
    epoch: int,
    stage1_epochs: int,
    stage2_epochs: int,
    stage3_epochs: int,
) -> int:
    """현재 epoch이 어느 stage에 속하는지 반환."""
    del stage3_epochs  # 마지막 stage는 무한히 적용 가능
    if epoch < stage1_epochs:
        return CurriculumStage.PAST_ONLY
    if epoch < stage1_epochs + stage2_epochs:
        return CurriculumStage.MAIN_ONLY
    return CurriculumStage.JOINT


def get_loss_weights(stage: int) -> dict[str, float]:
    if stage == CurriculumStage.PAST_ONLY:
        return {"past": 1.0, "main": 0.0}
    if stage == CurriculumStage.MAIN_ONLY:
        return {"past": 0.0, "main": 1.0}
    return {"past": 1.0, "main": 1.0}


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad
