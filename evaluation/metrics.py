"""Ensemble forecast 평가 metric (instruction.md §4.13)."""
from __future__ import annotations

import numpy as np
import torch


def compute_spread(ensemble: torch.Tensor) -> torch.Tensor:
    """Pixel-wise ensemble spread = std across members.

    Args:
        ensemble: (B, C, H, W)
    Returns:
        (C, H, W)
    """
    return ensemble.std(dim=0)


def compute_skill(
    ensemble_mean: torch.Tensor, ground_truth: torch.Tensor
) -> torch.Tensor:
    """Pixel-wise |ensemble_mean - gt|."""
    return (ensemble_mean - ground_truth).abs()


def spread_skill_ratio(
    ensemble: torch.Tensor, ground_truth: torch.Tensor
) -> float:
    """Domain-averaged spread/skill ratio. Ideal ≈ 1.

    < 1: underdispersive; > 1: overdispersive.
    """
    spread = compute_spread(ensemble)
    skill = compute_skill(ensemble.mean(dim=0), ground_truth)
    return float(spread.mean() / (skill.mean() + 1e-8))


def spatial_correlation(map1: torch.Tensor, map2: torch.Tensor) -> float:
    """Pearson correlation between two spatial maps (flattened)."""
    f1 = map1.flatten().float()
    f2 = map2.flatten().float()
    f1c = f1 - f1.mean()
    f2c = f2 - f2.mean()
    num = (f1c * f2c).sum()
    denom = torch.sqrt((f1c ** 2).sum() * (f2c ** 2).sum())
    return float(num / (denom + 1e-8))


def crps(
    ensemble: torch.Tensor, ground_truth: torch.Tensor
) -> torch.Tensor:
    """Continuous Ranked Probability Score (per-pixel).

    Args:
        ensemble: (B, C, H, W)
        ground_truth: (C, H, W)
    Returns:
        (C, H, W) — lower is better.
    """
    B = ensemble.shape[0]
    term1 = (ensemble - ground_truth.unsqueeze(0)).abs().mean(dim=0)

    # |X - X'| 항: B^2 메모리 주의. 64×64×B^2이라 적당함.
    diffs = (ensemble.unsqueeze(0) - ensemble.unsqueeze(1)).abs()
    term2 = 0.5 * diffs.mean(dim=(0, 1))
    del diffs

    _ = B  # silence linter
    return term1 - term2


def rank_histogram(
    ensembles: list[torch.Tensor],
    ground_truths: list[torch.Tensor],
    num_bins: int | None = None,
) -> np.ndarray:
    """Rank histogram for ensemble calibration.

    Args:
        ensembles: list of (B, C, H, W)
        ground_truths: list of (C, H, W)
    Returns:
        (num_bins,) histogram counts.
    """
    ranks: list[torch.Tensor] = []
    B_last = None
    for ens, gt in zip(ensembles, ground_truths):
        B_last = ens.shape[0]
        combined = torch.cat([gt.unsqueeze(0), ens], dim=0)   # (B+1, C, H, W)
        # GT(combined[0])의 rank를 각 픽셀별로 계산
        rank_of_gt = (combined < combined[0:1]).sum(dim=0)    # (C, H, W)
        ranks.append(rank_of_gt.flatten())

    all_ranks = torch.cat(ranks).cpu().numpy()
    bins = num_bins if num_bins is not None else (
        (B_last or 0) + 1
    )
    return np.histogram(all_ranks, bins=bins, range=(0, bins))[0]
