"""정규화 적용/역적용 함수 (학습/추론 시 on-the-fly 변환용)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import xarray as xr

from .config import HOUR_TO_IDX


def load_stats_to_torch(
    stats_path: str, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """통계량을 torch tensor로 로드.

    Returns:
        mean: (4, C, H, W) torch.Tensor
        std:  (4, C, H, W) torch.Tensor
    """
    stats = xr.open_zarr(stats_path).load()
    mean = torch.from_numpy(stats["mean"].values.astype(np.float32)).to(device)
    std = torch.from_numpy(stats["std"].values.astype(np.float32)).to(device)
    return mean, std


def _resolve_h_indices(
    times: np.ndarray, device: torch.device
) -> torch.Tensor:
    hours = pd.DatetimeIndex(np.atleast_1d(times)).hour.to_numpy()
    return torch.tensor(
        [HOUR_TO_IDX[int(h)] for h in hours],
        dtype=torch.long,
        device=device,
    )


def normalize_torch(
    x: torch.Tensor,
    times: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Torch tensor에 정규화 적용.

    Args:
        x: (B, C, H, W) or (C, H, W) raw data
        times: datetime64 array (B,) or scalar
        mean, std: (4, C, H, W) stats
    """
    if x.ndim == 3:
        hour = pd.Timestamp(times).hour
        h_idx = HOUR_TO_IDX[hour]
        return (x - mean[h_idx]) / std[h_idx]

    h_indices = _resolve_h_indices(times, x.device)
    mu_batch = mean[h_indices]    # (B, C, H, W)
    sigma_batch = std[h_indices]
    return (x - mu_batch) / sigma_batch


def denormalize_torch(
    x_norm: torch.Tensor,
    times: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """역정규화: 정규화된 텐서 → 원본 단위."""
    if x_norm.ndim == 3:
        hour = pd.Timestamp(times).hour
        h_idx = HOUR_TO_IDX[hour]
        return x_norm * std[h_idx] + mean[h_idx]

    h_indices = _resolve_h_indices(times, x_norm.device)
    mu_batch = mean[h_indices]
    sigma_batch = std[h_indices]
    return x_norm * sigma_batch + mu_batch
