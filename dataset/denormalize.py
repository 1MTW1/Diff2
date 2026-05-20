"""정규화된 텐서를 원본 단위로 역변환."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import xarray as xr


HOUR_TO_IDX = {0: 0, 6: 1, 12: 2, 18: 3}


def load_stats(
    stats_path: str = "data/normalization_stats.zarr",
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        mean: (4, C=3, H=64, W=64) torch.Tensor
        std:  (4, C=3, H=64, W=64) torch.Tensor
    """
    stats = xr.open_zarr(stats_path).load()
    mean = torch.from_numpy(stats["mean"].values.astype(np.float32)).to(device)
    std = torch.from_numpy(stats["std"].values.astype(np.float32)).to(device)
    return mean, std


def _hour_indices(times: np.ndarray, device: torch.device) -> torch.Tensor:
    hours = pd.DatetimeIndex(np.atleast_1d(times)).hour.to_numpy()
    return torch.tensor(
        [HOUR_TO_IDX[int(h)] for h in hours],
        dtype=torch.long,
        device=device,
    )


def denormalize(
    x_norm: torch.Tensor,
    times: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """역정규화: normalized → 원본 단위.

    Args:
        x_norm: (B, C, H, W) or (C, H, W)
        times: 각 sample의 datetime64 (시각 추출용)
        mean, std: (4, C, H, W)
    """
    if x_norm.ndim == 3:
        hour = pd.Timestamp(times).hour
        h_idx = HOUR_TO_IDX[hour]
        return x_norm * std[h_idx] + mean[h_idx]

    h_indices = _hour_indices(times, x_norm.device)
    mu_batch = mean[h_indices]    # (B, C, H, W)
    sigma_batch = std[h_indices]
    return x_norm * sigma_batch + mu_batch
