"""experiment.md용 공용 유틸: 캐시 I/O, denormalize, plot helper."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch

from dataset.denormalize import HOUR_TO_IDX, load_stats


VARIABLES = ("t", "u", "v")
TEMP_IDX = 0
U_IDX = 1
V_IDX = 2


@dataclass
class EnsembleSample:
    """단일 시점의 ensemble 캐시 데이터.

    Attributes:
        ensemble: (N, C, H, W) 정규화 공간
        log_var:  (C, H, W) ℓ_m raw log
        x_t_true: (C, H, W) 정규화 공간 GT
        time_t:   numpy.datetime64 scalar
        path:     원본 npz 경로
    """
    ensemble: np.ndarray
    log_var: np.ndarray
    x_t_true: np.ndarray
    time_t: np.datetime64
    path: Path


def load_ensemble_npz(path: Path | str) -> EnsembleSample:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    return EnsembleSample(
        ensemble=data["ensemble"].astype(np.float32),
        log_var=data["log_var"].astype(np.float32),
        x_t_true=data["x_t_true"].astype(np.float32),
        time_t=np.datetime64(str(data["time_t"])),
        path=path,
    )


def list_ensemble_files(ensemble_dir: Path | str) -> list[Path]:
    ensemble_dir = Path(ensemble_dir)
    files = sorted(ensemble_dir.glob("sample_*.npz"))
    if not files:
        raise FileNotFoundError(f"No sample_*.npz under {ensemble_dir}")
    return files


def iter_ensemble_samples(ensemble_dir: Path | str) -> Iterator[EnsembleSample]:
    for f in list_ensemble_files(ensemble_dir):
        yield load_ensemble_npz(f)


def denorm_array(
    x_norm: np.ndarray,
    time_t: np.datetime64,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> np.ndarray:
    """역정규화: (C,H,W) or (N,C,H,W) numpy. mean/std는 (4,C,H,W) torch."""
    hour = pd.Timestamp(time_t).hour
    h_idx = HOUR_TO_IDX[int(hour)]
    mu = mean[h_idx].cpu().numpy()      # (C, H, W)
    sigma = std[h_idx].cpu().numpy()
    return x_norm * sigma + mu


def denorm_spread(
    spread_norm: np.ndarray,
    time_t: np.datetime64,
    std: torch.Tensor,
) -> np.ndarray:
    """Spread/std는 mean 빼고 σ만 곱함 (additive shift 무효)."""
    hour = pd.Timestamp(time_t).hour
    h_idx = HOUR_TO_IDX[int(hour)]
    sigma = std[h_idx].cpu().numpy()
    return spread_norm * sigma


def load_norm_stats(
    stats_path: str = "data/normalization_stats.zarr",
) -> tuple[torch.Tensor, torch.Tensor]:
    return load_stats(stats_path, device="cpu")


def ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_plot_defaults() -> None:
    import matplotlib

    matplotlib.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "axes.grid": False,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
