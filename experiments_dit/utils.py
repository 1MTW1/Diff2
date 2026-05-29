"""experiments_dit용 공용 유틸: latent 캐시 I/O, denormalize, plot helper.

experiments/utils.py 의 v2 LDM/DiT 판. 캐시는 latent 공간 (C_z=12, 16×16)으로
저장되며, exp5 전용으로 디코딩된 픽셀 필드(`ensemble_pixel`, `x_t_true_pixel`)도
함께 들어 있을 수 있다. `EnsembleSample`은 추가 픽셀 키를 optional 필드로
보유하고, npz에 없으면 `None`으로 둔다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch

from dataset.denormalize import HOUR_TO_IDX, load_stats


VARIABLES = ("t", "u", "v")
# exp5의 픽셀 공간 composite map용 — 물리 변수 인덱스.
TEMP_IDX = 0
U_IDX = 1
V_IDX = 2
# latent 공간 분석에서 대표로 사용하는 latent 채널 (v1의 TEMP_IDX에 대응).
LATENT_CH = 0


@dataclass
class EnsembleSample:
    """단일 시점의 latent ensemble 캐시 데이터.

    Attributes:
        ensemble:        (N, C_z, H_z, W_z) 정규화 latent ẑ_0 앙상블
        log_var:         (C_z, H_z, W_z) latent dual-head log_var (멤버 평균)
        x_t_true:        (C_z, H_z, W_z) GT frame-pair의 latent 인코딩 (posterior μ)
        time_t:          numpy.datetime64 scalar
        path:            원본 npz 경로
        ensemble_pixel:  (N, C, H, W) 디코딩된 픽셀 앙상블 (exp5 전용; 없으면 None)
        x_t_true_pixel:  (C, H, W) GT 픽셀 필드 (exp5 전용; 없으면 None)
    """
    ensemble: np.ndarray
    log_var: np.ndarray
    x_t_true: np.ndarray
    time_t: np.datetime64
    path: Path
    ensemble_pixel: Optional[np.ndarray] = None
    x_t_true_pixel: Optional[np.ndarray] = None


def load_ensemble_npz(path: Path | str) -> EnsembleSample:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    keys = set(data.files)
    ensemble_pixel = (
        data["ensemble_pixel"].astype(np.float32)
        if "ensemble_pixel" in keys else None
    )
    x_t_true_pixel = (
        data["x_t_true_pixel"].astype(np.float32)
        if "x_t_true_pixel" in keys else None
    )
    return EnsembleSample(
        ensemble=data["ensemble"].astype(np.float32),
        log_var=data["log_var"].astype(np.float32),
        x_t_true=data["x_t_true"].astype(np.float32),
        time_t=np.datetime64(str(data["time_t"])),
        path=path,
        ensemble_pixel=ensemble_pixel,
        x_t_true_pixel=x_t_true_pixel,
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
