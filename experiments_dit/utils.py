"""experiments_dit용 공용 유틸: latent/픽셀 캐시 I/O, plot helper.

experiments/utils.py 의 v4 LDM/DiT 판. 캐시는 프레임별 latent 공간 (C_z=6, 16×16)으로
저장되며 (x̂_t latent), 디코딩·물리단위 복원된 픽셀 필드(`ensemble_pixel`=x̂_t,
`x_t_true_pixel`=GT x_t)도 함께 들어 있다. `EnsembleSample`은 추가 픽셀 키를 optional
필드로 보유하고, npz에 없으면 `None`으로 둔다.

★v4: 캐시 픽셀은 ensemble_inference 가 climatology 로 이미 물리단위 복원해 저장하므로,
별도 역정규화(normalization_stats)는 불필요 — 관련 denorm helper 는 제거됨.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


VARIABLES = ("t", "u", "v")
# exp5의 픽셀 공간 composite map용 — 물리 변수 인덱스.
TEMP_IDX = 0
U_IDX = 1
V_IDX = 2
# latent 공간 분석에서 대표로 사용하는 latent 채널 (v1의 TEMP_IDX에 대응).
LATENT_CH = 0

# ── 픽셀 그리드 좌표 (data/era5_normalized.zarr) ──────────────────────
#   lat: 69 → 6 °N (index 0..63, step −1°),  lon: 95 → 158 °E (step +1°)
# 한반도 박스 (lat 43→33°N, lon 124→132°E) = 11×9 — exp7/aavg 공용.
KOREA_LAT_SLICE = slice(26, 37)
KOREA_LON_SLICE = slice(29, 38)
# Seoul (37.56°N, 127°E) 최근접 픽셀 = (lat_idx 31, lon_idx 32) 중심 3×3.
SEOUL_LAT_IDX = 31
SEOUL_LON_IDX = 32
SEOUL_LAT_SLICE = slice(SEOUL_LAT_IDX - 1, SEOUL_LAT_IDX + 2)   # 39,38,37 °N
SEOUL_LON_SLICE = slice(SEOUL_LON_IDX - 1, SEOUL_LON_IDX + 2)   # 126,127,128 °E


@dataclass
class EnsembleSample:
    """단일 시점의 latent ensemble 캐시 데이터.

    Attributes:
        ensemble:        (N, C_z, H_z, W_z) 정규화 latent ẑ_0 앙상블 (x̂_t, C_z=6)
        log_var:         (C_z, H_z, W_z) latent dual-head log_var (멤버 평균)
        x_t_true:        (C_z, H_z, W_z) GT x_t 의 latent 인코딩 (posterior μ)
        time_t:          numpy.datetime64 scalar
        path:            원본 npz 경로
        ensemble_pixel:  (N, C, H, W) 디코딩된 x̂_t 픽셀 앙상블 (없으면 None)
        x_t_true_pixel:  (C, H, W) GT x_t 픽셀 필드 (없으면 None)
        ensemble_pixel_tm1: (N, C, H, W) DDPM_past 생성 x̂_{t-1} 멤버 앙상블 (없으면 None)
        ensemble_pixel_tp1: (N, C, H, W) DDPM_past 생성 x̂_{t+1} 멤버 앙상블 (없으면 None)
        x_tm1_true_pixel: (C, H, W) GT x_{t-1} 픽셀 필드 (없으면 None)
        x_tp1_true_pixel: (C, H, W) GT x_{t+1} 픽셀 필드 (없으면 None)
    """
    ensemble: np.ndarray
    log_var: np.ndarray
    x_t_true: np.ndarray
    time_t: np.datetime64
    path: Path
    ensemble_pixel: Optional[np.ndarray] = None
    x_t_true_pixel: Optional[np.ndarray] = None
    ensemble_pixel_tm1: Optional[np.ndarray] = None
    ensemble_pixel_tp1: Optional[np.ndarray] = None
    x_tm1_true_pixel: Optional[np.ndarray] = None
    x_tp1_true_pixel: Optional[np.ndarray] = None


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

    def _opt(key: str) -> Optional[np.ndarray]:
        return data[key].astype(np.float32) if key in keys else None

    return EnsembleSample(
        ensemble=data["ensemble"].astype(np.float32),
        log_var=data["log_var"].astype(np.float32),
        x_t_true=data["x_t_true"].astype(np.float32),
        time_t=np.datetime64(str(data["time_t"])),
        path=path,
        ensemble_pixel=ensemble_pixel,
        x_t_true_pixel=x_t_true_pixel,
        ensemble_pixel_tm1=_opt("ensemble_pixel_tm1"),
        ensemble_pixel_tp1=_opt("ensemble_pixel_tp1"),
        x_tm1_true_pixel=_opt("x_tm1_true_pixel"),
        x_tp1_true_pixel=_opt("x_tp1_true_pixel"),
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
