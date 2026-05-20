"""시각별 픽셀별 z-score 통계량 계산.

Welford's online algorithm (batch 버전) 으로 메모리 효율적으로
각 (hour, channel, lat, lon) 위치의 평균/표준편차를 계산한다.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from .config import (
    CHANNEL_NAMES, HOURS, HOUR_TO_IDX, PATCH, STATS_PATH,
    STATS_YEAR_MIN, STATS_YEAR_MAX, STD_EPS, VARIABLES,
)
from .utils import (
    is_leap_feb29, list_era5_files, load_chunk_from_zarr,
    resolve_patch_indices,
)


def _batch_welford_update(
    mean_acc: np.ndarray,
    m2_acc: np.ndarray,
    n_old: int,
    batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """배치 단위 Welford 업데이트.

    Args:
        mean_acc: (C, H, W) 누적 평균
        m2_acc:   (C, H, W) 누적 제곱편차 합
        n_old:    이전까지 누적된 표본 수
        batch:    (n_new, C, H, W) 새 배치

    Returns:
        new_mean, new_m2, n_total
    """
    n_new = batch.shape[0]
    if n_new == 0:
        return mean_acc, m2_acc, n_old
    n_total = n_old + n_new

    batch_mean = batch.mean(axis=0)
    batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)

    delta = batch_mean - mean_acc
    new_mean = (n_old * mean_acc + n_new * batch_mean) / n_total
    new_m2 = m2_acc + batch_m2 + (delta ** 2) * (n_old * n_new / n_total)
    return new_mean, new_m2, n_total


def compute_stats(
    output_path: str = STATS_PATH,
    year_min: int = STATS_YEAR_MIN,
    year_max: int = STATS_YEAR_MAX,
) -> xr.Dataset:
    """시각별 픽셀별 평균/표준편차 계산 후 zarr로 저장.

    Returns:
        xarray Dataset with mean, std (shape (4, C, H, W))
    """
    print(f"[1/4] Listing ERA5 files [{year_min}-{year_max}]...")
    files = list_era5_files(year_min, year_max)
    print(f"      Found {len(files)} files")

    # 한반도 patch 위치 결정
    lat_slice, lon_slice, lat_vals, lon_vals = resolve_patch_indices(files[0])
    print(f"      Patch: lat {lat_vals[0]:.1f}~{lat_vals[-1]:.1f}, "
          f"lon {lon_vals[0]:.1f}~{lon_vals[-1]:.1f}")

    n_hours = len(HOURS)
    C = len(VARIABLES)
    H = W = PATCH

    # 시각별 독립 누적기
    n_per_hour = np.zeros(n_hours, dtype=np.int64)
    mean_acc = np.zeros((n_hours, C, H, W), dtype=np.float64)
    m2_acc = np.zeros((n_hours, C, H, W), dtype=np.float64)

    print("[2/4] Accumulating statistics (batch Welford)...")
    n_leap_excluded = 0

    for fp in tqdm(files, desc="files"):
        data, times = load_chunk_from_zarr(fp, VARIABLES, lat_slice, lon_slice)

        # 윤년 2월 29일 제외
        leap_mask = is_leap_feb29(times)
        n_leap_excluded += int(leap_mask.sum())
        if leap_mask.any():
            data = data[~leap_mask]
            times = times[~leap_mask]

        hours_arr = pd.DatetimeIndex(times).hour.to_numpy()

        for h in HOURS:
            mask = (hours_arr == h)
            if not mask.any():
                continue
            chunk_h = data[mask].astype(np.float64, copy=False)  # (n_h, C, H, W)
            h_idx = HOUR_TO_IDX[h]

            new_mean, new_m2, n_total = _batch_welford_update(
                mean_acc[h_idx], m2_acc[h_idx], int(n_per_hour[h_idx]), chunk_h,
            )
            mean_acc[h_idx] = new_mean
            m2_acc[h_idx] = new_m2
            n_per_hour[h_idx] = n_total

    print(f"      Excluded {n_leap_excluded} leap day (Feb 29) timesteps")
    print(f"      Samples per hour: {dict(zip(HOURS, n_per_hour.tolist()))}")

    # ── 분산 → 표준편차 ───────────────────────────────────────────
    print("[3/4] Computing std from accumulated M2...")
    if (n_per_hour == 0).any():
        raise RuntimeError(
            f"No samples accumulated for some hour: {n_per_hour.tolist()}"
        )
    variance = m2_acc / n_per_hour[:, None, None, None]
    std = np.sqrt(variance).astype(np.float32)
    mean = mean_acc.astype(np.float32)

    # σ 안정화
    std = np.where(std < STD_EPS, STD_EPS, std).astype(np.float32)

    # ── 저장 ──────────────────────────────────────────────────────
    print(f"[4/4] Saving to {output_path}...")
    coords = {
        "hour": np.array(HOURS, dtype=np.int64),
        "channel": np.array(CHANNEL_NAMES),
        "lat": lat_vals.astype(np.float32),
        "lon": lon_vals.astype(np.float32),
    }
    stats = xr.Dataset(
        {
            "mean": (("hour", "channel", "lat", "lon"), mean),
            "std":  (("hour", "channel", "lat", "lon"), std),
        },
        coords=coords,
        attrs={
            "description": (
                "Hour-wise pixel-wise z-score normalization stats for ERA5 "
                "(Korea 64x64)"
            ),
            "stats_year_min": year_min,
            "stats_year_max": year_max,
            "n_files": len(files),
            "n_leap_excluded": n_leap_excluded,
            "std_eps": STD_EPS,
        },
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.rmtree(out)
    stats.to_zarr(out)

    print(f"      mean shape: {tuple(stats['mean'].shape)}")
    print(f"      std shape:  {tuple(stats['std'].shape)}")
    print(f"      mean range: [{float(stats['mean'].min()):.3f}, "
          f"{float(stats['mean'].max()):.3f}]")
    print(f"      std range:  [{float(stats['std'].min()):.3f}, "
          f"{float(stats['std'].max()):.3f}]")

    return stats


if __name__ == "__main__":
    compute_stats()
