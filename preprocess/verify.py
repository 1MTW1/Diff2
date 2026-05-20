"""전처리 결과 검증 스크립트.

§9 검증 통과 기준:
1. stats 파일 shape (4, 3, 64, 64), std ≥ STD_EPS
2. 정규화 데이터 shape (T, 3, 64, 64), train 기간 시각별 픽셀별 평균 ~0, std ~1
3. Diurnal cycle 가시성 (시각별 도메인 평균 변동)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .config import (
    HOUR_TO_IDX, NORMALIZED_PATH, PATCH, STATS_PATH, STATS_YEAR_MAX,
    STATS_YEAR_MIN, STD_EPS, VARIABLES,
)


def verify_stats(stats_path: str = STATS_PATH) -> None:
    print(f"\n=== Stats verification: {stats_path} ===")
    stats = xr.open_zarr(stats_path).load()

    expected_shape = (4, len(VARIABLES), PATCH, PATCH)
    assert stats["mean"].shape == expected_shape, (
        f"mean shape {stats['mean'].shape} != {expected_shape}"
    )
    assert stats["std"].shape == expected_shape, (
        f"std shape {stats['std'].shape} != {expected_shape}"
    )
    print(f"  mean shape: {stats['mean'].shape}  OK")
    print(f"  std  shape: {stats['std'].shape}   OK")
    print(f"  hour values: {stats.hour.values}")
    print(f"  channels:    {stats.channel.values}")

    min_std = float(stats["std"].min())
    assert min_std >= STD_EPS, f"min std {min_std} < STD_EPS {STD_EPS}"
    print(f"  min std = {min_std:.3e} (≥ STD_EPS {STD_EPS:.0e})  OK")

    for c_idx, c_name in enumerate(stats.channel.values):
        m = stats["mean"].isel(channel=c_idx).values
        s = stats["std"].isel(channel=c_idx).values
        print(f"  {c_name}: mean=[{m.min():.3f}, {m.max():.3f}], "
              f"std=[{s.min():.4f}, {s.max():.4f}]")

    # Diurnal cycle: temperature (channel 0)의 시각별 도메인 평균
    print("  Diurnal cycle check (temperature, domain-mean per hour):")
    for h in [0, 6, 12, 18]:
        h_idx = HOUR_TO_IDX[h]
        t_mean = float(stats["mean"].isel(hour=h_idx, channel=0).values.mean())
        print(f"    hour {h:02d} UTC: {t_mean:.2f} K")


def verify_normalized(
    norm_path: str = NORMALIZED_PATH,
    train_year_min: int = STATS_YEAR_MIN,
    train_year_max: int = STATS_YEAR_MAX,
    pixel_mean_tol: float = 0.01,
    pixel_std_tol: float = 0.05,
) -> None:
    print(f"\n=== Normalized verification: {norm_path} ===")
    norm = xr.open_zarr(norm_path)
    print(f"  shape: {norm['normalized'].shape}")
    print(f"  time range: {str(norm.time.values[0])[:19]} "
          f"~ {str(norm.time.values[-1])[:19]}")

    train = norm["normalized"].sel(
        time=slice(f"{train_year_min}-01-01", f"{train_year_max}-12-31")
    )
    overall_mean = float(train.mean())
    overall_std = float(train.std())
    print(f"  Train overall mean: {overall_mean:.4f} (target ~0.0)")
    print(f"  Train overall std:  {overall_std:.4f} (target ~1.0)")

    # 시각별 픽셀별 검증
    train_hours = pd.DatetimeIndex(train.time.values).hour.to_numpy()
    print("  Per-hour pixel-wise verification:")
    for h in [0, 6, 12, 18]:
        mask = (train_hours == h)
        if not mask.any():
            print(f"    hour {h:02d}: no samples")
            continue
        train_h = train.isel(time=np.where(mask)[0])
        px_mean_h = train_h.mean(dim="time").values  # (C, H, W)
        px_std_h = train_h.std(dim="time").values

        m_abs_max = float(np.abs(px_mean_h).max())
        s_dev_max = float(np.abs(px_std_h - 1.0).max())
        ok_m = m_abs_max < pixel_mean_tol
        ok_s = s_dev_max < pixel_std_tol
        flag = "OK" if (ok_m and ok_s) else "FAIL"
        print(
            f"    hour {h:02d}: "
            f"|mean|_max={m_abs_max:.4f} (tol {pixel_mean_tol}), "
            f"|std-1|_max={s_dev_max:.4f} (tol {pixel_std_tol})  {flag}"
        )


def main() -> None:
    verify_stats()
    verify_normalized()


if __name__ == "__main__":
    main()
