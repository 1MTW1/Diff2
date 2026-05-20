"""Preprocessing 공통 유틸리티."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from .config import (
    DATASET_GLOB, KOREA_LAT_CENTER, KOREA_LON_CENTER, PATCH,
)


def resolve_patch_indices(zarr_path: str) -> tuple[slice, slice, np.ndarray, np.ndarray]:
    """첫 zarr 파일에서 한반도 중심 64×64 슬라이스 인덱스 결정.

    Returns:
        lat_slice, lon_slice: numpy slice 객체
        lat_vals, lon_vals: 슬라이스에 해당하는 위경도 값 (저장용)
    """
    root = zarr.open(zarr_path, mode="r")
    lats = root["latitude"][:]
    lons = root["longitude"][:]
    half = PATCH // 2
    lat_c = int(np.argmin(np.abs(lats - KOREA_LAT_CENTER)))
    lon_c = int(np.argmin(np.abs(lons - KOREA_LON_CENTER)))
    lat_slice = slice(lat_c - half, lat_c + half)
    lon_slice = slice(lon_c - half, lon_c + half)
    return lat_slice, lon_slice, lats[lat_slice], lons[lon_slice]


def decode_times(root) -> np.ndarray:
    """zarr time variable을 datetime64로 디코딩.

    units: 'hours since YYYY-MM-DD HH:MM:SS'
    """
    units: str = root["time"].attrs["units"]
    base_str = units.split("since")[1].strip()
    base = pd.Timestamp(base_str)
    hours = root["time"][:].astype("int64")
    return (base + pd.to_timedelta(hours, unit="h")).to_numpy()


def filter_files_by_year(
    files: list[str], year_min: int, year_max: int
) -> list[str]:
    """파일명에서 연도 파싱하여 [year_min, year_max] 범위만 필터링.

    파일명 형식: era5_6h_1lv_*_YYYY_MM.zarr
    """
    keep = []
    for fp in files:
        try:
            year = int(Path(fp).name.split("_")[-2])
        except (ValueError, IndexError):
            continue
        if year_min <= year <= year_max:
            keep.append(fp)
    return keep


def list_era5_files(year_min: int, year_max: int) -> list[str]:
    """주어진 연도 범위에 해당하는 raw zarr 파일 목록 반환."""
    files = sorted(glob.glob(DATASET_GLOB))
    files = filter_files_by_year(files, year_min, year_max)
    if not files:
        raise FileNotFoundError(
            f"No files matched {DATASET_GLOB} in [{year_min}, {year_max}]"
        )
    return files


def load_chunk_from_zarr(
    zarr_path: str,
    variables: list[str],
    lat_slice: slice,
    lon_slice: slice,
) -> tuple[np.ndarray, np.ndarray]:
    """단일 zarr 파일에서 한반도 patch + 변수 추출.

    Returns:
        data: (T, C, H, W) float32 - NaN은 0으로 대체
        times: (T,) datetime64
    """
    root = zarr.open(zarr_path, mode="r")
    chans = []
    for v in variables:
        # (T, level=1, H_patch, W_patch) → squeeze level → (T, H, W)
        arr = root[v][:, :, lat_slice, lon_slice].astype(np.float32)
        chans.append(arr.squeeze(axis=1))
    data = np.stack(chans, axis=1)            # (T, C, H, W)
    np.nan_to_num(data, copy=False, nan=0.0)
    times = decode_times(root)
    return data, times


def is_leap_feb29(times: np.ndarray) -> np.ndarray:
    """각 시점이 윤년 2월 29일인지 boolean mask 반환."""
    idx = pd.DatetimeIndex(np.asarray(times))
    return np.asarray((idx.month == 2) & (idx.day == 29), dtype=bool)
