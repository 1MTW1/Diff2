"""ERA5 raw → 시각별 픽셀별 z-score 정규화 후 zarr 저장."""
from __future__ import annotations

import shutil
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from tqdm import tqdm

from .config import (
    CHANNEL_NAMES, HOUR_TO_IDX, KOREA_LAT_CENTER, KOREA_LON_CENTER,
    NORMALIZED_PATH, PATCH, STATS_PATH, VARIABLES, WRITE_CHUNK,
    YEAR_MAX, YEAR_MIN,
)
from .utils import (
    decode_times, list_era5_files, load_chunk_from_zarr,
    resolve_patch_indices,
)


def normalize_dataset(
    stats_path: str = STATS_PATH,
    output_path: str = NORMALIZED_PATH,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
) -> None:
    """ERA5 raw 데이터를 시각별 픽셀별 정규화하여 zarr로 저장.

    출력: (T, C, H, W) float32, 시간 순으로 정렬
    """
    print(f"[1/5] Loading stats: {stats_path}")
    stats = xr.open_zarr(stats_path).load()
    mu_arr = stats["mean"].values.astype(np.float32)    # (4, C, H, W)
    sigma_arr = stats["std"].values.astype(np.float32)
    print(f"      mu shape: {mu_arr.shape}, sigma shape: {sigma_arr.shape}")

    print(f"[2/5] Listing ERA5 files [{year_min}-{year_max}]...")
    files = list_era5_files(year_min, year_max)
    print(f"      Found {len(files)} files")

    # 첫 파일에서 patch / lat_vals / lon_vals 결정 (저장용)
    lat_slice, lon_slice, lat_vals, lon_vals = resolve_patch_indices(files[0])

    # ── 1차 패스: 총 timestep 수와 timestep별 메타데이터 수집 ─────
    print("[3/5] Pass 1: Counting timesteps and gathering metadata...")
    file_times: list[np.ndarray] = []
    for fp in tqdm(files, desc="metadata"):
        root = zarr.open(fp, mode="r")
        times = decode_times(root)
        file_times.append(times)

    all_times = np.concatenate(file_times)
    # 중복 검출
    unique_times, counts = np.unique(all_times, return_counts=True)
    n_dup = int(np.sum(counts > 1))
    if n_dup > 0:
        print(f"      Warning: {n_dup} duplicate timesteps found "
              f"(will use unique-sorted)")
    all_times_sorted = np.sort(unique_times)
    n_total = len(all_times_sorted)
    print(f"      Total timesteps: {n_total}")

    C = len(VARIABLES)
    H = W = PATCH

    # ── 출력 zarr skeleton 사전 할당 ─────────────────────────────
    print(f"[4/5] Initializing output zarr skeleton: {output_path}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.rmtree(out)

    placeholder = da.zeros(
        (n_total, C, H, W),
        chunks=(WRITE_CHUNK, C, H, W),
        dtype=np.float32,
    )
    skeleton = xr.Dataset(
        {"normalized": (("time", "channel", "lat", "lon"), placeholder)},
        coords={
            "time": all_times_sorted,
            "channel": np.array(CHANNEL_NAMES),
            "lat": lat_vals.astype(np.float32),
            "lon": lon_vals.astype(np.float32),
        },
        attrs={
            "description": (
                "ERA5 hour-wise pixel-wise z-score normalized (Korea 64x64)"
            ),
            "stats_source": stats_path,
            "year_min": year_min,
            "year_max": year_max,
            "patch_center_lat": KOREA_LAT_CENTER,
            "patch_center_lon": KOREA_LON_CENTER,
        },
    )
    skeleton.to_zarr(out, compute=False)

    # ── 2차 패스: 데이터 로드 + 정규화 + region write ────────────
    print(f"[5/5] Normalizing and writing {n_total} samples...")

    # all_times_sorted에서 각 timestep의 위치 lookup
    time_to_pos = {t: i for i, t in enumerate(all_times_sorted)}

    # 검증용 누적
    running_sum = 0.0
    running_sq = 0.0
    running_n = 0

    for fp in tqdm(files, desc="normalize"):
        data, times = load_chunk_from_zarr(fp, VARIABLES, lat_slice, lon_slice)
        # data: (T_file, C, H, W), times: (T_file,)

        # 시각별로 통계 lookup + 정규화
        hours_arr = pd.DatetimeIndex(times).hour.to_numpy()
        h_indices = np.array([HOUR_TO_IDX[h] for h in hours_arr])

        # Fancy indexing: mu[h_indices] → (T_file, C, H, W)
        mu_chunk = mu_arr[h_indices]
        sigma_chunk = sigma_arr[h_indices]

        normalized = ((data - mu_chunk) / sigma_chunk).astype(np.float32)

        # 시간 순서로 sorted된 위치에 region write
        positions = np.array([time_to_pos[t] for t in times])

        # 연속된 위치 (월별 파일이므로 일반적으로 성립)
        is_contiguous = (
            len(positions) > 0 and bool(np.all(np.diff(positions) == 1))
        )

        if is_contiguous:
            start = int(positions[0])
            end = int(positions[-1]) + 1
            ds_chunk = xr.Dataset(
                {"normalized": (
                    ("time", "channel", "lat", "lon"), normalized
                )},
            )
            ds_chunk.to_zarr(out, region={"time": slice(start, end)})
        else:
            print(f"      Warning: non-contiguous file {Path(fp).name}, "
                  f"writing individually")
            for i, pos in enumerate(positions):
                ds_one = xr.Dataset(
                    {"normalized": (
                        ("time", "channel", "lat", "lon"),
                        normalized[i:i + 1],
                    )},
                )
                ds_one.to_zarr(
                    out, region={"time": slice(int(pos), int(pos) + 1)},
                )

        running_sum += float(normalized.sum())
        running_sq += float((normalized ** 2).sum())
        running_n += int(normalized.size)

    overall_mean = running_sum / running_n
    overall_var = running_sq / running_n - overall_mean ** 2
    overall_std = overall_var ** 0.5
    print(f"      Saved → {output_path}")
    print(f"      Overall normalized mean ≈ {overall_mean:.4f} (target: 0.0)")
    print(f"      Overall normalized std  ≈ {overall_std:.4f} (target: 1.0)")
    print("      Note: train period (2000-2019) mean/std should be very close")
    print("            to (0, 1). val/test period may slightly deviate.")


if __name__ == "__main__":
    normalize_dataset()
