# Data Preprocessing 구현 지침

본 문서는 Diffusion²-based ensemble generation 모델 학습을 위한 ERA5 reanalysis 데이터 전처리 코드 구현 지침이다. 본 전처리는 **시각별 픽셀별 z-score 정규화 (hour-wise, pixel-wise standardization)** 방식을 사용한다.

---

## 1. 개요

### 1.1 전처리 목표

ERA5 raw 데이터를 모델 학습에 적합한 형태로 변환한다.

1. **공간 cropping**: 전 지구 데이터 → 한반도 중심 64×64 patch
2. **시각별 픽셀별 z-score 정규화**: VP diffusion의 N(0, I) 가정을 만족시키되, 시각별 diurnal cycle도 반영

### 1.2 정규화 방식

각 (hour, channel, lat, lon) 위치마다 독립적인 (μ, σ) 사용:

$$\tilde{x}_{t, c, i, j} = \frac{x_{t, c, i, j} - \mu_{h(t), c, i, j}}{\sigma_{h(t), c, i, j}}$$

여기서:
- h(t): 시점 t의 hour (0, 6, 12, 18 UTC 중 하나)
- μ, σ: 학습 기간(2000-2019)에서 계산된 통계량
- 통계량 shape: **(4, 3, 64, 64)** = (hour, channel, lat, lon)

이 방식의 의도:
- **VP diffusion의 N(0, I) 가정 충족**: 각 (h, c, i, j) 위치에서 정규화 후 평균 0, 분산 1
- **Diurnal cycle 반영**: 0/6/12/18 UTC가 다른 통계 가짐 (예: 낮 vs 밤 기온)
- **Spatial structure 보존**: 적도/극지의 변동성 차이가 통계에 반영

### 1.3 데이터 소스

- **원본 위치**: `/geodata2/MuLANWR/era5_6h_1lv_*.zarr` (월별 zarr 파일)
- **시간 해상도**: 6시간 (0/6/12/18 UTC)
- **공간 해상도**: 전 지구 1° (위경도)
- **변수**: temperature, u_component_of_wind, v_component_of_wind (3 channels)
- **압력층**: 단일 level (level dimension squeeze)

### 1.4 출력

```
data/
├── normalization_stats.zarr  # μ, σ 통계량 (4, 3, 64, 64)
└── era5_normalized.zarr      # 사전 정규화된 데이터 (T, 3, 64, 64)
```

**중요**: `data/era5_raw.zarr` 파일은 별도로 저장하지 않는다. 정규화 과정에서 직접 raw zarr 파일들을 읽어 처리하고 정규화 결과만 저장한다.

---

## 2. 프로젝트 구조

```
preprocess/
├── __init__.py
├── config.py              # 설정값 (도메인, 연도 등)
├── compute_stats.py       # 통계량 계산
├── normalize_dataset.py   # 정규화 + 저장
└── utils.py               # 공통 유틸 (path resolve, time decode 등)

data/
├── normalization_stats.zarr   # 생성됨
└── era5_normalized.zarr       # 생성됨
```

실행 순서:
1. `python -m preprocess.compute_stats` → `data/normalization_stats.zarr` 생성
2. `python -m preprocess.normalize_dataset` → `data/era5_normalized.zarr` 생성

---

## 3. 설정 (`preprocess/config.py`)

```python
"""Data preprocessing 공통 설정."""
from __future__ import annotations

# ── 입력 ─────────────────────────────────────────────────────────
DATASET_GLOB = "/geodata2/MuLANWR/era5_6h_1lv_*.zarr"

# ── 도메인 ───────────────────────────────────────────────────────
# 한반도 중심 64×64 patch (이전 연구와 동일)
KOREA_LAT_CENTER = 37.0
KOREA_LON_CENTER = 127.5
PATCH = 64

# ── 변수 ─────────────────────────────────────────────────────────
VARIABLES = ["temperature", "u_component_of_wind", "v_component_of_wind"]
CHANNEL_NAMES = ["t", "u", "v"]

# ── 연도 ─────────────────────────────────────────────────────────
YEAR_MIN = 2000     # 데이터 가용 시작
YEAR_MAX = 2022     # 데이터 가용 끝

# 통계량 계산은 train split만 사용 (leakage 방지)
STATS_YEAR_MIN = 2000
STATS_YEAR_MAX = 2019

# ── 시각 ─────────────────────────────────────────────────────────
HOURS = [0, 6, 12, 18]   # 6h 데이터의 4개 시각
HOUR_TO_IDX = {0: 0, 6: 1, 12: 2, 18: 3}

# ── Leap day (DoY 60 = Feb 29) ────────────────────────────────────
# 시각별 통계에서도 윤년 처리: Feb 29의 통계는 Feb 28과 Mar 1의 평균으로 대체
# (climatology.py와 동일한 정신)
LEAP_DATE = (2, 29)

# ── 출력 ─────────────────────────────────────────────────────────
STATS_PATH = "data/normalization_stats.zarr"
NORMALIZED_PATH = "data/era5_normalized.zarr"

# ── I/O ──────────────────────────────────────────────────────────
LOAD_CHUNK = 512       # 시간축 청크 크기 (RAM 부하 조절)
WRITE_CHUNK = 1024     # zarr 저장 청크 크기

# ── Numerical ────────────────────────────────────────────────────
STD_EPS = 1e-6         # σ가 0에 가까울 때 안정화
```

---

## 4. 공통 유틸 (`preprocess/utils.py`)

```python
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
    return ((idx.month == 2) & (idx.day == 29)).to_numpy()
```

---

## 5. 통계량 계산 (`preprocess/compute_stats.py`)

### 5.1 알고리즘

**입력**: ERA5 raw zarr 파일들 (2000-2019, train split)

**출력**: `data/normalization_stats.zarr`
- `mean`: (4, 3, 64, 64) float32 - 시각×변수×위경도별 평균
- `std`: (4, 3, 64, 64) float32 - 시각×변수×위경도별 표준편차

**알고리즘**:

1. **데이터 로드 (chunked)**:
   - 2000-2019 모든 zarr 파일을 순회
   - 각 파일에서 한반도 patch 추출 + NaN→0
   - 시간/시각 정보 저장

2. **시각별 분리**:
   - 각 timestep의 hour를 추출
   - 0, 6, 12, 18 UTC에 해당하는 timestep 그룹으로 분리

3. **윤년 처리 (Feb 29 제외)**:
   - 통계 계산에서 Feb 29 timestep은 제외
   - 마지막에 Feb 28과 Mar 1의 통계 평균으로 Feb 29 통계를 만들지는 않음 (시각별 정규화에서는 매 시각마다 통계량이 하나만 있으므로, Feb 29만의 별도 통계가 필요 없음)
   - **단순 처리**: Feb 29 timestep은 통계 계산에서 빠지지만, 추론/학습 시 Feb 29 데이터에는 (그 시각의 평균적인) 통계를 적용

4. **픽셀별 평균/표준편차 계산 (시각별)**:
   - 각 hour h에 대해:
     - `mu[h, c, i, j] = mean over time of data[t in hour h, c, i, j]`
     - `sigma[h, c, i, j] = std over time of data[t in hour h, c, i, j]`
   - 메모리 효율을 위해 Welford's online algorithm 또는 chunked 계산 사용

5. **σ 안정화**:
   - σ가 STD_EPS(=1e-6) 미만이면 STD_EPS로 clamp
   - Numerical 불안정 방지

6. **저장**:
   - `data/normalization_stats.zarr`에 xarray Dataset으로 저장
   - Coords: hour (4,), channel (3,), lat (64,), lon (64,)

### 5.2 구현 코드

```python
"""시각별 픽셀별 z-score 통계량 계산."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from .config import (
    CHANNEL_NAMES, HOURS, HOUR_TO_IDX, LOAD_CHUNK, PATCH, STATS_PATH,
    STATS_YEAR_MIN, STATS_YEAR_MAX, STD_EPS, VARIABLES,
)
from .utils import (
    is_leap_feb29, list_era5_files, load_chunk_from_zarr,
    resolve_patch_indices,
)


def compute_stats(
    output_path: str = STATS_PATH,
    year_min: int = STATS_YEAR_MIN,
    year_max: int = STATS_YEAR_MAX,
):
    """시각별 픽셀별 평균/표준편차 계산.
    
    Welford's online algorithm으로 메모리 효율적으로 계산.
    
    Returns:
        xarray Dataset with mean, std
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
    
    # ── Welford's online algorithm (시각별로 독립적으로 누적) ─────
    # M_k = mean after k samples
    # S_k = sum of squared deviations after k samples
    # 수식: 
    #   delta = x_k - M_{k-1}
    #   M_k = M_{k-1} + delta / k
    #   delta2 = x_k - M_k
    #   S_k = S_{k-1} + delta * delta2
    # Final: var = S_n / n, std = sqrt(var)
    
    n_per_hour = np.zeros(n_hours, dtype=np.int64)
    mean_acc = np.zeros((n_hours, C, H, W), dtype=np.float64)
    m2_acc = np.zeros((n_hours, C, H, W), dtype=np.float64)
    
    print(f"[2/4] Accumulating statistics (Welford's algorithm)...")
    n_leap_excluded = 0
    
    for fp in tqdm(files, desc="files"):
        data, times = load_chunk_from_zarr(fp, VARIABLES, lat_slice, lon_slice)
        
        # 윤년 2월 29일 제외
        leap_mask = is_leap_feb29(times)
        n_leap_excluded += int(leap_mask.sum())
        data = data[~leap_mask]
        times = times[~leap_mask]
        
        # 각 timestep의 hour 추출
        hours_arr = pd.DatetimeIndex(times).hour.to_numpy()
        
        # 시각별로 통계 누적
        for h in HOURS:
            mask = (hours_arr == h)
            if not mask.any():
                continue
            chunk_h = data[mask]                # (n_h, C, H, W)
            h_idx = HOUR_TO_IDX[h]
            
            # Welford's algorithm for batch
            for sample in chunk_h:
                n_per_hour[h_idx] += 1
                delta = sample - mean_acc[h_idx]
                mean_acc[h_idx] += delta / n_per_hour[h_idx]
                delta2 = sample - mean_acc[h_idx]
                m2_acc[h_idx] += delta * delta2
    
    print(f"      Excluded {n_leap_excluded} leap day (Feb 29) timesteps")
    print(f"      Samples per hour: {dict(zip(HOURS, n_per_hour.tolist()))}")
    
    # ── 분산 → 표준편차 ───────────────────────────────────────────
    print(f"[3/4] Computing std from accumulated M2...")
    variance = m2_acc / n_per_hour[:, None, None, None]
    std = np.sqrt(variance).astype(np.float32)
    mean = mean_acc.astype(np.float32)
    
    # σ 안정화
    std = np.where(std < STD_EPS, STD_EPS, std)
    
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
            "description": "Hour-wise pixel-wise z-score normalization stats for ERA5 (Korea 64x64)",
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
        import shutil
        shutil.rmtree(out)
    stats.to_zarr(out)
    
    # ── 검증 ──────────────────────────────────────────────────────
    print(f"      mean shape: {tuple(stats['mean'].shape)}")
    print(f"      std shape:  {tuple(stats['std'].shape)}")
    print(f"      mean range: [{float(stats['mean'].min()):.3f}, "
          f"{float(stats['mean'].max()):.3f}]")
    print(f"      std range:  [{float(stats['std'].min()):.3f}, "
          f"{float(stats['std'].max()):.3f}]")
    
    return stats


if __name__ == "__main__":
    compute_stats()
```

### 5.3 메모리 / 시간 고려

- **Welford's algorithm**: 한 timestep씩 누적하므로 RAM 사용량이 매우 적음 (통계 누적 array만 유지)
- **단점**: Python loop라 느릴 수 있음
- **최적화 가능**: chunk 단위 batch update로 가능하나, 본 task의 데이터 크기에서는 단순한 sample-wise loop로 충분
- **예상 시간**: 20년치 × 4 timestep/day × 365 day ≈ 29,200 timesteps. 각 timestep 처리는 단순 numpy 연산이므로 수 분 내 완료 예상

만약 시간이 너무 오래 걸린다면 다음 최적화 적용 가능:

```python
# Batch update version (시각 그룹 전체를 한 번에 업데이트)
for h in HOURS:
    mask = (hours_arr == h)
    if not mask.any():
        continue
    chunk_h = data[mask]                # (n_h, C, H, W)
    h_idx = HOUR_TO_IDX[h]
    
    # Batch Welford
    n_old = n_per_hour[h_idx]
    n_new = len(chunk_h)
    n_total = n_old + n_new
    
    batch_mean = chunk_h.mean(axis=0)
    batch_m2 = ((chunk_h - batch_mean) ** 2).sum(axis=0)
    
    delta = batch_mean - mean_acc[h_idx]
    new_mean = (n_old * mean_acc[h_idx] + n_new * batch_mean) / n_total
    new_m2 = m2_acc[h_idx] + batch_m2 + delta**2 * n_old * n_new / n_total
    
    mean_acc[h_idx] = new_mean
    m2_acc[h_idx] = new_m2
    n_per_hour[h_idx] = n_total
```

---

## 6. 정규화 데이터 저장 (`preprocess/normalize_dataset.py`)

### 6.1 알고리즘

**입력**:
- ERA5 raw zarr 파일들 (2000-2022, 전체 기간)
- `data/normalization_stats.zarr` (통계량)

**출력**: `data/era5_normalized.zarr`
- `normalized`: (T, 3, 64, 64) float32 - 정규화된 데이터
- Coords: time, channel, lat, lon

**알고리즘**:

1. **통계량 로드**:
   - `data/normalization_stats.zarr` → in-memory numpy arrays

2. **출력 zarr skeleton 사전 할당**:
   - 메타데이터만 먼저 저장 (compute=False)
   - 이후 region write로 chunk별 dump

3. **chunked 처리 (전체 기간)**:
   - 각 zarr 파일을 순회
   - 각 timestep의 hour → 통계량 lookup
   - 정규화 적용: (x - μ_h) / σ_h
   - 결과를 region write

4. **윤년 처리**:
   - **단순화**: 윤년 2월 29일 데이터도 정상 처리. 통계량이 Feb 29만의 것이 아니라 "그 시각의 평균적 통계"이므로 적용 가능.
   - 이는 climatology(DoY-based)와 다른 점. 시각별 통계는 monthday와 무관.

5. **검증**:
   - 정규화 후 전체 평균/분산이 ~ (0, 1)인지 확인

### 6.2 구현 코드

```python
"""ERA5 raw → 시각별 픽셀별 z-score 정규화 후 zarr 저장."""
from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from .config import (
    CHANNEL_NAMES, HOUR_TO_IDX, NORMALIZED_PATH, PATCH,
    STATS_PATH, VARIABLES, WRITE_CHUNK, YEAR_MAX, YEAR_MIN,
)
from .utils import (
    list_era5_files, load_chunk_from_zarr, resolve_patch_indices,
)


def normalize_dataset(
    stats_path: str = STATS_PATH,
    output_path: str = NORMALIZED_PATH,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
):
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
    # zarr 사전 할당을 위해 미리 알아야 함
    print(f"[3/5] Pass 1: Counting timesteps and gathering metadata...")
    file_times: list[np.ndarray] = []
    file_n: list[int] = []
    for fp in tqdm(files, desc="metadata"):
        # time만 읽어서 빠르게 카운트
        import zarr
        root = zarr.open(fp, mode="r")
        from .utils import decode_times
        times = decode_times(root)
        file_times.append(times)
        file_n.append(len(times))
    
    all_times = np.concatenate(file_times)
    order = np.argsort(all_times, kind="stable")
    all_times_sorted = all_times[order]
    n_total = len(all_times_sorted)
    print(f"      Total timesteps: {n_total}")
    
    C = len(VARIABLES)
    H = W = PATCH
    
    # ── 2차 패스: 데이터 로드 + 정규화 + region write ─────────────
    print(f"[4/5] Initializing output zarr skeleton: {output_path}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        import shutil
        shutil.rmtree(out)
    
    # dask zeros로 skeleton 만들기 (compute=False)
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
            "description": "ERA5 hour-wise pixel-wise z-score normalized (Korea 64x64)",
            "stats_source": stats_path,
            "year_min": year_min,
            "year_max": year_max,
            "patch_center_lat": 37.0,
            "patch_center_lon": 127.5,
        },
    )
    skeleton.to_zarr(out, compute=False)
    
    # ── 파일별로 로드 → 정규화 → 정렬된 위치에 region write ───────
    print(f"[5/5] Normalizing and writing {n_total} samples...")
    
    # all_times_sorted에서 각 timestep의 위치 lookup
    time_to_pos = {t: i for i, t in enumerate(all_times_sorted)}
    
    # 통계 누적 (검증용)
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
        # times가 파일 내에서는 정렬되어 있지만, 전체에서 정렬된 위치 찾기 필요
        positions = np.array([time_to_pos[t] for t in times])
        
        # 연속된 위치인지 확인 (파일 내 모든 timestep이 sorted 순서에서 인접한지)
        # ERA5 월별 파일이므로 일반적으로 연속됨
        is_contiguous = np.all(np.diff(positions) == 1)
        
        if is_contiguous:
            # 단순 region write
            start = int(positions[0])
            end = int(positions[-1]) + 1
            ds_chunk = xr.Dataset(
                {"normalized": (("time", "channel", "lat", "lon"), normalized)},
            )
            ds_chunk.to_zarr(out, region={"time": slice(start, end)})
        else:
            # Non-contiguous: 개별 timestep씩 write (예외적 케이스)
            print(f"      Warning: non-contiguous file {Path(fp).name}, "
                  f"writing individually")
            for i, pos in enumerate(positions):
                ds_one = xr.Dataset(
                    {"normalized": (
                        ("time", "channel", "lat", "lon"),
                        normalized[i:i+1],
                    )},
                )
                ds_one.to_zarr(out, region={"time": slice(int(pos), int(pos) + 1)})
        
        # 검증용 누적
        running_sum += float(normalized.sum())
        running_sq += float((normalized ** 2).sum())
        running_n += normalized.size
    
    # ── 검증 ──────────────────────────────────────────────────────
    overall_mean = running_sum / running_n
    overall_var = running_sq / running_n - overall_mean ** 2
    overall_std = overall_var ** 0.5
    print(f"      Saved → {output_path}")
    print(f"      Overall normalized mean ≈ {overall_mean:.4f} (target: 0.0)")
    print(f"      Overall normalized std  ≈ {overall_std:.4f} (target: 1.0)")
    print(f"      Note: train period (2000-2019) mean/std should be very close")
    print(f"            to (0, 1). val/test period may slightly deviate.")


if __name__ == "__main__":
    normalize_dataset()
```

### 6.3 메모리 / 시간 고려

- 각 zarr 파일 (월별, ~120 timesteps)을 메모리에 한 번에 로드
- 64×64 patch이므로 메모리 부담 적음 (월별 ~3.7 MB)
- 정규화는 numpy broadcasting으로 빠르게 수행
- region write로 zarr에 직접 dump하여 메모리 효율적

---

## 7. 정규화 적용/역적용 함수 (선택적, `preprocess/normalize_ops.py`)

학습 시 사전 저장된 데이터를 그대로 사용한다면 필요 없지만, on-the-fly 정규화가 필요하거나 추론 결과를 denormalize할 때 사용할 수 있는 유틸리티 함수.

```python
"""정규화 적용/역적용 함수 (학습/추론 시 사용)."""
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


def normalize_torch(
    x: torch.Tensor,
    times: np.ndarray,  # datetime64 array, shape (B,) or scalar
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Torch tensor에 정규화 적용.
    
    Args:
        x: (B, C, H, W) or (C, H, W) raw data
        times: datetime64 array (B,) or scalar
        mean, std: (4, C, H, W) stats
    
    Returns:
        normalized: same shape as x
    """
    if x.ndim == 3:
        # Single sample
        hour = pd.Timestamp(times).hour
        h_idx = HOUR_TO_IDX[hour]
        return (x - mean[h_idx]) / std[h_idx]
    
    # Batch
    hours = pd.DatetimeIndex(times).hour.to_numpy()
    h_indices = torch.tensor(
        [HOUR_TO_IDX[h] for h in hours],
        dtype=torch.long,
        device=x.device,
    )
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
    
    hours = pd.DatetimeIndex(times).hour.to_numpy()
    h_indices = torch.tensor(
        [HOUR_TO_IDX[h] for h in hours],
        dtype=torch.long,
        device=x_norm.device,
    )
    mu_batch = mean[h_indices]
    sigma_batch = std[h_indices]
    return x_norm * sigma_batch + mu_batch
```

---

## 8. 학습 시 사용 패턴

본 task의 학습/추론에서는 **사전 정규화된 데이터(`data/era5_normalized.zarr`)를 직접 로드**한다. 추가 정규화 연산 없이 그대로 모델에 입력.

5 시점 (x_{t-3}, x_{t-2}, x_{t-1}, x_t, x_{t+1})에는 동일한 정규화가 이미 적용되어 있으므로, 별도 처리 없이 시점별 인덱싱만 하면 된다.

Dataset 클래스 (학습 시 사용)는 별도 문서에서 구현하지만, 핵심 로직은 다음과 같다:

```python
import xarray as xr

class NormalizedERA5Dataset:
    def __init__(self, normalized_path: str = "data/era5_normalized.zarr"):
        self.ds = xr.open_zarr(normalized_path)
        # time, channel, lat, lon coords 사용 가능
    
    def __getitem__(self, idx):
        # 5 시점 슬라이싱: idx-3, idx-2, idx-1, idx, idx+1
        chunk = self.ds["normalized"].isel(
            time=slice(idx - 3, idx + 2)
        ).values  # (5, C, H, W)
        return {
            "x_tm3": torch.from_numpy(chunk[0]),
            "x_tm2": torch.from_numpy(chunk[1]),
            "x_tm1": torch.from_numpy(chunk[2]),
            "x_t":   torch.from_numpy(chunk[3]),
            "x_tp1": torch.from_numpy(chunk[4]),
        }
```

---

## 9. 검증 단계

전처리 완료 후 다음 검증을 반드시 수행해야 한다.

### 9.1 통계량 검증

```python
import xarray as xr
import numpy as np

stats = xr.open_zarr("data/normalization_stats.zarr")
print(f"mean shape: {stats['mean'].shape}")          # (4, 3, 64, 64)
print(f"std shape:  {stats['std'].shape}")           # (4, 3, 64, 64)
print(f"hour values: {stats.hour.values}")           # [0, 6, 12, 18]
print(f"channels: {stats.channel.values}")           # ['t', 'u', 'v']

# 변수별 mean/std의 전체 분포
for c_idx, c_name in enumerate(stats.channel.values):
    m = stats["mean"].isel(channel=c_idx).values
    s = stats["std"].isel(channel=c_idx).values
    print(f"  {c_name}: mean range=[{m.min():.3f}, {m.max():.3f}], "
          f"std range=[{s.min():.4f}, {s.max():.4f}]")

# 시각별 변화 (diurnal cycle 확인)
# 예: temperature의 시각별 평균 (도메인 평균)
for h in [0, 6, 12, 18]:
    h_idx = {0: 0, 6: 1, 12: 2, 18: 3}[h]
    t_mean = stats["mean"].isel(hour=h_idx, channel=0).values.mean()
    print(f"  hour {h:02d}: domain-mean temp = {t_mean:.2f}K")
```

### 9.2 정규화 결과 검증

```python
norm = xr.open_zarr("data/era5_normalized.zarr")

# Train period (2000-2019)에서 통계 확인
train = norm["normalized"].sel(time=slice("2000-01-01", "2019-12-31"))

# Train 전체 평균/분산: ~ (0, 1)
print(f"Train overall mean: {float(train.mean()):.4f} (target ~0.0)")
print(f"Train overall std:  {float(train.std()):.4f}  (target ~1.0)")

# 픽셀별 검증 (시간 평균 후)
pixel_mean = train.mean(dim="time").values  # (3, 64, 64)
pixel_std = train.std(dim="time").values

# 각 (hour, channel, lat, lon) 위치에서 정규화가 정확한지는
# 시각별로 분리해서 확인해야 함
print(f"Pixel mean range (all hours mixed): [{pixel_mean.min():.4f}, {pixel_mean.max():.4f}]")
print(f"Pixel std range (all hours mixed):  [{pixel_std.min():.4f}, {pixel_std.max():.4f}]")

# 시각별 픽셀별 검증
import pandas as pd
for h in [0, 6, 12, 18]:
    train_h = train.sel(time=pd.DatetimeIndex(train.time.values).hour == h)
    px_mean_h = train_h.mean(dim="time").values
    px_std_h = train_h.std(dim="time").values
    print(f"  hour {h:02d}: pixel_mean=[{px_mean_h.min():.4f}, {px_mean_h.max():.4f}], "
          f"pixel_std=[{px_std_h.min():.4f}, {px_std_h.max():.4f}]")
    # 각 시각별로 분리하면 픽셀 평균 ~0, 픽셀 std ~1이어야 함
```

### 9.3 검증 통과 기준

다음 조건을 모두 만족해야 한다:

1. **통계량 파일**:
   - `data/normalization_stats.zarr` 존재
   - shape: (4, 3, 64, 64) for both mean and std
   - `std`의 모든 값이 STD_EPS 이상

2. **정규화 데이터**:
   - `data/era5_normalized.zarr` 존재
   - shape: (T, 3, 64, 64), T ≈ 33,600 (2000-2022, 6시간 간격)
   - Train period (2000-2019)에서:
     - 시각별 픽셀별 평균: |mean| < 0.01
     - 시각별 픽셀별 표준편차: |std - 1| < 0.05

3. **시각 분리 효과**:
   - 변수 0 (temperature)의 시각별 평균이 일변동을 보임 (낮 vs 밤 차이)
   - 만약 시각별 평균이 거의 같다면 시각 분리가 의미 없거나 코드에 버그

---

## 10. 실행 절차

전처리 전체 절차:

```bash
# 1. 통계량 계산 (한 번만 실행, 약 수 분 ~ 십수 분 예상)
python -m preprocess.compute_stats

# 2. 정규화 데이터 저장 (한 번만 실행, 약 수 분 ~ 십수 분 예상)
python -m preprocess.normalize_dataset

# 3. (선택) 검증 스크립트 실행
python -m preprocess.verify_normalization
```

검증 스크립트는 필요시 별도로 만들 수 있다. 위 §9의 코드를 모아서 `preprocess/verify.py`로 만들면 된다.

---

## 11. 잠재적 문제 및 대응

### 11.1 메모리 부족

**증상**: 통계 계산 또는 정규화 중 OOM

**원인**: 
- 단일 파일이 너무 큰 경우 (월별 ~120 timesteps × 3 channels × 64×64 = ~3.7 MB이므로 일반적으로 안전)
- numpy float64 누적 array의 크기 (4 × 3 × 64 × 64 × 8 bytes = ~393 KB, 안전)

**대응**: 사실상 64×64 patch에서는 메모리 문제 발생 가능성 낮음. 만약 발생 시 LOAD_CHUNK를 줄임.

### 11.2 NaN 값 처리

**증상**: 정규화 결과에 NaN 발생

**원인**:
- 원본 데이터에 NaN이 있고 `np.nan_to_num`이 적용되지 않은 경우
- σ가 정확히 0인 위치 (STD_EPS clipping이 작동하지 않은 경우)

**대응**: 
- `load_chunk_from_zarr`에서 NaN→0 처리 확인
- `compute_stats`에서 `std = np.where(std < STD_EPS, STD_EPS, std)` 확인

### 11.3 시간 정렬 문제

**증상**: `data/era5_normalized.zarr`의 timestep이 누락되거나 중복

**원인**:
- 월별 zarr 파일 간 time 중복
- 정렬 순서 버그

**대응**:
- 1차 패스에서 모든 time을 수집한 후 unique + sort
- 중복 검출 로직 추가:
  ```python
  unique_times, counts = np.unique(all_times, return_counts=True)
  if np.any(counts > 1):
      print(f"Warning: {np.sum(counts > 1)} duplicate timesteps found")
  ```

### 11.4 윤년 데이터의 통계량

**의도된 동작**: Feb 29는 통계 계산에서 제외되지만, 정규화 적용 시에는 (그 시각의 일반적인) 통계량을 사용.

**예상 효과**: Feb 29 데이터의 정규화 결과가 약간 biased될 수 있으나, 4년에 하루이므로 무시할 수 있는 수준.

만약 더 정확한 처리를 원한다면, climatology.py처럼 Feb 28과 Mar 1의 평균으로 Feb 29 통계를 만드는 로직 추가 가능. 다만 시각별 통계의 경우 monthday와 무관한 글로벌 통계라서 이 처리가 의미가 적음.

---

## 12. 요약

본 전처리 파이프라인은 다음을 수행한다:

1. **공간 cropping**: 전 지구 → 한반도 64×64
2. **시각별 픽셀별 통계 계산**: (4, 3, 64, 64) shape의 μ, σ
3. **z-score 정규화**: x → (x - μ_h) / σ_h
4. **사전 저장**: 학습 시 추가 연산 없이 직접 사용 가능

핵심 디자인 결정:

| 결정사항 | 선택 | 이유 |
|---------|------|------|
| 통계량 spatial 구조 | 픽셀별 (H, W) | VP diffusion의 N(0, I) 가정 충족 |
| 시각 분리 | 4 hours (0/6/12/18) | Diurnal cycle 반영 |
| 통계 계산 기간 | 2000-2019 (train) | Leakage 방지 |
| 윤년 처리 | Feb 29 통계 제외 | 단순화 (영향 미미) |
| 출력 구조 | stats + normalized | 학습 속도 vs 디스크 trade-off에서 속도 선택 |
| 변수별 처리 | 동일 | 세 변수 모두 well-behaved |
| 학습 시 패턴 | 모든 시점 동일 정규화 | 사전 정규화된 데이터 그대로 사용 |
