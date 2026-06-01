"""사전 정규화된 ERA5 데이터 로딩.

새 구조(중심-프레임 복원형 Dual-DDPM)는 연속 3시점 `x_{t-1}, x_t, x_{t+1}` 만 쓴다:
  - DDPM_past : cond = x_t,            target = [x_{t-1}, x_{t+1}]
  - DDPM_main : cond = [x̂_{t-1}, x̂_{t+1}], target = x_t
VAE는 프레임별 단일 autoencoder라 학습/통계는 개별 프레임(`ERA5FrameDataset`)을 본다.
"""
from __future__ import annotations

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import collate, default_collate_fn_map


def _collate_datetime64(batch, *, collate_fn_map=None):
    """numpy.datetime64를 그대로 묶어서 datetime64 배열로 반환."""
    del collate_fn_map
    return np.array(batch)


# default_collate_fn_map에 datetime64 핸들러 추가한 맵.
_COLLATE_FN_MAP = {
    **default_collate_fn_map,
    np.datetime64: _collate_datetime64,
}


def collate_with_time(batch):
    """ERA5NormalizedDataset의 `time_t` (numpy.datetime64) 지원 collate."""
    return collate(batch, collate_fn_map=_COLLATE_FN_MAP)


class ERA5NormalizedDataset(Dataset):
    """사전 정규화된 ERA5 데이터를 연속 3시점 윈도우로 반환.

    데이터 구조:
        data/era5_normalized.zarr:
            normalized: (T, C=3, H=64, W=64), float32
            Coords: time (datetime64), channel=['t','u','v'], lat (64,), lon (64,)

    train/inference 모두 `x_{t-1}, x_t, x_{t+1}` 3프레임을 반환한다 (추론에서 이웃 두
    프레임은 GT 평가용). 유효 인덱스: abs_idx-1, abs_idx+1 이 모두 존재해야 하므로
    abs_idx ∈ [1, n_times-2].
    """

    SPLIT_RANGES = {
        "train":      ("2000-01-01", "2019-12-31"),
        "validation": ("2020-01-01", "2021-12-31"),
        "test":       ("2022-01-01", "2022-12-31"),
    }

    def __init__(
        self,
        normalized_path: str = "data/era5_normalized.zarr",
        mode: str = "train",           # 'train' or 'inference' (윈도우 동일)
        split: str = "train",          # 'train' / 'validation' / 'test'
        load_into_memory: bool = False,
    ):
        super().__init__()
        if mode not in ("train", "inference"):
            raise ValueError(f"Unknown mode: {mode}")
        if split not in self.SPLIT_RANGES:
            raise ValueError(f"Unknown split: {split}")
        self.mode = mode
        self.split = split

        ds = xr.open_zarr(normalized_path)
        start, end = self.SPLIT_RANGES[split]
        ds_split = ds.sel(time=slice(start, end))

        self.times = ds_split.time.values
        self.n_times = len(self.times)
        if self.n_times == 0:
            raise RuntimeError(
                f"No timesteps in split '{split}' of {normalized_path}"
            )

        if load_into_memory:
            self.data = ds_split["normalized"].values.astype(np.float32)
            self.zarr_handle = None
        else:
            self.data = None
            self.zarr_handle = ds_split["normalized"]

        # x_{t-1}, x_{t+1} 모두 유효 → abs_idx ∈ [1, n_times-2]
        self.valid_start = 1
        self.valid_end = self.n_times - 1  # exclusive

        if self.valid_end <= self.valid_start:
            raise RuntimeError(
                f"Split '{split}' too short: n_times={self.n_times}"
            )

    def __len__(self) -> int:
        return self.valid_end - self.valid_start

    def _get_window(self, start: int, end: int) -> np.ndarray:
        if self.data is not None:
            return self.data[start:end]
        return self.zarr_handle.isel(
            time=slice(start, end)
        ).values.astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        abs_idx = idx + self.valid_start
        chunk = self._get_window(abs_idx - 1, abs_idx + 2)   # (3, C, H, W)
        return {
            "x_tm1":  torch.from_numpy(chunk[0]),
            "x_t":    torch.from_numpy(chunk[1]),
            "x_tp1":  torch.from_numpy(chunk[2]),
            "time_t": self.times[abs_idx],
        }


class ERA5FrameDataset(Dataset):
    """단일 기상장 프레임을 반환 — 프레임별 VAE(Stage 0) 학습 / latent 통계용.

    VAE가 프레임별 단일 autoencoder(`(C,H,W)` → latent)이므로, split 내 모든 개별
    프레임의 분포를 보고 학습한다.

    반환: `(C, H, W)` = `(3, 64, 64)` 텐서.
    """

    SPLIT_RANGES = ERA5NormalizedDataset.SPLIT_RANGES

    def __init__(
        self,
        normalized_path: str = "data/era5_normalized.zarr",
        split: str = "train",
        load_into_memory: bool = False,
    ):
        super().__init__()
        if split not in self.SPLIT_RANGES:
            raise ValueError(f"Unknown split: {split}")
        self.split = split

        ds = xr.open_zarr(normalized_path)
        start, end = self.SPLIT_RANGES[split]
        ds_split = ds.sel(time=slice(start, end))
        self.times = ds_split.time.values
        self.n_times = len(self.times)
        if self.n_times == 0:
            raise RuntimeError(
                f"No timesteps in split '{split}' of {normalized_path}"
            )

        if load_into_memory:
            self.data = ds_split["normalized"].values.astype(np.float32)
            self.zarr_handle = None
        else:
            self.data = None
            self.zarr_handle = ds_split["normalized"]

    def __len__(self) -> int:
        return self.n_times

    def _get_frame(self, abs_idx: int) -> np.ndarray:
        if self.data is not None:
            return self.data[abs_idx]
        return self.zarr_handle.isel(time=abs_idx).values.astype(np.float32)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self._get_frame(idx))        # (C, H, W)
