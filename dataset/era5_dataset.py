"""사전 정규화된 ERA5 데이터를 5 시점 / 2 시점 단위로 로딩."""
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
    """사전 정규화된 ERA5 데이터를 학습/추론에 사용할 형태로 반환.

    데이터 구조:
        data/era5_normalized.zarr:
            normalized: (T, C=3, H=64, W=64), float32
            Coords: time (datetime64), channel=['t','u','v'], lat (64,), lon (64,)
    """

    SPLIT_RANGES = {
        "train":      ("2000-01-01", "2019-12-31"),
        "validation": ("2020-01-01", "2021-12-31"),
        "test":       ("2022-01-01", "2022-12-31"),
    }

    def __init__(
        self,
        normalized_path: str = "data/era5_normalized.zarr",
        mode: str = "train",           # 'train' or 'inference'
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

        # 유효 인덱스 범위
        # train: idx-3 ~ idx+1 모두 유효 → idx ∈ [3, n_times-2)
        # inference: idx-1, idx → idx ∈ [1, n_times)
        if mode == "train":
            self.valid_start = 3
            self.valid_end = self.n_times - 1  # exclusive (idx+1 필요)
        else:  # inference
            self.valid_start = 1
            self.valid_end = self.n_times

        if self.valid_end <= self.valid_start:
            raise RuntimeError(
                f"Split '{split}' too short for mode '{mode}': "
                f"n_times={self.n_times}"
            )

    def __len__(self) -> int:
        return self.valid_end - self.valid_start

    def _get_frame(self, abs_idx: int) -> np.ndarray:
        if self.data is not None:
            return self.data[abs_idx]
        return self.zarr_handle.isel(time=abs_idx).values.astype(np.float32)

    def _get_window(self, start: int, end: int) -> np.ndarray:
        if self.data is not None:
            return self.data[start:end]
        return self.zarr_handle.isel(
            time=slice(start, end)
        ).values.astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        abs_idx = idx + self.valid_start

        if self.mode == "train":
            chunk = self._get_window(abs_idx - 3, abs_idx + 2)  # (5, C, H, W)
            return {
                "x_tm3":  torch.from_numpy(chunk[0]),
                "x_tm2":  torch.from_numpy(chunk[1]),
                "x_tm1":  torch.from_numpy(chunk[2]),
                "x_t":    torch.from_numpy(chunk[3]),
                "x_tp1":  torch.from_numpy(chunk[4]),
                "time_t": self.times[abs_idx],
            }

        # inference
        x_tm1 = self._get_frame(abs_idx - 1)
        x_t = self._get_frame(abs_idx)
        return {
            "x_tm1":  torch.from_numpy(x_tm1),
            "x_t":    torch.from_numpy(x_t),
            "time_t": self.times[abs_idx],
        }


class ERA5PairDataset(Dataset):
    """연속 2시점 기상장을 채널 concat 형태로 반환 — VAE(Stage 0) 학습용.

    diffusion target은 항상 연속 2시점 쌍(`[x_{t-3},x_{t-2}]`, `[x_t,x_{t+1}]`)이므로
    VAE는 split 내 모든 연속 2시점 쌍의 분포를 보고 학습한다.

    반환: `(2*C, H, W)` = `(6, 64, 64)` 텐서 (frame0 채널 ‖ frame1 채널).
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
        if self.n_times < 2:
            raise RuntimeError(
                f"Split '{split}' too short for pairs: n_times={self.n_times}"
            )

        if load_into_memory:
            self.data = ds_split["normalized"].values.astype(np.float32)
            self.zarr_handle = None
        else:
            self.data = None
            self.zarr_handle = ds_split["normalized"]

    def __len__(self) -> int:
        return self.n_times - 1

    def _get_window(self, start: int, end: int) -> np.ndarray:
        if self.data is not None:
            return self.data[start:end]
        return self.zarr_handle.isel(
            time=slice(start, end)
        ).values.astype(np.float32)

    def __getitem__(self, idx: int) -> torch.Tensor:
        chunk = self._get_window(idx, idx + 2)        # (2, C, H, W)
        C, H, W = chunk.shape[1:]
        pair = chunk.reshape(2 * C, H, W)             # (6, 64, 64)
        return torch.from_numpy(pair)
