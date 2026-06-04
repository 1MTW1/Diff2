"""[2b/4] Pixelwise 통계 계산 — IMPLEMENTATION_SPEC v4 §3[2b], §2.2.

변수·위치별 **pixelwise mean/std** P_mu, P_sig `(C, H, W)` 를 계산한다 (doy 무관).
전 기간(2000~2019, train split)·전 doy 의 00시 스냅샷을 모아 각 (변수, 위치)에서
평균/표준편차를 낸다. condition 으로 줄 climatology 의 **pixelwise 표준화**
(c̃ = (c_mu[doy] − P_mu) / P_sig, §2.2)에 쓰인다.

자기통계(c_sig)로 표준화하면 0 으로 붕괴하므로 금지. pixelwise(위치별·doy무관) 통계로
표준화하면 변수·위치 스케일은 정규화하면서 **계절(doy) climatology 변동(절대 맥락)은 보존**.
주의: P_mu/P_sig 는 anomaly 표준화에 쓰는 c_mu/c_sig 와 **다른 통계**다(혼동 금지).

입력: [1] 00시 스냅샷 데이터 (2000~2019, train split). 시간축(axis=0)만 집계.
출력: `pixel_stats.npz` (P_mu, P_sig: 각 (C, H, W)). 추론에도 필요 → 보관.

실행:
    python -m scripts.compute_pixel_stats \\
        --input data/era5_00utc.zarr --var snapshot --output data/pixel_stats.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

# train split (leakage 방지) — compute_climatology.py 와 동일.
TRAIN_RANGE = ("2000-01-01", "2019-12-31")


def compute_pixel_stats(
    input_path: str,
    output_path: str,
    var: str = "snapshot",
    sigma_floor: float = 1e-6,
) -> None:
    print(f"[1/2] open {input_path} (var={var}), select train {TRAIN_RANGE}")
    ds = xr.open_zarr(input_path).sel(
        time=slice(TRAIN_RANGE[0], TRAIN_RANGE[1])
    )
    data = ds[var].values.astype(np.float64)          # (N, C, H, W)
    N, C, H, W = data.shape

    # 시간축(axis=0)만 집계 → (변수, 위치)별 pixelwise 통계. doy 에 무관.
    P_mu = data.mean(axis=0).astype(np.float32)                    # (C, H, W)
    P_sig = data.std(axis=0).astype(np.float32)                    # (C, H, W)
    P_sig = np.maximum(P_sig, sigma_floor)                         # 0 division 방어

    print(f"[2/2] save {output_path}")
    np.savez(output_path, P_mu=P_mu, P_sig=P_sig)
    print(f"      N={N} days, P_mu{P_mu.shape}  P_sig{P_sig.shape}")
    print(f"      P_mu  per-channel mean = "
          f"{np.round(P_mu.mean(axis=(1, 2)), 4).tolist()}")
    print(f"      P_sig per-channel mean = "
          f"{np.round(P_sig.mean(axis=(1, 2)), 4).tolist()}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/era5_00utc.zarr")
    p.add_argument("--var", type=str, default="snapshot")
    p.add_argument("--output", type=str, default="data/pixel_stats.npz")
    p.add_argument("--sigma_floor", type=float, default=1e-6)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    compute_pixel_stats(a.input, a.output, a.var, a.sigma_floor)
