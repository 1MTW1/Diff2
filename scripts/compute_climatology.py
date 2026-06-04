"""[2/4] Climatology 계산 (2단계: doy별 raw → ±7일 smoothing) — IMPLEMENTATION_SPEC v4 §3[2].

00시 스냅샷 데이터(2000~2019, train split)로부터 doy 별 climatology 를 만든다.
★v3/v4 핵심: "±7일×20년 샘플을 한꺼번에 pool" 하지 않고, **doy별 climatology 를 먼저
만든 뒤 그 climatology 시계열을 doy 축으로 smoothing** 한다 (SEEDS 가 timeseries 를
15일 centered window 로 smoothing 한 방식과 동일). 결과(특히 std)가 pool 방식과
다르므로 반드시 이 순서로 한다.

1단계 — doy별 raw climatology (smoothing 전):
  각 doy d 에 해당하는 20년치 00시 스냅샷 샘플들만 모아
    raw_mu[d]  = 평균장 (C, H, W)
    raw_sig[d] = 표준편차장 (C, H, W)
  → doy별 raw 시계열 raw_mu[0..365], raw_sig[0..365].

2단계 — 그 시계열에 ±w일 sliding window smoothing (doy 축):
    c_mu[d]  = mean( raw_mu[d-w .. d+w] )
    c_sig[d] = mean( raw_sig[d-w .. d+w] )

설계 포인트:
  - window 반경 w=7 (총 15일).
  - **연말-연초 wrap**: doy 축 윈도우는 366 ring 에서 modulo 순환.
  - **윤년 Feb 29(slot 59)**: smoothing 후 `0.5·(slot58+slot60)` 보간으로 덮어쓴다
    (±7d window 면 양옆이 이미 포함되므로 fallback/consistency 성격).
  - **빈 slot**: 1단계에서 샘플이 없는 doy 는 smoothing 평균에서 제외(데이터 있는 slot만).
  - **c_sig floor**: `max(c_sig, eps)` 로 0 division 방어.

출력: `data/climatology.npz` (c_mu, c_sig). **추론 역표준화에 필요 — 영구 보관.**

실행:
    python -m scripts.compute_climatology \\
        --input data/era5_00utc.zarr --var snapshot \\
        --output data/climatology.npz --window_radius 7
"""
from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

from models.climatology import _FEB29_SLOT, _N_SLOTS, doy_slot

# train split (leakage 방지) — dataset/era5_dataset.py 규약과 동일.
TRAIN_RANGE = ("2000-01-01", "2019-12-31")


def compute_climatology(
    input_path: str,
    output_path: str,
    var: str = "snapshot",
    window_radius: int = 7,
    sigma_floor: float = 1e-6,
) -> None:
    print(f"[1/4] open {input_path} (var={var}), select train {TRAIN_RANGE}")
    ds = xr.open_zarr(input_path).sel(
        time=slice(TRAIN_RANGE[0], TRAIN_RANGE[1])
    )
    data = ds[var].values.astype(np.float32)          # (N, C, H, W)
    slots = doy_slot(ds.time.values)                  # (N,) ∈ [0,365]
    N, C, H, W = data.shape
    print(f"      N={N} days, per-frame ({C},{H},{W}), w={window_radius}")

    # slot → 해당 doy 의 샘플 인덱스 목록.
    by_slot: list[list[int]] = [[] for _ in range(_N_SLOTS)]
    for i, s in enumerate(slots):
        by_slot[int(s)].append(i)

    # ── 1단계: doy별 raw climatology (smoothing 전) ─────────────────
    print("[2/4] stage1: per-doy raw climatology (no smoothing)")
    raw_mu = np.zeros((_N_SLOTS, C, H, W), dtype=np.float32)
    raw_sig = np.zeros((_N_SLOTS, C, H, W), dtype=np.float32)
    has_data = np.zeros(_N_SLOTS, dtype=bool)
    for d in range(_N_SLOTS):
        idxs = by_slot[d]
        if not idxs:
            continue
        samples = data[np.asarray(idxs)]              # (n_d, C, H, W)
        raw_mu[d] = samples.mean(axis=0)
        raw_sig[d] = samples.std(axis=0)              # population std (ddof=0)
        has_data[d] = True
    n_empty = int((~has_data).sum())
    print(f"      filled {_N_SLOTS - n_empty}/{_N_SLOTS} doy slots "
          f"({n_empty} empty)")

    # ── 2단계: doy 축 ±w일 smoothing (wrap, 빈 slot 제외) ───────────
    print(f"[3/4] stage2: ±{window_radius}d doy-axis smoothing (wrap)")
    c_mu = np.zeros_like(raw_mu)
    c_sig = np.zeros_like(raw_sig)
    for d in range(_N_SLOTS):
        acc_mu = np.zeros((C, H, W), dtype=np.float32)
        acc_sig = np.zeros((C, H, W), dtype=np.float32)
        cnt = 0
        for k in range(-window_radius, window_radius + 1):
            j = (d + k) % _N_SLOTS
            if has_data[j]:
                acc_mu += raw_mu[j]
                acc_sig += raw_sig[j]
                cnt += 1
        if cnt == 0:
            # 윈도우 전체가 비어있는 극단 — 가장 가까운 데이터 slot 으로 fallback.
            for r in range(1, _N_SLOTS):
                lo, hi = (d - r) % _N_SLOTS, (d + r) % _N_SLOTS
                if has_data[lo]:
                    c_mu[d], c_sig[d] = raw_mu[lo], raw_sig[lo]; break
                if has_data[hi]:
                    c_mu[d], c_sig[d] = raw_mu[hi], raw_sig[hi]; break
        else:
            c_mu[d] = acc_mu / cnt
            c_sig[d] = acc_sig / cnt

    # ── 윤년 Feb 29 (slot 59): 2/28·3/1 중간값으로 보간/일관화 ──────
    s58, s60 = _FEB29_SLOT - 1, _FEB29_SLOT + 1
    c_mu[_FEB29_SLOT] = 0.5 * (c_mu[s58] + c_mu[s60])
    c_sig[_FEB29_SLOT] = 0.5 * (c_sig[s58] + c_sig[s60])

    # c_sig floor.
    c_sig = np.maximum(c_sig, sigma_floor)

    print(f"[4/4] save {output_path}")
    np.savez(
        output_path, c_mu=c_mu, c_sig=c_sig,
        window_radius=np.int64(window_radius),
        sigma_floor=np.float32(sigma_floor),
    )
    print(f"      c_mu{c_mu.shape}  c_sig{c_sig.shape}")
    print(f"      c_mu  range [{c_mu.min():.4f}, {c_mu.max():.4f}]")
    print(f"      c_sig range [{c_sig.min():.4f}, {c_sig.max():.4f}]")
    # 검증: doy 축 smoothness (인접 slot 간 변화량) — 들쭉날쭉하지 않은지.
    raw_jit = np.abs(np.diff(raw_mu, axis=0)).mean()
    sm_jit = np.abs(np.diff(c_mu, axis=0)).mean()
    print(f"      doy-axis jitter (mean |Δ|): raw={raw_jit:.4e} "
          f"smoothed={sm_jit:.4e} (smoothed < raw expected)")
    mid = 0.5 * (c_mu[s58] + c_mu[s60])
    print(f"      Feb29 mid-check |c_mu[59]-mid|max="
          f"{np.abs(c_mu[_FEB29_SLOT]-mid).max():.2e}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/era5_00utc.zarr")
    p.add_argument("--var", type=str, default="snapshot")
    p.add_argument("--output", type=str, default="data/climatology.npz")
    p.add_argument("--window_radius", type=int, default=7)
    p.add_argument("--sigma_floor", type=float, default=1e-6)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    compute_climatology(
        a.input, a.output, a.var, a.window_radius, a.sigma_floor,
    )
