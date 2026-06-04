"""[3/4] 표준화 anomaly 계산 — IMPLEMENTATION_SPEC v4 §3[3].

00시 스냅샷 데이터 + climatology 로부터 표준화 anomaly 를 만든다. 이것이 VAE·diffusion 의
입력/타깃이 된다.

  x̃ = (x_00utc − c_mu[doy]) / c_sig[doy]

doy 인덱싱은 `models.climatology.doy_slot` 단일 규칙([2]와 동일)을 사용한다.
전 기간(2000~2022, train/val/test 모두)을 변환해 추론까지 커버한다.

출력: `data/era5_00utc_anomaly.zarr` (변수 `anomaly`, 시간축 1일 간격 00시).
      `dataset/era5_dataset.py` 가 `var_name='anomaly'` 로 그대로 로드한다.

실행:
    python -m scripts.compute_anomaly \\
        --snapshot data/era5_00utc.zarr --var snapshot \\
        --climatology data/climatology.npz \\
        --output data/era5_00utc_anomaly.zarr
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

from models.climatology import doy_slot


def compute_anomaly(
    snapshot_path: str,
    climatology_path: str,
    output_path: str,
    var: str = "snapshot",
    out_var: str = "anomaly",
) -> None:
    print(f"[1/4] open {snapshot_path} (var={var})")
    ds = xr.open_zarr(snapshot_path)
    data = ds[var].values.astype(np.float32)          # (N, C, H, W)
    slots = doy_slot(ds.time.values)                  # (N,)

    print(f"[2/4] load climatology {climatology_path}")
    clim = np.load(climatology_path)
    c_mu, c_sig = clim["c_mu"], clim["c_sig"]          # (366, C, H, W)

    print(f"[3/4] standardize: x̃ = (x − c_mu[doy]) / c_sig[doy]")
    mu_per = c_mu[slots]                               # (N, C, H, W)
    sig_per = c_sig[slots]
    anom = ((data - mu_per) / sig_per).astype(np.float32)

    out = xr.Dataset(
        {out_var: (("time", "channel", "lat", "lon"), anom)},
        coords={k: ds.coords[k] for k in ("time", "channel", "lat", "lon")
                if k in ds.coords},
        attrs={
            "description": "Standardized 00 UTC snapshot anomaly (x - c_mu)/c_sig",
            "snapshot_source": snapshot_path,
            "climatology_source": climatology_path,
        },
    )

    # 원본 coord 에서 상속된 zarr chunk encoding 이 dask chunk 와 어긋나는 것 방지.
    for v in list(out.variables):
        out[v].encoding.clear()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        shutil.rmtree(out_p)
    print(f"[4/4] write {output_path} (var={out_var})")
    out.to_zarr(out_p, mode="w")

    # 검증: 채널별 anomaly mean≈0, std≈1 (climatology 가 잘 동작하는지).
    chan_mean = anom.mean(axis=(0, 2, 3))
    chan_std = anom.std(axis=(0, 2, 3))
    print(f"      anomaly per-channel mean={np.round(chan_mean, 4).tolist()} "
          f"(target ~0)")
    print(f"      anomaly per-channel std ={np.round(chan_std, 4).tolist()} "
          f"(target ~1)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=str, default="data/era5_00utc.zarr")
    p.add_argument("--var", type=str, default="snapshot")
    p.add_argument("--climatology", type=str, default="data/climatology.npz")
    p.add_argument("--output", type=str, default="data/era5_00utc_anomaly.zarr")
    p.add_argument("--out_var", type=str, default="anomaly")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    compute_anomaly(a.snapshot, a.climatology, a.output, a.var, a.out_var)
