"""[1/4] 00 UTC 스냅샷 추출 (+ 원본 단위 역정규화) — IMPLEMENTATION_SPEC v4 §3[1].

매일 **00 UTC 스냅샷 1프레임만** 추출한다 (06/12/18 시각은 버림). daily mean 집계를
하지 않는다 (SEEDS 방식 — 일중 변동을 평균으로 뭉개지 않음). 태스크는 "00 UTC 스냅샷
기상장의 표준화 anomaly 예측" 으로 재정의된다.

★ 저장은 **원본 기상장(물리단위)** 로 한다. 입력 `era5_normalized.zarr` 는 시각별
pixelwise z-score(`(x−mean[hour])/std[hour]`) 로 정규화돼 있으므로, 00시 통계
(`stats.sel(hour=0)`) 로 역정규화하여 raw 단위로 되돌린다:
    x_raw = x_norm · std[00] + mean[00]
이렇게 하면 이후 climatology·anomaly 가 전부 물리단위에서 계산되고, 추론 시
`destandardize_anomaly` 가 곧장 물리단위를 복원하므로 normalization_stats 가 불필요해진다.
(anomaly x̃ 자체는 affine 불변이라 정규화 공간과 수치 동일 — 단지 단위를 물리로 통일.)

입력: 6시간 간격 정규화 데이터 (기본 `data/era5_normalized.zarr`, 변수 `normalized`).
      통계 (기본 `data/normalization_stats.zarr`, mean/std `(hour,C,H,W)`).
출력: `data/era5_00utc.zarr` (변수 `snapshot`, 1일 간격 00시, **물리단위**).

결측: 해당 날짜에 00시 프레임이 없으면 그 날은 자동 제외(필터에서 빠짐).

실행:
    python -m scripts.compute_daily_snapshot \\
        --input data/era5_normalized.zarr --var normalized \\
        --stats data/normalization_stats.zarr \\
        --output data/era5_00utc.zarr
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import xarray as xr


def compute_daily_snapshot(
    input_path: str,
    output_path: str,
    var: str = "normalized",
    out_var: str = "snapshot",
    stats_path: str | None = "data/normalization_stats.zarr",
) -> None:
    print(f"[1/4] open {input_path} (var={var})")
    ds = xr.open_zarr(input_path)
    if var not in ds:
        raise KeyError(
            f"variable '{var}' not in {input_path} (have {list(ds.data_vars)})"
        )
    da = ds[var]  # (T, C, H, W), lazy

    print("[2/4] select 00 UTC snapshots only (drop 06/12/18)")
    # 시각이 00시인 프레임만 — daily mean 안 함 (SEEDS 방식).
    da00 = da.sel(time=da.time.dt.hour == 0)
    n00 = da00.sizes["time"]
    if n00 == 0:
        raise RuntimeError(
            f"no 00 UTC frames found in {input_path} — 시간 coord 가 UTC 시각을 "
            f"담고 있는지 확인하세요."
        )

    # ── 00시 통계로 역정규화 → 원본 기상장(물리단위) ──────────────
    do_denorm = bool(stats_path) and str(stats_path).lower() != "none"
    if do_denorm:
        print(f"[3/4] denormalize with {stats_path} (hour=0): "
              f"x_raw = x_norm·std[00] + mean[00]")
        stats = xr.open_zarr(stats_path)
        # 시각축에서 00시 통계만 — coord 'hour' 가 [0,6,12,18].
        mean0 = stats["mean"].sel(hour=0)               # (C, H, W)
        std0 = stats["std"].sel(hour=0)
        # channel/lat/lon coord 정합 broadcast. 00시 데이터에만 적용.
        da00 = da00 * std0 + mean0
        denorm_attr = stats_path
    else:
        print("[3/4] denormalize skipped (stats=none) — output stays normalized")
        denorm_attr = "none"

    out = da00.to_dataset(name=out_var)
    out.attrs.update({
        "description": "00 UTC daily snapshot in physical units "
                       "(no daily mean; SEEDS-style)",
        "source": input_path,
        "source_var": var,
        "denormalized_with": denorm_attr,
    })
    # 원본에서 상속된 zarr chunk encoding 은 00시 sub-select 후 dask chunk 와
    # 어긋나 to_zarr 가 거부한다 → encoding 비우고 xarray 가 새로 청크하게 둔다.
    for v in list(out.variables):
        out[v].encoding.clear()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        shutil.rmtree(out_p)
    print(f"[4/4] write {output_path} (var={out_var}) — triggers compute")
    out.to_zarr(out_p, mode="w")

    print(f"      done. {n00} 00-UTC frames, shape per-frame "
          f"{tuple(out[out_var].shape[1:])}, "
          f"units={'physical' if do_denorm else 'normalized'}")
    print(f"      time range: {str(out.time.values[0])[:13]} ~ "
          f"{str(out.time.values[-1])[:13]}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/era5_normalized.zarr")
    p.add_argument("--var", type=str, default="normalized")
    p.add_argument("--stats", type=str, default="data/normalization_stats.zarr",
                   help="역정규화용 통계 zarr (mean/std (hour,C,H,W)). "
                        "'none' 이면 역정규화 생략(정규화 공간 유지).")
    p.add_argument("--output", type=str, default="data/era5_00utc.zarr")
    p.add_argument("--out_var", type=str, default="snapshot")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    compute_daily_snapshot(a.input, a.output, a.var, a.out_var, a.stats)
