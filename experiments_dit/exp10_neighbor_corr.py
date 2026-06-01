"""실험 10 (v2 LDM/DiT): Seoul 3×3 픽셀별 x̂_{t-1} ↔ x̂_{t+1} 시간상관 vs ERA5.

Seoul 중심 3×3 픽셀 박스에서, **각 픽셀·각 채널(T,u,v)** 마다 시간축(모든 test 시점)
에 걸친 Pearson 상관을 계산한다:
    r(p, c) = corr_t( field_{t-1}^c(p) ,  field_{t+1}^c(p) )
즉 한 시점이 만들어내는 이웃 쌍 (x_{t-1}, x_{t+1}) 의 같은 픽셀 값이 시점들 사이에서
얼마나 함께 움직이는가 (12h 간격 시간상관). 모델이 생성한 (x̂_{t-1}, x̂_{t+1}) 의 시간
구조가 ERA5 의 (x_{t-1}, x_{t+1}) 와 일치하는지 본다.

그림: 3×3 subplot grid
    row 0 = ERA5,  row 1 = ensemble (member 또는 ensemble mean),  row 2 = ERA5 − ensemble
    col   = T, u, v
각 cell 은 3×3 픽셀 상관 heatmap. row 0·1 은 공유 cbar [-1,1], row 2 는 diff diverging cbar.

ensemble row 는 기본적으로 단일 member (--member) 를 쓰고, --ensemble-mean 지정 시
멤버 평균 필드로 계산한다. Pearson 은 시각별 통계로 역정규화한 물리 공간에서 계산한다
(정규화가 hour-of-day 별 affine 이라 시간축 상관이 raw 와 달라지므로).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    SEOUL_LAT_IDX, SEOUL_LAT_SLICE, SEOUL_LON_IDX, SEOUL_LON_SLICE, VARIABLES,
    ensure_dir, list_ensemble_files, load_ensemble_npz, load_norm_stats,
    set_plot_defaults,
)
from .exp5_composite_maps import _denorm_field

_EPS = 1e-12


def _pearson_axis0(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """시간축(axis 0) Pearson 상관. a,b: (T, H, W) → (H, W)."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean(axis=0, keepdims=True)
    num = (ac * bc).sum(axis=0)
    den = np.sqrt((ac * ac).sum(axis=0) * (bc * bc).sum(axis=0)) + _EPS
    return num / den


def _collect_series(
    ensemble_dir: Path, member: int, use_ens_mean: bool,
) -> dict:
    """시점별 Seoul 박스 (t-1, t+1) 필드를 ERA5 / ensemble 각각 쌓는다.

    Returns dict with arrays shaped (T, C, 3, 3): gt_tm1, gt_tp1, en_tm1, en_tp1.
    """
    files = list_ensemble_files(ensemble_dir)
    mean, std = load_norm_stats()
    box = (slice(None), SEOUL_LAT_SLICE, SEOUL_LON_SLICE)   # (C, 3, 3)

    gt_tm1, gt_tp1, en_tm1, en_tp1 = [], [], [], []
    for f in files:
        es = load_ensemble_npz(f)
        need = [es.x_tm1_true_pixel, es.x_tp1_true_pixel,
                es.ensemble_pixel_tm1, es.ensemble_pixel_tp1]
        if any(x is None for x in need):
            raise KeyError(
                f"{f} 에 이웃 픽셀 필드가 없습니다 — 이웃 저장 버전 "
                f"experiments_dit.ensemble_inference 로 캐시를 재생성하세요."
            )
        tm1_time = es.time_t - np.timedelta64(6, "h")
        tp1_time = es.time_t + np.timedelta64(6, "h")

        # ERA5 (denorm) — (C,H,W) → Seoul box
        gt_tm1.append(_denorm_field(es.x_tm1_true_pixel, tm1_time, mean, std)[box])
        gt_tp1.append(_denorm_field(es.x_tp1_true_pixel, tp1_time, mean, std)[box])

        # ensemble: member 선택 또는 멤버 평균 → (C,H,W) → denorm → box
        if use_ens_mean:
            em1 = es.ensemble_pixel_tm1.mean(axis=0)
            ep1 = es.ensemble_pixel_tp1.mean(axis=0)
        else:
            em1 = es.ensemble_pixel_tm1[member]
            ep1 = es.ensemble_pixel_tp1[member]
        en_tm1.append(_denorm_field(em1, tm1_time, mean, std)[box])
        en_tp1.append(_denorm_field(ep1, tp1_time, mean, std)[box])

    return {
        "gt_tm1": np.stack(gt_tm1), "gt_tp1": np.stack(gt_tp1),
        "en_tm1": np.stack(en_tm1), "en_tp1": np.stack(en_tp1),
        "n": len(files),
    }


def _corr_maps(series: dict) -> dict:
    """채널별 (3,3) 시간상관 맵: era5, ensemble."""
    C = series["gt_tm1"].shape[1]
    era5 = np.stack([
        _pearson_axis0(series["gt_tm1"][:, c], series["gt_tp1"][:, c])
        for c in range(C)
    ])                                                      # (C, 3, 3)
    ens = np.stack([
        _pearson_axis0(series["en_tm1"][:, c], series["en_tp1"][:, c])
        for c in range(C)
    ])
    return {"era5": era5, "ensemble": ens, "diff": era5 - ens}


def _annot_heatmap(ax, data: np.ndarray, vmin, vmax, cmap, title):
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
    H, W = data.shape
    for i in range(H):
        for j in range(W):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="k")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def _save_grid(maps: dict, label_ens: str, out_path: Path) -> None:
    fig, axes = plt.subplots(3, len(VARIABLES), figsize=(11, 11))
    diff_max = max(float(np.abs(maps["diff"]).max()), 1e-6)
    im_corr = im_diff = None
    rows = [("ERA5", maps["era5"], "RdBu_r", -1.0, 1.0),
            (label_ens, maps["ensemble"], "RdBu_r", -1.0, 1.0),
            ("ERA5 − ens", maps["diff"], "PuOr", -diff_max, diff_max)]
    for ri, (rlabel, arr, cmap, vmin, vmax) in enumerate(rows):
        for ci, v in enumerate(VARIABLES):
            ax = axes[ri, ci]
            im = _annot_heatmap(ax, arr[ci], vmin, vmax, cmap,
                                f"{v.upper()}" if ri == 0 else "")
            if ci == 0:
                ax.set_ylabel(rlabel, fontsize=11)
            if ri < 2:
                im_corr = im
            else:
                im_diff = im

    fig.suptitle(
        "Exp10 — Seoul 3×3 pixelwise temporal corr  "
        "r( field(t-1), field(t+1) )   vs ERA5",
        y=0.98, fontsize=12,
    )
    fig.subplots_adjust(left=0.08, right=0.86, top=0.93, bottom=0.04,
                        hspace=0.15, wspace=0.10)
    cax_c = fig.add_axes([0.88, 0.40, 0.022, 0.50])
    fig.colorbar(im_corr, cax=cax_c).set_label("Pearson r")
    cax_d = fig.add_axes([0.88, 0.06, 0.022, 0.25])
    fig.colorbar(im_diff, cax=cax_d).set_label("Δr (ERA5 − ens)")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_exp10(
    ensemble_dir: str,
    figures_dir: str,
    metrics_dir: str,
    member: int = 0,
    ensemble_mean: bool = False,
) -> dict:
    set_plot_defaults()
    fig_dir = ensure_dir(figures_dir)
    met_dir = ensure_dir(metrics_dir)

    series = _collect_series(Path(ensemble_dir), member, ensemble_mean)
    maps = _corr_maps(series)
    label_ens = "ensemble mean" if ensemble_mean else f"member {member}"
    print(f"[exp10] {series['n']} timesteps · ensemble row = {label_ens}")

    tag = "ensmean" if ensemble_mean else f"member{member}"
    _save_grid(maps, label_ens, fig_dir / f"exp10_neighbor_corr_{tag}.png")

    stats = {
        "seoul_center_idx": [SEOUL_LAT_IDX, SEOUL_LON_IDX],
        "box": "3×3",
        "n_timesteps": series["n"],
        "ensemble_row": label_ens,
        "mean_abs_diff_per_var": {
            v: float(np.abs(maps["diff"][ci]).mean())
            for ci, v in enumerate(VARIABLES)
        },
        "era5_mean_r_per_var": {
            v: float(maps["era5"][ci].mean()) for ci, v in enumerate(VARIABLES)
        },
        "ensemble_mean_r_per_var": {
            v: float(maps["ensemble"][ci].mean())
            for ci, v in enumerate(VARIABLES)
        },
    }
    with open(met_dir / f"exp10_stats_{tag}.json", "w") as fp:
        json.dump(stats, fp, indent=2)
    print(f"[exp10] wrote exp10_neighbor_corr_{tag}.png + stats")
    return stats


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles_dit")
    p.add_argument("--figures_dir", default="outputs/figures_dit")
    p.add_argument("--metrics_dir", default="outputs/metrics_dit")
    p.add_argument("--member", type=int, default=0,
                   help="ensemble row 에 쓸 멤버 인덱스 (기본 0).")
    p.add_argument("--ensemble-mean", dest="ensemble_mean",
                   action="store_true",
                   help="지정 시 ensemble row 를 멤버 평균 필드로 계산.")
    args = p.parse_args()
    run_exp10(
        args.ensemble_dir, args.figures_dir, args.metrics_dir,
        member=args.member, ensemble_mean=args.ensemble_mean,
    )
