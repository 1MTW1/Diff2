"""실험 6: 전체 ensemble에 걸친 pixel-wise RMSE/Spread.

각 pixel (i,j)에 대해 sample 차원으로 먼저 RMS aggregate:
    spread²(i,j) = mean_s [ std_b(ens_{s,b,i,j})² ]
    rmse² (i,j) = mean_s [ (mean_b(ens_{s,b,i,j}) − gt_{s,i,j})² ]
    ratio (i,j) = sqrt(spread²) / sqrt(rmse²)

그 후 (H*W=4096) pixel 분포를 histogram으로 plot. 추가로 spread/rmse/ratio의
spatial map도 같이 저장.

exp3와의 차이:
    exp3: sample별 mean_(i,j) ratio → sample 분포 (per-sample summary).
    exp6: pixel별 시간 RMS → pixel 분포 (per-pixel summary, climatology view).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset.denormalize import HOUR_TO_IDX

from .utils import (
    TEMP_IDX, ensure_dir, list_ensemble_files, load_ensemble_npz,
    load_norm_stats, set_plot_defaults,
)


_EPS = 1e-12


def _temp_sigma(time_t, std) -> np.ndarray:
    hour = pd.Timestamp(time_t).hour
    h_idx = HOUR_TO_IDX[int(hour)]
    return std[h_idx, TEMP_IDX].cpu().numpy()


def _accumulate(files, std):
    """sample 차원을 미리 누적해 픽셀별 mean(spread²), mean(err²) 계산.

    Returns:
        sp_n2, er_n2 : (H, W) 정규화 공간 mean square
        sp_d2, er_d2 : (H, W) 역정규화 공간 mean square
    """
    n = len(files)
    sp_n2 = er_n2 = sp_d2 = er_d2 = None
    for f in files:
        es = load_ensemble_npz(f)
        ens_t = es.ensemble[:, TEMP_IDX]                       # (N, H, W)
        spread = ens_t.std(axis=0, ddof=1)                     # (H, W)
        err = ens_t.mean(axis=0) - es.x_t_true[TEMP_IDX]       # (H, W)

        sigma_t = _temp_sigma(es.time_t, std)                  # (H, W)
        spread_d = spread * sigma_t
        err_d = err * sigma_t

        if sp_n2 is None:
            sp_n2 = spread ** 2
            er_n2 = err ** 2
            sp_d2 = spread_d ** 2
            er_d2 = err_d ** 2
        else:
            sp_n2 += spread ** 2
            er_n2 += err ** 2
            sp_d2 += spread_d ** 2
            er_d2 += err_d ** 2

    return sp_n2 / n, er_n2 / n, sp_d2 / n, er_d2 / n


def _save_hist(ratio_n: np.ndarray, ratio_d: np.ndarray, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, title in zip(
        axes,
        [ratio_n.flatten(), ratio_d.flatten()],
        ["Normalized space", "Denormalized space"],
    ):
        ax.hist(data, bins=50, color="C5", edgecolor="white")
        med = float(np.median(data))
        mean = float(data.mean())
        ax.axvline(med, color="k", linestyle="--",
                   label=f"median={med:.3f}")
        ax.axvline(mean, color="C0", linestyle="-",
                   label=f"mean={mean:.3f}")
        ax.axvline(1.0, color="r", linestyle=":", label="ideal=1")
        ax.set_xlabel("pixel spread / RMSE")
        ax.set_ylabel("count")
        ax.set_title(f"Exp6: pixel-wise ratio — {title}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_maps(
    spread_d: np.ndarray,
    rmse_d: np.ndarray,
    ratio_d: np.ndarray,
    out_path: Path,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # spread / rmse: sequential, 같은 K scale로 비교 가능
    vmax_sr = float(max(spread_d.max(), rmse_d.max()))
    for ax, data, title in zip(
        axes[:2],
        [spread_d, rmse_d],
        ["pixel-wise Spread (K)", "pixel-wise RMSE (K)"],
    ):
        im = ax.imshow(data, cmap="viridis", vmin=0, vmax=vmax_sr)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    # ratio: 1 중심 diverging
    dev = float(max(abs(ratio_d.max() - 1.0), abs(ratio_d.min() - 1.0)))
    dev = max(dev, 0.1)
    ax = axes[2]
    im = ax.imshow(ratio_d, cmap="RdBu_r", vmin=1 - dev, vmax=1 + dev)
    ax.set_title("Spread / RMSE  (1 = calibrated)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_exp6(
    ensemble_dir: str,
    figures_dir: str,
    metrics_dir: str,
) -> dict:
    set_plot_defaults()
    fig_dir = ensure_dir(figures_dir)
    met_dir = ensure_dir(metrics_dir)
    _, std = load_norm_stats()

    files = list_ensemble_files(Path(ensemble_dir))
    print(f"[exp6] processing {len(files)} samples")
    sp_n2, er_n2, sp_d2, er_d2 = _accumulate(files, std)

    spread_n = np.sqrt(sp_n2)
    rmse_n = np.sqrt(er_n2)
    ratio_n = spread_n / (rmse_n + _EPS)

    spread_d = np.sqrt(sp_d2)
    rmse_d = np.sqrt(er_d2)
    ratio_d = spread_d / (rmse_d + _EPS)

    _save_hist(ratio_n, ratio_d, fig_dir / "exp6_pixel_ratio_hist.png")
    _save_maps(spread_d, rmse_d, ratio_d, fig_dir / "exp6_pixel_maps.png")

    def _stats(arr: np.ndarray) -> dict:
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }

    stats = {
        "pixel_ratio_normalized":   _stats(ratio_n),
        "pixel_ratio_denormalized": _stats(ratio_d),
        "pixel_spread_denorm_K":    _stats(spread_d),
        "pixel_rmse_denorm_K":      _stats(rmse_d),
        "n_samples": len(files),
        "n_pixels": int(ratio_n.size),
        "epsilon": _EPS,
    }
    with open(met_dir / "exp6_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[exp6] {stats}")
    return stats


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles")
    p.add_argument("--figures_dir", default="outputs/figures")
    p.add_argument("--metrics_dir", default="outputs/metrics")
    args = p.parse_args()
    run_exp6(args.ensemble_dir, args.figures_dir, args.metrics_dir)
