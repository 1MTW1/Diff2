"""실험 6 (v2 LDM/DiT): latent 공간 pixel-wise RMSE/Spread.

각 latent pixel (i,j)에 대해 sample 차원으로 먼저 RMS aggregate:
    spread²(i,j) = mean_s [ std_b(ens_{s,b,i,j})² ]
    rmse² (i,j) = mean_s [ (mean_b(ens_{s,b,i,j}) − gt_{s,i,j})² ]
    ratio (i,j) = sqrt(spread²) / sqrt(rmse²)

그 후 (H_z*W_z=256) latent pixel 분포를 histogram으로 plot. 추가로
spread/rmse/ratio 의 spatial map도 latent 단위로 저장한다.

v1 exp6는 정규화 공간과 역정규화(K) 공간을 모두 계산했으나, v2 분석은 diffusion
latent 공간에서만 이루어지므로 K-space 절반을 제거하고 latent ratio만 보고한다.

exp3와의 차이:
    exp3: sample별 mean_(i,j) ratio → sample 분포 (per-sample summary).
    exp6: latent pixel별 시간 RMS → pixel 분포 (per-pixel summary).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    LATENT_CH, ensure_dir, list_ensemble_files, load_ensemble_npz,
    set_plot_defaults,
)


_EPS = 1e-12


def _accumulate(files):
    """sample 차원을 미리 누적해 latent pixel별 mean(spread²), mean(err²) 계산.

    Returns:
        sp_n2, er_n2 : (H_z, W_z) latent 공간 mean square
    """
    n = len(files)
    sp_n2 = er_n2 = None
    for f in files:
        es = load_ensemble_npz(f)
        ens_t = es.ensemble[:, LATENT_CH]                      # (N, H_z, W_z)
        spread = ens_t.std(axis=0, ddof=1)                     # (H_z, W_z)
        err = ens_t.mean(axis=0) - es.x_t_true[LATENT_CH]      # (H_z, W_z)

        if sp_n2 is None:
            sp_n2 = spread ** 2
            er_n2 = err ** 2
        else:
            sp_n2 += spread ** 2
            er_n2 += err ** 2

    return sp_n2 / n, er_n2 / n


def _save_hist(ratio_n: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    data = ratio_n.flatten()
    ax.hist(data, bins=50, color="C5", edgecolor="white")
    med = float(np.median(data))
    mean = float(data.mean())
    ax.axvline(med, color="k", linestyle="--", label=f"median={med:.3f}")
    ax.axvline(mean, color="C0", linestyle="-", label=f"mean={mean:.3f}")
    ax.axvline(1.0, color="r", linestyle=":", label="ideal=1")
    ax.set_xlabel("latent pixel spread / RMSE")
    ax.set_ylabel("count")
    ax.set_title("Exp6: latent pixel-wise ratio — latent space")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_maps(
    spread_n: np.ndarray,
    rmse_n: np.ndarray,
    ratio_n: np.ndarray,
    out_path: Path,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # spread / rmse: sequential, 같은 latent scale로 비교 가능
    vmax_sr = float(max(spread_n.max(), rmse_n.max()))
    for ax, data, title in zip(
        axes[:2],
        [spread_n, rmse_n],
        ["latent pixel-wise Spread", "latent pixel-wise RMSE"],
    ):
        im = ax.imshow(data, cmap="viridis", vmin=0, vmax=vmax_sr)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    # ratio: 1 중심 diverging
    dev = float(max(abs(ratio_n.max() - 1.0), abs(ratio_n.min() - 1.0)))
    dev = max(dev, 0.1)
    ax = axes[2]
    im = ax.imshow(ratio_n, cmap="RdBu_r", vmin=1 - dev, vmax=1 + dev)
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

    files = list_ensemble_files(Path(ensemble_dir))
    print(f"[exp6] processing {len(files)} samples")
    sp_n2, er_n2 = _accumulate(files)

    spread_n = np.sqrt(sp_n2)
    rmse_n = np.sqrt(er_n2)
    ratio_n = spread_n / (rmse_n + _EPS)

    _save_hist(ratio_n, fig_dir / "exp6_pixel_ratio_hist.png")
    _save_maps(spread_n, rmse_n, ratio_n, fig_dir / "exp6_pixel_maps.png")

    def _stats(arr: np.ndarray) -> dict:
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }

    stats = {
        "pixel_ratio_latent":  _stats(ratio_n),
        "pixel_spread_latent": _stats(spread_n),
        "pixel_rmse_latent":   _stats(rmse_n),
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
    p.add_argument("--ensemble_dir", default="outputs/ensembles_dit")
    p.add_argument("--figures_dir", default="outputs/figures_dit")
    p.add_argument("--metrics_dir", default="outputs/metrics_dit")
    args = p.parse_args()
    run_exp6(args.ensemble_dir, args.figures_dir, args.metrics_dir)
