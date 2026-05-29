"""실험 3 (v2 LDM/DiT): latent 공간 Spread / RMSE ratio (calibration).

experiment.md §4.3 — 픽셀별 ratio를 먼저 구하고 평균. v1 exp3은 정규화 공간과
역정규화(K) 공간을 모두 계산했으나, v2 분석은 diffusion latent 공간에서만
이루어지므로 역정규화/K-space 절반을 제거하고 latent 공간 ratio만 보고한다.
추가로 √mean(s²)/√mean(e²) (aggregate ratio) 도 함께 보고한다.
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


_EPS = 1e-6


def _pixel_ratio_mean(spread: np.ndarray, err: np.ndarray) -> float:
    return float((spread / (np.abs(err) + _EPS)).mean())
# pixelwise ratio를 먼저 구한 후 평균. pixel별 ratio의 분포가 넓을 수 있으므로, per-pixel ratio를

def _aggregate_ratio(spread: np.ndarray, err: np.ndarray) -> float:
    num = float(np.sqrt((spread ** 2).mean()))
    den = float(np.sqrt((err ** 2).mean())) + _EPS
    return num / den
# spread²와 err²의 평균을 먼저 구한 후 루트. pixel별 ratio의 분포가 넓어도, spread²와 err²의 평균은 극단값에 덜 민감할 수 있다.

def _save_hist(norm_ratios: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(norm_ratios, bins=40, color="C4", edgecolor="white")
    ax.axvline(float(np.median(norm_ratios)), color="k", linestyle="--",
               label=f"median={np.median(norm_ratios):.3f}")
    ax.axvline(1.0, color="r", linestyle=":", label="ideal=1")
    ax.set_xlabel("mean_(i,j) [ s(i,j) / (|e(i,j)| + ε) ]")
    ax.set_ylabel("count")
    ax.set_title("Exp3: per-sample ratio — latent space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_exp3(
    ensemble_dir: str,
    figures_dir: str,
    metrics_dir: str,
) -> dict:
    set_plot_defaults()
    fig_dir = ensure_dir(figures_dir)
    met_dir = ensure_dir(metrics_dir)

    files = list_ensemble_files(Path(ensemble_dir))
    n = len(files)
    norm_ratios = np.empty(n, dtype=np.float64)
    norm_agg = np.empty(n, dtype=np.float64)

    for i, f in enumerate(files):
        es = load_ensemble_npz(f)
        ens_t = es.ensemble[:, LATENT_CH]                      # (N, H_z, W_z)
        spread = ens_t.std(axis=0, ddof=1)                     # (H_z, W_z)
        err = ens_t.mean(axis=0) - es.x_t_true[LATENT_CH]      # (H_z, W_z)

        norm_ratios[i] = _pixel_ratio_mean(spread, err)
        norm_agg[i] = _aggregate_ratio(spread, err)

    _save_hist(norm_ratios, fig_dir / "exp3_ratio_hist.png")

    stats = {
        "pixel_mean_ratio_latent": {
            "mean": float(norm_ratios.mean()),
            "median": float(np.median(norm_ratios)),
            "std": float(norm_ratios.std()),
            "n_samples": int(n),
        },
        "aggregate_ratio_latent": {
            "mean": float(norm_agg.mean()),
            "median": float(np.median(norm_agg)),
            "std": float(norm_agg.std()),
        },
        "epsilon": _EPS,
    }
    with open(met_dir / "exp3_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[exp3] {stats}")
    return stats


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles_dit")
    p.add_argument("--figures_dir", default="outputs/figures_dit")
    p.add_argument("--metrics_dir", default="outputs/metrics_dit")
    args = p.parse_args()
    run_exp3(args.ensemble_dir, args.figures_dir, args.metrics_dir)
