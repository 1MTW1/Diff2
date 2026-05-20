"""실험 3: Spread / RMSE ratio (calibration).

experiment.md §4.3 — 픽셀별 ratio를 먼저 구하고 평균. 분모 폭발 가능성에 따라
정규화 / 역정규화 공간 모두 계산. 추가로 √mean(s²)/√mean(e²) 도 보고.
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


def _temp_sigma(time_t, std) -> np.ndarray:
    """해당 시각의 t 채널 std (H, W)."""
    hour = pd.Timestamp(time_t).hour
    h_idx = HOUR_TO_IDX[int(hour)]
    return std[h_idx, TEMP_IDX].cpu().numpy()


_EPS = 1e-6


def _pixel_ratio_mean(spread: np.ndarray, err: np.ndarray) -> float:
    return float((spread / (np.abs(err) + _EPS)).mean())


def _aggregate_ratio(spread: np.ndarray, err: np.ndarray) -> float:
    num = float(np.sqrt((spread ** 2).mean()))
    den = float(np.sqrt((err ** 2).mean())) + _EPS
    return num / den


def _save_hist(
    norm_ratios: np.ndarray,
    denorm_ratios: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, title in zip(
        axes,
        [norm_ratios, denorm_ratios],
        ["Normalized space", "Denormalized space (K)"],
    ):
        ax.hist(data, bins=40, color="C4", edgecolor="white")
        ax.axvline(float(np.median(data)), color="k", linestyle="--",
                   label=f"median={np.median(data):.3f}")
        ax.axvline(1.0, color="r", linestyle=":", label="ideal=1")
        ax.set_xlabel("mean_(i,j) [ s(i,j) / (|e(i,j)| + ε) ]")
        ax.set_ylabel("count")
        ax.set_title(f"Exp3: per-sample ratio — {title}")
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
    _, std = load_norm_stats()

    files = list_ensemble_files(Path(ensemble_dir))
    n = len(files)
    norm_ratios = np.empty(n, dtype=np.float64)
    denorm_ratios = np.empty(n, dtype=np.float64)
    norm_agg = np.empty(n, dtype=np.float64)
    denorm_agg = np.empty(n, dtype=np.float64)

    for i, f in enumerate(files):
        es = load_ensemble_npz(f)
        ens_t = es.ensemble[:, TEMP_IDX]                       # (N, H, W)
        spread = ens_t.std(axis=0, ddof=1)                     # (H, W)
        err = ens_t.mean(axis=0) - es.x_t_true[TEMP_IDX]       # (H, W)

        norm_ratios[i] = _pixel_ratio_mean(spread, err)
        norm_agg[i] = _aggregate_ratio(spread, err)

        sigma_t = _temp_sigma(es.time_t, std)                  # (H, W)
        spread_d = spread * sigma_t
        err_d = err * sigma_t                                  # mean shift은 무관
        denorm_ratios[i] = _pixel_ratio_mean(spread_d, err_d)
        denorm_agg[i] = _aggregate_ratio(spread_d, err_d)

    _save_hist(norm_ratios, denorm_ratios, fig_dir / "exp3_ratio_hist.png")

    stats = {
        "pixel_mean_ratio_normalized": {
            "mean": float(norm_ratios.mean()),
            "median": float(np.median(norm_ratios)),
            "std": float(norm_ratios.std()),
            "n_samples": int(n),
        },
        "pixel_mean_ratio_denormalized": {
            "mean": float(denorm_ratios.mean()),
            "median": float(np.median(denorm_ratios)),
            "std": float(denorm_ratios.std()),
            "n_samples": int(n),
        },
        "aggregate_ratio_normalized": {
            "mean": float(norm_agg.mean()),
            "median": float(np.median(norm_agg)),
            "std": float(norm_agg.std()),
        },
        "aggregate_ratio_denormalized": {
            "mean": float(denorm_agg.mean()),
            "median": float(np.median(denorm_agg)),
            "std": float(denorm_agg.std()),
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
    p.add_argument("--ensemble_dir", default="outputs/ensembles")
    p.add_argument("--figures_dir", default="outputs/figures")
    p.add_argument("--metrics_dir", default="outputs/metrics")
    args = p.parse_args()
    run_exp3(args.ensemble_dir, args.figures_dir, args.metrics_dir)
