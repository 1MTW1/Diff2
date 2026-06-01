"""실험 1 (v2 LDM/DiT): latent log_var 의 pixelwise 다양성 & 시점 간 차이.

latent 캐시의 `log_var` 채널 0 (LATENT_CH) 에서 exp(ℓ) 의 공간 다양성(CV)과
무작위 sample 쌍 공간 상관(flow-dependence)을 본다.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import (
    LATENT_CH, ensure_dir, list_ensemble_files, load_ensemble_npz,
    set_plot_defaults,
)


def _load_logvar_stack(
    ensemble_dir: Path,
) -> tuple[np.ndarray, list[np.datetime64]]:
    """모든 sample의 exp(ℓ) (H_z, W_z) → stack (S, H_z, W_z) + 시간 리스트."""
    files = list_ensemble_files(ensemble_dir)
    arrs: list[np.ndarray] = []
    times: list[np.datetime64] = []
    for f in files:
        es = load_ensemble_npz(f)
        arrs.append(np.exp(es.log_var[LATENT_CH]).astype(np.float32))
        times.append(es.time_t)
    return np.stack(arrs, axis=0), times


def _pixelwise_cv(stack: np.ndarray) -> np.ndarray:
    """sample별 spatial CV. stack: (S, H, W) → (S,)."""
    flat = stack.reshape(stack.shape[0], -1)
    m = flat.mean(axis=1)
    s = flat.std(axis=1)
    return s / (np.abs(m) + 1e-12)


def _pairwise_spatial_corr(
    stack: np.ndarray, n_pairs: int, rng: np.random.Generator,
) -> np.ndarray:
    """무작위 sample 쌍 n_pairs개의 픽셀-flatten Pearson correlation."""
    S, H, W = stack.shape
    flat = stack.reshape(S, -1).astype(np.float64)
    centered = flat - flat.mean(axis=1, keepdims=True)
    norms = np.sqrt((centered ** 2).sum(axis=1))

    rhos = np.empty(n_pairs, dtype=np.float64)
    cnt = 0
    while cnt < n_pairs:
        i, j = int(rng.integers(0, S)), int(rng.integers(0, S))
        if i == j:
            continue
        num = float((centered[i] * centered[j]).sum())
        den = float(norms[i] * norms[j]) + 1e-12
        rhos[cnt] = num / den
        cnt += 1
    return rhos


def _save_heatmaps(
    stack: np.ndarray,
    times: list[np.datetime64],
    out_path: Path,
    rng: np.random.Generator,
    n_show: int = 6,
) -> None:
    S = stack.shape[0]
    pick = rng.choice(S, size=min(n_show, S), replace=False)
    vmin = float(stack[pick].min())
    vmax = float(stack[pick].max())

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for ax, idx in zip(axes.flat, pick):
        im = ax.imshow(stack[idx], vmin=vmin, vmax=vmax, cmap="magma")
        ts = pd.Timestamp(times[idx]).strftime("%Y-%m-%d %H:%M")
        ax.set_title(f"#{idx}  {ts}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("exp(ℓ) latent ch0 — 6 random test samples", y=1.00)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        fraction=0.025, pad=0.02)
    cbar.set_label("exp(ℓ)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_cv_hist(cvs: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(cvs, bins=40, color="C0", edgecolor="white")
    ax.axvline(float(np.median(cvs)), color="k", linestyle="--",
               label=f"median={np.median(cvs):.3f}")
    ax.set_xlabel("Spatial CV of exp(ℓ) per sample")
    ax.set_ylabel("count")
    ax.set_title("Exp1: pixelwise diversity of latent ℓ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_pair_hist(rhos: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(rhos, bins=40, color="C3", edgecolor="white")
    ax.axvline(float(np.median(rhos)), color="k", linestyle="--",
               label=f"median={np.median(rhos):.3f}")
    ax.set_xlabel("Pearson corr of exp(ℓ) between random sample pairs")
    ax.set_ylabel("count")
    ax.set_title("Exp1: cross-sample spatial correlation (flow-dependence)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_exp1(
    ensemble_dir: str,
    figures_dir: str,
    metrics_dir: str,
    n_pairs: int = 1000,
    seed: int = 42,
) -> dict:
    set_plot_defaults()
    fig_dir = ensure_dir(figures_dir)
    met_dir = ensure_dir(metrics_dir)
    rng = np.random.default_rng(seed)

    stack, times = _load_logvar_stack(Path(ensemble_dir))
    print(f"[exp1] loaded stack shape={stack.shape}")

    cvs = _pixelwise_cv(stack)
    rhos = _pairwise_spatial_corr(stack, n_pairs=n_pairs, rng=rng)

    _save_heatmaps(stack, times, fig_dir / "exp1_logvar_heatmaps.png", rng)
    _save_cv_hist(cvs, fig_dir / "exp1_pixelwise_cv_hist.png")
    _save_pair_hist(rhos, fig_dir / "exp1_pairwise_correlation_hist.png")

    stats = {
        "cv": {
            "mean": float(cvs.mean()),
            "median": float(np.median(cvs)),
            "std": float(cvs.std()),
            "n": int(cvs.size),
        },
        "pairwise_correlation": {
            "mean": float(rhos.mean()),
            "median": float(np.median(rhos)),
            "std": float(rhos.std()),
            "n_pairs": int(rhos.size),
        },
    }
    with open(met_dir / "exp1_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[exp1] {stats}")
    return stats


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles_dit")
    p.add_argument("--figures_dir", default="outputs/figures_dit")
    p.add_argument("--metrics_dir", default="outputs/metrics_dit")
    p.add_argument("--n_pairs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_exp1(args.ensemble_dir, args.figures_dir, args.metrics_dir,
             args.n_pairs, args.seed)
