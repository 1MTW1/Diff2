"""실험 2: ℓ_m vs ensemble spread correlation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    TEMP_IDX, ensure_dir, list_ensemble_files, load_ensemble_npz,
    set_plot_defaults,
)


def _per_sample_corr(
    sigma_ell_flat: np.ndarray, spread_flat: np.ndarray,
) -> float:
    a = sigma_ell_flat - sigma_ell_flat.mean()
    b = spread_flat - spread_flat.mean()
    num = float((a * b).sum())
    den = float(np.sqrt((a * a).sum() * (b * b).sum())) + 1e-12
    return num / den


def _ensemble_spread(ens: np.ndarray) -> np.ndarray:
    """ens: (N, C, H, W) → (C, H, W) std (ddof=1)."""
    return ens.std(axis=0, ddof=1)


def _save_scatter_examples(
    items: list[tuple[int, np.ndarray, np.ndarray, float]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (idx, x, y, r) in zip(axes, items):
        ax.scatter(x, y, s=4, alpha=0.35, color="C0")
        lim = max(float(x.max()), float(y.max()))
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("σ_ℓ = sqrt(exp(ℓ_m))")
        ax.set_ylabel("ensemble spread")
        ax.set_title(f"sample #{idx}  r={r:.3f}")
    fig.suptitle("Exp2: σ_ℓ vs ensemble spread (temperature)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_corr_hist(corrs: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(corrs, bins=40, color="C2", edgecolor="white")
    ax.axvline(float(np.median(corrs)), color="k", linestyle="--",
               label=f"median={np.median(corrs):.3f}")
    ax.set_xlabel("Pearson r per sample")
    ax.set_ylabel("count")
    ax.set_title("Exp2: per-sample (σ_ℓ, spread) correlation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _pairwise_spatial_corr(
    stack: np.ndarray, n_pairs: int, rng: np.random.Generator,
) -> np.ndarray:
    """stack: (S, H, W) → 무작위 sample 쌍 spatial Pearson 분포 (n_pairs,)."""
    S = stack.shape[0]
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


def _load_gt_temp_stack(files: list[Path]) -> np.ndarray:
    """모든 sample의 GT 온도장 → (S, H, W)."""
    arrs = [load_ensemble_npz(f).x_t_true[TEMP_IDX].astype(np.float32)
            for f in files]
    return np.stack(arrs, axis=0)


def _save_comparison_hist(
    per_sample_corrs: np.ndarray,
    gt_pair_corrs: np.ndarray,
    out_path: Path,
) -> None:
    """exp2의 per-sample 상관과 GT 시점간 공간상관을 같은 axis에 overlay."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(
        per_sample_corrs, bins=40, alpha=0.65, color="C2",
        density=True,
        label=f"σ_ℓ ↔ spread  per sample  "
              f"(median={np.median(per_sample_corrs):.3f})",
    )
    ax.hist(
        gt_pair_corrs, bins=40, alpha=0.55, color="C7",
        density=True,
        label=f"GT(t) cross-sample baseline  "
              f"(median={np.median(gt_pair_corrs):.3f})",
    )
    ax.axvline(0.0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("density")
    ax.set_title(
        "Exp2: σ_ℓ↔spread vs GT temperature cross-sample correlation"
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_exp2(
    ensemble_dir: str,
    figures_dir: str,
    metrics_dir: str,
    seed: int = 42,
    n_gt_pairs: int = 1000,
) -> dict:
    set_plot_defaults()
    fig_dir = ensure_dir(figures_dir)
    met_dir = ensure_dir(metrics_dir)
    rng = np.random.default_rng(seed)

    files = list_ensemble_files(Path(ensemble_dir))
    corrs = np.empty(len(files), dtype=np.float64)
    cache: list[tuple[int, np.ndarray, np.ndarray]] = []

    for i, f in enumerate(files):
        es = load_ensemble_npz(f)
        sigma_ell = np.sqrt(np.exp(es.log_var[TEMP_IDX]))
        spread = _ensemble_spread(es.ensemble)[TEMP_IDX]
        x = sigma_ell.flatten()
        y = spread.flatten()
        corrs[i] = _per_sample_corr(x, y)
        cache.append((i, x, y))

    # 3 random scatter examples
    picks = rng.choice(len(files), size=min(3, len(files)), replace=False)
    items = [(cache[p][0], cache[p][1], cache[p][2], float(corrs[p]))
             for p in picks]
    _save_scatter_examples(items, fig_dir / "exp2_scatter_examples.png")
    _save_corr_hist(corrs, fig_dir / "exp2_correlation_hist.png")

    # ── GT 시점간 공간 상관 분포 (baseline) ──
    gt_stack = _load_gt_temp_stack(files)
    gt_pair_rhos = _pairwise_spatial_corr(gt_stack, n_gt_pairs, rng)
    _save_comparison_hist(
        corrs, gt_pair_rhos, fig_dir / "exp2_vs_gt_baseline.png",
    )

    stats = {
        "per_sample_correlation": {
            "mean": float(corrs.mean()),
            "median": float(np.median(corrs)),
            "std": float(corrs.std()),
            "n_samples": int(corrs.size),
        },
        "gt_pair_correlation": {
            "mean": float(gt_pair_rhos.mean()),
            "median": float(np.median(gt_pair_rhos)),
            "std": float(gt_pair_rhos.std()),
            "n_pairs": int(gt_pair_rhos.size),
        },
    }
    with open(met_dir / "exp2_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[exp2] {stats}")
    return stats


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles")
    p.add_argument("--figures_dir", default="outputs/figures")
    p.add_argument("--metrics_dir", default="outputs/metrics")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_gt_pairs", type=int, default=1000)
    args = p.parse_args()
    run_exp2(args.ensemble_dir, args.figures_dir, args.metrics_dir,
             seed=args.seed, n_gt_pairs=args.n_gt_pairs)
