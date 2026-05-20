"""실험 4: 특정 시점에서 low-ℓ_m / high-ℓ_m 픽셀의 spread 비교 (bar plot).

이전 버전은 시계열이었으나, "한 시각에서 ℓ_m 큰 픽셀과 작은 픽셀이 실제
ensemble spread 차이를 보이는가" 가 본질이라 단일 시점 bar plot 으로 단순화.

각 그룹 (low / high) 에서 percentile pool 내 무작위 K개 픽셀 추출 후,
픽셀마다 두 양을 나란히 비교:
  - σ_ℓ  = √exp(ℓ_m(i,j))    (모델 예측 std)
  - ens_spread(i,j) = std_b(ens_{b,i,j})  (실제 멤버 분산)

Convolution padding artifact를 피하려 picker pool은 interior에서만.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset.denormalize import HOUR_TO_IDX

from .utils import (
    TEMP_IDX, ensure_dir, list_ensemble_files, load_ensemble_npz,
    load_norm_stats, set_plot_defaults,
)


def _temp_sigma_t(time_t, std) -> np.ndarray:
    """(H, W) — 해당 시각의 t 채널 denorm σ."""
    hour = pd.Timestamp(time_t).hour
    h_idx = HOUR_TO_IDX[int(hour)]
    return std[h_idx, TEMP_IDX].cpu().numpy()


def _pick_low_high_pools(
    rank_map: np.ndarray,
    percentile: float,
    margin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """interior 픽셀들 중 rank_map 하위/상위 percentile 전체를 pool로 반환.

    Args:
        rank_map: (H, W) ranking에 사용할 양수 map (예: K-space σ_ℓ).

    Returns:
        low_flat_idx, high_flat_idx: 1D flatten 인덱스 (각 ~percentile * interior).
    """
    H, W = rank_map.shape
    if margin * 2 >= min(H, W):
        raise ValueError(f"margin={margin} too large for ({H}, {W})")
    interior = np.zeros((H, W), dtype=bool)
    if margin > 0:
        interior[margin:H - margin, margin:W - margin] = True
    else:
        interior[:, :] = True
    valid_idx = np.where(interior.flatten())[0]
    valid_vals = rank_map.flatten()[valid_idx]

    n_valid = valid_idx.size
    cutoff = max(int(round(n_valid * percentile)), 1)
    order = np.argsort(valid_vals)
    low_pool = valid_idx[order[:cutoff]]
    high_pool = valid_idx[order[-cutoff:]]
    return low_pool, high_pool


def _pool_values(
    flat_indices: np.ndarray,
    lv_map: np.ndarray,
    ensemble: np.ndarray,
    sigma_t: np.ndarray,
) -> dict:
    """pool 픽셀에 대해 normalized 및 K-space σ_ℓ, ensemble spread 동시 반환."""
    H, W = lv_map.shape
    rows = (flat_indices // W).astype(int)
    cols = (flat_indices % W).astype(int)
    sigma_t_pool = sigma_t[rows, cols]
    sigma_ell_n = np.sqrt(lv_map[rows, cols])
    spread_n = ensemble[:, TEMP_IDX, rows, cols].std(axis=0, ddof=1)
    return {
        "ij": list(zip(rows.tolist(), cols.tolist())),
        "sigma_ell_norm":  sigma_ell_n.astype(np.float64),
        "ens_spread_norm": spread_n.astype(np.float64),
        "sigma_ell_K":     (sigma_ell_n * sigma_t_pool).astype(np.float64),
        "ens_spread_K":    (spread_n * sigma_t_pool).astype(np.float64),
    }


def _draw_panel_from_arrays(
    ax,
    arrs: list,
    percentile: float,
    group_label_low: str,
    group_label_high: str,
    title: str, ylabel: str,
) -> None:
    """1 panel: 4 arrays = [LOW σ_ℓ, LOW spread, HIGH σ_ℓ, HIGH spread]."""
    means = [float(a.mean()) for a in arrs]
    stds = [float(a.std()) for a in arrs]
    medians = [float(np.median(a)) for a in arrs]
    colors = ["C2", "C3", "C2", "C3"]
    x = np.arange(4)

    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors, edgecolor="k", alpha=0.55, width=0.65,
                  zorder=2)

    rng = np.random.default_rng(0)
    for xi, vals in zip(x, arrs):
        jitter = rng.uniform(-0.18, 0.18, size=vals.size)
        ax.scatter(xi + jitter, vals, s=5, alpha=0.35,
                   color="k", zorder=3, edgecolors="none")

    ax.axvline(1.5, color="k", linestyle="--", alpha=0.5)
    upper = max(float(a.max()) for a in arrs) * 1.18
    ax.set_ylim(0, upper)

    ax.text(0.5, upper * 0.96, group_label_low,
            ha="center", va="top", fontsize=9,
            color="C0", fontweight="bold")
    ax.text(2.5, upper * 0.96, group_label_high,
            ha="center", va="top", fontsize=9,
            color="C3", fontweight="bold")

    for b, m, med in zip(bars, means, medians):
        ax.text(b.get_x() + b.get_width() / 2, m + b.get_height() * 0.03,
                f"μ={m:.2g}\nmd={med:.2g}",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(["σ_ℓ", "spread", "σ_ℓ", "spread"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)


def _draw_panel(
    ax,
    low: dict, high: dict,
    sigma_key: str, spread_key: str,
    percentile: float,
    title: str, ylabel: str,
) -> None:
    """단일 sample용 — 픽셀 array를 그대로 _draw_panel_from_arrays 로 위임."""
    arrs = [low[sigma_key], low[spread_key],
            high[sigma_key], high[spread_key]]
    n_low = arrs[0].size
    n_high = arrs[2].size
    _draw_panel_from_arrays(
        ax, arrs, percentile,
        group_label_low=f"LOW ℓ_m\nbottom {percentile * 100:.0f}%   N={n_low}",
        group_label_high=f"HIGH ℓ_m\ntop {percentile * 100:.0f}%   N={n_high}",
        title=title, ylabel=ylabel,
    )


def _save_bar(
    low_norm: dict, high_norm: dict,
    low_K: dict, high_K: dict,
    sample_idx: int, time_t, out_path: Path,
    percentile: float, margin: int,
) -> None:
    """1×2 subplot: normalized space ranking + K-space ranking 비교."""
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    _draw_panel(
        axes[0], low_norm, high_norm,
        sigma_key="sigma_ell_norm", spread_key="ens_spread_norm",
        percentile=percentile,
        title="Normalized space  (rank by σ_ℓ_norm)",
        ylabel="std (normalized)",
    )
    _draw_panel(
        axes[1], low_K, high_K,
        sigma_key="sigma_ell_K", spread_key="ens_spread_K",
        percentile=percentile,
        title="Denormalized space  (rank by σ_ℓ_K)",
        ylabel="std (K)",
    )

    handles = [
        Patch(facecolor="C2", alpha=0.6, label="σ_ℓ = √exp(ℓ_m)"),
        Patch(facecolor="C3", alpha=0.6, label="ensemble spread"),
    ]
    fig.legend(handles=handles, loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.00), fontsize=10)

    ts = pd.Timestamp(time_t).strftime("%Y-%m-%d %H:%M")
    fig.suptitle(
        f"Exp4 — sample #{sample_idx}  {ts}  margin={margin}",
        y=1.05, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _summarize(arr: np.ndarray) -> dict:
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "n": int(arr.size),
    }


def run_exp4(
    ensemble_dir: str,
    figures_dir: str,
    percentile: float = 0.1,
    margin: int = 8,
    sample_idx: int | None = None,
    seed: int = 42,
) -> dict:
    """
    Args:
        percentile : pool 비율 (예: 0.1 = 하위/상위 10%). 그룹 전체 pool 사용.
        margin     : interior 한정 (boundary 띠 두께). 64x64면 8 권장.
        sample_idx : 특정 시점 인덱스. None이면 seed로 무작위 1개.
    """
    set_plot_defaults()
    fig_dir = ensure_dir(figures_dir)
    _, std = load_norm_stats()

    files = list_ensemble_files(Path(ensemble_dir))
    rng = np.random.default_rng(seed)
    if sample_idx is None:
        sample_idx = int(rng.integers(0, len(files)))

    es = load_ensemble_npz(files[sample_idx])
    lv_map = np.exp(es.log_var[TEMP_IDX])                       # (H, W)
    sigma_t = _temp_sigma_t(es.time_t, std)                     # (H, W)

    # Pool 두 종류: normalized σ_ℓ_n / K-space σ_ℓ_K 각각 ranking.
    sigma_ell_n_map = np.sqrt(lv_map)
    sigma_ell_K_map = sigma_ell_n_map * sigma_t

    low_n_idx, high_n_idx = _pick_low_high_pools(
        sigma_ell_n_map, percentile=percentile, margin=margin,
    )
    low_K_idx, high_K_idx = _pick_low_high_pools(
        sigma_ell_K_map, percentile=percentile, margin=margin,
    )

    low_norm = _pool_values(low_n_idx, lv_map, es.ensemble, sigma_t)
    high_norm = _pool_values(high_n_idx, lv_map, es.ensemble, sigma_t)
    low_K = _pool_values(low_K_idx, lv_map, es.ensemble, sigma_t)
    high_K = _pool_values(high_K_idx, lv_map, es.ensemble, sigma_t)

    out_path = fig_dir / f"exp4_bar_sample{sample_idx:05d}.png"
    _save_bar(
        low_norm, high_norm, low_K, high_K,
        sample_idx, es.time_t, out_path,
        percentile=percentile, margin=margin,
    )

    stats = {
        "sample_idx": sample_idx,
        "time_t": str(es.time_t),
        "percentile": percentile,
        "margin": margin,
        "normalized": {
            "low": {
                "sigma_ell": _summarize(low_norm["sigma_ell_norm"]),
                "ens_spread": _summarize(low_norm["ens_spread_norm"]),
            },
            "high": {
                "sigma_ell": _summarize(high_norm["sigma_ell_norm"]),
                "ens_spread": _summarize(high_norm["ens_spread_norm"]),
            },
        },
        "denormalized_K": {
            "low": {
                "sigma_ell": _summarize(low_K["sigma_ell_K"]),
                "ens_spread": _summarize(low_K["ens_spread_K"]),
            },
            "high": {
                "sigma_ell": _summarize(high_K["sigma_ell_K"]),
                "ens_spread": _summarize(high_K["ens_spread_K"]),
            },
        },
        "out_path": str(out_path),
    }
    print(f"[exp4] sample={sample_idx} time={es.time_t}")
    print(f"[exp4] norm  LOW  σ_ℓ μ={stats['normalized']['low']['sigma_ell']['mean']:.4f}  "
          f"spread μ={stats['normalized']['low']['ens_spread']['mean']:.4f}")
    print(f"[exp4] norm  HIGH σ_ℓ μ={stats['normalized']['high']['sigma_ell']['mean']:.4f}  "
          f"spread μ={stats['normalized']['high']['ens_spread']['mean']:.4f}")
    print(f"[exp4] K     LOW  σ_ℓ μ={stats['denormalized_K']['low']['sigma_ell']['mean']:.3f}  "
          f"spread μ={stats['denormalized_K']['low']['ens_spread']['mean']:.3f}")
    print(f"[exp4] K     HIGH σ_ℓ μ={stats['denormalized_K']['high']['sigma_ell']['mean']:.3f}  "
          f"spread μ={stats['denormalized_K']['high']['ens_spread']['mean']:.3f}")
    return stats


def _save_bar_aggregate(
    per_sample: dict,
    out_path: Path,
    percentile: float, margin: int, n_samples: int,
) -> None:
    """전체 sample 평균 1×2 subplot: 각 bar = sample 평균의 mean±std, 점=각 sample."""
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

    label_low = (f"LOW ℓ_m\nbottom {percentile * 100:.0f}%\n"
                 f"{n_samples} samples")
    label_high = (f"HIGH ℓ_m\ntop {percentile * 100:.0f}%\n"
                  f"{n_samples} samples")

    arrs_norm = [
        per_sample["norm_low_sigma"],
        per_sample["norm_low_spread"],
        per_sample["norm_high_sigma"],
        per_sample["norm_high_spread"],
    ]
    arrs_K = [
        per_sample["K_low_sigma"],
        per_sample["K_low_spread"],
        per_sample["K_high_sigma"],
        per_sample["K_high_spread"],
    ]
    _draw_panel_from_arrays(
        axes[0], arrs_norm, percentile,
        group_label_low=label_low, group_label_high=label_high,
        title="Normalized space  (per-sample pool means; rank by σ_ℓ_norm)",
        ylabel="std (normalized)",
    )
    _draw_panel_from_arrays(
        axes[1], arrs_K, percentile,
        group_label_low=label_low, group_label_high=label_high,
        title="Denormalized space  (per-sample pool means; rank by σ_ℓ_K)",
        ylabel="std (K)",
    )

    handles = [
        Patch(facecolor="C2", alpha=0.6, label="σ_ℓ = √exp(ℓ_m)"),
        Patch(facecolor="C3", alpha=0.6, label="ensemble spread"),
    ]
    fig.legend(handles=handles, loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.00), fontsize=10)
    fig.suptitle(
        f"Exp4 aggregate — {n_samples} samples  "
        f"percentile={percentile * 100:.0f}%  margin={margin}",
        y=1.05, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_exp4_aggregate(
    ensemble_dir: str,
    figures_dir: str,
    percentile: float = 0.1,
    margin: int = 8,
) -> dict:
    """모든 sample에 대해 LOW/HIGH pool의 sample별 평균을 모아 통계.

    각 sample s 마다:
        per-sample pool mean (4개씩 in normalized / K) 계산.
    그 결과 (S,) 짜리 array 8개를 sample 차원으로 통계 + plot.
    """
    set_plot_defaults()
    fig_dir = ensure_dir(figures_dir)
    _, std = load_norm_stats()

    files = list_ensemble_files(Path(ensemble_dir))
    keys = [
        "norm_low_sigma", "norm_low_spread",
        "norm_high_sigma", "norm_high_spread",
        "K_low_sigma", "K_low_spread",
        "K_high_sigma", "K_high_spread",
    ]
    per_sample = {k: [] for k in keys}

    for f in files:
        es = load_ensemble_npz(f)
        lv_map = np.exp(es.log_var[TEMP_IDX])
        sigma_t = _temp_sigma_t(es.time_t, std)
        sigma_ell_n_map = np.sqrt(lv_map)
        sigma_ell_K_map = sigma_ell_n_map * sigma_t

        low_n_idx, high_n_idx = _pick_low_high_pools(
            sigma_ell_n_map, percentile=percentile, margin=margin,
        )
        low_K_idx, high_K_idx = _pick_low_high_pools(
            sigma_ell_K_map, percentile=percentile, margin=margin,
        )

        low_n = _pool_values(low_n_idx, lv_map, es.ensemble, sigma_t)
        high_n = _pool_values(high_n_idx, lv_map, es.ensemble, sigma_t)
        low_K = _pool_values(low_K_idx, lv_map, es.ensemble, sigma_t)
        high_K = _pool_values(high_K_idx, lv_map, es.ensemble, sigma_t)

        per_sample["norm_low_sigma"].append(float(low_n["sigma_ell_norm"].mean()))
        per_sample["norm_low_spread"].append(float(low_n["ens_spread_norm"].mean()))
        per_sample["norm_high_sigma"].append(float(high_n["sigma_ell_norm"].mean()))
        per_sample["norm_high_spread"].append(float(high_n["ens_spread_norm"].mean()))
        per_sample["K_low_sigma"].append(float(low_K["sigma_ell_K"].mean()))
        per_sample["K_low_spread"].append(float(low_K["ens_spread_K"].mean()))
        per_sample["K_high_sigma"].append(float(high_K["sigma_ell_K"].mean()))
        per_sample["K_high_spread"].append(float(high_K["ens_spread_K"].mean()))

    for k in keys:
        per_sample[k] = np.array(per_sample[k])

    out_path = fig_dir / (
        f"exp4_bar_aggregate_p{int(percentile * 100):02d}"
        f"_m{margin}.png"
    )
    _save_bar_aggregate(
        per_sample, out_path,
        percentile=percentile, margin=margin, n_samples=len(files),
    )

    stats = {
        "n_samples": len(files),
        "percentile": percentile,
        "margin": margin,
        "normalized": {
            "low": {
                "sigma_ell": _summarize(per_sample["norm_low_sigma"]),
                "ens_spread": _summarize(per_sample["norm_low_spread"]),
            },
            "high": {
                "sigma_ell": _summarize(per_sample["norm_high_sigma"]),
                "ens_spread": _summarize(per_sample["norm_high_spread"]),
            },
        },
        "denormalized_K": {
            "low": {
                "sigma_ell": _summarize(per_sample["K_low_sigma"]),
                "ens_spread": _summarize(per_sample["K_low_spread"]),
            },
            "high": {
                "sigma_ell": _summarize(per_sample["K_high_sigma"]),
                "ens_spread": _summarize(per_sample["K_high_spread"]),
            },
        },
        "out_path": str(out_path),
    }
    print(f"[exp4-aggregate] n_samples={len(files)}")
    print(f"[exp4-aggregate] norm  LOW  σ_ℓ μ={stats['normalized']['low']['sigma_ell']['mean']:.4f}  "
          f"spread μ={stats['normalized']['low']['ens_spread']['mean']:.4f}")
    print(f"[exp4-aggregate] norm  HIGH σ_ℓ μ={stats['normalized']['high']['sigma_ell']['mean']:.4f}  "
          f"spread μ={stats['normalized']['high']['ens_spread']['mean']:.4f}")
    print(f"[exp4-aggregate] K     LOW  σ_ℓ μ={stats['denormalized_K']['low']['sigma_ell']['mean']:.3f}  "
          f"spread μ={stats['denormalized_K']['low']['ens_spread']['mean']:.3f}")
    print(f"[exp4-aggregate] K     HIGH σ_ℓ μ={stats['denormalized_K']['high']['sigma_ell']['mean']:.3f}  "
          f"spread μ={stats['denormalized_K']['high']['ens_spread']['mean']:.3f}")
    return stats


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles")
    p.add_argument("--figures_dir", default="outputs/figures")
    p.add_argument("--percentile", type=float, default=0.1,
                   help="하위/상위 percentile pool 비율 (예: 0.1 = 10%).")
    p.add_argument("--margin", type=int, default=8,
                   help="interior boundary 띠 두께. 0이면 전체 grid.")
    p.add_argument("--sample_idx", type=int, default=None,
                   help="고정 시점 인덱스. None이면 seed로 무작위.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--aggregate", action="store_true",
                   help="전체 sample 평균 모드. 설정 시 sample_idx 무시.")
    args = p.parse_args()
    if args.aggregate:
        run_exp4_aggregate(
            args.ensemble_dir, args.figures_dir,
            percentile=args.percentile, margin=args.margin,
        )
    else:
        run_exp4(
            args.ensemble_dir, args.figures_dir,
            percentile=args.percentile,
            margin=args.margin,
            sample_idx=args.sample_idx,
            seed=args.seed,
        )
