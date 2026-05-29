"""실험 4 (v2 LDM/DiT): 한 시점에서 low-/high-log_var latent 픽셀의 spread 비교.

"한 시각에서 log_var 큰 latent 픽셀과 작은 픽셀이 실제 ensemble spread 차이를
보이는가" 를 단일 시점 bar plot 으로 본다.

각 그룹 (low / high) 에서 percentile pool 내 latent 픽셀을 모아, 픽셀마다
두 양을 나란히 비교:
  - σ_ℓ  = √exp(log_var(i,j))       (모델 예측 std)
  - ens_spread(i,j) = std_b(ens_{b,i,j})  (실제 멤버 분산)

v1 exp4는 정규화 공간과 K-space 두 panel을 그렸으나, v2 분석은 diffusion latent
공간에서만 이루어지므로 K-space 절반을 제거하고 latent 공간 단일 panel로 단순화.
latent 격자가 16×16 으로 작으므로 boundary margin 기본값을 줄였다.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import (
    LATENT_CH, ensure_dir, list_ensemble_files, load_ensemble_npz,
    set_plot_defaults,
)


def _pick_low_high_pools(
    rank_map: np.ndarray,
    percentile: float,
    margin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """interior 픽셀들 중 rank_map 하위/상위 percentile 전체를 pool로 반환.

    Args:
        rank_map: (H, W) ranking에 사용할 양수 map (latent σ_ℓ).

    Returns:
        low_flat_idx, high_flat_idx: 1D flatten 인덱스.
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
) -> dict:
    """pool 픽셀에 대해 latent σ_ℓ, ensemble spread 반환."""
    H, W = lv_map.shape
    rows = (flat_indices // W).astype(int)
    cols = (flat_indices % W).astype(int)
    sigma_ell = np.sqrt(lv_map[rows, cols])
    spread = ensemble[:, LATENT_CH, rows, cols].std(axis=0, ddof=1)
    return {
        "ij": list(zip(rows.tolist(), cols.tolist())),
        "sigma_ell": sigma_ell.astype(np.float64),
        "ens_spread": spread.astype(np.float64),
    }


def _draw_panel_from_arrays(
    ax,
    arrs: list,
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
    upper = max(upper, 1e-9)
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
    percentile: float,
    title: str, ylabel: str,
) -> None:
    """단일 sample용 — 픽셀 array를 그대로 _draw_panel_from_arrays 로 위임."""
    arrs = [low["sigma_ell"], low["ens_spread"],
            high["sigma_ell"], high["ens_spread"]]
    n_low = arrs[0].size
    n_high = arrs[2].size
    _draw_panel_from_arrays(
        ax, arrs,
        group_label_low=f"LOW log_var\nbottom {percentile * 100:.0f}%   "
                        f"N={n_low}",
        group_label_high=f"HIGH log_var\ntop {percentile * 100:.0f}%   "
                         f"N={n_high}",
        title=title, ylabel=ylabel,
    )


def _save_bar(
    low: dict, high: dict,
    sample_idx: int, time_t, out_path: Path,
    percentile: float, margin: int,
) -> None:
    """단일 panel: latent 공간 ranking 비교."""
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    _draw_panel(
        ax, low, high,
        percentile=percentile,
        title="Latent space  (rank by σ_ℓ)",
        ylabel="std (latent)",
    )

    handles = [
        Patch(facecolor="C2", alpha=0.6, label="σ_ℓ = √exp(log_var)"),
        Patch(facecolor="C3", alpha=0.6, label="ensemble spread"),
    ]
    fig.legend(handles=handles, loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.00), fontsize=10)

    ts = pd.Timestamp(time_t).strftime("%Y-%m-%d %H:%M")
    fig.suptitle(
        f"Exp4 — sample #{sample_idx}  {ts}  margin={margin}",
        y=1.06, fontsize=12,
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
    margin: int = 2,
    sample_idx: int | None = None,
    seed: int = 42,
) -> dict:
    """
    Args:
        percentile : pool 비율 (예: 0.1 = 하위/상위 10%). 그룹 전체 pool 사용.
        margin     : interior 한정 (boundary 띠 두께). latent 16x16이라 2 권장.
        sample_idx : 특정 시점 인덱스. None이면 seed로 무작위 1개.
    """
    set_plot_defaults()
    fig_dir = ensure_dir(figures_dir)

    files = list_ensemble_files(Path(ensemble_dir))
    rng = np.random.default_rng(seed)
    if sample_idx is None:
        sample_idx = int(rng.integers(0, len(files)))

    es = load_ensemble_npz(files[sample_idx])
    lv_map = np.exp(es.log_var[LATENT_CH])                      # (H_z, W_z)
    sigma_ell_map = np.sqrt(lv_map)

    low_idx, high_idx = _pick_low_high_pools(
        sigma_ell_map, percentile=percentile, margin=margin,
    )
    low = _pool_values(low_idx, lv_map, es.ensemble)
    high = _pool_values(high_idx, lv_map, es.ensemble)

    out_path = fig_dir / f"exp4_bar_sample{sample_idx:05d}.png"
    _save_bar(
        low, high, sample_idx, es.time_t, out_path,
        percentile=percentile, margin=margin,
    )

    stats = {
        "sample_idx": sample_idx,
        "time_t": str(es.time_t),
        "percentile": percentile,
        "margin": margin,
        "latent": {
            "low": {
                "sigma_ell": _summarize(low["sigma_ell"]),
                "ens_spread": _summarize(low["ens_spread"]),
            },
            "high": {
                "sigma_ell": _summarize(high["sigma_ell"]),
                "ens_spread": _summarize(high["ens_spread"]),
            },
        },
        "out_path": str(out_path),
    }
    print(f"[exp4] sample={sample_idx} time={es.time_t}")
    print(f"[exp4] latent LOW  σ_ℓ μ={stats['latent']['low']['sigma_ell']['mean']:.4f}  "
          f"spread μ={stats['latent']['low']['ens_spread']['mean']:.4f}")
    print(f"[exp4] latent HIGH σ_ℓ μ={stats['latent']['high']['sigma_ell']['mean']:.4f}  "
          f"spread μ={stats['latent']['high']['ens_spread']['mean']:.4f}")
    return stats


def _save_bar_aggregate(
    per_sample: dict,
    out_path: Path,
    percentile: float, margin: int, n_samples: int,
) -> None:
    """전체 sample 평균 단일 panel: 각 bar = sample 평균의 mean±std, 점=각 sample."""
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    label_low = (f"LOW log_var\nbottom {percentile * 100:.0f}%\n"
                 f"{n_samples} samples")
    label_high = (f"HIGH log_var\ntop {percentile * 100:.0f}%\n"
                  f"{n_samples} samples")

    arrs = [
        per_sample["low_sigma"],
        per_sample["low_spread"],
        per_sample["high_sigma"],
        per_sample["high_spread"],
    ]
    _draw_panel_from_arrays(
        ax, arrs,
        group_label_low=label_low, group_label_high=label_high,
        title="Latent space  (per-sample pool means; rank by σ_ℓ)",
        ylabel="std (latent)",
    )

    handles = [
        Patch(facecolor="C2", alpha=0.6, label="σ_ℓ = √exp(log_var)"),
        Patch(facecolor="C3", alpha=0.6, label="ensemble spread"),
    ]
    fig.legend(handles=handles, loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.00), fontsize=10)
    fig.suptitle(
        f"Exp4 aggregate — {n_samples} samples  "
        f"percentile={percentile * 100:.0f}%  margin={margin}",
        y=1.06, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_exp4_aggregate(
    ensemble_dir: str,
    figures_dir: str,
    percentile: float = 0.1,
    margin: int = 2,
) -> dict:
    """모든 sample에 대해 LOW/HIGH pool의 sample별 평균을 모아 통계.

    각 sample s 마다 per-sample pool mean (4개) 계산. 그 결과 (S,) 짜리
    array 4개를 sample 차원으로 통계 + plot.
    """
    set_plot_defaults()
    fig_dir = ensure_dir(figures_dir)

    files = list_ensemble_files(Path(ensemble_dir))
    keys = ["low_sigma", "low_spread", "high_sigma", "high_spread"]
    per_sample = {k: [] for k in keys}

    for f in files:
        es = load_ensemble_npz(f)
        lv_map = np.exp(es.log_var[LATENT_CH])
        sigma_ell_map = np.sqrt(lv_map)

        low_idx, high_idx = _pick_low_high_pools(
            sigma_ell_map, percentile=percentile, margin=margin,
        )
        low = _pool_values(low_idx, lv_map, es.ensemble)
        high = _pool_values(high_idx, lv_map, es.ensemble)

        per_sample["low_sigma"].append(float(low["sigma_ell"].mean()))
        per_sample["low_spread"].append(float(low["ens_spread"].mean()))
        per_sample["high_sigma"].append(float(high["sigma_ell"].mean()))
        per_sample["high_spread"].append(float(high["ens_spread"].mean()))

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
        "latent": {
            "low": {
                "sigma_ell": _summarize(per_sample["low_sigma"]),
                "ens_spread": _summarize(per_sample["low_spread"]),
            },
            "high": {
                "sigma_ell": _summarize(per_sample["high_sigma"]),
                "ens_spread": _summarize(per_sample["high_spread"]),
            },
        },
        "out_path": str(out_path),
    }
    print(f"[exp4-aggregate] n_samples={len(files)}")
    print(f"[exp4-aggregate] latent LOW  σ_ℓ μ={stats['latent']['low']['sigma_ell']['mean']:.4f}  "
          f"spread μ={stats['latent']['low']['ens_spread']['mean']:.4f}")
    print(f"[exp4-aggregate] latent HIGH σ_ℓ μ={stats['latent']['high']['sigma_ell']['mean']:.4f}  "
          f"spread μ={stats['latent']['high']['ens_spread']['mean']:.4f}")
    return stats


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles_dit")
    p.add_argument("--figures_dir", default="outputs/figures_dit")
    p.add_argument("--percentile", type=float, default=0.1,
                   help="하위/상위 percentile pool 비율 (예: 0.1 = 10%).")
    p.add_argument("--margin", type=int, default=2,
                   help="interior boundary 띠 두께. 0이면 전체 latent grid.")
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
