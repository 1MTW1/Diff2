"""실험 5 (v2 LDM/DiT): composite map (quiver + shading) — 픽셀 공간.

6 experiment 중 유일하게 **픽셀 공간**에서 그려진다. latent 캐시가 아닌 디코딩된
픽셀 필드(`ensemble_pixel`, `x_t_true_pixel`)를 사용하여 GT + 14 멤버의
diff-from-mean + ensemble mean 의 4×4 (16-panel) composite map 과 spread map 을
그린다. 픽셀 필드는 물리 단위(K)로 역정규화된다 (`_denorm_field`).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset.denormalize import HOUR_TO_IDX

from .utils import (
    TEMP_IDX, U_IDX, V_IDX, ensure_dir, list_ensemble_files,
    load_ensemble_npz, load_norm_stats, set_plot_defaults,
)


_QUIVER_STRIDE = 8           # 64/8 = 8 arrows per side
_QUIVER_SCALE = 400.0        # m/s per axis-width; ↑ 클수록 화살표 짧음
_QUIVER_CLIP_MAG = 60.0      # m/s, 시각화용 magnitude cap
_QUIVER_KEY_MAGNITUDE = 20.0  # m/s, 화살표 길이 reference


def _denorm_field(
    x_norm: np.ndarray, time_t, mean, std,
) -> np.ndarray:
    """(C, H, W) 또는 (N, C, H, W) → 역정규화."""
    hour = pd.Timestamp(time_t).hour
    h_idx = HOUR_TO_IDX[int(hour)]
    mu = mean[h_idx].cpu().numpy()       # (C, H, W)
    sig = std[h_idx].cpu().numpy()
    return x_norm * sig + mu


def _denorm_spread_field(
    s_norm: np.ndarray, time_t, std,
) -> np.ndarray:
    """spread는 mean shift 없음."""
    hour = pd.Timestamp(time_t).hour
    h_idx = HOUR_TO_IDX[int(hour)]
    sig = std[h_idx].cpu().numpy()
    return s_norm * sig


def _clip_wind(u: np.ndarray, v: np.ndarray, max_mag: float):
    """방향 보존하면서 magnitude를 max_mag로 cap."""
    mag = np.sqrt(u * u + v * v)
    scale = np.minimum(1.0, max_mag / (mag + 1e-9))
    return u * scale, v * scale


def _plot_one_panel(ax, temp, u, v, vmin, vmax, title, cmap="RdBu_r"):
    im = ax.pcolormesh(temp, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="auto")
    H, W = temp.shape
    ys, xs = np.mgrid[0:H, 0:W]
    s = _QUIVER_STRIDE
    u_c, v_c = _clip_wind(u, v, _QUIVER_CLIP_MAG)
    q = ax.quiver(
        xs[::s, ::s], ys[::s, ::s],
        u_c[::s, ::s], v_c[::s, ::s],
        color="k",
        scale=_QUIVER_SCALE, scale_units="width",
        width=0.0035, headwidth=4, headlength=5,
        pivot="middle",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return im, q


def _save_composite(
    sample: dict, out_path: Path,
) -> None:
    """16 panel: GT(abs) + 14 members(diff from mean) + ensemble mean(abs)."""
    fig, axes = plt.subplots(4, 4, figsize=(17, 16))
    flat = axes.flatten()

    gt = sample["gt"]                                  # (C, H, W)
    members = sample["members"]                        # (14, C, H, W)
    mean_field = sample["mean"]                        # (C, H, W)

    # ── 색범위 ────────────────────────────────────────────────────
    abs_pool = np.concatenate(
        [gt[TEMP_IDX].flatten(), mean_field[TEMP_IDX].flatten()]
    )
    abs_vmin = float(np.percentile(abs_pool, 1))
    abs_vmax = float(np.percentile(abs_pool, 99))

    diff_pool = (members[:, TEMP_IDX] - mean_field[TEMP_IDX]).flatten()
    diff_max = float(np.percentile(np.abs(diff_pool), 99))
    diff_max = max(diff_max, 1e-6)

    im_abs = None
    im_diff = None
    last_q = None

    # 0: GT (절대)
    im_abs, last_q = _plot_one_panel(
        flat[0],
        gt[TEMP_IDX], gt[U_IDX], gt[V_IDX],
        abs_vmin, abs_vmax, "Ground Truth", cmap="RdBu_r",
    )
    # 1..14: members (diff = member − mean)
    for k in range(14):
        m = members[k]
        diff_t = m[TEMP_IDX] - mean_field[TEMP_IDX]
        im_diff, last_q = _plot_one_panel(
            flat[1 + k],
            diff_t, m[U_IDX], m[V_IDX],
            -diff_max, diff_max, f"Member {k + 1} − Mean", cmap="RdBu_r",
        )
    # 15: ensemble mean (절대)
    im_abs, last_q = _plot_one_panel(
        flat[15],
        mean_field[TEMP_IDX], mean_field[U_IDX], mean_field[V_IDX],
        abs_vmin, abs_vmax, "Ensemble Mean", cmap="RdBu_r",
    )

    fig.suptitle(
        f"Exp5 composite — "
        f"{pd.Timestamp(sample['time_t']).strftime('%Y-%m-%d %H:%M')}",
        y=0.98, fontsize=13,
    )

    # ── 수동 레이아웃: 오른쪽에 두 개의 colorbar + 하단 quiver key ──
    fig.subplots_adjust(
        left=0.03, right=0.86, top=0.94, bottom=0.07,
        hspace=0.18, wspace=0.05,
    )

    cax_abs = fig.add_axes([0.88, 0.55, 0.020, 0.34])
    cb_abs = fig.colorbar(im_abs, cax=cax_abs)
    cb_abs.set_label("absolute T (K)  — GT & Mean")

    cax_diff = fig.add_axes([0.88, 0.12, 0.020, 0.34])
    cb_diff = fig.colorbar(im_diff, cax=cax_diff)
    cb_diff.set_label("member − mean  (K)")

    # quiver key (figure 좌하단)
    if last_q is not None:
        axes[0, 0].quiverkey(
            last_q, 0.04, 0.02,
            _QUIVER_KEY_MAGNITUDE,
            f"{_QUIVER_KEY_MAGNITUDE:.0f} m/s  (cap {_QUIVER_CLIP_MAG:.0f})",
            labelpos="E", coordinates="figure",
        )

    fig.savefig(out_path)
    plt.close(fig)


def _save_spread(
    spread_norm: np.ndarray, spread_denorm: np.ndarray,
    time_t, out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, data, title, unit in zip(
        axes,
        [spread_norm[TEMP_IDX], spread_denorm[TEMP_IDX]],
        [f"Normalized spread (t)",
         f"Denormalized spread (t)"],
        ["", "K"],
    ):
        im = ax.imshow(data, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        if unit:
            cb.set_label(unit)
    fig.suptitle(
        f"Exp5 spread — {pd.Timestamp(time_t).strftime('%Y-%m-%d %H:%M')}",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_exp5(
    ensemble_dir: str,
    figures_dir: str,
    n_samples: int = 3,
    seed: int = 42,
) -> dict:
    set_plot_defaults()
    fig_dir = ensure_dir(figures_dir)
    mean, std = load_norm_stats()

    files = list_ensemble_files(Path(ensemble_dir))
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(files), size=min(n_samples, len(files)),
                       replace=False)

    chosen = []
    for k, idx in enumerate(picks, start=1):
        es = load_ensemble_npz(files[idx])
        if es.ensemble_pixel is None or es.x_t_true_pixel is None:
            raise KeyError(
                f"{files[idx]} 에 픽셀 캐시 (ensemble_pixel/x_t_true_pixel) 가 "
                f"없습니다 — experiments_dit.ensemble_inference 로 생성하세요."
            )
        gt_d = _denorm_field(es.x_t_true_pixel, es.time_t, mean, std)     # (C,H,W)
        members_d = _denorm_field(es.ensemble_pixel, es.time_t, mean, std)  # (N,C,H,W)
        mean_d = members_d.mean(axis=0)                                   # (C,H,W)

        sample = {
            "gt": gt_d,
            "members": members_d[:14],
            "mean": mean_d,
            "time_t": es.time_t,
        }
        _save_composite(sample, fig_dir / f"exp5_composite_sample{k}.png")

        spread_norm = es.ensemble_pixel.std(axis=0, ddof=1)               # (C,H,W)
        spread_d = _denorm_spread_field(spread_norm, es.time_t, std)
        _save_spread(
            spread_norm, spread_d, es.time_t,
            fig_dir / f"exp5_spread_sample{k}.png",
        )
        chosen.append({
            "idx": int(idx),
            "time_t": str(es.time_t),
            "file": str(files[idx]),
        })

    print(f"[exp5] picked={chosen}")
    return {"picked": chosen}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles_dit")
    p.add_argument("--figures_dir", default="outputs/figures_dit")
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_exp5(args.ensemble_dir, args.figures_dir, args.n_samples, args.seed)
