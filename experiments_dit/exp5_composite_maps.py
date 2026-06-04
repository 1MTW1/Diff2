"""실험 5 (v2 LDM/DiT): composite map (quiver + shading) — 픽셀 공간.

세 lead (t-1, t, t+1) 각각에 대해, n_samples 개의 시점에서 4×4(16-panel)
composite map 을 그린다 (lead × sample 당 1 그림):
    - panel 0      : GT (절대 T 음영 + GT 바람 quiver)
    - panel 1..14  : GT − member_k (T 차이 음영 + (GT−member) 바람 차이 quiver)
    - panel 15     : ensemble mean (절대 T 음영 + 평균 바람 quiver)
공유 colorbar 2개:
    - 절대(panel 0·15) : {GT_T, mean_T} 의 (min, max)
    - 차이(panel 1..14): 모든 (GT_T − member_T) 의 (min, max)   ← literal min/max
픽셀 필드는 물리 단위(climatology 역표준화 완료). 이웃(t±1)은 ±1일(±24h) 시점.
입력 캐시는 이웃 저장 버전 ensemble_inference 로 생성해야 한다.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from models.climatology import doy_slot

from .utils import (
    TEMP_IDX, U_IDX, V_IDX, ensure_dir, list_ensemble_files,
    load_ensemble_npz, set_plot_defaults,
)


_QUIVER_STRIDE = 8           # 64/8 = 8 arrows per side
_QUIVER_SCALE = 400.0        # m/s per axis-width; ↑ 클수록 화살표 짧음
_QUIVER_CLIP_MAG = 60.0      # m/s, 시각화용 magnitude cap
_QUIVER_KEY_MAGNITUDE = 20.0  # m/s, 화살표 길이 reference

_N_MEMBERS_SHOWN = 14        # panel 1..14
# (라벨, 파일명 토큰, 멤버 픽셀 키, GT 픽셀 키, 중심으로부터 시간 오프셋[h])
# 이웃은 ±1일(±24h) — v4 데이터는 매일 00 UTC 스냅샷.
_LEADS = [
    ("t-1", "tm1", "ensemble_pixel_tm1", "x_tm1_true_pixel", -24),
    ("t",   "t",   "ensemble_pixel",     "x_t_true_pixel",     0),
    ("t+1", "tp1", "ensemble_pixel_tp1", "x_tp1_true_pixel",  24),
]


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
    """16 panel: GT(abs) + 14×(GT − member)(diff) + ensemble mean(abs).

    sample: {gt (C,H,W), members (≥14,C,H,W), mean (C,H,W), time_t, lead}.
    음영=T, quiver=바람. diff panel 의 quiver 는 (GT − member) 바람 차이.
    """
    fig, axes = plt.subplots(4, 4, figsize=(17, 16))
    flat = axes.flatten()

    gt = sample["gt"]                                  # (C, H, W)
    members = sample["members"][:_N_MEMBERS_SHOWN]     # (14, C, H, W)
    mean_field = sample["mean"]                        # (C, H, W)
    lead = sample["lead"]

    # ── 색범위 (명시 spec: literal min/max) ───────────────────────
    abs_pool = np.concatenate(
        [gt[TEMP_IDX].flatten(), mean_field[TEMP_IDX].flatten()]
    )
    abs_vmin, abs_vmax = float(abs_pool.min()), float(abs_pool.max())

    diff_pool = (gt[TEMP_IDX] - members[:, TEMP_IDX]).flatten()
    diff_vmin, diff_vmax = float(diff_pool.min()), float(diff_pool.max())
    if diff_vmin == diff_vmax:                         # degenerate guard
        diff_vmin, diff_vmax = diff_vmin - 1e-6, diff_vmax + 1e-6

    im_abs = im_diff = last_q = None

    # 0: GT (절대)
    im_abs, last_q = _plot_one_panel(
        flat[0],
        gt[TEMP_IDX], gt[U_IDX], gt[V_IDX],
        abs_vmin, abs_vmax, "Ground Truth", cmap="RdBu_r",
    )
    # 1..14: GT − member_k  (음영=T 차이, quiver=바람 차이)
    for k in range(_N_MEMBERS_SHOWN):
        m = members[k]
        diff_t = gt[TEMP_IDX] - m[TEMP_IDX]
        du, dv = gt[U_IDX] - m[U_IDX], gt[V_IDX] - m[V_IDX]
        im_diff, last_q = _plot_one_panel(
            flat[1 + k],
            diff_t, du, dv,
            diff_vmin, diff_vmax, f"GT − Member {k + 1}", cmap="RdBu_r",
        )
    # 15: ensemble mean (절대)
    im_abs, last_q = _plot_one_panel(
        flat[15],
        mean_field[TEMP_IDX], mean_field[U_IDX], mean_field[V_IDX],
        abs_vmin, abs_vmax, "Ensemble Mean", cmap="RdBu_r",
    )

    fig.suptitle(
        f"Exp5 composite — lead {lead}  ·  "
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
    cb_diff.set_label("GT − member  (K)")

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
    spread_phys: np.ndarray, spread_norm: np.ndarray, time_t, out_path: Path,
) -> None:
    """앙상블 멤버 spread 비교 — 물리공간 vs 정규화공간 (2행 × T/u/v 3열).

    상단: 물리단위 spread (K, m/s) = std(members).
    하단: 정규화 spread (무차원) = climatology σ(doy) 로 표준화한 멤버의 std
          = 물리 spread / c_sig[doy]  (std((x−c_mu)/c_sig) = std(x)/c_sig).
    """
    var_cols = [(TEMP_IDX, "T", "K"), (U_IDX, "u", "m/s"), (V_IDX, "v", "m/s")]
    rows = [(spread_phys, "physical"), (spread_norm, "normalized")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for i, (field, space) in enumerate(rows):
        for j, (ci, name, unit) in enumerate(var_cols):
            ax = axes[i, j]
            im = ax.imshow(field[ci], cmap="viridis")
            ax.set_title(f"{space} spread {name}")
            ax.set_xticks([]); ax.set_yticks([])
            cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
            cb.set_label(unit if space == "physical" else "σ (climatology)")
    fig.suptitle(
        f"Exp5 spread (physical vs normalized) — "
        f"{pd.Timestamp(time_t).strftime('%Y-%m-%d %H:%M')}",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_exp5(
    ensemble_dir: str,
    figures_dir: str,
    n_samples: int = 3,
    seed: int = 42,
    config_path: str = "config/default.yaml",
) -> dict:
    set_plot_defaults()
    fig_dir = ensure_dir(figures_dir)

    # climatology c_sig(doy) — 정규화공간 spread 계산용 (물리 spread / c_sig[doy]).
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    c_sig = np.load(cfg["climatology"]["path"])["c_sig"]   # (366, C, H, W)

    files = list_ensemble_files(Path(ensemble_dir))
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(files), size=min(n_samples, len(files)),
                       replace=False)

    chosen = []
    for k, idx in enumerate(picks, start=1):
        es = load_ensemble_npz(files[idx])
        for lead, tok, mem_key, gt_key, off_h in _LEADS:
            members = getattr(es, mem_key)               # (N,C,H,W) 물리단위
            gt = getattr(es, gt_key)                      # (C,H,W) 물리단위
            if members is None or gt is None:
                raise KeyError(
                    f"{files[idx]} 에 '{mem_key}'/'{gt_key}' 가 없습니다 — 이웃 저장 "
                    f"버전 experiments_dit.ensemble_inference 로 캐시를 재생성하세요."
                )
            lead_time = es.time_t + np.timedelta64(off_h, "h")

            sample = {
                "gt": gt,
                "members": members,
                "mean": members.mean(axis=0),             # (C,H,W)
                "time_t": lead_time,
                "lead": lead,
            }
            _save_composite(
                sample, fig_dir / f"exp5_composite_sample{k}_{tok}.png",
            )

            spread_phys = members.std(axis=0, ddof=1)     # (C,H,W) 물리단위 멤버 spread
            # 정규화공간 spread: climatology σ(doy)로 표준화 = 물리 spread / c_sig[doy].
            doy = int(doy_slot(lead_time)[0])
            spread_norm = spread_phys / c_sig[doy]         # (C,H,W) 무차원
            _save_spread(
                spread_phys, spread_norm, lead_time,
                fig_dir / f"exp5_spread_sample{k}_{tok}.png",
            )
        chosen.append({
            "idx": int(idx),
            "time_t": str(es.time_t),
            "file": str(files[idx]),
        })

    print(f"[exp5] picked={chosen}  (×3 leads each)")
    return {"picked": chosen}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles_dit")
    p.add_argument("--figures_dir", default="outputs/figures_dit")
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default="config/default.yaml",
                   help="climatology.path(c_sig) 로드용 config")
    args = p.parse_args()
    run_exp5(args.ensemble_dir, args.figures_dir, args.n_samples,
             args.seed, args.config)
