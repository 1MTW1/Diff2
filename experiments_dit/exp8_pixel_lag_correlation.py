"""실험 8 (v2 LDM/DiT): 서울 중심 3×3 영역 pixelwise time-lag correlation.

exp7 의 spatial Pearson (한 시점 안 픽셀들의 cross-corr) 과 대비:
각 픽셀 p 에 대해 lag 별 valid 시점 시계열 u(·, p) 와 T(·+lag·6h, p) 두 시계열을
모아 시간 차원 Pearson r 을 계산한다 → 픽셀당 한 값 → 3×3 spatial map.

데이터 좌표 확인 (dataset/era5_dataset.py):
    lat: 64 px, 69° → 6°N (1° 간격, 북→남 인덱스 증가)
    lon: 64 px, 95° → 158°E (1° 간격, 서→동 인덱스 증가)
서울 (≈37.5°N, 127°E) 중심 3×3:
    lat 인덱스 [31, 32, 33] → 38, 37, 36 °N
    lon 인덱스 [31, 32, 33] → 126, 127, 128 °E

출력은 exp7 과 같은 3 PNG (member A / member B / ensemble mean), 각 PNG 가
3×7 grid (row = GT / ensemble / diff, col = lag −3 ~ +3). 각 cell 은
시계열 plot 이 아니라 3×3 spatial heatmap 이다.

주의: ensemble cache 의 "member k" 는 시점마다 독립 noise — 시간 일관성 없음.
픽셀별 시간 Pearson 도 동일 인덱스를 잇는 임의 trajectory 위에서 계산된다.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    TEMP_IDX, U_IDX, ensure_dir, list_ensemble_files, load_ensemble_npz,
    set_plot_defaults,
)


# 서울 (≈37.5°N, 127°E) 중심 3×3
LAT_SLICE = slice(31, 34)   # 38, 37, 36 °N
LON_SLICE = slice(31, 34)   # 126, 127, 128 °E
BOX_H, BOX_W = 3, 3

LAGS = [-3, -2, -1, 0, 1, 2, 3]
STEP_HOURS = 6
LAG_HOURS = [k * STEP_HOURS for k in LAGS]

_EPS = 1e-12


# ───────────────────────── 계산 ─────────────────────────

def _build_time_map(files: list[Path]) -> dict[np.datetime64, Path]:
    out: dict[np.datetime64, Path] = {}
    for f in files:
        es = load_ensemble_npz(f)
        out[es.time_t] = f
    return out


def _valid_times_per_lag(
    time_map: dict[np.datetime64, Path],
) -> dict[int, list[np.datetime64]]:
    """lag 별로 t+lag·6h 가 cache 안에 존재하는 시점 리스트."""
    sorted_t = sorted(time_map.keys())
    per_lag: dict[int, list[np.datetime64]] = {}
    for lag, hours in zip(LAGS, LAG_HOURS):
        off = np.timedelta64(hours, "h")
        per_lag[lag] = [t for t in sorted_t if (t + off) in time_map]
    return per_lag


def _pick_members(n_members: int, seed: int) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    pick = rng.choice(n_members, size=2, replace=False)
    return int(pick[0]), int(pick[1])


def _box(arr: np.ndarray, ch: int) -> np.ndarray:
    """(*, C, H, W) → (..., 3, 3) — 서울 중심만 슬라이스."""
    return arr[..., ch, LAT_SLICE, LON_SLICE]


def _pixelwise_time_pearson(
    u_series: np.ndarray, t_series: np.ndarray,
) -> np.ndarray:
    """u_series, t_series: (T, H, W) → 픽셀별 시간 Pearson (H, W).

    각 픽셀에서 두 시계열 사이의 표준 Pearson 정의:
        r_p = sum_t (u(t,p) - ū_p)(T(t,p) - T̄_p)
              / sqrt( sum_t (u(t,p)-ū_p)² · sum_t (T(t,p)-T̄_p)² )
    """
    u_mean = u_series.mean(axis=0, keepdims=True)
    t_mean = t_series.mean(axis=0, keepdims=True)
    uc = u_series - u_mean
    tc = t_series - t_mean
    num = (uc * tc).sum(axis=0)
    den = np.sqrt((uc * uc).sum(axis=0) * (tc * tc).sum(axis=0)) + _EPS
    return num / den


def _load_box_series(
    time_map: dict[np.datetime64, Path],
    times: list[np.datetime64],
    member_a: int,
    member_b: int,
) -> dict[np.datetime64, dict[str, np.ndarray]]:
    """필요한 시점들의 6종 box (gt_u/gt_t, memA_u/memA_t, memB_u/memB_t, mean_u/mean_t) 로드."""
    cache: dict[np.datetime64, dict[str, np.ndarray]] = {}
    for t in times:
        if t in cache:
            continue
        es = load_ensemble_npz(time_map[t])
        mean_pix = es.ensemble_pixel.mean(axis=0)
        cache[t] = {
            "gt_u":    _box(es.x_t_true_pixel, U_IDX),
            "gt_t":    _box(es.x_t_true_pixel, TEMP_IDX),
            "memA_u":  _box(es.ensemble_pixel[member_a], U_IDX),
            "memA_t":  _box(es.ensemble_pixel[member_a], TEMP_IDX),
            "memB_u":  _box(es.ensemble_pixel[member_b], U_IDX),
            "memB_t":  _box(es.ensemble_pixel[member_b], TEMP_IDX),
            "mean_u":  _box(mean_pix, U_IDX),
            "mean_t":  _box(mean_pix, TEMP_IDX),
        }
    return cache


def _compute_lag_maps(
    time_map: dict[np.datetime64, Path],
    valid_per_lag: dict[int, list[np.datetime64]],
    member_a: int,
    member_b: int,
) -> dict[int, dict[str, np.ndarray]]:
    """lag 별로 4종 subject 의 3×3 픽셀 corr map 계산."""
    # 모든 lag 에 걸쳐 필요한 시점 = base t ∪ (t + lag·6h)
    needed: set[np.datetime64] = set()
    for lag, hours in zip(LAGS, LAG_HOURS):
        off = np.timedelta64(hours, "h")
        for t in valid_per_lag[lag]:
            needed.add(t)
            needed.add(t + off)
    cache = _load_box_series(time_map, sorted(needed), member_a, member_b)

    maps: dict[int, dict[str, np.ndarray]] = {}
    for lag, hours in zip(LAGS, LAG_HOURS):
        ts = valid_per_lag[lag]
        off = np.timedelta64(hours, "h")
        if len(ts) < 2:
            nan_map = np.full((BOX_H, BOX_W), np.nan)
            maps[lag] = {k: nan_map.copy() for k in
                         ("gt", "memA", "memB", "mean")}
            maps[lag]["n"] = len(ts)
            continue

        # u 와 T 두 시계열 stack — 같은 valid t 셋, T 만 +lag offset
        def _stack(key_u: str, key_t: str) -> tuple[np.ndarray, np.ndarray]:
            u_ser = np.stack([cache[t][key_u]       for t in ts], axis=0)
            t_ser = np.stack([cache[t + off][key_t] for t in ts], axis=0)
            return u_ser, t_ser

        maps[lag] = {
            "gt":   _pixelwise_time_pearson(*_stack("gt_u",   "gt_t")),
            "memA": _pixelwise_time_pearson(*_stack("memA_u", "memA_t")),
            "memB": _pixelwise_time_pearson(*_stack("memB_u", "memB_t")),
            "mean": _pixelwise_time_pearson(*_stack("mean_u", "mean_t")),
            "n":    len(ts),
        }
    return maps


# ───────────────────────── plot ─────────────────────────

def _draw_map(ax, m: np.ndarray, vmin: float, vmax: float, cmap: str) -> None:
    """3×3 픽셀 heatmap + 픽셀당 r 값 annotate."""
    im = ax.imshow(m, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            if not np.isfinite(v):
                continue
            txt_color = "white" if abs(v) > 0.55 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color=txt_color, fontsize=8)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["126°E", "127°E", "128°E"], fontsize=6)
    ax.set_yticklabels(["38°N", "37°N", "36°N"],   fontsize=6)


def _save_subject(
    maps: dict[int, dict[str, np.ndarray]],
    bottom_key: str,
    label_top: str, label_bottom: str,
    suptitle: str,
    out_path: Path,
) -> None:
    """3×7 grid: row 0=GT, row 1=ensemble subject, row 2=diff. 각 cell=3×3 corr map."""
    fig, axes = plt.subplots(3, 7, figsize=(18, 8.0))

    # diff symmetric vmax — 데이터 99-percentile
    diff_pool = []
    for lag in LAGS:
        d = maps[lag][bottom_key] - maps[lag]["gt"]
        diff_pool.append(d[np.isfinite(d)].flatten())
    if diff_pool:
        diff_arr = np.concatenate(diff_pool)
        if diff_arr.size:
            dmax = float(np.percentile(np.abs(diff_arr), 99))
            dmax = max(dmax, 0.05)
        else:
            dmax = 0.5
    else:
        dmax = 0.5

    last_im_corr = None
    last_im_diff = None
    for j, lag in enumerate(LAGS):
        m_gt   = maps[lag]["gt"]
        m_ens  = maps[lag][bottom_key]
        m_diff = m_ens - m_gt

        _draw_map(axes[0, j], m_gt,  -1.0, 1.0, "RdBu_r")
        _draw_map(axes[1, j], m_ens, -1.0, 1.0, "RdBu_r")
        _draw_map(axes[2, j], m_diff, -dmax, dmax, "PuOr_r")

        axes[0, j].set_title(
            f"lag = {lag:+d}  ({LAG_HOURS[j]:+d}h)\nN={maps[lag]['n']}",
            fontsize=9,
        )

        # 마지막 column 의 imshow handle 보관 (colorbar 용)
        last_im_corr = axes[1, j].images[0]
        last_im_diff = axes[2, j].images[0]

    axes[0, 0].set_ylabel(label_top,    fontsize=10)
    axes[1, 0].set_ylabel(label_bottom, fontsize=10)
    axes[2, 0].set_ylabel(f"{label_bottom} − {label_top}", fontsize=10)

    fig.suptitle(suptitle, y=0.995, fontsize=12)
    fig.subplots_adjust(left=0.05, right=0.92, top=0.90, bottom=0.05,
                        hspace=0.30, wspace=0.20)

    # 우측 colorbar 두 개: row0,1 의 corr ([-1,1]) 와 row2 의 diff
    cax_corr = fig.add_axes([0.935, 0.42, 0.012, 0.46])
    cb_corr = fig.colorbar(last_im_corr, cax=cax_corr)
    cb_corr.set_label("pixelwise time Pearson  r", fontsize=8)
    cb_corr.ax.tick_params(labelsize=7)

    cax_diff = fig.add_axes([0.935, 0.06, 0.012, 0.28])
    cb_diff = fig.colorbar(last_im_diff, cax=cax_diff)
    cb_diff.set_label("Δr  (ensemble − GT)", fontsize=8)
    cb_diff.ax.tick_params(labelsize=7)

    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ───────────────────────── stats ─────────────────────────

def _summarize_map(m: np.ndarray) -> dict:
    flat = m[np.isfinite(m)].flatten()
    if flat.size == 0:
        return {k: None for k in ("mean", "median", "std", "min", "max", "n")}
    return {
        "mean":   float(flat.mean()),
        "median": float(np.median(flat)),
        "std":    float(flat.std()),
        "min":    float(flat.min()),
        "max":    float(flat.max()),
        "n":      int(flat.size),
    }


def _summary_per_subject(maps, key, diff_against: str | None = None) -> dict:
    out = {}
    for lag in LAGS:
        m = maps[lag][key]
        if diff_against is not None:
            m = m - maps[lag][diff_against]
        out[str(lag)] = {
            **_summarize_map(m),
            "lag_hours":     STEP_HOURS * lag,
            "n_valid_times": int(maps[lag]["n"]),
        }
    return out


# ───────────────────────── entry ─────────────────────────

def run_exp8(
    ensemble_dir: str,
    figures_dir: str,
    metrics_dir: str,
    seed: int = 42,
    member_a: int | None = None,
    member_b: int | None = None,
) -> dict:
    set_plot_defaults()
    fig_dir = ensure_dir(figures_dir)
    met_dir = ensure_dir(metrics_dir)

    files = list_ensemble_files(Path(ensemble_dir))
    print(f"[exp8] {len(files)} cache files")
    time_map = _build_time_map(files)

    sample = load_ensemble_npz(files[0])
    n_members = sample.ensemble_pixel.shape[0]
    if member_a is None or member_b is None:
        idx_a, idx_b = _pick_members(n_members, seed)
    else:
        idx_a, idx_b = int(member_a), int(member_b)
    print(f"[exp8] members = ({idx_a}, {idx_b})  out of N={n_members}")

    valid_per_lag = _valid_times_per_lag(time_map)
    for lag in LAGS:
        print(f"[exp8] lag={lag:+d} ({STEP_HOURS*lag:+d}h)  "
              f"→ {len(valid_per_lag[lag])} valid timesteps")

    maps = _compute_lag_maps(time_map, valid_per_lag, idx_a, idx_b)

    _save_subject(
        maps, "memA",
        label_top="GT", label_bottom=f"Member {idx_a}",
        suptitle=f"Exp8 — Seoul 3×3 pixel-wise time corr  (member {idx_a} vs GT)",
        out_path=fig_dir / "exp8_memberA.png",
    )
    _save_subject(
        maps, "memB",
        label_top="GT", label_bottom=f"Member {idx_b}",
        suptitle=f"Exp8 — Seoul 3×3 pixel-wise time corr  (member {idx_b} vs GT)",
        out_path=fig_dir / "exp8_memberB.png",
    )
    _save_subject(
        maps, "mean",
        label_top="GT", label_bottom="Ensemble mean",
        suptitle="Exp8 — Seoul 3×3 pixel-wise time corr  (ensemble mean vs GT)",
        out_path=fig_dir / "exp8_mean.png",
    )

    stats = {
        "domain": {
            "lat_range_deg": [36, 38], "lon_range_deg": [126, 128],
            "lat_slice": [LAT_SLICE.start, LAT_SLICE.stop],
            "lon_slice": [LON_SLICE.start, LON_SLICE.stop],
            "box_hw": [BOX_H, BOX_W],
            "center_label": "Seoul (≈37.5°N, 127°E)",
        },
        "lags_step":  LAGS,
        "lags_hours": LAG_HOURS,
        "n_members":  n_members,
        "member_indices": {"A": idx_a, "B": idx_b},
        "n_valid_per_lag": {str(lag): len(valid_per_lag[lag]) for lag in LAGS},
        "n_cache_files": len(files),
        "summary": {
            "gt":            _summary_per_subject(maps, "gt"),
            "memberA":       _summary_per_subject(maps, "memA"),
            "memberB":       _summary_per_subject(maps, "memB"),
            "ensemble_mean": _summary_per_subject(maps, "mean"),
        },
        "diff_summary": {
            "memberA_minus_gt":       _summary_per_subject(maps, "memA", "gt"),
            "memberB_minus_gt":       _summary_per_subject(maps, "memB", "gt"),
            "ensemble_mean_minus_gt": _summary_per_subject(maps, "mean", "gt"),
        },
        # 픽셀 9개의 raw r 값도 보존 (재현·후처리용)
        "raw_maps": {
            str(lag): {
                k: maps[lag][k].tolist()
                for k in ("gt", "memA", "memB", "mean")
            }
            for lag in LAGS
        },
    }
    with open(met_dir / "exp8_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[exp8] wrote 3 PNG → {fig_dir}")
    print(f"[exp8] wrote exp8_stats.json → {met_dir}")
    return stats


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles_dit")
    p.add_argument("--figures_dir",  default="outputs/figures_dit")
    p.add_argument("--metrics_dir",  default="outputs/metrics_dit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--member_a", type=int, default=None,
                   help="고정 멤버 인덱스 A — 미지정 시 seed 로 random")
    p.add_argument("--member_b", type=int, default=None)
    args = p.parse_args()
    run_exp8(
        args.ensemble_dir, args.figures_dir, args.metrics_dir,
        seed=args.seed, member_a=args.member_a, member_b=args.member_b,
    )
