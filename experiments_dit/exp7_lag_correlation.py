"""실험 7 (v2 LDM/DiT): 한반도 영역 u ↔ T time-lag spatial correlation.

각 valid 시점 t 에 대해 한반도 영역 (lat 33–43°N, lon 124–132°E, 11×9=99 px) 의
픽셀들 사이에서 u(t,p) 와 T(t+lag·6h, p) 의 spatial Pearson r 을 lag ∈ [-3,3]
step (= ±18h) 으로 계산하고, GT 시계열과 ensemble 시계열을 위/아래 sub-panel
로 비교한다. 출력은 3개 PNG: member A, member B, ensemble mean.

데이터는 ensemble cache (sample_*.npz) 의 `x_t_true_pixel` (GT) 과
`ensemble_pixel` (N 멤버) 을 사용. spatial Pearson 은 affine 불변이라
정규화 공간에서 그대로 계산한다.

주의: ensemble cache 의 "member k" 는 시점마다 독립 noise 로 생성되므로
시간적으로 일관된 trajectory 가 아니다. 본 실험은 사용자가 명시한 디자인대로
고정 인덱스를 시점 차원으로 잇는 비교를 수행한다 (plot caption 에 명시).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import (
    TEMP_IDX, U_IDX, ensure_dir, list_ensemble_files, load_ensemble_npz,
    set_plot_defaults,
)


# 한반도 영역 (인덱스 — dataset/era5_dataset.py 확인: lat 69→6, lon 95→158)
LAT_SLICE = slice(26, 37)   # lat 43 → 33 °N
LON_SLICE = slice(29, 38)   # lon 124 → 132 °E

LAGS = [-3, -2, -1, 0, 1, 2, 3]
STEP_HOURS = 6
LAG_HOURS = [k * STEP_HOURS for k in LAGS]

_EPS = 1e-12


# ───────────────────────── 계산 ─────────────────────────

def _spatial_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """a, b: (H, W) — 동일 shape, flatten 후 Pearson r."""
    af, bf = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    ac = af - af.mean()
    bc = bf - bf.mean()
    den = float(np.sqrt((ac * ac).sum() * (bc * bc).sum())) + _EPS
    return float((ac * bc).sum() / den)


def _build_time_map(files: list[Path]) -> dict[np.datetime64, Path]:
    out: dict[np.datetime64, Path] = {}
    for f in files:
        es = load_ensemble_npz(f)
        out[es.time_t] = f
    return out


def _valid_times_per_lag(
    time_map: dict[np.datetime64, Path],
) -> dict[int, list[np.datetime64]]:
    """lag 별로 t+lag·6h 가 cache 안에 존재하는 시점 리스트.

    캐시가 Mon/Wed/Fri 일자별로 dense (4 timestep) 라 inter-day lag 는 결측 →
    lag 마다 valid t 셋이 다르다. 각 lag 별로 독립 시계열을 plot.
    """
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
    """(*, C, H, W) → (..., 11, 9) — Korea 영역만 슬라이스."""
    return arr[..., ch, LAT_SLICE, LON_SLICE]


def _compute_all(
    time_map: dict[np.datetime64, Path],
    valid_per_lag: dict[int, list[np.datetime64]],
    member_a: int,
    member_b: int,
) -> dict:
    """lag 별로 valid t 가 다르므로, lag 마다 (times, r) array 를 따로 반환."""
    loaded: dict[np.datetime64, dict] = {}

    def _get(t):
        if t not in loaded:
            es = load_ensemble_npz(time_map[t])
            mean_pix = es.ensemble_pixel.mean(axis=0)               # (3, 64, 64)
            loaded[t] = {
                "gt_u":   _box(es.x_t_true_pixel, U_IDX),           # (11, 9)
                "gt_t":   _box(es.x_t_true_pixel, TEMP_IDX),
                "mem_u":  _box(es.ensemble_pixel, U_IDX),           # (N, 11, 9)
                "mem_t":  _box(es.ensemble_pixel, TEMP_IDX),
                "mean_u": _box(mean_pix, U_IDX),
                "mean_t": _box(mean_pix, TEMP_IDX),
            }
        return loaded[t]

    series: dict[int, dict[str, np.ndarray]] = {}
    for lag, hours in zip(LAGS, LAG_HOURS):
        ts = valid_per_lag[lag]
        off = np.timedelta64(hours, "h")
        T = len(ts)
        r_gt  = np.empty(T); r_a = np.empty(T); r_b = np.empty(T); r_m = np.empty(T)
        for i, t in enumerate(ts):
            base = _get(t)
            tgt  = _get(t + off)
            r_gt[i] = _spatial_pearson(base["gt_u"],   tgt["gt_t"])
            r_a [i] = _spatial_pearson(base["mem_u`"][member_a],
                                       tgt["mem_t"][member_a])
            r_b [i] = _spatial_pearson(base["mem_u"][member_b],
                                       tgt["mem_t"][member_b])
            r_m [i] = _spatial_pearson(base["mean_u"], tgt["mean_t"])
        series[lag] = {
            "times": np.array(ts), "r_gt": r_gt,
            "r_mem_a": r_a, "r_mem_b": r_b, "r_mean": r_m,
        }
    return series


# ───────────────────────── plot ─────────────────────────

def _format_time_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_fontsize(7)


def _draw_lag_panel(
    ax, times, r_series: np.ndarray, color: str,
    ylim: tuple[float, float] = (-1.05, 1.05),
) -> None:
    """한 cell — 시계열 1개."""
    dts = pd.to_datetime(times)
    ax.plot(
        dts, r_series, color=color, linewidth=0.6, alpha=0.85,
        marker="o", markersize=1.5, markeredgewidth=0,
    )
    ax.axhline(0, color="k", linewidth=0.4, alpha=0.5)
    med = float(np.median(r_series))
    ax.axhline(med, color=color, linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=7)


def _diff_ylim(series, bottom_key: str) -> tuple[float, float]:
    """모든 lag column 의 diff (ens - GT) 절댓값 99-percentile 로 symmetric ylim."""
    pool = np.concatenate([
        series[lag][bottom_key] - series[lag]["r_gt"]
        for lag in LAGS if series[lag]["times"].size
    ])
    if pool.size == 0:
        return (-0.5, 0.5)
    m = float(np.percentile(np.abs(pool), 99))
    m = max(m, 0.05) * 1.15
    return (-m, m)


def _save_pair(
    series: dict[int, dict[str, np.ndarray]],
    bottom_key: str,
    label_top: str, label_bottom: str,
    suptitle: str,
    out_path: Path,
) -> None:
    """3×7 grid: row 0 = GT, row 1 = ensemble, row 2 = diff (ensemble − GT)."""
    fig, axes = plt.subplots(
        3, 7, figsize=(22, 7.6), sharex=True,
    )
    diff_lim = _diff_ylim(series, bottom_key)

    for j, lag in enumerate(LAGS):
        s = series[lag]
        diff = s[bottom_key] - s["r_gt"]
        _draw_lag_panel(axes[0, j], s["times"], s["r_gt"],     color="C0")
        _draw_lag_panel(axes[1, j], s["times"], s[bottom_key], color="C3")
        _draw_lag_panel(axes[2, j], s["times"], diff,          color="C2",
                        ylim=diff_lim)
        axes[0, j].set_title(
            f"lag = {lag:+d}  ({LAG_HOURS[j]:+d}h)   N={len(s['times'])}",
            fontsize=9,
        )

    # row 0, 1 은 [-1, 1] sharey 묶기 (sharey=True 면 row2 도 묶이므로 수동 처리)
    for j in range(1, 7):
        axes[0, j].sharey(axes[0, 0])
        axes[1, j].sharey(axes[1, 0])
        axes[2, j].sharey(axes[2, 0])

    axes[0, 0].set_ylabel(label_top,    fontsize=10)
    axes[1, 0].set_ylabel(label_bottom, fontsize=10)
    axes[2, 0].set_ylabel(f"{label_bottom} − {label_top}", fontsize=10)

    for ax in axes[2, :]:
        _format_time_axis(ax)
    for row in (0, 1):
        for ax in axes[row, :]:
            ax.tick_params(axis="x", labelbottom=False)

    fig.suptitle(suptitle, y=0.995, fontsize=12)
    fig.text(
        0.005, 0.66,
        "spatial Pearson  r ( u(t) ,  T(t+lag) )   over Korea box (11×9)",
        rotation=90, va="center", fontsize=9,
    )
    fig.text(
        0.005, 0.18,
        "Δr  ( ensemble − GT )",
        rotation=90, va="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.05, right=0.995, top=0.93, bottom=0.08,
                        hspace=0.15, wspace=0.10)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ───────────────────────── stats ─────────────────────────

def _summarize_per_lag(
    series: dict[int, dict[str, np.ndarray]], key: str,
    diff_key: str | None = None,
) -> dict:
    """key 자체의 통계. diff_key 가 주어지면 (series[lag][key] - series[lag][diff_key]).
    """
    out = {}
    for j, lag in enumerate(LAGS):
        if diff_key is None:
            col = series[lag][key]
        else:
            col = series[lag][key] - series[lag][diff_key]
        out[str(lag)] = {
            "mean":   float(col.mean()) if col.size else None,
            "median": float(np.median(col)) if col.size else None,
            "std":    float(col.std()) if col.size else None,
            "p10":    float(np.percentile(col, 10)) if col.size else None,
            "p90":    float(np.percentile(col, 90)) if col.size else None,
            "lag_hours": LAG_HOURS[j],
            "n":      int(col.size),
        }
    return out


# ───────────────────────── entry ─────────────────────────

def run_exp7(
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
    print(f"[exp7] {len(files)} cache files")
    time_map = _build_time_map(files)

    # 멤버 인덱스
    sample = load_ensemble_npz(files[0])
    n_members = sample.ensemble_pixel.shape[0]
    if member_a is None or member_b is None:
        idx_a, idx_b = _pick_members(n_members, seed)
    else:
        idx_a, idx_b = int(member_a), int(member_b)
    print(f"[exp7] members = ({idx_a}, {idx_b})  out of N={n_members}")

    valid_per_lag = _valid_times_per_lag(time_map)
    for lag in LAGS:
        print(f"[exp7] lag={lag:+d} ({LAG_HOURS[LAGS.index(lag)]:+d}h)  "
              f"→ {len(valid_per_lag[lag])} valid timesteps")
    if all(len(v) == 0 for v in valid_per_lag.values()):
        raise RuntimeError("어떤 lag 도 valid 시점이 없습니다.")

    series = _compute_all(time_map, valid_per_lag, idx_a, idx_b)

    _save_pair(
        series, "r_mem_a",
        label_top="GT", label_bottom=f"Member {idx_a}",
        suptitle=f"Exp7 — Korea u↔T lag corr  (member {idx_a} vs GT)",
        out_path=fig_dir / "exp7_memberA.png",
    )
    _save_pair(
        series, "r_mem_b",
        label_top="GT", label_bottom=f"Member {idx_b}",
        suptitle=f"Exp7 — Korea u↔T lag corr  (member {idx_b} vs GT)",
        out_path=fig_dir / "exp7_memberB.png",
    )
    _save_pair(
        series, "r_mean",
        label_top="GT", label_bottom="Ensemble mean",
        suptitle="Exp7 — Korea u↔T lag corr  (ensemble mean vs GT)",
        out_path=fig_dir / "exp7_mean.png",
    )

    stats = {
        "domain": {
            "lat_range_deg": [33, 43], "lon_range_deg": [124, 132],
            "lat_slice": [LAT_SLICE.start, LAT_SLICE.stop],
            "lon_slice": [LON_SLICE.start, LON_SLICE.stop],
            "box_hw": [LAT_SLICE.stop - LAT_SLICE.start,
                       LON_SLICE.stop - LON_SLICE.start],
        },
        "lags_step":  LAGS,
        "lags_hours": LAG_HOURS,
        "n_members":  n_members,
        "member_indices": {"A": idx_a, "B": idx_b},
        "n_valid_per_lag": {str(lag): len(valid_per_lag[lag]) for lag in LAGS},
        "n_cache_files": len(files),
        "summary": {
            "gt":            _summarize_per_lag(series, "r_gt"),
            "memberA":       _summarize_per_lag(series, "r_mem_a"),
            "memberB":       _summarize_per_lag(series, "r_mem_b"),
            "ensemble_mean": _summarize_per_lag(series, "r_mean"),
        },
        "diff_summary": {
            "memberA_minus_gt":       _summarize_per_lag(series, "r_mem_a", "r_gt"),
            "memberB_minus_gt":       _summarize_per_lag(series, "r_mem_b", "r_gt"),
            "ensemble_mean_minus_gt": _summarize_per_lag(series, "r_mean",  "r_gt"),
        },
    }
    with open(met_dir / "exp7_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[exp7] wrote 3 PNG → {fig_dir}")
    print(f"[exp7] wrote exp7_stats.json → {met_dir}")
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
    run_exp7(
        args.ensemble_dir, args.figures_dir, args.metrics_dir,
        seed=args.seed, member_a=args.member_a, member_b=args.member_b,
    )
