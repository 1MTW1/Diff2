"""실험 9 (v2 LDM/DiT): 한반도 영역평균(aavg) 시계열 + 앙상블 spread shading.

세 lead (t-1, t, t+1) × 세 변수 (T, u, v) 에 대해, 한반도 박스(11×9)의 **영역평균**
시계열을 그린다. 각 패널은:
    - GT line                (ERA5 관측 area-average)
    - ensemble-mean line     (멤버별 area-average 의 평균)
    - ±1σ shading            (멤버별 area-average 의 멤버 간 std)

즉 spread 는 "영역평균의 멤버 std" (먼저 멤버별로 한반도 평균을 낸 뒤, 그 스칼라들의
멤버 간 표준편차). 모든 값은 물리 단위(climatology 역표준화 완료). 이웃 프레임(t±1)은
±1일(±24h) 시점이다.

데이터: ensemble cache 의 픽셀 필드 (ensemble_pixel{,_tm1,_tp1}, x_{t,tm1,tp1}_true_pixel).
입력 캐시는 experiments_dit.ensemble_inference (이웃 저장 버전) 로 생성해야 한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import (
    KOREA_LAT_SLICE, KOREA_LON_SLICE, VARIABLES, ensure_dir,
    list_ensemble_files, load_ensemble_npz, set_plot_defaults,
)


# (라벨, 멤버 픽셀 키, GT 픽셀 키, 중심으로부터의 시간 오프셋[h])
# 이웃은 ±1일(±24h) — v4 데이터는 매일 00 UTC 스냅샷.
_LEADS = [
    ("t-1", "ensemble_pixel_tm1", "x_tm1_true_pixel", -24),
    ("t",   "ensemble_pixel",     "x_t_true_pixel",     0),
    ("t+1", "ensemble_pixel_tp1", "x_tp1_true_pixel",  24),
]
_UNITS = {"t": "K", "u": "m/s", "v": "m/s"}


def _box_aavg(field: np.ndarray, ch: int) -> np.ndarray:
    """(C,H,W) 또는 (N,C,H,W) → 한반도 박스 영역평균. 스칼라 또는 (N,)."""
    box = field[..., ch, KOREA_LAT_SLICE, KOREA_LON_SLICE]
    return box.mean(axis=(-2, -1))


def _collect(ensemble_dir: Path) -> dict:
    """lead·변수별 (times, gt, ens_mean, ens_std) 시계열을 모은다."""
    files = list_ensemble_files(ensemble_dir)

    # series[lead_label][var_name] = {"times":[], "gt":[], "mean":[], "std":[]}
    series: dict[str, dict[str, dict[str, list]]] = {
        lead: {v: {"times": [], "gt": [], "mean": [], "std": []}
               for v in VARIABLES}
        for lead, *_ in _LEADS
    }

    for f in files:
        es = load_ensemble_npz(f)
        for lead, mem_key, gt_key, off_h in _LEADS:
            mem_d = getattr(es, mem_key)                  # (N,C,H,W) 물리단위
            gt_d = getattr(es, gt_key)                     # (C,H,W) 물리단위
            if mem_d is None or gt_d is None:
                raise KeyError(
                    f"{f} 에 '{mem_key}'/'{gt_key}' 가 없습니다 — 이웃 저장 버전의 "
                    f"ensemble_inference 로 캐시를 재생성하세요."
                )
            lead_time = es.time_t + np.timedelta64(off_h, "h")
            for ci, v in enumerate(VARIABLES):
                gt_a = float(_box_aavg(gt_d, ci))
                mem_a = _box_aavg(mem_d, ci)                           # (N,)
                s = series[lead][v]
                s["times"].append(lead_time)
                s["gt"].append(gt_a)
                s["mean"].append(float(mem_a.mean()))
                s["std"].append(float(mem_a.std(ddof=1)))

    # list → sorted numpy arrays (시간 정렬)
    for lead, *_ in _LEADS:
        for v in VARIABLES:
            s = series[lead][v]
            order = np.argsort(np.array(s["times"]))
            for k in ("times", "gt", "mean", "std"):
                s[k] = np.array(s[k])[order]
    return series


def _format_time_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_fontsize(7)


# lead 라벨 → 파일명 토큰 ("t-1"→tm1 등; '+','-' 회피).
_LEAD_TOK = {"t-1": "tm1", "t": "t", "t+1": "tp1"}


def _save_lead(series: dict, lead: str, out_path: Path) -> None:
    """단일 lead 의 area-avg 시계열 (1×3 변수: T,u,v) → 하나의 PNG."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2), sharex=True)
    for ci, v in enumerate(VARIABLES):
        ax = axes[ci]
        s = series[lead][v]
        dts = pd.to_datetime(s["times"])
        mean_arr = s["mean"]
        std_arr = s["std"]
        ax.fill_between(
            dts, mean_arr - std_arr, mean_arr + std_arr,
            color="C3", alpha=0.25, linewidth=0, label="±1σ (member)",
        )
        ax.plot(dts, mean_arr, color="C3", linewidth=0.8, label="ens mean")
        ax.plot(dts, s["gt"], color="C0", linewidth=0.8, label="GT (ERA5)")
        ax.set_title(f"{v.upper()}  ({_UNITS[v]})", fontsize=11)
        ax.set_ylabel("area-avg", fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
        _format_time_axis(ax)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"Exp9 — Korea-box area-average timeseries · lead {lead}  "
        "(GT vs ensemble mean ±1σ)",
        y=1.0, fontsize=13,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.13,
                        wspace=0.16)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _summarize(series: dict) -> dict:
    """lead·변수별 RMSE(ens mean − GT) 와 평균 spread."""
    out: dict[str, dict[str, dict]] = {}
    for lead, *_ in _LEADS:
        out[lead] = {}
        for v in VARIABLES:
            s = series[lead][v]
            err = s["mean"] - s["gt"]
            out[lead][v] = {
                "n": int(s["gt"].size),
                "rmse_ensmean_vs_gt": float(np.sqrt(np.mean(err ** 2)))
                if err.size else None,
                "bias_ensmean_minus_gt": float(err.mean()) if err.size else None,
                "mean_spread_1sigma": float(s["std"].mean())
                if s["std"].size else None,
                "unit": _UNITS[v],
            }
    return out


def run_exp9(
    ensemble_dir: str,
    figures_dir: str,
    metrics_dir: str,
) -> dict:
    set_plot_defaults()
    fig_dir = ensure_dir(figures_dir)
    met_dir = ensure_dir(metrics_dir)

    series = _collect(Path(ensemble_dir))
    n = series["t"]["t"]["gt"].size
    print(f"[exp9] {n} timesteps · 3 leads × {len(VARIABLES)} vars")

    written = []
    for lead, *_ in _LEADS:
        tok = _LEAD_TOK[lead]
        out_p = fig_dir / f"exp9_aavg_spread_{tok}.png"
        _save_lead(series, lead, out_p)
        written.append(out_p.name)

    stats = {
        "domain": {"box": "Korea 11×9",
                   "lat_slice": [KOREA_LAT_SLICE.start, KOREA_LAT_SLICE.stop],
                   "lon_slice": [KOREA_LON_SLICE.start, KOREA_LON_SLICE.stop]},
        "spread_def": "std over members of per-member area-average",
        "summary": _summarize(series),
    }
    with open(met_dir / "exp9_stats.json", "w") as fp:
        json.dump(stats, fp, indent=2)
    print(f"[exp9] wrote {', '.join(written)} + exp9_stats.json")
    return stats


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles_dit")
    p.add_argument("--figures_dir", default="outputs/figures_dit")
    p.add_argument("--metrics_dir", default="outputs/metrics_dit")
    args = p.parse_args()
    run_exp9(args.ensemble_dir, args.figures_dir, args.metrics_dir)
