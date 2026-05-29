"""exp_xtm15_diff per-sample CSV → ensemble spread grouped barplot.

x̂_{t-2} (baseline) condition 과 x̂_{t-15} (alt) condition 의 앙상블 spread 를
변수(t/u/v)별로 나란히 비교한다. 막대 높이 = 샘플 평균, errorbar = 표준편차.

spread 는 exp_xtm15_diff 가 역정규화(물리 단위) 후 ddof=1 로 계산한 per-pixel std
의 공간 평균이다 (CSV 의 {var}__spread_{baseline,alt}_mean). T 는 K, u/v 는 m/s.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir, set_plot_defaults

_VARS = ("t", "u", "v")
_VAR_LABELS = {"t": "T (K)", "u": "u (m/s)", "v": "v (m/s)"}
_COLOR_BASE = "#4C72B0"
_COLOR_ALT = "#C44E52"


def plot_spread_bar(csv_path: str | Path, out_path: str | Path) -> dict:
    df = pd.read_csv(csv_path)
    n = len(df)
    ddof = 1 if n > 1 else 0

    base_mean, base_std, alt_mean, alt_std = [], [], [], []
    for v in _VARS:
        b = df[f"{v}__spread_baseline_mean"]
        a = df[f"{v}__spread_alt_mean"]
        base_mean.append(float(b.mean()))
        base_std.append(float(b.std(ddof=ddof)))
        alt_mean.append(float(a.mean()))
        alt_std.append(float(a.std(ddof=ddof)))

    x = np.arange(len(_VARS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(
        x - w / 2, base_mean, w, yerr=base_std, capsize=4,
        label=r"$\hat{x}_{t-2}$ (baseline)", color=_COLOR_BASE,
        edgecolor="black", linewidth=0.5,
    )
    b2 = ax.bar(
        x + w / 2, alt_mean, w, yerr=alt_std, capsize=4,
        label=r"$\hat{x}_{t-15}$ (alt)", color=_COLOR_ALT,
        edgecolor="black", linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_VAR_LABELS[v] for v in _VARS])
    ax.set_ylabel("ensemble spread  (per-pixel std, spatial mean; T:K, u/v:m/s)")
    ax.set_title(
        "Ensemble spread by main condition: "
        r"$\hat{x}_{t-2}$ vs $\hat{x}_{t-15}$"
        f"   (mean ± std, n={n})"
    )
    ax.legend()
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "n_samples": n,
        "baseline_mean": dict(zip(_VARS, base_mean)),
        "alt_mean": dict(zip(_VARS, alt_mean)),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="exp_xtm15_diff spread grouped barplot"
    )
    p.add_argument(
        "--csv",
        default="outputs/metrics_dit/xtm15_diff/exp_xtm15_diff_per_sample.csv",
    )
    p.add_argument(
        "--out",
        default="outputs/figures_dit/xtm15_diff/exp_xtm15_spread_bar.png",
    )
    args = p.parse_args()

    set_plot_defaults()
    out = Path(args.out)
    ensure_dir(out.parent)
    info = plot_spread_bar(args.csv, out)
    print(f"[done] n={info['n_samples']} → {out}")


if __name__ == "__main__":
    main()
