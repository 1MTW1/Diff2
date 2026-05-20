"""Exp6 (t-2): pixel-wise RMSE/Spread distribution at x_{t-2}."""
from __future__ import annotations

from experiments.exp6_pixel_ratio import run_exp6 as _run


def run_exp6(
    ensemble_dir: str, figures_dir: str, metrics_dir: str,
) -> dict:
    print("[exp6 t-2] running")
    return _run(ensemble_dir, figures_dir, metrics_dir)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles_tm2")
    p.add_argument("--figures_dir",  default="outputs/figures_tm2")
    p.add_argument("--metrics_dir",  default="outputs/metrics_tm2")
    args = p.parse_args()
    run_exp6(args.ensemble_dir, args.figures_dir, args.metrics_dir)
