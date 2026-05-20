"""Exp3 (t-2): spread/RMSE ratio (calibration) at x_{t-2}."""
from __future__ import annotations

from experiments.exp3_spread_skill import run_exp3 as _run


def run_exp3(
    ensemble_dir: str, figures_dir: str, metrics_dir: str,
) -> dict:
    print("[exp3 t-2] running")
    return _run(ensemble_dir, figures_dir, metrics_dir)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles_tm2")
    p.add_argument("--figures_dir",  default="outputs/figures_tm2")
    p.add_argument("--metrics_dir",  default="outputs/metrics_tm2")
    args = p.parse_args()
    run_exp3(args.ensemble_dir, args.figures_dir, args.metrics_dir)
