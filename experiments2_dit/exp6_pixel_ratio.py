"""Exp6 (v2 LDM/DiT, t-2): latent pixel-wise RMSE/Spread at x̂_{t-2}."""
from __future__ import annotations

from experiments_dit.exp6_pixel_ratio import run_exp6 as _run


def run_exp6(
    ensemble_dir: str, figures_dir: str, metrics_dir: str,
) -> dict:
    print("[exp6 dit t-2] running")
    return _run(ensemble_dir, figures_dir, metrics_dir)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles_dit_tm2")
    p.add_argument("--figures_dir",  default="outputs/figures_dit_tm2")
    p.add_argument("--metrics_dir",  default="outputs/metrics_dit_tm2")
    args = p.parse_args()
    run_exp6(args.ensemble_dir, args.figures_dir, args.metrics_dir)
