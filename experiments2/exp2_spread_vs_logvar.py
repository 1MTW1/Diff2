"""Exp2 (t-2): σ_ℓ_past vs ensemble spread of x_{t-2}."""
from __future__ import annotations

from experiments.exp2_spread_vs_logvar import run_exp2 as _run


def run_exp2(
    ensemble_dir: str, figures_dir: str, metrics_dir: str,
    seed: int = 42, n_gt_pairs: int = 1000,
) -> dict:
    print("[exp2 t-2] running")
    return _run(ensemble_dir, figures_dir, metrics_dir,
                seed=seed, n_gt_pairs=n_gt_pairs)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles_tm2")
    p.add_argument("--figures_dir",  default="outputs/figures_tm2")
    p.add_argument("--metrics_dir",  default="outputs/metrics_tm2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_gt_pairs", type=int, default=1000)
    args = p.parse_args()
    run_exp2(args.ensemble_dir, args.figures_dir, args.metrics_dir,
             seed=args.seed, n_gt_pairs=args.n_gt_pairs)
