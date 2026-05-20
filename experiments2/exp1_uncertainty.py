"""Exp1 (t-2): ℓ_past_{t-2} 의 pixelwise diversity & 시점 간 차이."""
from __future__ import annotations

from experiments.exp1_uncertainty import run_exp1 as _run


def run_exp1(
    ensemble_dir: str, figures_dir: str, metrics_dir: str,
    n_pairs: int = 1000, seed: int = 42,
) -> dict:
    print("[exp1 t-2] running")
    return _run(ensemble_dir, figures_dir, metrics_dir, n_pairs, seed)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles_tm2")
    p.add_argument("--figures_dir",  default="outputs/figures_tm2")
    p.add_argument("--metrics_dir",  default="outputs/metrics_tm2")
    p.add_argument("--n_pairs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_exp1(args.ensemble_dir, args.figures_dir, args.metrics_dir,
             args.n_pairs, args.seed)
