"""Exp5 (t-2): composite map (4×4 GT/members/mean) at x_{t-2}."""
from __future__ import annotations

from experiments.exp5_composite_maps import run_exp5 as _run


def run_exp5(
    ensemble_dir: str, figures_dir: str,
    n_samples: int = 3, seed: int = 42,
) -> dict:
    print("[exp5 t-2] running")
    return _run(ensemble_dir, figures_dir, n_samples=n_samples, seed=seed)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles_tm2")
    p.add_argument("--figures_dir",  default="outputs/figures_tm2")
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_exp5(args.ensemble_dir, args.figures_dir, args.n_samples, args.seed)
