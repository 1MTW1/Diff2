"""Exp4 (v2 LDM/DiT, t-2): bar plot LOW vs HIGH latent log_var_past 픽셀."""
from __future__ import annotations

from experiments_dit.exp4_pixel_timeseries import (
    run_exp4 as _run, run_exp4_aggregate as _run_agg,
)


def run_exp4(
    ensemble_dir: str, figures_dir: str,
    percentile: float = 0.1, margin: int = 2,
    sample_idx: int | None = None, seed: int = 42,
) -> dict:
    print("[exp4 dit t-2] single-sample")
    return _run(
        ensemble_dir, figures_dir,
        percentile=percentile, margin=margin,
        sample_idx=sample_idx, seed=seed,
    )


def run_exp4_aggregate(
    ensemble_dir: str, figures_dir: str,
    percentile: float = 0.1, margin: int = 2,
) -> dict:
    print("[exp4 dit t-2] aggregate over all samples")
    return _run_agg(ensemble_dir, figures_dir,
                    percentile=percentile, margin=margin)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles_dit_tm2")
    p.add_argument("--figures_dir",  default="outputs/figures_dit_tm2")
    p.add_argument("--percentile", type=float, default=0.1)
    p.add_argument("--margin", type=int, default=2)
    p.add_argument("--sample_idx", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--aggregate", action="store_true")
    args = p.parse_args()
    if args.aggregate:
        run_exp4_aggregate(args.ensemble_dir, args.figures_dir,
                           percentile=args.percentile, margin=args.margin)
    else:
        run_exp4(args.ensemble_dir, args.figures_dir,
                 percentile=args.percentile, margin=args.margin,
                 sample_idx=args.sample_idx, seed=args.seed)
