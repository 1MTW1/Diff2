"""experiments/ 의 exp1-5 일괄 실행."""
from __future__ import annotations

import argparse

from experiments.exp1_uncertainty import run_exp1
from experiments.exp2_spread_vs_logvar import run_exp2
from experiments.exp3_spread_skill import run_exp3
from experiments.exp4_pixel_timeseries import run_exp4
from experiments.exp5_composite_maps import run_exp5
from experiments.exp6_pixel_ratio import run_exp6


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles")
    p.add_argument("--figures_dir", default="outputs/figures")
    p.add_argument("--metrics_dir", default="outputs/metrics")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", nargs="*", default=None,
                   help="실행할 실험 번호만 명시 (예: --only 1 2)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    only = {int(x) for x in args.only} if args.only else {1, 2, 3, 4, 5, 6}

    if 1 in only:
        run_exp1(args.ensemble_dir, args.figures_dir, args.metrics_dir,
                 n_pairs=1000, seed=args.seed)
    if 2 in only:
        run_exp2(args.ensemble_dir, args.figures_dir, args.metrics_dir,
                 seed=args.seed)
    if 3 in only:
        run_exp3(args.ensemble_dir, args.figures_dir, args.metrics_dir)
    if 4 in only:
        run_exp4(args.ensemble_dir, args.figures_dir,
                 percentile=0.1, margin=8,
                 sample_idx=None, seed=args.seed)
    if 5 in only:
        run_exp5(args.ensemble_dir, args.figures_dir,
                 n_samples=3, seed=args.seed)
    if 6 in only:
        run_exp6(args.ensemble_dir, args.figures_dir, args.metrics_dir)


if __name__ == "__main__":
    main()
