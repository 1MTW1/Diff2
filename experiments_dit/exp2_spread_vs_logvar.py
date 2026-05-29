"""실험 2 (v2 LDM/DiT): latent log_var vs ensemble spread correlation.

experiments/exp2_spread_vs_logvar.py 의 thin wrapper. v1 exp2는 캐시의
`ensemble`/`log_var`/`x_t_true` 를 채널 0 에서 분석하는데, v2 latent 캐시에서는
그 채널 0 이 곧 latent 채널 0 (LATENT_CH) 이므로 그대로 재사용한다.
"""
from __future__ import annotations

from experiments.exp2_spread_vs_logvar import run_exp2  # noqa: F401


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles_dit")
    p.add_argument("--figures_dir", default="outputs/figures_dit")
    p.add_argument("--metrics_dir", default="outputs/metrics_dit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_gt_pairs", type=int, default=1000)
    args = p.parse_args()
    run_exp2(args.ensemble_dir, args.figures_dir, args.metrics_dir,
             seed=args.seed, n_gt_pairs=args.n_gt_pairs)
