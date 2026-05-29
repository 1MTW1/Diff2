"""실험 1 (v2 LDM/DiT): latent log_var 의 pixelwise 다양성 & 시점 간 차이.

experiments/exp1_uncertainty.py 의 thin wrapper. v1 exp1은 캐시의
`ensemble`/`log_var` 를 채널 0 에서 분석하는데, v2 latent 캐시에서는 그 채널 0
이 곧 latent 채널 0 (LATENT_CH) 이므로 그대로 재사용해도 의미가 맞다.
"""
from __future__ import annotations

from experiments.exp1_uncertainty import run_exp1  # noqa: F401


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", default="outputs/ensembles_dit")
    p.add_argument("--figures_dir", default="outputs/figures_dit")
    p.add_argument("--metrics_dir", default="outputs/metrics_dit")
    p.add_argument("--n_pairs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_exp1(args.ensemble_dir, args.figures_dir, args.metrics_dir,
             args.n_pairs, args.seed)
