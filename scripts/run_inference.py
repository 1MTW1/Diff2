"""experiments/ensemble_inference 의 thin wrapper.

experiment.md §7.1 — 한 번 실행으로 test set ensemble을 캐시.
"""
from __future__ import annotations

import argparse

from experiments.ensemble_inference import run_inference


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/ensembles")
    p.add_argument("--n_members", type=int, default=30)
    p.add_argument("--past_steps", type=int, default=50)
    p.add_argument("--main_steps", type=int, default=200)
    p.add_argument("--days_per_week", type=int, default=3,
                   help="현재는 3 (Mon/Wed/Fri) 만 지원. 다른 값이면 전체 사용.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sub_sample = args.days_per_week == 3
    if not sub_sample:
        print("[warn] days_per_week != 3 → 전체 test set 사용")
    run_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        n_members=args.n_members,
        past_steps=args.past_steps,
        main_steps=args.main_steps,
        sub_sample=sub_sample,
        limit=args.limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
