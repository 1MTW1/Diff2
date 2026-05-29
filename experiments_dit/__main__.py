"""experiments_dit 일괄 실행 진입점 — exp1~exp6를 한 번에 (instruction_v2 §4, x_t 평가).

캐시된 앙상블(`outputs/ensembles_dit/`)을 읽어 6개 실험 모듈을 순차 실행한다.
각 실험은 독립 프로세스로 실행되어 하나가 실패해도 나머지는 계속된다.

선행: `python -m experiments_dit.ensemble_inference ...` 로 앙상블 캐시 생성.

실행:
    python -m experiments_dit
    python -m experiments_dit --ensemble_dir <dir> --figures_dir <dir> --metrics_dir <dir>
"""
from __future__ import annotations

import argparse
import subprocess
import sys

# (모듈명, --metrics_dir 인자 지원 여부) — exp4·exp5는 metrics 출력이 없다.
_EXPERIMENTS: list[tuple[str, bool]] = [
    ("exp1_uncertainty", True),
    ("exp2_spread_vs_logvar", True),
    ("exp3_spread_skill", True),
    ("exp4_pixel_timeseries", False),
    ("exp5_composite_maps", False),
    ("exp6_pixel_ratio", True),
    ("exp7_lag_correlation", True),
    ("exp8_pixel_lag_correlation", True),
]


def main() -> None:
    p = argparse.ArgumentParser(
        description="experiments_dit exp1~exp6 일괄 실행")
    p.add_argument("--ensemble_dir", default="outputs/ensembles_dit")
    p.add_argument("--figures_dir", default="outputs/figures_dit")
    p.add_argument("--metrics_dir", default="outputs/metrics_dit")
    args = p.parse_args()

    failed: list[str] = []
    for mod, has_metrics in _EXPERIMENTS:
        cmd = [sys.executable, "-m", f"experiments_dit.{mod}",
               "--ensemble_dir", args.ensemble_dir,
               "--figures_dir", args.figures_dir]
        if has_metrics:
            cmd += ["--metrics_dir", args.metrics_dir]
        print(f"\n{'═' * 18} {mod} {'═' * 18}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(mod)
            print(f"!!! {mod} FAILED")

    n = len(_EXPERIMENTS)
    print(f"\n[done] {n - len(failed)}/{n} succeeded"
          + (f" — failed: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
