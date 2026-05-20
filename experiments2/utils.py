"""experiments/utils 의 t-2 별칭.

캐시 schema는 동일 (ensemble, log_var, x_t_true, time_t). 의미만 t-2 로 해석:
    ensemble  : x_{t-2} 예측 멤버
    log_var   : ℓ_past 의 x_{t-2} 채널부 (멤버 평균)
    x_t_true  : GT x_{t-2}
    time_t    : t-2 timestamp (denormalize 시 정확한 hour 사용을 위해)

이렇게 하면 experiments/exp{1..6}*.py 가 그대로 재사용 가능.
"""
from experiments.utils import (  # noqa: F401
    EnsembleSample,
    TEMP_IDX, U_IDX, V_IDX, VARIABLES,
    denorm_array, denorm_spread,
    ensure_dir,
    iter_ensemble_samples,
    list_ensemble_files,
    load_ensemble_npz,
    load_norm_stats,
    set_plot_defaults,
)
