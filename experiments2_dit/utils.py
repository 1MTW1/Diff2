"""experiments_dit/utils 의 t-2 별칭.

캐시 schema는 동일 (ensemble, log_var, x_t_true, ensemble_pixel,
x_t_true_pixel, time_t). 의미만 t-2 로 해석:
    ensemble        : x̂_{t-2} 예측 멤버의 latent (= z_past)
    log_var         : ℓ_past 의 latent dual-head log_var (멤버 평균)
    x_t_true        : GT [x_{t-3}, x_{t-2}] 블록의 latent 인코딩
    ensemble_pixel  : 디코딩된 x̂_{t-2} 픽셀 앙상블 (exp5 전용)
    x_t_true_pixel  : GT x_{t-2} 픽셀 필드 (exp5 전용)
    time_t          : t-2 timestamp

이렇게 하면 experiments_dit/exp{1..6}*.py 가 그대로 재사용 가능하다.
"""
from experiments_dit.utils import (  # noqa: F401
    EnsembleSample,
    LATENT_CH,
    TEMP_IDX, U_IDX, V_IDX, VARIABLES,
    denorm_array, denorm_spread,
    ensure_dir,
    iter_ensemble_samples,
    list_ensemble_files,
    load_ensemble_npz,
    load_norm_stats,
    set_plot_defaults,
)
