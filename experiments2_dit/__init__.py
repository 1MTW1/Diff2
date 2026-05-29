"""t-2 시점에서의 v2 LDM/DiT 평가 실험 패키지.

- experiments_dit/ 는 DDPM_main 의 x_t 예측을 평가.
- experiments2_dit/ 는 DDPM_past 의 x̂_{t-2} 예측을 평가.
  past 모델은 학습 시 reverse chain에 dual-head log_var noise를 주입한 분포로
  학습되므로 (instruction_v2 §4.1), inference 에서도 `inject_uncertainty=True`
  를 그대로 켜는 것이 분포 일치 측면에서 자연스럽다. 캐시 schema 와 exp1~exp6
  모듈은 experiments_dit/ 와 동일하며 의미만 t-2 로 해석된다.
"""
