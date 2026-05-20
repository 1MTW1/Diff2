"""t-2 시점에서의 평가 실험 패키지.

- experiments/는 model_main 의 x_t prediction을 평가.
- experiments2/는 model_past 의 x_{t-2} prediction을 평가.
  past 모델은 학습 시 reverse chain에 learned ℓ를 사용한 분포로 학습되었으므로
  inference 에서도 learned variance 를 그대로 켜는 것이 분포 일치 측면에서 자연.
"""
