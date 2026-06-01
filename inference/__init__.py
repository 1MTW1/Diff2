"""추론 패키지.

LDM (instruction_v2):
  - LatentVDMSampler        : latent 공간 연속시간 VDM sampler (+ 불확실성 주입)
  - generate_future_ensemble: past가 이웃 [x̂_{t-1},x̂_{t+1}] 생성 → main이 x̂_t 복원하는 앙상블

"""
from .sampling import LatentVDMSampler, generate_future_ensemble

__all__ = [
    "LatentVDMSampler",
    "generate_future_ensemble",
]
