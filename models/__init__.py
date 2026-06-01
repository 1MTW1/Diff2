"""Diffusion²-LDM 모델 패키지 (instruction_v2.md).

LDM 변환 후 핵심 컴포넌트:
  - WeatherVAE        : 기상장 프레임 ↔ latent (프레임별, Stage 0 선학습, frozen)
  - LatentNormalizer  : latent 사후 정규화
  - VDMSchedule       : 연속시간 VP noise schedule
  - ConditionEncoder  : 프레임별 2D conv condition 인코더 (past 1프레임 / main 2프레임)
  - DualHeadDiT       : Transformer diffusion backbone (U-Net 대체)
"""
from .dit import DualHeadDiT, build_dit
from .encoder import ConditionEncoder, build_encoder
from .latent_norm import LatentNormalizer
from .pos_emb import sinusoidal_2d_pos_emb
from .schedule import VDMSchedule
from .time_embedding import SinusoidalTimeEmbedding
from .vae import WeatherVAE, build_vae, weather_vae_loss

__all__ = [
    "WeatherVAE",
    "build_vae",
    "weather_vae_loss",
    "LatentNormalizer",
    "VDMSchedule",
    "ConditionEncoder",
    "build_encoder",
    "DualHeadDiT",
    "build_dit",
    "SinusoidalTimeEmbedding",
    "sinusoidal_2d_pos_emb",
]
