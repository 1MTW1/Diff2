"""Diffusion²-LDM 모델 패키지 (instruction_v2.md).

LDM 변환 후 핵심 컴포넌트:
  - WeatherVAE       : 기상장 ↔ latent (Stage 0 선학습, frozen)
  - LatentNormalizer : latent 사후 정규화
  - VDMSchedule      : 연속시간 VP noise schedule
  - TemporalEncoder  : 2시점 condition → 토큰
  - DualHeadDiT      : Transformer diffusion backbone (U-Net 대체)

레거시 픽셀 공간 모듈(UNet, SpatialSelfAttention, DualHeadDDPM)은 파일은
유지되지만 패키지 export에서는 제외됨 — 필요 시 `models.unet` 등으로 직접 import.
"""
from .dit import DualHeadDiT, build_dit
from .encoder import TemporalEncoder, build_encoder
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
    "TemporalEncoder",
    "build_encoder",
    "DualHeadDiT",
    "build_dit",
    "SinusoidalTimeEmbedding",
    "sinusoidal_2d_pos_emb",
]
