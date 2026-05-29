"""Latent 정규화 (instruction_v2 §2.2).

VAE 학습 완료 후 latent의 1·2차 모멘트만 평균0/분산1로 맞춘다.
**분포를 가우시안으로 바꾸는 게 아니다** — latent ẑ는 diffusion 입장에서 픽셀
이미지와 같은 "데이터"일 뿐이며, 가우시안일 필요가 없다. 분산만 ≈1이면 VP
noise schedule이 의도대로 동작하고 유한 step에서도 z_T ≈ N(0,I)가 성립한다.

지원 mode (저장된 통계 텐서 shape으로 자동 구분):
  - channel_pixelwise: μ,σ ∈ (C_z, H_z, W_z)  — 위치별·채널별
  - channelwise:       μ,σ ∈ (C_z, 1, 1)      — 채널별만 (데이터가 적을 때 fallback)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LatentNormalizer(nn.Module):
    """ẑ = (z − μ)/σ,  z = ẑ·σ + μ.  통계는 buffer로 보관 (학습 대상 아님)."""

    def __init__(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        mode: str = "channel_pixelwise",
        sigma_floor: float = 1e-6,
    ):
        super().__init__()
        if mu.shape != sigma.shape:
            raise ValueError(
                f"mu/sigma shape mismatch: {mu.shape} vs {sigma.shape}"
            )
        self.mode = mode
        # batch 차원을 추가해 (1, C_z, *, *) — z (B, C_z, H_z, W_z)에 broadcast.
        mu = mu.float().unsqueeze(0)
        sigma = sigma.float().clamp_min(sigma_floor).unsqueeze(0)
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        """raw latent z → 정규화 latent ẑ."""
        return (z - self.mu) / self.sigma

    def denormalize(self, z_norm: torch.Tensor) -> torch.Tensor:
        """정규화 latent ẑ → raw latent z."""
        return z_norm * self.sigma + self.mu

    @classmethod
    def from_file(
        cls,
        path: str,
        map_location: torch.device | str = "cpu",
    ) -> "LatentNormalizer":
        """`compute_latent_stats.py`가 저장한 latent_stats.pt 로드."""
        stats = torch.load(path, map_location=map_location)
        return cls(
            mu=stats["mu"],
            sigma=stats["sigma"],
            mode=stats.get("mode", "channel_pixelwise"),
        )
