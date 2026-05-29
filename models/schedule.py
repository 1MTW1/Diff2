"""VDM noise schedule — 연속시간 variance-preserving (instruction_v2 §2.5).

연속 시점 t∈[0,1] → (α(t), σ(t)). 고정 선형 log-SNR.

부호 규약 (코드 전반에서 일관):
    γ(t) = γ_min + t·(γ_max − γ_min)      — t에 대해 단조 증가
    α(t)² = sigmoid(−γ(t)),  σ(t)² = sigmoid(γ(t))
    ⇒ α² + σ² = 1                          (VP 보장; sigmoid(−γ)+sigmoid(γ)=1)
    ⇒ SNR(t) = α²/σ² = exp(−γ(t))

즉 γ는 −log-SNR이다. t가 커질수록 γ↑ → σ↑ → 노이즈↑ (t=0 거의 clean,
t=1 거의 순수 노이즈). ε-prediction과 완벽히 호환된다.
"""
from __future__ import annotations

import torch


def _to_bchw(v: torch.Tensor) -> torch.Tensor:
    """(B,) → (B,1,1,1) — (B,C,H,W) latent에 broadcast."""
    return v.view(-1, 1, 1, 1)


class VDMSchedule:
    """선형 log-SNR VP schedule. 상태 없는 순수 함수 모음."""

    def __init__(self, gamma_min: float = -6.0, gamma_max: float = 6.0):
        if gamma_max <= gamma_min:
            raise ValueError(
                f"gamma_max ({gamma_max}) must exceed gamma_min ({gamma_min})"
            )
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)

    # ── 기본 함수 ───────────────────────────────────────────────────
    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """선형 −log-SNR γ(t).  (B,) → (B,1,1,1)."""
        g = self.gamma_min + t * (self.gamma_max - self.gamma_min)
        return _to_bchw(g)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """신호 계수 α(t) = sqrt(sigmoid(−γ)).  (B,) → (B,1,1,1)."""
        return torch.sqrt(torch.sigmoid(-self.gamma(t)))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """노이즈 계수 σ(t) = sqrt(sigmoid(γ)).  (B,) → (B,1,1,1)."""
        return torch.sqrt(torch.sigmoid(self.gamma(t)))

    def alpha_sigma(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(α(t), σ(t)) 동시 반환."""
        g = self.gamma(t)
        return (
            torch.sqrt(torch.sigmoid(-g)),
            torch.sqrt(torch.sigmoid(g)),
        )

    # ── Forward (noising) ──────────────────────────────────────────
    def forward_noise(
        self,
        z_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """z_t = α(t)·z_0 + σ(t)·ε,  ε ~ N(0, I).

        Args:
            z_0:   (B, C, H, W) 깨끗한 (정규화된) latent
            t:     (B,) ∈ [0, 1] 연속 시점
            noise: 외부 주입 ε (None이면 새로 샘플링)
        Returns:
            (z_t, ε)
        """
        if noise is None:
            noise = torch.randn_like(z_0)
        alpha, sigma = self.alpha_sigma(t)
        return alpha * z_0 + sigma * noise, noise
