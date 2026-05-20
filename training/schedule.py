"""Linear noise schedule + reverse process 관련 계산 (instruction.md §3.2~3.5, §4.7)."""
from __future__ import annotations

import torch


def _broadcast(scalar_per_b: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B,)을 target.dim()에 맞게 trailing dim 추가."""
    return scalar_per_b.view(-1, *([1] * (target.dim() - 1)))


class LinearNoiseSchedule:
    """β_m linear schedule. forward/reverse 계산 유틸 포함."""

    def __init__(
        self,
        M: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.05,
        device: str | torch.device = "cuda",
    ):
        self.M = M
        self.device = torch.device(device)

        self.betas = torch.linspace(beta_start, beta_end, M, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sigma_sq = 1.0 - self.alphas_cumprod          # σ_m^2

    # ── 기본 lookup ─────────────────────────────────────────────────
    def get_alpha_bar(self, m: torch.Tensor) -> torch.Tensor:
        return self.alphas_cumprod[m]

    def get_sigma_sq(self, m: torch.Tensor) -> torch.Tensor:
        return self.sigma_sq[m]

    def get_alpha_mn(self, m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """α_{m|n} = sqrt(ᾱ_m / ᾱ_n). n < 0이면 sqrt(ᾱ_m)."""
        if isinstance(n, torch.Tensor) and torch.any(n < 0):
            # 마지막 step (n=-1): ᾱ_n = 1 (정의상 x_0 자체)
            n_safe = n.clamp_min(0)
            alpha_n = torch.where(
                n < 0,
                torch.ones_like(self.alphas_cumprod[n_safe]),
                self.alphas_cumprod[n_safe],
            )
        else:
            alpha_n = self.alphas_cumprod[n]
        return torch.sqrt(self.alphas_cumprod[m] / alpha_n)

    def get_sigma_mn_sq(
        self, m: torch.Tensor, n: torch.Tensor
    ) -> torch.Tensor:
        """σ_{m|n}^2 = σ_m^2 - α_{m|n}^2 * σ_n^2. n<0이면 σ_n^2=0."""
        sigma_m_sq = self.get_sigma_sq(m)
        if isinstance(n, torch.Tensor) and torch.any(n < 0):
            n_safe = n.clamp_min(0)
            sigma_n_sq = torch.where(
                n < 0,
                torch.zeros_like(self.sigma_sq[n_safe]),
                self.sigma_sq[n_safe],
            )
        else:
            sigma_n_sq = self.sigma_sq[n]
        alpha_mn = self.get_alpha_mn(m, n)
        return sigma_m_sq - alpha_mn ** 2 * sigma_n_sq

    # ── Forward / reverse ──────────────────────────────────────────
    def forward_noise(
        self,
        x_0: torch.Tensor,
        m: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x_m = sqrt(ᾱ_m) x_0 + sqrt(1-ᾱ_m) ε."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = _broadcast(self.sqrt_alphas_cumprod[m], x_0)
        sqrt_omab = _broadcast(self.sqrt_one_minus_alphas_cumprod[m], x_0)
        return sqrt_ab * x_0 + sqrt_omab * noise, noise

    def reverse_mean(
        self,
        x_m: torch.Tensor,
        eps_pred: torch.Tensor,
        m: torch.Tensor,
    ) -> torch.Tensor:
        """μ_θ 계산 (Eq. 9). sampling 시 m은 batch 내 동일하다고 가정."""
        n = m - 1

        sigma_m_sq_b = self.get_sigma_sq(m)
        sigma_m_sq = _broadcast(sigma_m_sq_b, x_m)
        alpha_m_b = self.sqrt_alphas_cumprod[m]
        alpha_m = _broadcast(alpha_m_b, x_m)
        sigma_m = torch.sqrt(sigma_m_sq)

        # 마지막 step: x_0 prediction
        if torch.all(n < 0):
            return (x_m - sigma_m * eps_pred) / alpha_m

        # 일반 step (sampling은 batch-uniform m이므로 분기 가능)
        sigma_n_sq_b = self.get_sigma_sq(n)
        sigma_n_sq = _broadcast(sigma_n_sq_b, x_m)
        alpha_mn_b = self.get_alpha_mn(m, n)
        alpha_mn = _broadcast(alpha_mn_b, x_m)
        sigma_mn_sq_b = self.get_sigma_mn_sq(m, n)
        sigma_mn_sq = _broadcast(sigma_mn_sq_b, x_m)
        sqrt_alpha_n_b = self.sqrt_alphas_cumprod[n]
        sqrt_alpha_n = _broadcast(sqrt_alpha_n_b, x_m)

        term1 = (alpha_mn * sigma_n_sq / sigma_m_sq) * x_m
        x0_hat = (x_m - sigma_m * eps_pred) / alpha_m
        term2 = (sigma_mn_sq * sqrt_alpha_n / sigma_m_sq) * x0_hat
        return term1 + term2

    def reverse_variance_scheduler_term(
        self, m: torch.Tensor
    ) -> torch.Tensor:
        """Scheduler baseline variance σ_n^2 σ_{m|n}^2 / σ_m^2. 마지막 step이면 0."""
        n = m - 1
        if torch.all(n < 0):
            return torch.zeros_like(self.sigma_sq[m])

        sigma_m_sq = self.get_sigma_sq(m)
        sigma_n_sq = self.get_sigma_sq(n)
        sigma_mn_sq = self.get_sigma_mn_sq(m, n)
        return sigma_n_sq * sigma_mn_sq / sigma_m_sq
