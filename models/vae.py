"""WeatherVAE — 2시점 기상장 ↔ latent 변환 (instruction_v2 §2.1).

(B, 6, 64, 64)  ──encode──▶  posterior (μ_z, logσ²_z), 각 (B, 12, 16, 16)
(B, 12, 16, 16) ──decode──▶  (B, 6, 64, 64)

공간 64→16 (×4 downsample), 채널 6→12 → 총 8배 압축.
diffusion과 분리되어 Stage 0에서 개별 학습 후 freeze 된다.
KL은 매우 약하게 (압축기 역할; latent를 가우시안으로 강제하지 않음 — §2.1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """채널 수에 안전한 GroupNorm 헬퍼."""
    g = max_groups
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class _ResBlock(nn.Module):
    """Pre-norm residual conv block (time embedding 없음 — VAE는 비-diffusion)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = _gn(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = _gn(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class _Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class _Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class WeatherVAE(nn.Module):
    """2시점 기상장용 KL-regularized autoencoder.

    Args:
        in_channels:     입력 채널 (= target_frames × C, 기본 6)
        latent_channels: latent 채널 C_z (기본 12)
        base_channels:   encoder/decoder base width
        ch_mult:         level별 채널 배수 (len == downsample 횟수)
        logvar_clip:     posterior logσ² clamp 범위 (수치 안정)
    """

    def __init__(
        self,
        in_channels: int = 6,
        latent_channels: int = 12,
        base_channels: int = 128,
        ch_mult: tuple[int, ...] = (1, 2),
        logvar_clip: tuple[float, float] = (-30.0, 20.0),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.logvar_clip = logvar_clip

        # ── Encoder: 64 → 32 → 16 ──────────────────────────────────
        enc: list[nn.Module] = [nn.Conv2d(in_channels, base_channels, 3, padding=1)]
        ch = base_channels
        for mult in ch_mult:
            out_ch = base_channels * mult
            enc.append(_ResBlock(ch, out_ch))
            enc.append(_Downsample(out_ch))
            ch = out_ch
        enc.append(_ResBlock(ch, ch))
        self.encoder = nn.Sequential(*enc)
        self.enc_norm = _gn(ch)
        # posterior: μ_z 와 logσ²_z 동시 출력 → 2·latent_channels
        self.to_posterior = nn.Conv2d(ch, 2 * latent_channels, 3, padding=1)

        # ── Decoder: 16 → 32 → 64 (encoder 대칭) ────────────────────
        self.from_latent = nn.Conv2d(latent_channels, ch, 3, padding=1)
        dec: list[nn.Module] = [_ResBlock(ch, ch)]
        rev_mult = list(reversed(ch_mult))
        for i in range(len(rev_mult)):
            dec.append(_Upsample(ch))
            # 다음 레벨의 목표 채널 (마지막 레벨은 base_channels)
            out_ch = (
                base_channels * rev_mult[i + 1]
                if i + 1 < len(rev_mult) else base_channels
            )
            dec.append(_ResBlock(ch, out_ch))
            ch = out_ch
        self.decoder = nn.Sequential(*dec)
        self.dec_norm = _gn(ch)
        self.to_output = nn.Conv2d(ch, in_channels, 3, padding=1)

    # ── 핵심 메서드 ─────────────────────────────────────────────────
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, in_channels, 64, 64) → (μ_z, logσ²_z), 각 (B, C_z, 16, 16)."""
        h = self.encoder(x)
        h = self.to_posterior(F.silu(self.enc_norm(h)))
        mu, log_var = h.chunk(2, dim=1)
        log_var = torch.clamp(log_var, *self.logvar_clip)
        return mu, log_var

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """z = μ + exp(0.5·logσ²)·ε,  ε ~ N(0, I)."""
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, C_z, 16, 16) → (B, in_channels, 64, 64)."""
        h = self.from_latent(z)
        h = self.decoder(h)
        return self.to_output(F.silu(self.dec_norm(h)))

    def forward(
        self, x: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            recon:   (B, in_channels, 64, 64)
            mu:      (B, C_z, 16, 16)
            log_var: (B, C_z, 16, 16)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var) if sample else mu
        recon = self.decode(z)
        return recon, mu, log_var


def build_vae(vae_cfg: dict) -> WeatherVAE:
    """config["vae"] dict로부터 WeatherVAE 생성 (학습/추론 공통)."""
    return WeatherVAE(
        in_channels=int(vae_cfg["in_channels"]),
        latent_channels=int(vae_cfg["latent_channels"]),
        base_channels=int(vae_cfg.get("base_channels", 128)),
        ch_mult=tuple(vae_cfg.get("ch_mult", (1, 2))),
        logvar_clip=tuple(vae_cfg.get("logvar_clip", (-30.0, 20.0))),
    )


def _spatial_gradient_loss(
    recon: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """인접 격자 차분(∂x, ∂y) 매칭 — 재구성의 선명도 보조항."""
    rx = recon[..., :, 1:] - recon[..., :, :-1]
    tx = target[..., :, 1:] - target[..., :, :-1]
    ry = recon[..., 1:, :] - recon[..., :-1, :]
    ty = target[..., 1:, :] - target[..., :-1, :]
    return F.mse_loss(rx, tx) + F.mse_loss(ry, ty)


def weather_vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 1e-6,
    grad_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L = L_recon + λ_kl·L_kl (+ λ_grad·L_grad).

    L_kl: 표준 정규분포를 향한 KL. λ_kl는 ~1e-6 (LDM/SD 관례, §2.1) —
    이보다 크면 posterior collapse 위험. latent 분포 정규화는 §2.2의
    사후 통계로 처리하므로 KL은 약한 발산 방지용일 뿐이다.
    """
    recon_loss = F.mse_loss(recon, target)
    # 표준 VAE KL: 0.5·Σ(μ² + σ² − logσ² − 1), 요소별 평균
    kl = 0.5 * (mu.pow(2) + log_var.exp() - log_var - 1.0).mean()

    total = recon_loss + kl_weight * kl
    logs = {"recon": float(recon_loss.detach()), "kl": float(kl.detach())}
    if grad_weight > 0:
        grad_loss = _spatial_gradient_loss(recon, target)
        total = total + grad_weight * grad_loss
        logs["grad"] = float(grad_loss.detach())
    logs["vae_total"] = float(total.detach())
    return total, logs
