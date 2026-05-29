"""Condition Encoder — 2시점 condition 기상장을 토큰 시퀀스로 변환 (instruction_v2 §2.3).

(B, 2, C, 64, 64)  ──Conv3d──▶  (B, C', 64, 64)
                   ──patchify(p=4)──▶  (B, 16×16, D)  + 2D sinusoidal PE

기존 `TemporalEncoder`의 3D Conv 구조는 유지하되 입력 시점 수를 **항상 2**로
고정하고, 출력을 patchify하여 DiT의 cross-attention용 토큰으로 만든다.
past/main이 동일 인스턴스를 공유하며 **모든 stage에서 학습**된다 (VAE만 frozen).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .pos_emb import sinusoidal_2d_pos_emb


class TemporalEncoder(nn.Module):
    """2시점 condition → condition 토큰 `(B, N_tok, D)`.

    Args:
        in_channels:     변수 수 C (기본 3)
        hidden_channels: Conv3d 출력 채널 C' (기본 64)
        num_layers:      Conv3d 레이어 수
        patch_size:      patchify 패치 크기 p (64 → 64/p 토큰 격자)
        token_dim:       토큰 임베딩 차원 D
        spatial_size:    입력 공간 해상도 (H, W)
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 2,
        patch_size: int = 4,
        token_dim: int = 384,
        spatial_size: tuple[int, int] = (64, 64),
    ):
        super().__init__()
        H, W = spatial_size
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(
                f"spatial_size {spatial_size} must be divisible by "
                f"patch_size {patch_size}"
            )
        self.patch_size = patch_size
        self.token_dim = token_dim
        self.grid_h = H // patch_size
        self.grid_w = W // patch_size
        self.num_tokens = self.grid_h * self.grid_w

        # ── 3D Conv 시공간 feature 추출 (기존 구조 유지) ───────────
        layers: list[nn.Module] = []
        c_in = in_channels
        for i in range(num_layers):
            c_out = (
                hidden_channels if i == num_layers - 1
                else max(hidden_channels // 2, in_channels)
            )
            layers.append(
                nn.Conv3d(c_in, c_out, kernel_size=(3, 3, 3),
                          padding=(1, 1, 1))
            )
            layers.append(nn.GroupNorm(min(8, c_out), c_out))
            layers.append(nn.SiLU())
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))

        # ── Patchify: (B, C', H, W) → (B, D, H/p, W/p) ─────────────
        self.patchify = nn.Conv2d(
            hidden_channels, token_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # ── 2D sinusoidal positional encoding (16×16 격자) ─────────
        pos = sinusoidal_2d_pos_emb(self.grid_h, self.grid_w, token_dim)
        self.register_buffer("pos_emb", pos.unsqueeze(0))   # (1, N_tok, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, C, H, W) — 항상 2시점
        Returns:
            (B, N_tok, D) condition 토큰
        """
        if x.shape[1] != 2:
            raise ValueError(
                f"TemporalEncoder expects exactly 2 timesteps, got {x.shape[1]}"
            )
        # (B, 2, C, H, W) → (B, C, 2, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv(x)                       # (B, C', 2, H, W)
        x = self.temporal_pool(x).squeeze(2)   # (B, C', H, W)

        x = self.patchify(x)                   # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)       # (B, N_tok, D), row-major
        return x + self.pos_emb


def build_encoder(config: dict) -> TemporalEncoder:
    """config dict로부터 TemporalEncoder 생성 (학습/추론 공통)."""
    enc_cfg = config["encoder"]
    return TemporalEncoder(
        in_channels=int(config["data"]["n_channels"]),
        hidden_channels=int(enc_cfg["hidden_channels"]),
        num_layers=int(enc_cfg["num_layers"]),
        patch_size=int(enc_cfg["patch_size"]),
        token_dim=int(enc_cfg["token_dim"]),
        spatial_size=tuple(config["data"]["spatial"]),
    )
