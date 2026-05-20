"""Diffusion U-Net backbone.

(B, T*C_data + C', H, W) → (B, 2*T*C_data, H, W) — dual-head용.
하위 2 resolution에 self-attention (instruction.md §4.5).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SpatialSelfAttention
from .time_embedding import SinusoidalTimeEmbedding


def _gn(ch: int, max_groups: int = 8) -> nn.GroupNorm:
    """채널 수에 안전한 GroupNorm 헬퍼."""
    g = max_groups
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)


class ResBlock(nn.Module):
    """Pre-norm residual block with time embedding injection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = _gn(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = _gn(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class _DownStage(nn.Module):
    """ResBlock(+attention) 또는 Downsample을 캡슐화."""

    def __init__(
        self,
        kind: str,
        block: nn.Module | None = None,
        attn: nn.Module | None = None,
        down: nn.Module | None = None,
    ):
        super().__init__()
        self.kind = kind
        if block is not None:
            self.block = block
        if attn is not None:
            self.attn = attn
        if down is not None:
            self.down = down


class UNet(nn.Module):
    """Diffusion U-Net.

    Input: (B, T*C_data, H, W) noisy target. Condition (B, C', H, W)이 forward에서
    channel-concat 되어 in_channels로 들어옴.
    Output: (B, 2*T*C_data, H, W) — dual-head (ε̂, ℓ).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (0, 1),
        time_emb_dim: int = 256,
        dropout: float = 0.1,
        pos_emb_channels: int = 0,
        spatial_size: tuple[int, int] | None = None,
    ):
        super().__init__()
        n_levels = len(channel_mults)
        # attention_resolutions는 bottom-indexed (0 = 가장 깊은 level)
        attn_levels = {n_levels - 1 - i for i in attention_resolutions}

        # ── Time embedding MLP ──────────────────────────────────────
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ── Position embedding (학습 가능한 2D 절대 위치 prior) ─────
        # 고정 도메인(한반도 64×64)에서 climatology 위치 의존성을 모델에 주입.
        self.pos_emb_channels = pos_emb_channels
        if pos_emb_channels > 0:
            if spatial_size is None:
                raise ValueError(
                    "spatial_size must be provided when pos_emb_channels > 0"
                )
            H, W = spatial_size
            self.pos_emb = nn.Parameter(
                torch.zeros(pos_emb_channels, H, W)
            )
            nn.init.normal_(self.pos_emb, std=0.02)
        else:
            self.pos_emb = None

        # ── Input conv ──────────────────────────────────────────────
        self.input_conv = nn.Conv2d(
            in_channels + pos_emb_channels, base_channels, 3, padding=1,
        )

        # ── Down path ───────────────────────────────────────────────
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()       # 같은 인덱스에서 사용
        self.down_kinds: list[str] = []         # 'res' or 'down'

        ch = base_channels
        skip_channels: list[int] = [ch]         # input_conv 출력
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResBlock(ch, out_ch, time_emb_dim, dropout)
                )
                attn = (
                    SpatialSelfAttention(out_ch)
                    if level in attn_levels else nn.Identity()
                )
                self.down_attns.append(attn)
                self.down_kinds.append("res")
                ch = out_ch
                skip_channels.append(ch)
            if level != n_levels - 1:
                self.down_blocks.append(Downsample(ch))
                self.down_attns.append(nn.Identity())
                self.down_kinds.append("down")
                skip_channels.append(ch)

        # ── Middle ──────────────────────────────────────────────────
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = SpatialSelfAttention(ch)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # ── Up path ─────────────────────────────────────────────────
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.up_kinds: list[str] = []

        for level in reversed(range(n_levels)):
            out_ch = base_channels * channel_mults[level]
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                self.up_blocks.append(
                    ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout)
                )
                attn = (
                    SpatialSelfAttention(out_ch)
                    if level in attn_levels else nn.Identity()
                )
                self.up_attns.append(attn)
                self.up_kinds.append("res")
                ch = out_ch
            if level != 0:
                self.up_blocks.append(Upsample(ch))
                self.up_attns.append(nn.Identity())
                self.up_kinds.append("up")

        assert not skip_channels, (
            f"Unmatched skips after up path: {skip_channels}"
        )

        # ── Output ──────────────────────────────────────────────────
        self.out_norm = _gn(ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

        # Zero-init last conv for stable training start
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, T*C_data, H, W)
            t:    (B,) diffusion timestep
            cond: (B, C', H, W) condition from Encoder
        Returns:
            (B, out_channels, H, W)
        """
        parts = [x, cond]
        if self.pos_emb is not None:
            pos = self.pos_emb.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            parts.append(pos)
        h = torch.cat(parts, dim=1)
        t_emb = self.time_embed(t)

        h = self.input_conv(h)
        skips = [h]

        for block, attn, kind in zip(
            self.down_blocks, self.down_attns, self.down_kinds
        ):
            if kind == "res":
                h = block(h, t_emb)
                h = attn(h)
            else:  # down
                h = block(h)
            skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for block, attn, kind in zip(
            self.up_blocks, self.up_attns, self.up_kinds
        ):
            if kind == "res":
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
                h = attn(h)
            else:  # up
                h = block(h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
