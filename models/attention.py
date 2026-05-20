"""Spatial self-attention block for U-Net feature maps."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SpatialSelfAttention(nn.Module):
    """(B, C, H, W) 위에서 동작하는 multi-head self-attention.

    Pre-norm + residual; QKV/output projection은 1×1 conv.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)                                    # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)                        # (B, C, H, W) each

        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)

        attn = torch.einsum("bhdi,bhdj->bhij", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)
        return x + out
