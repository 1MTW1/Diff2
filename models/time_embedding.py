"""표준 sinusoidal positional encoding (Vaswani et al., 2017)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Diffusion timestep을 (B, dim) embedding으로 매핑."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) diffusion timestep (정수)
        Returns:
            (B, dim) embedding
        """
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
