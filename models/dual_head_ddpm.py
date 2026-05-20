"""Dual-head DDPM wrapper.

U-Net 출력 (2 * target_channels) → (ε̂, ℓ) 분리. ℓ는 clipping (§3.7).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DualHeadDDPM(nn.Module):
    """U-Net 위에서 dual-head (ε̂, ℓ) 출력을 분리해 반환."""

    def __init__(
        self,
        unet: nn.Module,
        target_channels: int,
        log_var_clip: tuple[float, float] = (-10.0, 10.0),
    ):
        super().__init__()
        self.unet = unet
        self.target_channels = target_channels
        self.log_var_clip = log_var_clip

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (B, target_channels, H, W) noisy target
            t:    (B,) timestep
            cond: (B, C', H, W) condition
        Returns:
            eps_pred: (B, target_channels, H, W)
            log_var:  (B, target_channels, H, W), clipped
        """
        out = self.unet(x, t, cond)
        eps_pred = out[:, : self.target_channels]
        log_var = out[:, self.target_channels :]
        log_var = torch.clamp(log_var, *self.log_var_clip)
        return eps_pred, log_var
