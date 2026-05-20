"""Temporal Encoder: 여러 시점의 기상장을 condition feature map으로 변환."""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """(B, T, C, H, W) → (B, C', H, W).

    Conv3d로 시공간 feature를 추출한 뒤 시간축을 AdaptiveAvgPool로 collapse.
    같은 인스턴스를 T=2 (past) / T=3 (main) 양쪽에 재사용 가능.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
    ):
        super().__init__()
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
        self.out_channels = hidden_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            (B, C', H, W)
        """
        # (B, T, C, H, W) → (B, C, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv(x)                  # (B, C', T, H, W)
        x = self.temporal_pool(x)         # (B, C', 1, H, W)
        return x.squeeze(2)               # (B, C', H, W)
