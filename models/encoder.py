"""Condition Encoder — 기상장 프레임을 condition 토큰으로 변환 (instruction_v2 §2.3, 수정).

새 구조에서 condition은 프레임 수가 다르다:
  - DDPM_past : cond = x_t                 (1프레임) → (B, 256, D)
  - DDPM_main : cond = [x̂_{t-1}, x̂_{t+1}]  (2프레임) → (B, 512, D)

따라서 3D conv를 버리고 **프레임별 2D conv** 인코더를 쓴다. 각 프레임을
`(B, C, 64, 64) → patchify(p=4) → (B, 256, D) + 2D sinusoidal PE` 로 토큰화한 뒤,
멀티프레임(main)일 때만 학습된 time embedding(0=t-1, 1=t+1)을 슬롯별로 더하고
sequence 축으로 concat한다. past/main이 동일 인스턴스를 공유하며 모든 stage에서 학습.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .pos_emb import sinusoidal_2d_pos_emb


class ConditionEncoder(nn.Module):
    """기상장 프레임 → condition 토큰. `(B, F, C, H, W)`, F∈{1,2}.

    Args:
        in_channels:     변수 수 C (기본 3)
        hidden_channels: 2D conv 출력 채널 C' (기본 64)
        num_layers:      2D conv 레이어 수
        patch_size:      patchify 패치 크기 p (64 → 64/p 토큰 격자)
        token_dim:       토큰 임베딩 차원 D
        spatial_size:    입력 공간 해상도 (H, W)
        max_frames:      time embedding 슬롯 수 (main의 2프레임 구분용)
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 2,
        patch_size: int = 4,
        token_dim: int = 384,
        spatial_size: tuple[int, int] = (64, 64),
        max_frames: int = 2,
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
        self.max_frames = max_frames
        self.grid_h = H // patch_size
        self.grid_w = W // patch_size
        self.num_tokens = self.grid_h * self.grid_w

        # ── 프레임별 2D conv feature 추출 ──────────────────────────
        layers: list[nn.Module] = []
        c_in = in_channels
        for i in range(num_layers):
            c_out = (
                hidden_channels if i == num_layers - 1
                else max(hidden_channels // 2, in_channels)
            )
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(min(8, c_out), c_out))
            layers.append(nn.SiLU())
            c_in = c_out
        self.conv = nn.Sequential(*layers)

        # ── Patchify: (B, C', H, W) → (B, D, H/p, W/p) ─────────────
        self.patchify = nn.Conv2d(
            hidden_channels, token_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # ── 2D sinusoidal positional encoding (16×16 격자) ─────────
        pos = sinusoidal_2d_pos_emb(self.grid_h, self.grid_w, token_dim)
        self.register_buffer("pos_emb", pos.unsqueeze(0))   # (1, N_tok, D)

        # ── time embedding — main의 2프레임(t-1, t+1) 구분 (F>1에서만) ─
        self.time_emb = nn.Embedding(max_frames, token_dim)

    def _encode_one(self, frame: torch.Tensor) -> torch.Tensor:
        """단일 프레임 `(B, C, H, W)` → 토큰 `(B, N_tok, D)` (+2D PE)."""
        h = self.conv(frame)                   # (B, C', H, W)
        h = self.patchify(h)                   # (B, D, H/p, W/p)
        h = h.flatten(2).transpose(1, 2)       # (B, N_tok, D), row-major
        return h + self.pos_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, F, C, H, W) — F=1(past, x_t) 또는 F=2(main, [x̂_{t-1}, x̂_{t+1}])
        Returns:
            (B, F·N_tok, D) condition 토큰
        """
        if x.dim() != 5:
            raise ValueError(f"expected (B,F,C,H,W), got {tuple(x.shape)}")
        F = x.shape[1]
        if F > self.max_frames:
            raise ValueError(f"F={F} exceeds max_frames={self.max_frames}")

        toks = [self._encode_one(x[:, f]) for f in range(F)]   # 각 (B, N_tok, D)
        if F == 1:
            return toks[0]

        # F>1 (main): 슬롯별 time embedding 주입 후 sequence 축 concat
        idx = torch.arange(F, device=x.device)
        te = self.time_emb(idx)                                # (F, D)
        toks = [t + te[f].view(1, 1, -1) for f, t in enumerate(toks)]
        return torch.cat(toks, dim=1)                          # (B, F·N_tok, D)


def build_encoder(config: dict) -> ConditionEncoder:
    """config dict로부터 ConditionEncoder 생성 (학습/추론 공통)."""
    enc_cfg = config["encoder"]
    return ConditionEncoder(
        in_channels=int(config["data"]["n_channels"]),
        hidden_channels=int(enc_cfg["hidden_channels"]),
        num_layers=int(enc_cfg["num_layers"]),
        patch_size=int(enc_cfg["patch_size"]),
        token_dim=int(enc_cfg["token_dim"]),
        spatial_size=tuple(config["data"]["spatial"]),
    )
