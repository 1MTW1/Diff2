"""Condition Encoder — snapshot 기반 condition 인코더 (IMPLEMENTATION_SPEC v2 §5).

SEEDS(Li et al., 2024) S2.3 을 우리 규모로 단순화: anomaly·climatology snapshot 들을
**공유 conv 인코더**로 처리하고, snapshot 종류는 **type(categorical) embedding** 으로
구분한 뒤, snapshot 간 **self-attention** 으로 결합한다. SEEDS 의 변수축·공간축 axial
attention 은 conv/patchify 로 대체하고 핵심(type embedding + snapshot self-attention)만 채택.

condition 구성 규칙 (§1 — "자기가 생성하는 시점의 climatology" 를 받음):
  - DDPM_past : { x̃_t(ANOM_CENTER), c̃_{t-1}(CLIM_PREV), c̃_{t+1}(CLIM_NEXT) }
  - DDPM_main : { x̃_{t-1}(ANOM_PREV), x̃_{t+1}(ANOM_NEXT), c̃_t(CLIM_CENTER) }

각 snapshot 은 표준화된 픽셀장 `(B,C,64,64)`:
  conv → patchify(p=4) → (B,256,D) + 2D PE + type embedding[type]
  → 모든 snapshot concat → snapshot self-attention → condition tokens (B, N_snap·256, D)

`forward` 입력은 `past_snapshots`/`main_snapshots` 빌더가 만든 [(field, type_id), ...]
리스트다 (§1 규칙).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .pos_emb import sinusoidal_2d_pos_emb


# ── snapshot type enum (anomaly vs climatology × prev/center/next) ──────
class SnapshotType:
    ANOM_CENTER = 0   # x̃_t  (past condition)
    ANOM_PREV = 1     # x̃_{t-1}
    ANOM_NEXT = 2     # x̃_{t+1}
    CLIM_PREV = 3     # c̃_{t-1}
    CLIM_CENTER = 4   # c̃_t
    CLIM_NEXT = 5     # c̃_{t+1}


NUM_SNAPSHOT_TYPES = 6


class _SnapshotAttnBlock(nn.Module):
    """snapshot 토큰 간 pre-norm self-attention + MLP (condition 결합)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.norm2(x))


class ConditionEncoder(nn.Module):
    """snapshot 들 → condition 토큰. type embedding + snapshot self-attention.

    Args:
        in_channels:        변수 수 C (기본 3)
        hidden_channels:    2D conv 출력 채널 C' (기본 64)
        num_layers:         2D conv 레이어 수
        patch_size:         patchify 패치 크기 p (64 → 64/p 토큰 격자)
        token_dim:          토큰 임베딩 차원 D
        spatial_size:       입력 공간 해상도 (H, W)
        num_snapshot_types: type embedding 종류 수 (기본 6)
        snapshot_attn_layers: snapshot self-attention 블록 수 (기본 2)
        num_heads:          snapshot self-attention head 수 (기본 6)
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 2,
        patch_size: int = 4,
        token_dim: int = 384,
        spatial_size: tuple[int, int] = (64, 64),
        num_snapshot_types: int = NUM_SNAPSHOT_TYPES,
        snapshot_attn_layers: int = 2,
        num_heads: int = 6,
    ):
        super().__init__()
        H, W = spatial_size
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(
                f"spatial_size {spatial_size} must be divisible by "
                f"patch_size {patch_size}"
            )
        grid_h, grid_w = H // patch_size, W // patch_size

        # ── 프레임별 2D conv feature 추출 (모든 snapshot 공유) ──────
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
        pos = sinusoidal_2d_pos_emb(grid_h, grid_w, token_dim)
        self.register_buffer("pos_emb", pos.unsqueeze(0))   # (1, N_tok, D)

        # ── type embedding — snapshot 종류 구분 (anomaly/climatology × 시점) ─
        self.type_emb = nn.Embedding(num_snapshot_types, token_dim)

        # ── snapshot 간 self-attention (SEEDS s-축 transformer) ────
        self.snapshot_attn = nn.ModuleList([
            _SnapshotAttnBlock(token_dim, num_heads)
            for _ in range(snapshot_attn_layers)
        ])

    # ── 단일 snapshot 공간 인코딩 (+2D PE, type 없음) ───────────────
    def _encode_one(self, frame: torch.Tensor) -> torch.Tensor:
        """`(B, C, H, W)` → 토큰 `(B, N_tok, D)` (+2D PE)."""
        h = self.conv(frame)                   # (B, C', H, W)
        h = self.patchify(h)                   # (B, D, H/p, W/p)
        h = h.flatten(2).transpose(1, 2)       # (B, N_tok, D), row-major
        return h + self.pos_emb

    def _type_vec(self, type_id: int) -> torch.Tensor:
        """type embedding 벡터 (1, 1, D) — snapshot 토큰에 broadcast 더함."""
        return self.type_emb.weight[type_id].view(1, 1, -1)

    # ── forward — snapshot 리스트 → condition 토큰 (DDP 호환 단일 진입점) ──
    def forward(
        self, snapshots: list[tuple[torch.Tensor, int]],
    ) -> torch.Tensor:
        """snapshots: [(field (B,C,H,W), type_id), ...] → (B, N_snap·N_tok, D).

        §1 규칙대로 `past_snapshots`/`main_snapshots` 빌더로 구성해 넘긴다.
          [단계 A] 각 snapshot 공간 인코딩 + type embedding broadcast
          [단계 B] 전 snapshot 토큰 concat → snapshot self-attention
        accelerate.prepare(DDP) 의 grad 동기화를 위해 반드시 이 forward 를 호출한다.
        """
        if not snapshots:
            raise ValueError("snapshots must be non-empty")
        toks = [
            self._encode_one(field) + self._type_vec(type_id)
            for field, type_id in snapshots
        ]
        x = torch.cat(toks, dim=1)             # (B, N_snap·N_tok, D)
        for blk in self.snapshot_attn:
            x = blk(x)
        return x


# ── §1 규칙 snapshot 빌더 (DDP-safe: encoder(forward) 입력 구성) ─────────
def past_snapshots(
    x_t_anom: torch.Tensor,
    c_tm1: torch.Tensor | None = None,
    c_tp1: torch.Tensor | None = None,
) -> list[tuple[torch.Tensor, int]]:
    """DDPM_past condition snapshot 리스트: 중심 anomaly + 이웃 climatology."""
    snaps: list[tuple[torch.Tensor, int]] = [
        (x_t_anom, SnapshotType.ANOM_CENTER)
    ]
    if c_tm1 is not None:
        snaps.append((c_tm1, SnapshotType.CLIM_PREV))
    if c_tp1 is not None:
        snaps.append((c_tp1, SnapshotType.CLIM_NEXT))
    return snaps


def main_snapshots(
    x_tm1_anom: torch.Tensor,
    x_tp1_anom: torch.Tensor,
    c_t: torch.Tensor | None = None,
) -> list[tuple[torch.Tensor, int]]:
    """DDPM_main condition snapshot 리스트: 이웃 anomaly + 중심 climatology."""
    snaps: list[tuple[torch.Tensor, int]] = [
        (x_tm1_anom, SnapshotType.ANOM_PREV),
        (x_tp1_anom, SnapshotType.ANOM_NEXT),
    ]
    if c_t is not None:
        snaps.append((c_t, SnapshotType.CLIM_CENTER))
    return snaps


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
        num_snapshot_types=int(enc_cfg.get("num_snapshot_types", NUM_SNAPSHOT_TYPES)),
        snapshot_attn_layers=int(enc_cfg.get("snapshot_attn_layers", 2)),
        num_heads=int(enc_cfg.get("snapshot_attn_heads",
                                  config.get("dit", {}).get("num_heads", 6))),
    )
