"""DualHeadDiT — Transformer diffusion backbone (instruction_v2 §2.4).

기존 픽셀 공간 U-Net을 대체한다. latent target과 condition 토큰을 Transformer로
결합하여 dual-head `(eps_pred, log_var)`를 출력한다.

    z_t (B, C_z, 16, 16)  ──patchify(p=1)──▶  256 토큰 (각 latent 픽셀 = 1 토큰)
    + 2D sinusoidal PE
    + condition 토큰 (B, 256, D)  ──cross-attention──▶
    + 연속 t  ──sinusoidal+MLP──▶ AdaLN-zero scale/shift/gate
    ──▶ unpatchify ──▶ (B, 2·C_z, 16, 16) = (eps_pred ‖ log_var)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .pos_emb import sinusoidal_2d_pos_emb
from .time_embedding import SinusoidalTimeEmbedding


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """AdaLN modulation: x·(1+scale) + shift.  x:(B,N,D), shift/scale:(B,D)."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """DiT Transformer block — self-attn(target) + cross-attn(target↔cond) + MLP.

    DiT(Peebles & Xie, 2023)의 AdaLN-zero 방식으로 연속 시점 t를 주입한다.
    cross-attention으로 condition 토큰을 결합 (SD3 MM-DiT 철학 — §2.4).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        # AdaLN-zero: t_emb → (shift,scale,gate) × 3 (self/cross/mlp)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        sh1, sc1, g1, sh2, sc2, g2, sh3, sc3, g3 = (
            self.adaLN(t_emb).chunk(9, dim=1)
        )
        # self-attention (target 토큰 간)
        h = _modulate(self.norm1(x), sh1, sc1)
        attn, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + g1.unsqueeze(1) * attn
        # cross-attention (target query ← condition key/value)
        h = _modulate(self.norm2(x), sh2, sc2)
        attn, _ = self.cross_attn(h, cond, cond, need_weights=False)
        x = x + g2.unsqueeze(1) * attn
        # MLP
        h = _modulate(self.norm3(x), sh3, sc3)
        x = x + g3.unsqueeze(1) * self.mlp(h)
        return x


class _FinalLayer(nn.Module):
    """AdaLN + linear projection → dual-head 채널. 출력 projection zero-init."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(t_emb).chunk(2, dim=1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class DualHeadDiT(nn.Module):
    """Latent diffusion Transformer with dual-head 출력.

    Args:
        latent_channels: latent 채널 C_z (기본 12)
        latent_size:     latent 공간 해상도 (H_z, W_z) (기본 16×16)
        token_dim:       Transformer hidden dim D
        depth:           Transformer block 수
        num_heads:       attention head 수
        mlp_ratio:       MLP hidden 배수
        dropout:         dropout 비율
        log_var_clip:    log_var clamp 범위
        time_scale:      연속 t∈[0,1]를 sinusoidal embedding 전 스케일
                         (정수 timestep용 embedding 재사용을 위함 — §2.4)
    """

    def __init__(
        self,
        latent_channels: int = 12,
        latent_size: tuple[int, int] = (16, 16),
        token_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        log_var_clip: tuple[float, float] = (-10.0, 10.0),
        time_scale: float = 1000.0,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.grid_h, self.grid_w = latent_size
        self.num_tokens = self.grid_h * self.grid_w
        self.log_var_clip = log_var_clip
        self.time_scale = time_scale

        # ── latent 토큰화 (patch_size=1: 각 latent 픽셀 = 1 토큰) ───
        self.x_embed = nn.Conv2d(latent_channels, token_dim, kernel_size=1)
        pos = sinusoidal_2d_pos_emb(self.grid_h, self.grid_w, token_dim)
        self.register_buffer("pos_emb", pos.unsqueeze(0))   # (1, N_tok, D)

        # ── 연속 시점 임베딩 ───────────────────────────────────────
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbedding(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )

        # ── Transformer blocks ─────────────────────────────────────
        self.blocks = nn.ModuleList([
            DiTBlock(token_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.final = _FinalLayer(token_dim, 2 * latent_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        def _basic(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)
        # AdaLN-zero: 각 block의 변조 출력을 0으로 → block이 identity로 시작
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        # 최종 layer: 변조·출력 projection 모두 zero-init (학습 초기 안정성)
        nn.init.zeros_(self.final.adaLN[-1].weight)
        nn.init.zeros_(self.final.adaLN[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_t:         (B, C_z, H_z, W_z) noisy latent target
            t:           (B,) ∈ [0, 1] 연속 시점
            cond_tokens: (B, N_tok, D) condition 토큰 (encoder 출력)
        Returns:
            eps_pred: (B, C_z, H_z, W_z) latent 노이즈 예측
            log_var:  (B, C_z, H_z, W_z) latent 요소별 log 분산 (clip 적용)
        """
        B = z_t.shape[0]
        # 토큰화 + 2D PE
        x = self.x_embed(z_t)                       # (B, D, H_z, W_z)
        x = x.flatten(2).transpose(1, 2)            # (B, N_tok, D), row-major
        x = x + self.pos_emb

        # 연속 t → 임베딩 (정수용 sinusoidal 재사용 위해 time_scale 적용)
        t_emb = self.t_embed(t.float() * self.time_scale)   # (B, D)

        for block in self.blocks:
            x = block(x, cond_tokens, t_emb)
        x = self.final(x, t_emb)                    # (B, N_tok, 2·C_z)

        # unpatchify: (B, N_tok, 2·C_z) → (B, 2·C_z, H_z, W_z)
        out = x.transpose(1, 2).reshape(
            B, 2 * self.latent_channels, self.grid_h, self.grid_w
        )
        eps_pred = out[:, : self.latent_channels]
        log_var = torch.clamp(out[:, self.latent_channels:], *self.log_var_clip)
        return eps_pred, log_var


def build_dit(config: dict) -> DualHeadDiT:
    """config dict로부터 DualHeadDiT 생성 (학습/추론 공통)."""
    vae_cfg = config["vae"]
    dit_cfg = config["dit"]
    head_cfg = config["dual_head"]
    return DualHeadDiT(
        latent_channels=int(vae_cfg["latent_channels"]),
        latent_size=tuple(vae_cfg["latent_spatial"]),
        token_dim=int(dit_cfg["token_dim"]),
        depth=int(dit_cfg["depth"]),
        num_heads=int(dit_cfg["num_heads"]),
        mlp_ratio=float(dit_cfg.get("mlp_ratio", 4.0)),
        dropout=float(dit_cfg.get("dropout", 0.1)),
        log_var_clip=tuple(head_cfg["log_var_clip"]),
    )
