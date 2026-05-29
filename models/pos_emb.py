"""공유 positional encoding 유틸.

2D sinusoidal positional encoding — learnable pos_emb를 대체한다 (instruction_v2 §0).
고정 파라미터가 없어 해상도 변경(예: 0.1° 확장)에 일반화가 유리하다.
condition encoder(§2.3)와 DiT(§2.4) 양쪽이 같은 함수를 사용한다.
"""
from __future__ import annotations

import torch


def sinusoidal_2d_pos_emb(
    height: int,
    width: int,
    dim: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """(height·width, dim) 2D sinusoidal positional embedding.

    dim을 4등분하여 [sin(y), cos(y), sin(x), cos(x)]로 채운다.
    토큰 순서는 row-major (y가 바깥 루프) — patchify의 flatten 규약과 일치.

    Args:
        height, width: 토큰 격자 크기 (예: 16, 16)
        dim: 임베딩 차원 (4의 배수여야 함)
    Returns:
        (height*width, dim) 텐서
    """
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4, got {dim}")
    quarter = dim // 4

    omega = torch.arange(quarter, device=device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / quarter))          # (quarter,)

    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")  # (H, W)
    grid_y = grid_y.reshape(-1)                           # (H*W,) row-major
    grid_x = grid_x.reshape(-1)

    arg_y = grid_y[:, None] * omega[None, :]              # (H*W, quarter)
    arg_x = grid_x[:, None] * omega[None, :]
    emb = torch.cat(
        [torch.sin(arg_y), torch.cos(arg_y),
         torch.sin(arg_x), torch.cos(arg_x)],
        dim=1,
    )                                                     # (H*W, dim)
    return emb.to(dtype)
