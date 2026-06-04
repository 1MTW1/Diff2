"""Climatology bank — day-of-year climatology (IMPLEMENTATION_SPEC v4 §4).

Climatology-anomaly 전환의 결정론적 anchor. **픽셀 공간(00 UTC 스냅샷장)** 에서 동작한다.
`climatology.npz` (c_mu, c_sig: 각 `(366, C, H, W)`; 2단계 raw→smoothing 산출) 와
`pixel_stats.npz` (P_mu, P_sig: 각 `(C, H, W)`; pixelwise·doy무관) 를 buffer 로
보관한다 (학습 파라미터 없음).

두 종류의 표준화 (역할 다름 — §2):
  ① **anomaly 정규화 (local; doy·위치별)**: x̃ = (x_00utc − c_mu[doy]) / c_sig[doy]
     위치·계절별 표준화. VAE·diffusion 의 입력/타깃. 복원에도 같은 통계 사용.
  ② **climatology 정규화 (pixelwise; 위치별·doy무관)**: c̃ = (c_mu[doy] − P_mu) / P_sig
     ★v3/v4: 위치별 상수(P_mu,P_sig)로 표준화. **condition 으로 줄 climatology** 용.
     자기통계(c_sig)로 표준화하면 0 으로 붕괴하므로 금지 — pixelwise(doy무관) 통계로
     변수·위치 스케일은 정규화하되 **계절(doy)에 따른 climatology 변동(절대 맥락)은 보존**.

이 모듈은 **doy 인덱싱 규칙의 단일 정의처**다 (전처리 스크립트·학습·추론이 모두
`doy_slot` 을 import 해서 동일 규칙을 쓴다).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 366-slot 테이블의 기준 윤년. (month, day) → 이 해의 ordinal 로 매핑하면
# 평년/윤년이 동일 (월,일)에 대해 같은 slot 을 받는다. Feb 29 → slot 59.
_LEAP_REF_YEAR = 2000
_N_SLOTS = 366
_FEB29_SLOT = (date(_LEAP_REF_YEAR, 2, 29).toordinal()
               - date(_LEAP_REF_YEAR, 1, 1).toordinal())   # = 59


def doy_slot(times) -> np.ndarray:
    """datetime64(스칼라 또는 배열) → 366-slot day-of-year 인덱스 ∈ [0, 365].

    기준 윤년 2000 의 (month, day) ordinal 로 매핑한다:
        slot = ordinal(2000, m, d) − ordinal(2000, 1, 1)
    → 평년/윤년 모두 같은 (월,일)이면 같은 slot. Feb 29 는 윤년에만 등장(slot 59).
    평년의 Mar 1 은 slot 60 (Feb 29 슬롯을 건너뜀) 으로, 달력 날짜 정합이 유지된다.
    """
    idx = pd.DatetimeIndex(np.asarray(times).reshape(-1))
    base = date(_LEAP_REF_YEAR, 1, 1).toordinal()
    # (month, day) 조합 수가 적어 memo 로 가속.
    memo: dict[tuple[int, int], int] = {}
    out = np.empty(len(idx), dtype=np.int64)
    for i, (m, d) in enumerate(zip(idx.month, idx.day)):
        key = (int(m), int(d))
        slot = memo.get(key)
        if slot is None:
            slot = date(_LEAP_REF_YEAR, key[0], key[1]).toordinal() - base
            memo[key] = slot
        out[i] = slot
    return out


class ClimatologyBank(nn.Module):
    """일평균장의 sliding-window climatology (c_mu, c_sig) 보관 + 표준화/역표준화.

    Args:
        c_mu:        (366, C, H, W) 평균장
        c_sig:       (366, C, H, W) 표준편차장
        sigma_floor: c_sig 하한 (0 division 방어)
    """

    def __init__(
        self,
        c_mu: torch.Tensor | np.ndarray,
        c_sig: torch.Tensor | np.ndarray,
        p_mu: torch.Tensor | np.ndarray | None = None,
        p_sig: torch.Tensor | np.ndarray | None = None,
        sigma_floor: float = 1e-6,
    ):
        super().__init__()
        c_mu = torch.as_tensor(np.asarray(c_mu), dtype=torch.float32)
        c_sig = torch.as_tensor(np.asarray(c_sig), dtype=torch.float32)
        if c_mu.shape != c_sig.shape:
            raise ValueError(
                f"c_mu/c_sig shape mismatch: {tuple(c_mu.shape)} vs {tuple(c_sig.shape)}"
            )
        if c_mu.shape[0] != _N_SLOTS:
            raise ValueError(
                f"climatology must have {_N_SLOTS} doy slots, got {c_mu.shape[0]}"
            )
        self.register_buffer("c_mu", c_mu)
        self.register_buffer("c_sig", c_sig.clamp_min(sigma_floor))

        # pixelwise 통계 (climatology pixel-norm 용; (C,H,W) → (1,C,H,W) broadcast).
        # doy 무관·위치별 상수. anomaly 표준화의 c_mu/c_sig 와 **다른 통계**다(혼동 금지).
        if p_mu is not None and p_sig is not None:
            p_mu = torch.as_tensor(np.asarray(p_mu), dtype=torch.float32)
            p_sig = torch.as_tensor(np.asarray(p_sig), dtype=torch.float32)
            if p_mu.shape != c_mu.shape[1:]:
                raise ValueError(
                    f"pixel_stats shape {tuple(p_mu.shape)} != per-frame "
                    f"climatology shape {tuple(c_mu.shape[1:])}"
                )
            self.register_buffer("p_mu", p_mu.unsqueeze(0))
            self.register_buffer("p_sig", p_sig.clamp_min(sigma_floor).unsqueeze(0))
        else:
            self.p_mu = None
            self.p_sig = None

    # ── 로딩 ────────────────────────────────────────────────────────
    @classmethod
    def from_file(
        cls,
        path: str,
        pixel_stats_path: str | None = None,
        sigma_floor: float = 1e-6,
    ) -> "ClimatologyBank":
        """climatology.npz (+ 선택적 pixel_stats.npz) 로드."""
        d = np.load(path)
        p_mu = p_sig = None
        if pixel_stats_path is not None:
            g = np.load(pixel_stats_path)
            p_mu, p_sig = g["P_mu"], g["P_sig"]
        return cls(
            d["c_mu"], d["c_sig"],
            p_mu=p_mu, p_sig=p_sig, sigma_floor=sigma_floor,
        )

    # ── doy 조회 helper ─────────────────────────────────────────────
    def doy_from_times(self, times) -> torch.Tensor:
        """datetime64 → buffer device 의 LongTensor doy 인덱스 (B,)."""
        slots = doy_slot(times)
        return torch.from_numpy(slots).to(self.c_mu.device)

    def _gather(self, doy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """doy (B,) → (c_mu, c_sig) 각 (B, C, H, W)."""
        doy = doy.to(self.c_mu.device).long()
        return self.c_mu[doy], self.c_sig[doy]

    @staticmethod
    def _as_doy_tensor(doy, batch: int, device) -> torch.Tensor:
        """스칼라/배열/텐서 doy → (batch,) LongTensor 로 정규화."""
        if not torch.is_tensor(doy):
            doy = torch.as_tensor(np.asarray(doy).reshape(-1), dtype=torch.long)
        doy = doy.to(device).long().reshape(-1)
        if doy.numel() == 1 and batch > 1:
            doy = doy.expand(batch)
        return doy

    # ── ① anomaly 정규화 (local, 위치·계절별) ──────────────────────
    def standardize_anomaly(self, x_daily: torch.Tensor, doy) -> torch.Tensor:
        """픽셀 00시 스냅샷장 → 표준화 anomaly.  x̃ = (x − c_mu[doy]) / c_sig[doy]."""
        doy = self._as_doy_tensor(doy, x_daily.shape[0], x_daily.device)
        mu, sig = self._gather(doy)
        return (x_daily - mu) / sig

    def destandardize_anomaly(self, x_tilde: torch.Tensor, doy) -> torch.Tensor:
        """표준화 anomaly → 픽셀 00시 스냅샷장.  x̃ · c_sig[doy] + c_mu[doy]."""
        doy = self._as_doy_tensor(doy, x_tilde.shape[0], x_tilde.device)
        mu, sig = self._gather(doy)
        return x_tilde * sig + mu

    # ── ② climatology 정규화 (pixelwise, condition 용) ─────────────
    def climatology_pixel_norm(self, doy) -> torch.Tensor:
        """condition 으로 줄 climatology.  c̃ = (c_mu[doy] − P_mu) / P_sig.

        반환 (B, C, H, W). 자기통계(c_sig) 가 아니라 **pixelwise(위치별·doy무관)**
        통계로 표준화한다 (자기통계면 0 으로 붕괴; 전역 스칼라보다 위치별이 스케일 정합↑).
        doy 는 (B,) 또는 스칼라.
        """
        if self.p_mu is None or self.p_sig is None:
            raise RuntimeError(
                "pixel_stats 미로드 — ClimatologyBank.from_file 에 "
                "pixel_stats_path 를 넘기거나 config climatology.pixel_stats_path 설정."
            )
        doy = self._as_doy_tensor(doy, 1, self.c_mu.device).reshape(-1)
        c_mu = self.c_mu[doy]                               # (B, C, H, W)
        return (c_mu - self.p_mu) / self.p_sig


def build_climatology(config: dict) -> ClimatologyBank:
    """config["climatology"] dict 로부터 ClimatologyBank 생성 (학습/추론 공통)."""
    cc = config["climatology"]
    return ClimatologyBank.from_file(
        cc["path"],
        pixel_stats_path=cc.get("pixel_stats_path"),
        sigma_floor=float(cc.get("sigma_floor", 1e-6)),
    )
