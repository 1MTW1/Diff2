"""x_{t-2} 예측에 대한 30-member ensemble 생성 + 캐시.

파이프라인:
    (x_{t-1}, x_t)
       │ encoder
       ▼
    cond_past
       │ DDPMSampler (50 step, learned variance ON — 학습 분포와 일치)
       ▼
    (x̂_{t-3}, x̂_{t-2}), ℓ_past

평가 대상: x_{t-2} → past_sample[:, 1] (C channels).
ℓ_past 의 x_{t-2} 부분 = log_var[:, C:2C].
GT  : sample["x_tm2"] (mode='train' 사용).

캐시 schema는 experiments/ 와 동일하여 exp1-6 모듈을 그대로 재사용.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from dataset.era5_dataset import ERA5NormalizedDataset
from inference.ensemble import _build_models_from_ckpt
from inference.sampler import DDPMSampler
from training.schedule import LinearNoiseSchedule

from .utils import ensure_dir


def _maybe_accelerator():
    try:
        from accelerate import Accelerator
    except Exception:
        return None
    return Accelerator()


_LOG_VAR_MEMBER_REDUCE = "mean"
_TM2_SHIFT = np.timedelta64(12, "h")        # 2 × 6h


def _select_mon_wed_fri_indices(times: np.ndarray) -> np.ndarray:
    """월(0)/수(2)/금(4) 인덱스."""
    dow = pd.DatetimeIndex(times).dayofweek.to_numpy()
    return np.where((dow == 0) | (dow == 2) | (dow == 4))[0]


@torch.no_grad()
def _generate_past_ensemble(
    x_tm1: torch.Tensor,
    x_t: torch.Tensor,
    encoder,
    model_past,
    sampler_past: DDPMSampler,
    n_members: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(1, C, H, W) 입력 → x_{t-2} ensemble (N, C, H, W) + ℓ_past[t-2] (C, H, W)."""
    _, C, H, W = x_tm1.shape
    B = n_members

    cond_past_1 = encoder(torch.stack([x_tm1, x_t], dim=1))    # (1, C', H, W)
    cond_past_B = cond_past_1.expand(B, -1, -1, -1)

    past_sample, _, ell_past = sampler_past.sample(
        model_past, cond=cond_past_B,
        shape=(B, 2 * C, H, W), device=device,
    )
    past_sample = past_sample.reshape(B, 2, C, H, W)
    x_tm2_hat = past_sample[:, 1]                              # (B, C, H, W)

    # ell_past: (B, 2*C, H, W) → x_{t-2} 채널은 뒤쪽 C개
    ell_tm2 = ell_past[:, C:2 * C]
    if _LOG_VAR_MEMBER_REDUCE == "mean":
        log_var_tm2 = ell_tm2.mean(dim=0)                      # (C, H, W)
    else:
        log_var_tm2 = ell_tm2[0]
    return x_tm2_hat.cpu(), log_var_tm2.cpu()


def run_inference(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    n_members: int,
    past_steps: int = 50,
    sub_sample: bool = True,
    limit: int | None = None,
    seed: int = 42,
    use_learned_variance: bool = True,
) -> None:
    accelerator = _maybe_accelerator()
    rank = accelerator.process_index if accelerator is not None else 0
    world = accelerator.num_processes if accelerator is not None else 1
    is_main = accelerator.is_main_process if accelerator is not None else True

    def info(msg: str) -> None:
        if is_main:
            print(msg)

    if is_main:
        ensure_dir(output_dir)
    if accelerator is not None:
        accelerator.wait_for_everyone()
    out_dir = Path(output_dir)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = (accelerator.device if accelerator is not None
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in ckpt:
        ckpt["config"] = config
    encoder, model_past, _model_main = _build_models_from_ckpt(ckpt, device)

    diff_cfg = config["diffusion"]
    schedule = LinearNoiseSchedule(
        M=diff_cfg["M"],
        beta_start=float(diff_cfg["beta_start"]),
        beta_end=float(diff_cfg["beta_end"]),
        device=device,
    )
    # 학습 aux_sampler 와 동일 — past 는 learned ℓ 분포로 학습됨.
    sampler_past = DDPMSampler(
        schedule, num_inference_steps=past_steps,
        use_learned_variance=use_learned_variance,
    )
    info(f"[info] sampler_past: {past_steps}-step DDPM, "
         f"use_learned_variance={use_learned_variance}")

    # mode='train': x_tm3..x_tp1 모두 가용. 우리는 x_tm1, x_t (입력), x_tm2 (GT) 만 사용.
    ds = ERA5NormalizedDataset(
        normalized_path=config["data"]["normalized_path"],
        mode="train",
        split="test",
        load_into_memory=False,
    )

    if sub_sample:
        abs_times = ds.times
        keep_abs = _select_mon_wed_fri_indices(abs_times)
        # train mode 의 valid 범위 [valid_start, valid_end) 에 맞춤
        keep_abs = keep_abs[
            (keep_abs >= ds.valid_start) & (keep_abs < ds.valid_end)
        ]
        keep_rel = keep_abs - ds.valid_start
    else:
        keep_rel = np.arange(len(ds))

    if limit is not None:
        keep_rel = keep_rel[:limit]

    all_positions = np.arange(len(keep_rel))
    my_positions = all_positions[rank::world]
    my_indices = keep_rel[rank::world]

    info(f"[info] checkpoint={checkpoint_path}")
    info(f"[info] {n_members}-member past ensemble")
    info(f"[info] {len(keep_rel)} timesteps total, "
         f"world={world} → {len(my_positions)} per process (rank0)")

    torch.manual_seed(seed + rank)

    pbar = tqdm(
        zip(my_positions, my_indices),
        total=len(my_positions),
        desc=f"ensemble[rank{rank}]",
        disable=not is_main,
    )
    for save_idx, idx in pbar:
        sample = ds[int(idx)]
        x_tm1 = sample["x_tm1"].unsqueeze(0).to(device)
        x_t = sample["x_t"].unsqueeze(0).to(device)
        x_tm2_gt = sample["x_tm2"]
        time_t = sample["time_t"]                              # anchor t
        time_tm2 = np.datetime64(time_t) - _TM2_SHIFT          # t-2 timestamp

        ensemble, log_var_tm2 = _generate_past_ensemble(
            x_tm1, x_t, encoder, model_past, sampler_past,
            n_members=n_members, device=device,
        )

        out_path = out_dir / f"sample_{int(save_idx):05d}.npz"
        np.savez(
            out_path,
            ensemble=ensemble.numpy().astype(np.float32),
            log_var=log_var_tm2.numpy().astype(np.float32),
            x_t_true=x_tm2_gt.numpy().astype(np.float32),
            time_t=np.array(str(time_tm2)),
        )

    if accelerator is not None:
        accelerator.wait_for_everyone()
    info(f"[done] saved {len(keep_rel)} files → {out_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/ensembles_tm2")
    p.add_argument("--n_members", type=int, default=30)
    p.add_argument("--past_steps", type=int, default=50)
    p.add_argument("--no_subsample", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_learned_variance", action="store_true",
                   help="설정 시 past reverse chain에서도 learned ℓ noise OFF.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        n_members=args.n_members,
        past_steps=args.past_steps,
        sub_sample=not args.no_subsample,
        limit=args.limit,
        seed=args.seed,
        use_learned_variance=not args.no_learned_variance,
    )


if __name__ == "__main__":
    main()
