"""Test set sub-sample (Mon/Wed/Fri)에 대한 30-member ensemble 생성 + 캐시.

학습된 model_past, model_main을 두 개의 분리된 DDPMSampler로 호출:
- model_past: 50-step DDPM (학습 시 aux_sampler와 동일)
- model_main: 200-step full DDPM (품질)
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
    """accelerate가 launch한 multi-process 컨텍스트면 Accelerator를, 아니면 None.

    accelerate 미설치/단일 process 일 경우 graceful fallback.
    """
    try:
        from accelerate import Accelerator
    except Exception:
        return None
    return Accelerator()


# experiment.md §1.4: 캐시는 (C=3, H=64, W=64) 단일 ℓ_m. 멤버별로 차이가 있을 수
# 있으나 (cond_main이 멤버별 past sample에 의존) ℓ_m은 spec상 single map.
# → 멤버 평균으로 대표값 저장.
_LOG_VAR_MEMBER_REDUCE = "mean"


def _select_mon_wed_fri_indices(times: np.ndarray) -> np.ndarray:
    """월(0)/수(2)/금(4) timestamp에 해당하는 인덱스만 반환."""
    dow = pd.DatetimeIndex(times).dayofweek.to_numpy()
    keep = np.where((dow == 0) | (dow == 2) | (dow == 4))[0]
    return keep


@torch.no_grad()
def _generate_two_stage(
    x_tm1: torch.Tensor,
    x_t: torch.Tensor,
    encoder,
    model_past,
    model_main,
    sampler_past: DDPMSampler,
    sampler_main: DDPMSampler,
    n_members: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """단일 시점 (1, C, H, W) 입력 → (N, C, H, W) ensemble + (C, H, W) ℓ_m.

    Returns:
        ensemble: (N, C, H, W) main DDPM x_t 부분 (정규화 공간)
        log_var_xt: (C, H, W) 멤버 평균 ℓ_m (x_t 채널부)
    """
    _, C, H, W = x_tm1.shape
    B = n_members

    cond_past_1 = encoder(torch.stack([x_tm1, x_t], dim=1))   # (1, C', H, W)
    cond_past_B = cond_past_1.expand(B, -1, -1, -1)

    past_sample, _, _ = sampler_past.sample(
        model_past, cond=cond_past_B,
        shape=(B, 2 * C, H, W), device=device,
    )
    past_sample = past_sample.reshape(B, 2, C, H, W)
    x_tm3_hat = past_sample[:, 0]
    x_tm2_hat = past_sample[:, 1]

    x_tm1_B = x_tm1.expand(B, -1, -1, -1)
    cond_main = encoder(
        torch.stack([x_tm3_hat, x_tm2_hat, x_tm1_B], dim=1)
    )

    main_sample, _, ell_main = sampler_main.sample(
        model_main, cond=cond_main,
        shape=(B, 2 * C, H, W), device=device,
    )
    main_sample = main_sample.reshape(B, 2, C, H, W)
    ensemble = main_sample[:, 0]                              # (B, C, H, W)

    # ell_main: (B, 2*C, H, W) → x_t 채널만 (앞 C) → 멤버 reduce
    ell_xt = ell_main[:, :C]
    if _LOG_VAR_MEMBER_REDUCE == "mean":
        log_var_xt = ell_xt.mean(dim=0)                       # (C, H, W)
    else:
        log_var_xt = ell_xt[0]
    return ensemble.cpu(), log_var_xt.cpu()


def run_inference(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    n_members: int,
    past_steps: int,
    main_steps: int,
    sub_sample: bool = True,
    limit: int | None = None,
    seed: int = 42,
    main_use_learned_var: bool = False,
) -> None:
    accelerator = _maybe_accelerator()
    rank = accelerator.process_index if accelerator is not None else 0
    world = accelerator.num_processes if accelerator is not None else 1
    is_main = accelerator.is_main_process if accelerator is not None else True

    def info(msg: str) -> None:
        if is_main:
            print(msg)

    # 출력 디렉토리는 main process만 생성, 이후 동기화
    if is_main:
        ensure_dir(output_dir)
    if accelerator is not None:
        accelerator.wait_for_everyone()
    out_dir = Path(output_dir)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if accelerator is not None:
        device = accelerator.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in ckpt:
        ckpt["config"] = config
    encoder, model_past, model_main = _build_models_from_ckpt(ckpt, device)

    diff_cfg = config["diffusion"]
    schedule = LinearNoiseSchedule(
        M=diff_cfg["M"],
        beta_start=float(diff_cfg["beta_start"]),
        beta_end=float(diff_cfg["beta_end"]),
        device=device,
    )
    # Past sampler: 학습 시 aux_sampler와 동일하게 learned ℓ 포함 (main이 이
    # 분포로 학습되었으므로 일관성 유지).
    sampler_past = DDPMSampler(
        schedule, num_inference_steps=past_steps,
        use_learned_variance=True,
    )
    # Main sampler: 기본은 learned ℓ 없이 scheduler baseline noise만. 다운스트림
    # 모델이 main의 reverse 분포를 학습한 적 없으므로 calibration에 유리.
    sampler_main = DDPMSampler(
        schedule, num_inference_steps=main_steps,
        use_learned_variance=main_use_learned_var,
    )
    info(f"[info] main use_learned_variance={main_use_learned_var}")

    ds = ERA5NormalizedDataset(
        normalized_path=config["data"]["normalized_path"],
        mode="inference",
        split="test",
        load_into_memory=False,
    )

    if sub_sample:
        abs_times = ds.times
        keep_abs = _select_mon_wed_fri_indices(abs_times)
        keep_rel = keep_abs[keep_abs >= ds.valid_start] - ds.valid_start
    else:
        keep_rel = np.arange(len(ds))

    if limit is not None:
        keep_rel = keep_rel[:limit]

    # ── Timestep sharding: rank r은 keep_rel[r::world] 담당 ────────
    # save_idx는 전역 위치(0..len(keep_rel)-1)와 1:1 대응시켜 모든 process의
    # 출력 파일 union이 단일 process 결과와 동일하도록 함.
    all_positions = np.arange(len(keep_rel))
    my_positions = all_positions[rank::world]
    my_indices = keep_rel[rank::world]

    info(f"[info] checkpoint={checkpoint_path}")
    info(f"[info] {n_members}-member, "
         f"past={past_steps}step, main={main_steps}step")
    info(f"[info] {len(keep_rel)} timesteps total, "
         f"world={world} → {len(my_positions)} per process (rank0)")

    # 각 process마다 다른 seed → cross-process 중복 noise 방지.
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
        time_t = sample["time_t"]

        ensemble, log_var_xt = _generate_two_stage(
            x_tm1, x_t, encoder, model_past, model_main,
            sampler_past, sampler_main, n_members=n_members, device=device,
        )

        out_path = out_dir / f"sample_{int(save_idx):05d}.npz"
        np.savez(
            out_path,
            ensemble=ensemble.numpy().astype(np.float32),
            log_var=log_var_xt.numpy().astype(np.float32),
            x_t_true=sample["x_t"].numpy().astype(np.float32),
            time_t=np.array(str(time_t)),
        )

    if accelerator is not None:
        accelerator.wait_for_everyone()
    info(f"[done] saved {len(keep_rel)} files → {out_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/ensembles")
    p.add_argument("--n_members", type=int, default=30)
    p.add_argument("--past_steps", type=int, default=50)
    p.add_argument("--main_steps", type=int, default=200)
    p.add_argument("--no_subsample", action="store_true",
                   help="설정 시 모든 test 시점 사용")
    p.add_argument("--limit", type=int, default=None,
                   help="디버깅용: 처음 N개 시점만 생성")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--main_use_learned_var", action="store_true",
                   help="설정 시 main DDPM의 reverse step에서도 학습된 ℓ를 "
                        "noise로 사용 (기본: scheduler baseline noise만 사용)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        n_members=args.n_members,
        past_steps=args.past_steps,
        main_steps=args.main_steps,
        sub_sample=not args.no_subsample,
        limit=args.limit,
        seed=args.seed,
        main_use_learned_var=args.main_use_learned_var,
    )


if __name__ == "__main__":
    main()
