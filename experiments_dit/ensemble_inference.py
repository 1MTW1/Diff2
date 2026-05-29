"""v2 LDM/DiT — test set sub-sample (Mon/Wed/Fri)에 대한 N-member ensemble 생성.

experiments/ensemble_inference.py (v1 픽셀 공간 dual-DDPM) 의 LDM 변환판.
DDPM_main 의 x_t 예측을 평가하기 위한 latent 앙상블을 캐시한다.

파이프라인 (inference/sampling.py:generate_future_ensemble 와 동일):
    (x_{t-1}, x_t)  ──encoder──▶  cond_past
        │ DDPM_past (inject_uncertainty=True)
        ▼
    z_past ──VAE decode──▶ x̂_{t-2}
        │
    (x̂_{t-2}, x_{t-1})  ──encoder──▶  cond_main
        │ DDPM_main (inject_uncertainty=False)
        ▼
    z_main, log_var_main   ← 평가 대상

캐시 schema (sample_{idx:05d}.npz):
    ensemble        (N, 12, 16, 16)  정규화 latent ẑ_0 앙상블 = z_main
    log_var         (12, 16, 16)     latent dual-head log_var (멤버 평균)
    x_t_true        (12, 16, 16)     GT [x_t, x_{t+1}] 블록의 latent 인코딩 (posterior μ)
    ensemble_pixel  (N, 3, 64, 64)   디코딩된 x_t 픽셀 앙상블 (exp5 전용)
    x_t_true_pixel  (3, 64, 64)      GT x_t 픽셀 필드 (exp5 전용)
    time_t          str(timestamp)
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
from inference.sampling import LatentVDMSampler, _load_models
from models.schedule import VDMSchedule

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


# latent dual-head log_var는 멤버별로 다를 수 있으나 (cond_main이 멤버별 past
# sample에 의존) spec상 single map → 멤버 평균으로 대표값 저장.
_LOG_VAR_MEMBER_REDUCE = "mean"


def _select_mon_wed_fri_indices(times: np.ndarray) -> np.ndarray:
    """월(0)/수(2)/금(4) timestamp에 해당하는 인덱스만 반환."""
    dow = pd.DatetimeIndex(times).dayofweek.to_numpy()
    keep = np.where((dow == 0) | (dow == 2) | (dow == 4))[0]
    return keep


@torch.no_grad()
def _generate_main_ensemble(
    x_tm1: torch.Tensor,
    x_t: torch.Tensor,
    x_tp1: torch.Tensor,
    encoder,
    dit_past,
    dit_main,
    vae,
    normalizer,
    sampler: LatentVDMSampler,
    n_members: int,
    device: torch.device,
    past_num_steps: int | None = None,
    main_num_steps: int | None = None,
    inject_uncertainty_mode: str = "all",
) -> dict:
    """단일 시점 입력 → DDPM_main 의 x_t latent 앙상블 + 진단 캐시.

    Args:
        x_tm1, x_t, x_tp1: (1, C, H, W) 관측 3시점 (x_tp1은 GT latent 인코딩용).

    Returns:
        dict — 캐시 키 (ensemble, log_var, x_t_true, ensemble_pixel,
        x_t_true_pixel) 를 모두 cpu 텐서로 담는다.
    """
    B = n_members
    x_tm1 = x_tm1.to(device)
    x_t = x_t.to(device)
    x_tp1 = x_tp1.to(device)
    _, C, H, W = x_tm1.shape
    C_z = vae.latent_channels
    H_z, W_z = normalizer.mu.shape[-2:]

    # ── [1] DDPM_past: 다양한 과거 생성 (불확실성 주입) ────────────
    cond_past = encoder(torch.stack([x_tm1, x_t], dim=1))          # (1, N_tok, D)
    cond_past = cond_past.expand(B, -1, -1)                         # → (B, N_tok, D)
    z_past, _ = sampler.sample(
        dit_past, cond_past, (B, C_z, H_z, W_z), device,
        inject_uncertainty_mode=inject_uncertainty_mode,
        num_steps=past_num_steps,
    )
    # latent → 기상장: denormalize → decode
    x_past = vae.decode(normalizer.denormalize(z_past))            # (B, 2C, H, W)
    x_past = x_past.reshape(B, 2, C, H, W)
    x_tm2_hat = x_past[:, 1]                                       # x̂_{t-2}

    # ── [2] DDPM_main: future 생성 (불확실성 주입 없음) ────────────
    x_tm1_B = x_tm1.expand(B, -1, -1, -1)
    cond_main = encoder(torch.stack([x_tm2_hat, x_tm1_B], dim=1))
    z_main, lv_main = sampler.sample(
        dit_main, cond_main, (B, C_z, H_z, W_z), device,
        inject_uncertainty=False,
        num_steps=main_num_steps,
    )
    # main 픽셀 앙상블: decode → frame 0 (= x_t)
    x_main = vae.decode(normalizer.denormalize(z_main))            # (B, 2C, H, W)
    x_main = x_main.reshape(B, 2, C, H, W)
    ensemble_pixel = x_main[:, 0]                                  # (B, C, H, W)

    # log_var 멤버 reduce
    if _LOG_VAR_MEMBER_REDUCE == "mean":
        log_var = lv_main.mean(dim=0)                              # (C_z, H_z, W_z)
    else:
        log_var = lv_main[0]

    # GT [x_t, x_{t+1}] 블록의 latent 인코딩 (posterior μ, 샘플 아님)
    gt_pair = torch.cat([x_t, x_tp1], dim=1)                       # (1, 2C, H, W)
    mu_gt, _ = vae.encode(gt_pair)                                 # (1, C_z, H_z, W_z)
    x_t_true = normalizer.normalize(mu_gt)[0]                      # (C_z, H_z, W_z)

    return {
        "ensemble": z_main.detach().cpu(),
        "log_var": log_var.detach().cpu(),
        "x_t_true": x_t_true.detach().cpu(),
        "ensemble_pixel": ensemble_pixel.detach().cpu(),
        "x_t_true_pixel": x_t[0].detach().cpu(),
    }


def run_inference(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    n_members: int,
    past_num_steps: int | None = None,
    main_num_steps: int | None = None,
    sub_sample: bool = True,
    limit: int | None = None,
    seed: int = 42,
    inject_uncertainty_mode: str = "all",
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
    if "config" in ckpt:
        config = ckpt["config"]

    encoder, dit_past, dit_main, vae, normalizer = _load_models(
        config, ckpt, device,
    )

    schedule = VDMSchedule(
        gamma_min=float(config["schedule"]["gamma_min"]),
        gamma_max=float(config["schedule"]["gamma_max"]),
    )
    # config는 체크포인트에서 로드되므로 past_num_steps/main_num_steps 키가
    # 없을 수 있다 → 기존 num_steps(추론 step)로 안전하게 fallback.
    samp_cfg = config["sampling"]
    _default_steps = int(samp_cfg.get("num_steps", 50))
    if past_num_steps is None:
        past_num_steps = int(samp_cfg.get("past_num_steps", _default_steps))
    if main_num_steps is None:
        main_num_steps = int(samp_cfg.get("main_num_steps", _default_steps))
    sampler = LatentVDMSampler(schedule, num_steps=past_num_steps)

    # mode='train': x_tp1 가용 → GT [x_t, x_{t+1}] 블록 latent 인코딩에 사용.
    ds = ERA5NormalizedDataset(
        normalized_path=config["data"]["normalized_path"],
        mode="train",
        split="test",
        load_into_memory=False,
    )

    if sub_sample:
        abs_times = ds.times
        keep_abs = _select_mon_wed_fri_indices(abs_times)
        keep_abs = keep_abs[
            (keep_abs >= ds.valid_start) & (keep_abs < ds.valid_end)
        ]
        keep_rel = keep_abs - ds.valid_start
    else:
        keep_rel = np.arange(len(ds))

    if limit is not None:
        keep_rel = keep_rel[:limit]

    # ── Timestep sharding: rank r은 keep_rel[r::world] 담당 ────────
    all_positions = np.arange(len(keep_rel))
    my_positions = all_positions[rank::world]
    my_indices = keep_rel[rank::world]

    info(f"[info] checkpoint={checkpoint_path}")
    info(f"[info] {n_members}-member, DDPM_past inject_mode="
         f"{inject_uncertainty_mode!r}, latent VDM sampler "
         f"past_steps={past_num_steps} main_steps={main_num_steps}")
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
        x_tm1 = sample["x_tm1"].unsqueeze(0)
        x_t = sample["x_t"].unsqueeze(0)
        x_tp1 = sample["x_tp1"].unsqueeze(0)
        time_t = sample["time_t"]

        cache = _generate_main_ensemble(
            x_tm1, x_t, x_tp1, encoder, dit_past, dit_main, vae, normalizer,
            sampler, n_members=n_members, device=device,
            past_num_steps=past_num_steps, main_num_steps=main_num_steps,
            inject_uncertainty_mode=inject_uncertainty_mode,
        )

        out_path = out_dir / f"sample_{int(save_idx):05d}.npz"
        np.savez(
            out_path,
            ensemble=cache["ensemble"].numpy().astype(np.float32),
            log_var=cache["log_var"].numpy().astype(np.float32),
            x_t_true=cache["x_t_true"].numpy().astype(np.float32),
            ensemble_pixel=cache["ensemble_pixel"].numpy().astype(np.float32),
            x_t_true_pixel=cache["x_t_true_pixel"].numpy().astype(np.float32),
            time_t=np.array(str(time_t)),
        )

    if accelerator is not None:
        accelerator.wait_for_everyone()
    info(f"[done] saved {len(keep_rel)} files → {out_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="diffusion 체크포인트 (encoder/dit_past/dit_main)")
    p.add_argument("--output_dir", type=str, default="outputs/ensembles_dit")
    p.add_argument("--n_members", type=int, default=30)
    p.add_argument("--past_num_steps", type=int, default=None,
                   help="DDPM_past denoising step 수 "
                        "(미지정 시 config['sampling']['past_num_steps']).")
    p.add_argument("--main_num_steps", type=int, default=None,
                   help="DDPM_main denoising step 수 "
                        "(미지정 시 config['sampling']['main_num_steps']).")
    p.add_argument("--no_subsample", action="store_true",
                   help="설정 시 모든 test 시점 사용")
    p.add_argument("--limit", type=int, default=None,
                   help="디버깅용: 처음 N개 시점만 생성")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inject_uncertainty_mode", default="all",
                   choices=["all", "last", "none"],
                   help="DDPM_past 의 dual-head log_var 노이즈 주입 schedule. "
                        "'all'(매 step, 기본) / 'last'(마지막 step만) / 'none'.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        n_members=args.n_members,
        past_num_steps=args.past_num_steps,
        main_num_steps=args.main_num_steps,
        sub_sample=not args.no_subsample,
        limit=args.limit,
        seed=args.seed,
        inject_uncertainty_mode=args.inject_uncertainty_mode,
    )


if __name__ == "__main__":
    main()
