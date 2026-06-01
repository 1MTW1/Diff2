"""v2 LDM/DiT — test set sub-sample (Mon/Wed/Fri)에 대한 N-member ensemble 생성.

중심-프레임 복원형 Dual-DDPM. DDPM_main 의 x_t 복원을 평가하기 위한 latent 앙상블을
캐시한다 (inference/sampling.py:generate_future_ensemble 와 동일 파이프라인).

파이프라인:
    x_t  ──encoder──▶  cond_past
        │ DDPM_past (마지막 step만 logvar 주입)
        ▼
    z_past(12ch) ──VAE decode──▶ [x̂_{t-1}, x̂_{t+1}]   (멤버별 다양)
        │
    [x̂_{t-1}, x̂_{t+1}]  ──encoder──▶  cond_main
        │ DDPM_main (주입 없음)
        ▼
    z_main(6ch), log_var_main   ← 평가 대상 = 복원된 x̂_t

캐시 schema (sample_{idx:05d}.npz):
    ensemble            (N, 6, 16, 16)   정규화 latent ẑ_0 앙상블 = z_main (x̂_t)
    log_var             (6, 16, 16)      latent dual-head log_var (멤버 평균)
    x_t_true            (6, 16, 16)      GT x_t 의 latent 인코딩 (posterior μ)
    ensemble_pixel      (N, 3, 64, 64)   디코딩된 x̂_t 픽셀 앙상블 (exp5/7/8)
    x_t_true_pixel      (3, 64, 64)      GT x_t 픽셀 필드
    ensemble_pixel_tm1  (N, 3, 64, 64)   DDPM_past 가 생성한 x̂_{t-1} 멤버 앙상블
    ensemble_pixel_tp1  (N, 3, 64, 64)   DDPM_past 가 생성한 x̂_{t+1} 멤버 앙상블
    x_tm1_true_pixel    (3, 64, 64)      GT x_{t-1} 픽셀 필드
    x_tp1_true_pixel    (3, 64, 64)      GT x_{t+1} 픽셀 필드
    time_t              str(timestamp)
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
from inference.sampling import LatentVDMSampler, _decode_pair, _load_models
from models.schedule import VDMSchedule

from .utils import ensure_dir


def _maybe_accelerator():
    """accelerate가 launch한 multi-process 컨텍스트면 Accelerator를, 아니면 None.

    accelerate 미설치/단일 process 일 경우 graceful fallback. 끝 배리어에서
    rank 별 추론 시간 편차로 NCCL 기본 타임아웃(10분)을 넘겨 죽는 일을 막기 위해
    배리어 타임아웃을 4시간으로 늘린다 (training.train.run_infer 와 동일).
    """
    try:
        from datetime import timedelta

        from accelerate import Accelerator
        from accelerate.utils import InitProcessGroupKwargs
    except Exception:
        return None
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=4))
    return Accelerator(kwargs_handlers=[init_kwargs])


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
    x_t: torch.Tensor,
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
    inject_uncertainty_mode: str = "last",
) -> dict:
    """관측 x_t → DDPM_main 의 x̂_t latent 앙상블 + 진단 캐시.

    Args:
        x_t: (1, C, H, W) 관측 중심 프레임.

    Returns:
        dict — 캐시 키 (ensemble, log_var, x_t_true, ensemble_pixel,
        x_t_true_pixel, ensemble_pixel_tm1, ensemble_pixel_tp1) 를 모두
        cpu 텐서로 담는다. GT 이웃(x_{t±1})은 dataset 에서 직접 읽어 run_inference
        에서 저장한다.
    """
    B = n_members
    x_t = x_t.to(device)
    _, C, H, W = x_t.shape
    cz = vae.latent_channels                                       # per-frame = 6
    H_z, W_z = normalizer.mu.shape[-2:]

    # ── [1] DDPM_past: 다양한 이웃 [x̂_{t-1}, x̂_{t+1}] 생성 ─────────
    cond_past = encoder(x_t.unsqueeze(1))                          # (1, 256, D), F=1
    cond_past = cond_past.expand(B, -1, -1)                        # → (B, 256, D)
    z_past, _ = sampler.sample(
        dit_past, cond_past, (B, 2 * cz, H_z, W_z), device,
        inject_uncertainty_mode=inject_uncertainty_mode,
        num_steps=past_num_steps,
    )
    x_tm1_hat, x_tp1_hat = _decode_pair(vae, normalizer, z_past)   # 각 (B, C, H, W)

    # ── [2] DDPM_main: x̂_t 복원 (불확실성 주입 없음) ──────────────
    cond_main = encoder(torch.stack([x_tm1_hat, x_tp1_hat], dim=1))  # (B, 512, D), F=2
    z_main, lv_main = sampler.sample(
        dit_main, cond_main, (B, cz, H_z, W_z), device,
        inject_uncertainty=False,
        num_steps=main_num_steps,
    )
    ensemble_pixel = vae.decode(normalizer.denormalize(z_main))    # (B, C, H, W) = x̂_t

    # log_var 멤버 reduce
    if _LOG_VAR_MEMBER_REDUCE == "mean":
        log_var = lv_main.mean(dim=0)                              # (C_z, H_z, W_z)
    else:
        log_var = lv_main[0]

    # GT x_t 의 latent 인코딩 (posterior μ, 샘플 아님)
    mu_gt, _ = vae.encode(x_t)                                    # (1, C_z, H_z, W_z)
    x_t_true = normalizer.normalize(mu_gt)[0]                      # (C_z, H_z, W_z)

    return {
        "ensemble": z_main.detach().cpu(),
        "log_var": log_var.detach().cpu(),
        "x_t_true": x_t_true.detach().cpu(),
        "ensemble_pixel": ensemble_pixel.detach().cpu(),
        "x_t_true_pixel": x_t[0].detach().cpu(),
        # DDPM_past 가 멤버별로 생성한 이웃 프레임 (mode=last 주입) — exp 신규 분석용.
        "ensemble_pixel_tm1": x_tm1_hat.detach().cpu(),    # (B, C, H, W) = x̂_{t-1}
        "ensemble_pixel_tp1": x_tp1_hat.detach().cpu(),    # (B, C, H, W) = x̂_{t+1}
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

    # 3시점 윈도우(x_{t-1}, x_t, x_{t+1}) 반환 — 여기선 관측 x_t 만 사용.
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
        x_t = sample["x_t"].unsqueeze(0)
        time_t = sample["time_t"]

        cache = _generate_main_ensemble(
            x_t, encoder, dit_past, dit_main, vae, normalizer,
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
            ensemble_pixel_tm1=cache["ensemble_pixel_tm1"].numpy().astype(np.float32),
            ensemble_pixel_tp1=cache["ensemble_pixel_tp1"].numpy().astype(np.float32),
            x_tm1_true_pixel=sample["x_tm1"].numpy().astype(np.float32),
            x_tp1_true_pixel=sample["x_tp1"].numpy().astype(np.float32),
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
    p.add_argument("--inject_uncertainty_mode", default="last",
                   choices=["all", "last", "none"],
                   help="DDPM_past 의 dual-head log_var 노이즈 주입 schedule. "
                        "'last'(마지막 step만, 기본·학습과 일치) / 'all' / 'none'.")
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
