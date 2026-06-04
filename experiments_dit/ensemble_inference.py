"""v4 LDM/DiT — test set 전체 시점에 대한 N-member ensemble 생성.

중심-프레임 복원형 Dual-DDPM (climatology-anomaly 전환). DDPM_main 의 x_t 복원을
평가하기 위한 latent 앙상블을 캐시한다 (inference/sampling.py:generate_future_ensemble
와 동일 파이프라인 — §1 condition 규칙, §9 추론).

파이프라인 (입력 x̃_t = 표준화 anomaly):
    x̃_t (+ 이웃 climatology c̃_{t±1})  ──encoder──▶  cond_past
        │ DDPM_past (매 step logvar 주입, scale 0.1 — SPEC §9.1)
        ▼
    z_past(12ch) ──VAE decode──▶ [x̃_{t-1}, x̃_{t+1}]   (멤버별 다양, anomaly)
        │ (+ 중심 climatology c̃_t)
    [x̃_{t-1}, x̃_{t+1}]  ──encoder──▶  cond_main
        │ DDPM_main (주입 없음)
        ▼
    z_main(6ch), log_var_main   ← 평가 대상 = 복원된 x̂_t
        │ destandardize_anomaly (local, 물리단위 복원)

캐시 schema (sample_{idx:05d}.npz):
    ensemble            (N, 6, 16, 16)   정규화 anomaly latent ẑ_0 앙상블 = z_main
    log_var             (6, 16, 16)      latent dual-head log_var (멤버 평균)
    x_t_true            (6, 16, 16)      GT x̃_t 의 latent 인코딩 (posterior μ)
    ensemble_pixel      (N, 3, 64, 64)   복원된 x̂_t 픽셀 앙상블 (clim on → 물리단위)
    x_t_true_pixel      (3, 64, 64)      GT x_t 픽셀 필드 (clim on → 물리단위)
    ensemble_pixel_tm1  (N, 3, 64, 64)   DDPM_past 가 생성한 x̂_{t-1} 멤버 앙상블
    ensemble_pixel_tp1  (N, 3, 64, 64)   DDPM_past 가 생성한 x̂_{t+1} 멤버 앙상블
    x_tm1_true_pixel    (3, 64, 64)      GT x_{t-1} 픽셀 필드 (clim on → 물리단위)
    x_tp1_true_pixel    (3, 64, 64)      GT x_{t+1} 픽셀 필드 (clim on → 물리단위)
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

from dataset.era5_dataset import ERA5NormalizedDataset, resolve_data_source
from inference.sampling import LatentVDMSampler, _decode_pair, _load_models
from models.climatology import build_climatology
from models.encoder import main_snapshots, past_snapshots
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
    inject_uncertainty_mode: str = "all",
    inject_scale: float = 0.1,
    clim_bank=None,
    doy_t=None,
    doy_tm1=None,
    doy_tp1=None,
) -> dict:
    """관측 x̃_t(표준화 anomaly) → DDPM_main 의 x̂_t latent 앙상블 + 진단 캐시.

    v4 climatology-anomaly 파이프라인 (inference/sampling.py:generate_future_ensemble
    와 동일 규칙, §1/§9). past=중심 anomaly + 이웃 climatology, main=이웃 anomaly +
    중심 climatology. clim_bank 가 있으면 픽셀 캐시는 **물리단위로 역표준화** 된다.

    Args:
        x_t: (1, C, H, W) 관측 중심 프레임의 표준화 anomaly x̃_t.
        clim_bank: ClimatologyBank (None이면 legacy anomaly-only, 역표준화 생략).
        doy_t/tm1/tp1: climatology 조회·역표준화용 doy (스칼라 또는 (1,)).

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

    # ── climatology condition 조회 (§1: 생성 대상 시점). 전부 lookup ──
    if clim_bank is not None:
        c_tm1 = clim_bank.climatology_pixel_norm(doy_tm1)          # (1, C, H, W)
        c_tp1 = clim_bank.climatology_pixel_norm(doy_tp1)
        c_t = clim_bank.climatology_pixel_norm(doy_t)
    else:
        c_tm1 = c_tp1 = c_t = None

    # ── [1] DDPM_past: 다양한 이웃 anomaly [x̃_{t-1}, x̃_{t+1}] 생성 ──
    # past condition (§1) = 중심 anomaly x̃_t + 이웃 climatology c̃_{t±1}(pixel-norm).
    cond_past = encoder(past_snapshots(x_t, c_tm1, c_tp1))         # (1, N_cond, D)
    cond_past = cond_past.expand(B, -1, -1)                        # → (B, N_cond, D)
    z_past, _ = sampler.sample(
        dit_past, cond_past, (B, 2 * cz, H_z, W_z), device,
        inject_uncertainty_mode=inject_uncertainty_mode,          # 'all' (SPEC §9.1)
        inject_scale=inject_scale,                                # 0.1
        num_steps=past_num_steps,
    )
    x_tm1_hat, x_tp1_hat = _decode_pair(vae, normalizer, z_past)   # anomaly 각 (B,C,H,W)

    # ── [2] DDPM_main: x̃_t 복원 (불확실성 주입 없음) ──────────────
    # main condition (§1) = 이웃 anomaly (past 생성물) + 중심 climatology c̃_t.
    c_t_b = c_t.expand(B, -1, -1, -1) if c_t is not None else None
    cond_main = encoder(main_snapshots(x_tm1_hat, x_tp1_hat, c_t_b))
    z_main, lv_main = sampler.sample(
        dit_main, cond_main, (B, cz, H_z, W_z), device,
        inject_uncertainty_mode="none",                           # main은 주입 안 함
        num_steps=main_num_steps,
    )
    x_t_hat = vae.decode(normalizer.denormalize(z_main))          # anomaly (B, C, H, W)

    # log_var 멤버 reduce
    if _LOG_VAR_MEMBER_REDUCE == "mean":
        log_var = lv_main.mean(dim=0)                              # (C_z, H_z, W_z)
    else:
        log_var = lv_main[0]

    # GT x_t 의 latent 인코딩 (posterior μ, 샘플 아님) — anomaly latent.
    mu_gt, _ = vae.encode(x_t)                                    # (1, C_z, H_z, W_z)
    x_t_true = normalizer.normalize(mu_gt)[0]                      # (C_z, H_z, W_z)

    # ── 픽셀 캐시: clim 있으면 anomaly → 물리단위 역표준화 (local, §9.2) ──
    if clim_bank is not None:
        ensemble_pixel = clim_bank.destandardize_anomaly(x_t_hat, doy_t)
        x_t_true_pixel = clim_bank.destandardize_anomaly(x_t, doy_t)[0]
        x_tm1_hat = clim_bank.destandardize_anomaly(x_tm1_hat, doy_tm1)
        x_tp1_hat = clim_bank.destandardize_anomaly(x_tp1_hat, doy_tp1)
    else:
        ensemble_pixel = x_t_hat
        x_t_true_pixel = x_t[0]

    return {
        "ensemble": z_main.detach().cpu(),
        "log_var": log_var.detach().cpu(),
        "x_t_true": x_t_true.detach().cpu(),
        "ensemble_pixel": ensemble_pixel.detach().cpu(),
        "x_t_true_pixel": x_t_true_pixel.detach().cpu(),
        # DDPM_past 가 멤버별로 생성한 이웃 프레임 (clim 있으면 물리단위) — exp 분석용.
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
    sub_sample: bool = False,
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

    # DDPM_past 주입: CLI override 우선, 없으면 config (기본 all × 0.1, SPEC §9.1).
    if inject_uncertainty_mode is None:
        inject_uncertainty_mode = samp_cfg.get("past_inject_mode", "all")
    inject_scale = float(samp_cfg.get("past_inject_scale", 0.1))

    # use_00utc_only(표준화 anomaly) 이면 climatology condition 조회 + 물리단위 복원.
    use_anom = config["data"].get("use_00utc_only", False)
    clim_bank = build_climatology(config).to(device) if use_anom else None

    # 3시점 윈도우(x̃_{t-1}, x̃_t, x̃_{t+1}) 반환 — past cond=x̃_t, 이웃은 GT 평가용.
    src_path, src_var = resolve_data_source(config)
    ds = ERA5NormalizedDataset(
        normalized_path=src_path,
        var_name=src_var,
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
    info(f"[info] {n_members}-member, DDPM_past inject="
         f"{inject_uncertainty_mode!r}×{inject_scale}, "
         f"clim={'on' if clim_bank is not None else 'off'}, latent VDM sampler "
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

        # 역표준화·climatology 조회용 doy (각 프레임의 실제 시각 기준).
        if clim_bank is not None:
            doy_t = clim_bank.doy_from_times(sample["time_t"])
            doy_tm1 = clim_bank.doy_from_times(sample["time_tm1"])
            doy_tp1 = clim_bank.doy_from_times(sample["time_tp1"])
        else:
            doy_t = doy_tm1 = doy_tp1 = None

        cache = _generate_main_ensemble(
            x_t, encoder, dit_past, dit_main, vae, normalizer,
            sampler, n_members=n_members, device=device,
            past_num_steps=past_num_steps, main_num_steps=main_num_steps,
            inject_uncertainty_mode=inject_uncertainty_mode,
            inject_scale=inject_scale,
            clim_bank=clim_bank, doy_t=doy_t, doy_tm1=doy_tm1, doy_tp1=doy_tp1,
        )

        # GT 이웃은 dataset 의 anomaly x̃ → ensemble_pixel 과 같은 공간(물리단위)으로 저장.
        if clim_bank is not None:
            x_tm1_true_pixel = clim_bank.destandardize_anomaly(
                sample["x_tm1"].unsqueeze(0).to(device), doy_tm1)[0].cpu()
            x_tp1_true_pixel = clim_bank.destandardize_anomaly(
                sample["x_tp1"].unsqueeze(0).to(device), doy_tp1)[0].cpu()
        else:
            x_tm1_true_pixel = sample["x_tm1"]
            x_tp1_true_pixel = sample["x_tp1"]

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
            x_tm1_true_pixel=x_tm1_true_pixel.numpy().astype(np.float32),
            x_tp1_true_pixel=x_tp1_true_pixel.numpy().astype(np.float32),
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
    p.add_argument("--subsample_mwf", action="store_true",
                   help="설정 시 월/수/금 시점만 사용 (기본: 전체 test 시점).")
    p.add_argument("--limit", type=int, default=None,
                   help="디버깅용: 처음 N개 시점만 생성")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inject_uncertainty_mode", default=None,
                   choices=["all", "last", "none"],
                   help="DDPM_past 의 dual-head log_var 노이즈 주입 schedule. "
                        "미지정 시 config sampling.past_inject_mode (기본 'all', SPEC §9.1).")
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
        sub_sample=args.subsample_mwf,
        limit=args.limit,
        seed=args.seed,
        inject_uncertainty_mode=args.inject_uncertainty_mode,
    )


if __name__ == "__main__":
    main()
