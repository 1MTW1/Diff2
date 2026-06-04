"""Diffusion² LDM 학습 — 4-phase 파이프라인 (past → infer → main → joint).

중심-프레임 복원형 Dual-DDPM:
  DDPM_past : cond = x_t,                 target = [x_{t-1}, x_{t+1}]  (latent 12ch)
  DDPM_main : cond = [x̂_{t-1}, x̂_{t+1}],  target = x_t                (latent 6ch)
VAE는 프레임별 단일 autoencoder(frozen). past 타깃 latent은 두 프레임을 각각
인코딩·정규화 후 채널 concat(12ch), main 타깃은 x_t 한 프레임(6ch). **validation 없음.**

  phase=past : Condition Encoder + DiT_past(12ch) 만 heteroscedastic NLL로 학습.
               (VAE는 Stage 0에서 frozen) → `<out_dir>/past.pt`.

  phase=infer: past.pt 로 **train split 전체**에 대해 sample당 ensemble_size(=M) 개의
               z_past(12ch) 를 샘플링하여 fp16 memmap `(N, M, 12, H_z, W_z)` 에 저장.
               불확실성 주입은 **마지막 denoising step에서만**(inject_mode='last')
               logvar 비례 노이즈를 준다. **train data 전용.** multi-GPU 시 각 프로세스가
               train split을 연속 구간으로 나눠 같은 memmap에 기록한다(row 순서 보존).

  phase=main : DiT_main(6ch) + Encoder 를 학습. epoch e 는 미리 만든 member `e % M` 의
               z_past 를 decode → [x̂_{t-1}, x̂_{t+1}] → cond_main 으로 사용(각 epoch마다
               다른 과거). main 은 노이즈 주입을 받지 않으며 dual-head NLL을 유지한다.
               → `<out_dir>/main.pt` (encoder/dit_past/dit_main 번들).

  phase=joint: DiT_past + Encoder + DiT_main 을 end-to-end로 학습. **매 batch마다 DiT_past가
               동적으로 생성한** cond(mode='last')을 main에 쓴다. L = L_past + L_main.
               → `<out_dir>/checkpoint_final.pt` (inference 호환 번들).

선행 단계(반드시 먼저): Stage 0 VAE 학습 + latent 통계 계산.

실행 예:
    # 1) past 학습 (multi-GPU)
    accelerate launch --config_file config/accelerate_config.yaml -m training.train --phase past --output_dir outputs/middle --past_epochs 50
    # 2) 조건 생성 (multi-GPU). train split을 프로세스 수로 나눠 병렬 샘플링.
    accelerate launch --config_file config/accelerate_config.yaml \\
        -m training.train --phase infer --output_dir outputs/decoupled_1 \\
        --ensemble_size 100 --inject_mode last --batch_size 20
    # 3) main 학습 (multi-GPU)
    accelerate launch --config_file config/accelerate_config.yaml -m training.train --phase main --output_dir outputs/middle --main_epochs 100 --ensemble_size 100
    # 4) end-to-end joint 학습 (multi-GPU)
    accelerate launch --config_file config/accelerate_config.yaml -m training.train --phase joint --output_dir outputs/middle --joint_epochs 50 --inject_mode last
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, InitProcessGroupKwargs
from datetime import timedelta
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset.era5_dataset import (
    ERA5NormalizedDataset, collate_with_time, resolve_data_source,
)
from inference.sampling import LatentVDMSampler
from models.climatology import build_climatology, doy_slot
from models.dit import build_dit
from models.encoder import build_encoder, main_snapshots, past_snapshots
from models.latent_norm import LatentNormalizer
from models.schedule import VDMSchedule
from models.vae import build_vae

from .loss import heteroscedastic_nll_loss


# ─── 공통 helper ────────────────────────────────────────────────────
def _load_frozen_vae(
    config: dict, device: torch.device,
) -> tuple[torch.nn.Module, LatentNormalizer]:
    """Stage 0에서 학습된 VAE + latent 정규화 통계 로드 (둘 다 frozen)."""
    vae_ckpt = torch.load(config["vae"]["checkpoint"], map_location="cpu")
    vae = build_vae(vae_ckpt.get("config", config)["vae"]).to(device)
    vae.load_state_dict(vae_ckpt["vae"])
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    normalizer = LatentNormalizer.from_file(
        config["latent_norm"]["stats_path"], map_location=device,
    ).to(device)
    return vae, normalizer


def _build_schedule(config: dict) -> VDMSchedule:
    sc = config["schedule"]
    return VDMSchedule(
        gamma_min=float(sc["gamma_min"]), gamma_max=float(sc["gamma_max"]),
    )


def _load_climatology(config: dict, device: torch.device):
    """use_00utc_only 이면 ClimatologyBank(frozen) 로드 — climatology condition 용 (§1).

    아니면 None (legacy anomaly-only 경로). 학습 파라미터 없음.
    """
    if not config["data"].get("use_00utc_only", False):
        return None
    return build_climatology(config).to(device)


def _clim_cond(clim, doy):
    """climatology pixel-norm condition (없으면 None)."""
    return None if clim is None else clim.climatology_pixel_norm(doy)


def _resolve_past_inject(
    config: dict, args: argparse.Namespace,
) -> tuple[str, float]:
    """DDPM_past 샘플링 주입 (mode, scale) 결정 — train↔inference OOD 정합.

    CLI `--inject_mode` 가 명시되면 우선, 아니면 config sampling.past_inject_mode.
    scale 은 config sampling.past_inject_scale (기본 0.1, SPEC §7.1).
    """
    sc = config["sampling"]
    mode = args.inject_mode or sc.get("past_inject_mode", "all")
    scale = float(sc.get("past_inject_scale", 0.1))
    return mode, scale


def _per_frame_channels(config: dict) -> int:
    """프레임별 VAE latent 채널 C_z (=6)."""
    return int(config["vae"]["latent_channels"])


def _past_latent_shape(config: dict) -> tuple[int, int, int]:
    """DDPM_past 타깃 latent shape (2×C_z, H_z, W_z) = (12,16,16)."""
    cz = _per_frame_channels(config)
    H_z, W_z = (int(s) for s in config["vae"]["latent_spatial"])
    return 2 * cz, H_z, W_z


def _train_window_loader(config: dict) -> DataLoader:
    """3시점(x_{t-1},x_t,x_{t+1}) train split DataLoader (past / joint 공용)."""
    tc = config["training"]
    src_path, src_var = resolve_data_source(config)
    ds = ERA5NormalizedDataset(
        normalized_path=src_path, var_name=src_var,
        mode="train", split="train", load_into_memory=True,
    )
    return DataLoader(
        ds, batch_size=tc["batch_size"], shuffle=True,
        num_workers=tc["num_workers"], pin_memory=True, drop_last=True,
        collate_fn=collate_with_time,
    )


@torch.no_grad()
def _encode_frame(
    vae: torch.nn.Module,
    normalizer: LatentNormalizer,
    frame: torch.Tensor,         # (B, C, H, W)
) -> torch.Tensor:
    """단일 프레임 → VAE.encode → 결정론적 latent z=μ → 정규화 latent ẑ (B,6,16,16).

    VAE를 autoencoder처럼 deterministic하게 쓴다 (z=μ, 노이즈 샘플링 없음).
    **compute_latent_stats.py도 동일하게 z=μ로 통계를 내야** 정규화가 평균0/분산1에
    정합한다 — 재매개변수화 통계(Var(μ)+E[exp(logσ²)])와 섞으면 분산<1로 under-dispersed.
    """
    mu, _ = vae.encode(frame)
    return normalizer.normalize(mu)


@torch.no_grad()
def _encode_pair_concat(
    vae: torch.nn.Module,
    normalizer: LatentNormalizer,
    f0: torch.Tensor,
    f1: torch.Tensor,
) -> torch.Tensor:
    """두 프레임을 각각 인코딩·정규화 후 채널 concat → (B,12,16,16).

    채널 규약: [0:6]=ẑ(f0), [6:12]=ẑ(f1). past 타깃은 f0=x_{t-1}, f1=x_{t+1}.
    """
    return torch.cat(
        [_encode_frame(vae, normalizer, f0),
         _encode_frame(vae, normalizer, f1)], dim=1,
    )


def _decode_pair(
    vae: torch.nn.Module,
    normalizer: LatentNormalizer,
    z: torch.Tensor,             # (B, 12, 16, 16)
) -> tuple[torch.Tensor, torch.Tensor]:
    """past latent (12ch) → 채널 split → denorm → 프레임별 decode → (x̂0, x̂1)."""
    cz = z.shape[1] // 2
    x0 = vae.decode(normalizer.denormalize(z[:, :cz]))
    x1 = vae.decode(normalizer.denormalize(z[:, cz:]))
    return x0, x1


def _diffusion_nll(
    dit: torch.nn.Module,
    schedule: VDMSchedule,
    z0: torch.Tensor,
    cond_tokens: torch.Tensor,
) -> torch.Tensor:
    """단일 diffusion step의 heteroscedastic NLL (latent 공간). t ~ Uniform[0,1]."""
    B = z0.shape[0]
    t = torch.rand(B, device=z0.device)
    z_t, eps = schedule.forward_noise(z0, t)
    eps_pred, log_var = dit(z_t, t, cond_tokens)
    return heteroscedastic_nll_loss(eps, eps_pred, log_var)


@torch.no_grad()
def _sample_past_condition(
    raw_dit_past: torch.nn.Module,
    raw_encoder: torch.nn.Module,
    vae: torch.nn.Module,
    normalizer: LatentNormalizer,
    past_sampler: LatentVDMSampler,
    x_t: torch.Tensor,
    latent_shape: tuple[int, int, int],
    mode: str,
    inject_scale: float = 0.1,
    c_tm1: torch.Tensor | None = None,
    c_tp1: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """past를 실제 샘플링·decode하여 main의 condition 재료 (x̂_{t-1}, x̂_{t+1})을 만든다.

    teacher forcing 금지 — past **생성물**을 cond로 써 cond-distribution OOD를
    방지한다. eval 모드로 샘플링하고 sampling 전체는 no_grad (역전파 안 함).
    반환 두 프레임은 상수 텐서이며, main은 이를 (grad 흐르는) encoder로 재인코딩한다.

    past condition (§1): 중심 anomaly x̃_t + 이웃 climatology c̃_{t-1}, c̃_{t+1}.
    """
    B = x_t.shape[0]
    C_z, H_z, W_z = latent_shape                          # (12, 16, 16)
    device = x_t.device

    was_training = raw_dit_past.training
    raw_dit_past.eval()
    cond_past = raw_encoder(past_snapshots(x_t, c_tm1, c_tp1))
    z_past, _ = past_sampler.sample(
        raw_dit_past, cond_past, (B, C_z, H_z, W_z), device,
        inject_uncertainty_mode=mode, inject_scale=inject_scale,
    )
    raw_dit_past.train(was_training)

    return _decode_pair(vae, normalizer, z_past)          # (x̂_{t-1}, x̂_{t+1})


def _find_latest_ckpt(output_dir: Path, prefix: str) -> Path | None:
    last = output_dir / f"{prefix}_last.pt"
    return last if last.exists() else None


# ════════════════════════════════════════════════════════════════════
# Phase: past — Encoder + DiT_past 학습
# ════════════════════════════════════════════════════════════════════
def run_past(config: dict, out_dir: Path, args: argparse.Namespace) -> None:
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp])
    set_seed(42)
    device = accelerator.device
    accelerator.print(
        f"[past] device={device} procs={accelerator.num_processes} "
        f"mp={accelerator.mixed_precision} epochs={args.past_epochs}"
    )
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    encoder = build_encoder(config)
    dit_past = build_dit(config, latent_channels=2 * _per_frame_channels(config))
    vae, normalizer = _load_frozen_vae(config, device)
    clim = _load_climatology(config, device)       # climatology condition (§1)
    schedule = _build_schedule(config)

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(dit_past.parameters()),
        lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]),
    )
    train_loader = _train_window_loader(config)

    start_epoch = _maybe_resume(
        args, out_dir, "past", {"encoder": encoder, "dit_past": dit_past},
        optimizer, accelerator,
    )

    encoder, dit_past, optimizer, train_loader = accelerator.prepare(
        encoder, dit_past, optimizer, train_loader,
    )
    grad_clip = train_cfg.get("grad_clip", 1.0)
    log_every = config["logging"]["log_every"]
    log_path = out_dir / "train_log_past.jsonl"

    for epoch in range(start_epoch, args.past_epochs):
        encoder.train(); dit_past.train()
        running, t0 = 0.0, time.time()
        for it, batch in enumerate(train_loader):
            x_tm1, x_t, x_tp1 = batch["x_tm1"], batch["x_t"], batch["x_tp1"]
            # past (§1): cond = 중심 anomaly x̃_t + 이웃 climatology c̃_{t±1},
            #            target = [ẑ_{t-1}‖ẑ_{t+1}] (12ch). 생성 대상=t±1 → 이웃 clim.
            if clim is not None:
                c_tm1 = clim.climatology_pixel_norm(
                    clim.doy_from_times(batch["time_tm1"]))
                c_tp1 = clim.climatology_pixel_norm(
                    clim.doy_from_times(batch["time_tp1"]))
            else:
                c_tm1 = c_tp1 = None
            cond = encoder(past_snapshots(x_t, c_tm1, c_tp1))
            z0 = _encode_pair_concat(vae, normalizer, x_tm1, x_tp1)
            loss = _diffusion_nll(dit_past, schedule, z0, cond)

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            if grad_clip and grad_clip > 0:
                accelerator.clip_grad_norm_(
                    list(encoder.parameters()) + list(dit_past.parameters()),
                    max_norm=grad_clip,
                )
            optimizer.step()
            running += float(loss.detach())
            if (it + 1) % log_every == 0:
                accelerator.print(
                    f"  [past] ep={epoch} it={it+1}/{len(train_loader)} "
                    f"L_past={running / (it + 1):.4f}"
                )

        dt = time.time() - t0
        epoch_log = {
            "phase": "past", "epoch": epoch, "elapsed_sec": dt,
            "L_past": running / max(len(train_loader), 1),
        }
        accelerator.print(
            f"[past ep {epoch}] time={dt:.1f}s L_past={epoch_log['L_past']:.4f}"
        )
        if (epoch + 1) % train_cfg["checkpoint_every"] == 0:
            _save_bundle(out_dir / "past_last.pt", epoch, config, accelerator,
                         optimizer, encoder=encoder, dit_past=dit_past)
        if accelerator.is_main_process:
            with open(log_path, "a") as f:
                f.write(json.dumps(epoch_log) + "\n")

    # 최종 산출물: infer/main 이 사용하는 past.pt (+ resume용 past_last.pt)
    _save_bundle(out_dir / "past.pt", args.past_epochs - 1, config, accelerator,
                 optimizer, encoder=encoder, dit_past=dit_past)
    _save_bundle(out_dir / "past_last.pt", args.past_epochs - 1, config,
                 accelerator, optimizer, encoder=encoder, dit_past=dit_past)
    accelerator.print(f"[past] done → {out_dir / 'past.pt'}")


# ════════════════════════════════════════════════════════════════════
# Phase: infer — train split 전체에 대해 M개 z_past 생성·저장
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_infer(config: dict, out_dir: Path, args: argparse.Namespace) -> None:
    """train split을 프로세스 수만큼 연속 구간으로 나눠 각자 같은 memmap에 기록.

    다른 phase와 동일하게 `accelerate launch` 로 실행한다. 모델은 추론 전용이라
    DDP wrap(prepare) 없이 각 프로세스가 자기 GPU에서 독립적으로 샘플링한다.
    memmap row i ↔ dataset index i 정합을 위해 **연속 구간**으로 분할한다
    (accelerate의 배치 round-robin/padding은 순서를 깨므로 쓰지 않는다).
    """
    # rank별 추론 시간 편차가 커서, 먼저 끝난 rank가 line 393 배리어에서
    # 나머지를 기다리는 시간이 NCCL 기본 타임아웃(10분)을 넘으면 전체가 죽는다.
    # → 배리어 타임아웃을 4시간으로 늘려 느린 rank를 끝까지 기다린다.
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=4))
    accelerator = Accelerator(kwargs_handlers=[init_kwargs])
    device = accelerator.device
    rank, world = accelerator.process_index, accelerator.num_processes
    M = args.ensemble_size
    mode, inject_scale = _resolve_past_inject(config, args)
    bs = max(1, args.batch_size)             # sampler 호출당 sample 수 (batch=bs×M)

    past_path = out_dir / "past.pt"
    if not past_path.exists():
        raise FileNotFoundError(f"{past_path} 없음 — 먼저 --phase past 를 실행하세요.")
    ck = torch.load(past_path, map_location="cpu")

    encoder = build_encoder(config).to(device)
    dit_past = build_dit(
        config, latent_channels=2 * _per_frame_channels(config),
    ).to(device)
    encoder.load_state_dict(ck["encoder"])
    dit_past.load_state_dict(ck["dit_past"])
    encoder.eval(); dit_past.eval()

    # infer는 정규화 latent ẑ_0 를 그대로 저장한다 (decode는 main/joint에서). VAE 불필요.
    clim = _load_climatology(config, device)       # climatology condition (§1)
    schedule = _build_schedule(config)
    steps = int(config["sampling"]["past_sampling_steps_train"])
    sampler = LatentVDMSampler(schedule, num_steps=steps)
    C_z, H_z, W_z = _past_latent_shape(config)         # (12, 16, 16)

    src_path, src_var = resolve_data_source(config)
    ds = ERA5NormalizedDataset(
        normalized_path=src_path, var_name=src_var,
        mode="train", split="train", load_into_memory=True,
    )
    N = len(ds)
    cond_path = out_dir / "cond_latents.npy"

    # 이 프로세스가 담당할 연속 행 구간 [lo, hi).
    per = (N + world - 1) // world
    lo = rank * per
    hi = min(N, lo + per)

    # 공유 memmap: main process가 전체 파일을 할당하고, 배리어 후 모두 r+로 연다.
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        alloc = np.lib.format.open_memmap(
            cond_path, mode="w+", dtype=np.float16, shape=(N, M, C_z, H_z, W_z),
        )
        alloc.flush()
        del alloc
    accelerator.wait_for_everyone()
    mm = np.lib.format.open_memmap(cond_path, mode="r+")

    accelerator.print(
        f"[infer] world={world} N={N} M={M} steps={steps} "
        f"inject={mode}×{inject_scale} "
        f"batch={bs}×M={bs * M} → {cond_path} ({mm.nbytes / 1e9:.1f} GB fp16 total)"
    )
    print(f"[infer] rank {rank}/{world} rows[{lo}:{hi}] ({hi - lo} samples)")

    for start in tqdm(range(lo, hi, bs), desc=f"infer[r{rank}]",
                      total=(hi - lo + bs - 1) // bs,
                      disable=not accelerator.is_main_process):
        S = min(bs, hi - start)
        items = [ds[start + s] for s in range(S)]
        x_t = torch.stack([it["x_t"] for it in items]).to(device)
        # past (§1): cond = 중심 anomaly x̃_t + 이웃 climatology c̃_{t±1}
        if clim is not None:
            t_tm1 = np.array([it["time_tm1"] for it in items])
            t_tp1 = np.array([it["time_tp1"] for it in items])
            c_tm1 = clim.climatology_pixel_norm(clim.doy_from_times(t_tm1))
            c_tp1 = clim.climatology_pixel_norm(clim.doy_from_times(t_tp1))
        else:
            c_tm1 = c_tp1 = None
        cond = encoder(past_snapshots(x_t, c_tm1, c_tp1))    # (S, N_cond, D)
        n_tok, d = cond.shape[1], cond.shape[2]
        cond = cond.unsqueeze(1).expand(S, M, n_tok, d).reshape(S * M, n_tok, d)
        z, _ = sampler.sample(
            dit_past, cond, (S * M, C_z, H_z, W_z), device,
            inject_uncertainty_mode=mode, inject_scale=inject_scale,
        )
        z = z.reshape(S, M, C_z, H_z, W_z).to(torch.float16).cpu().numpy()
        mm[start:start + S] = z
    mm.flush()

    # 모든 rank가 자기 구간을 다 쓴 뒤에만 meta(전역 정보)를 기록한다.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        meta = {
            "n_samples": N, "ensemble_size": M, "latent_shape": [C_z, H_z, W_z],
            "split": "train", "inject_mode": mode,
            "inject_scale": inject_scale, "past_steps": steps,
            "times": [str(ds.times[i + ds.valid_start]) for i in range(N)],
        }
        with open(out_dir / "cond_meta.json", "w") as f:
            json.dump(meta, f)
        accelerator.print(f"[infer] done → {cond_path}")


# ════════════════════════════════════════════════════════════════════
# Phase: main — DiT_main + Encoder 학습 (미리 만든 조건 사용)
# ════════════════════════════════════════════════════════════════════
class MainCondDataset(Dataset):
    """train 프레임 + 미리 생성한 past 조건 latent(member별)을 묶어 반환.

    epoch마다 `self.member` 를 바꿔 각 epoch이 서로 다른 과거 조건을 받게 한다.
    base dataset 의 index와 memmap row 는 동일 순서(infer가 순서대로 기록)로 정렬된다.
    """

    def __init__(self, base: ERA5NormalizedDataset, cond_path: Path):
        self.base = base
        self.cond = np.load(cond_path, mmap_mode="r")   # (N, M, C_z, H_z, W_z) fp16
        if self.cond.shape[0] != len(base):
            raise RuntimeError(
                f"cond rows({self.cond.shape[0]}) != train samples({len(base)}) "
                f"— infer를 같은 데이터로 다시 실행하세요."
            )
        self.member = 0

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        s = self.base[i]
        z = np.asarray(self.cond[i, self.member], dtype=np.float32)
        # 중심 climatology c_t 조회용 doy (int; default collate 호환).
        doy_t = int(doy_slot(s["time_t"])[0])
        return {"x_t": s["x_t"], "z_past": torch.from_numpy(z), "doy_t": doy_t}


def _main_cond_to_target(
    batch: dict, encoder, vae, normalizer, clim=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """member latent(12ch) → (x̂_{t-1}, x̂_{t+1}) decode → cond_main, target z0_main(ẑ_t).

    main condition (§1): 이웃 anomaly x̃_{t±1} + 중심 climatology c̃_t (생성 대상=t).
    """
    z_past = batch["z_past"]                # (B, 12, 16, 16) member 과거 latent
    x_t = batch["x_t"]
    with torch.no_grad():
        x_tm1_hat, x_tp1_hat = _decode_pair(vae, normalizer, z_past)
    c_t = _clim_cond(clim, batch["doy_t"]) if clim is not None else None
    cond_main = encoder(main_snapshots(x_tm1_hat, x_tp1_hat, c_t))
    z0_main = _encode_frame(vae, normalizer, x_t)                     # (B,6,16,16)
    return cond_main, z0_main


def run_main(config: dict, out_dir: Path, args: argparse.Namespace) -> None:
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp])
    set_seed(42)
    device = accelerator.device

    cond_path = out_dir / "cond_latents.npy"
    past_path = out_dir / "past.pt"
    if not cond_path.exists():
        raise FileNotFoundError(f"{cond_path} 없음 — 먼저 --phase infer 를 실행하세요.")
    if not past_path.exists():
        raise FileNotFoundError(f"{past_path} 없음 — 먼저 --phase past 를 실행하세요.")
    ck_past = torch.load(past_path, map_location="cpu")

    # encoder는 past 단계 가중치로 초기화 (조건 분포 정합). dit_main은 fresh.
    encoder = build_encoder(config)
    encoder.load_state_dict(ck_past["encoder"])
    dit_main = build_dit(config, latent_channels=_per_frame_channels(config))
    vae, normalizer = _load_frozen_vae(config, device)
    clim = _load_climatology(config, device)       # climatology condition (§1)
    schedule = _build_schedule(config)

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        list(dit_main.parameters()) + list(encoder.parameters()),
        lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]),
    )

    src_path, src_var = resolve_data_source(config)
    base = ERA5NormalizedDataset(
        normalized_path=src_path, var_name=src_var,
        mode="train", split="train", load_into_memory=True,
    )
    M = args.ensemble_size
    train_ds = MainCondDataset(base, cond_path)
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    accelerator.print(
        f"[main] device={device} procs={accelerator.num_processes} "
        f"N={len(base)} M={M} epochs={args.main_epochs}"
    )
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    start_epoch = _maybe_resume(
        args, out_dir, "main", {"encoder": encoder, "dit_main": dit_main},
        optimizer, accelerator,
    )

    encoder, dit_main, optimizer, train_loader = accelerator.prepare(
        encoder, dit_main, optimizer, train_loader,
    )
    grad_clip = train_cfg.get("grad_clip", 1.0)
    log_every = config["logging"]["log_every"]
    log_path = out_dir / "train_log_main.jsonl"

    for epoch in range(start_epoch, args.main_epochs):
        member = epoch % M
        train_ds.member = member          # 이 epoch이 사용할 과거 조건
        encoder.train(); dit_main.train()
        running, t0 = 0.0, time.time()
        for it, batch in enumerate(train_loader):
            cond_main, z0_main = _main_cond_to_target(
                batch, encoder, vae, normalizer, clim
            )
            loss = _diffusion_nll(dit_main, schedule, z0_main, cond_main)

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            if grad_clip and grad_clip > 0:
                accelerator.clip_grad_norm_(
                    list(dit_main.parameters()) + list(encoder.parameters()),
                    max_norm=grad_clip,
                )
            optimizer.step()
            running += float(loss.detach())
            if (it + 1) % log_every == 0:
                accelerator.print(
                    f"  [main] ep={epoch} member={member} "
                    f"it={it+1}/{len(train_loader)} "
                    f"L_main={running / (it + 1):.4f}"
                )

        dt = time.time() - t0
        epoch_log = {
            "phase": "main", "epoch": epoch, "member": member, "elapsed_sec": dt,
            "L_main": running / max(len(train_loader), 1),
        }
        accelerator.print(
            f"[main ep {epoch}] member={member} time={dt:.1f}s "
            f"L_main={epoch_log['L_main']:.4f}"
        )
        if (epoch + 1) % train_cfg["checkpoint_every"] == 0:
            _save_bundle(out_dir / "main_last.pt", epoch, config, accelerator,
                         optimizer, encoder=encoder, dit_main=dit_main,
                         dit_past_state=ck_past["dit_past"])
        if accelerator.is_main_process:
            with open(log_path, "a") as f:
                f.write(json.dumps(epoch_log) + "\n")

    # 최종 번들: joint(및 inference)이 사용하는 encoder/dit_past/dit_main
    _save_bundle(out_dir / "main.pt", args.main_epochs - 1, config, accelerator,
                 optimizer, encoder=encoder, dit_main=dit_main,
                 dit_past_state=ck_past["dit_past"])
    _save_bundle(out_dir / "main_last.pt", args.main_epochs - 1, config,
                 accelerator, optimizer, encoder=encoder, dit_main=dit_main,
                 dit_past_state=ck_past["dit_past"])
    accelerator.print(f"[main] done → {out_dir / 'main.pt'}")


# ════════════════════════════════════════════════════════════════════
# Phase: joint — DiT_past + Encoder + DiT_main end-to-end (동적 cond)
# ════════════════════════════════════════════════════════════════════
def run_joint(config: dict, out_dir: Path, args: argparse.Namespace) -> None:
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp])
    set_seed(42)
    device = accelerator.device

    main_path = out_dir / "main.pt"
    if not main_path.exists():
        raise FileNotFoundError(f"{main_path} 없음 — 먼저 --phase main 을 실행하세요.")
    ck_main = torch.load(main_path, map_location="cpu")

    cz = _per_frame_channels(config)
    encoder = build_encoder(config)
    dit_past = build_dit(config, latent_channels=2 * cz)   # 12ch
    dit_main = build_dit(config, latent_channels=cz)       # 6ch
    encoder.load_state_dict(ck_main["encoder"])
    dit_past.load_state_dict(ck_main["dit_past"])
    dit_main.load_state_dict(ck_main["dit_main"])

    vae, normalizer = _load_frozen_vae(config, device)
    clim = _load_climatology(config, device)       # climatology condition (§1)
    schedule = _build_schedule(config)
    past_sampler = LatentVDMSampler(
        schedule, num_steps=int(config["sampling"]["past_sampling_steps_train"]),
    )
    latent_shape = _past_latent_shape(config)              # (12, 16, 16)
    inject_mode, inject_scale = _resolve_past_inject(config, args)

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(dit_past.parameters())
        + list(dit_main.parameters()),
        lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]),
    )
    train_loader = _train_window_loader(config)
    accelerator.print(
        f"[joint] device={device} procs={accelerator.num_processes} "
        f"epochs={args.joint_epochs} inject={inject_mode}×{inject_scale}"
    )
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    start_epoch = _maybe_resume(
        args, out_dir, "joint",
        {"encoder": encoder, "dit_past": dit_past, "dit_main": dit_main},
        optimizer, accelerator,
    )

    encoder, dit_past, dit_main, optimizer, train_loader = accelerator.prepare(
        encoder, dit_past, dit_main, optimizer, train_loader,
    )
    grad_clip = train_cfg.get("grad_clip", 1.0)
    log_every = config["logging"]["log_every"]
    log_path = out_dir / "train_log_joint.jsonl"

    for epoch in range(start_epoch, args.joint_epochs):
        encoder.train(); dit_past.train(); dit_main.train()
        run_p = run_m = 0.0
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            x_tm1, x_t, x_tp1 = batch["x_tm1"], batch["x_t"], batch["x_tp1"]
            # climatology condition 조회 (생성 대상 시점 — §1).
            if clim is not None:
                c_tm1 = clim.climatology_pixel_norm(
                    clim.doy_from_times(batch["time_tm1"]))
                c_tp1 = clim.climatology_pixel_norm(
                    clim.doy_from_times(batch["time_tp1"]))
                c_t = clim.climatology_pixel_norm(
                    clim.doy_from_times(batch["time_t"]))
            else:
                c_tm1 = c_tp1 = c_t = None

            # ── L_past: cond=중심 anomaly+이웃 clim, target=[ẑ_{t-1}‖ẑ_{t+1}] ──
            cond_past = encoder(past_snapshots(x_t, c_tm1, c_tp1))
            z0_past = _encode_pair_concat(vae, normalizer, x_tm1, x_tp1)
            L_past = _diffusion_nll(dit_past, schedule, z0_past, cond_past)

            # ── L_main: past가 동적 생성한 이웃 anomaly + 중심 clim, target=ẑ_t ──
            x_tm1_hat, x_tp1_hat = _sample_past_condition(
                accelerator.unwrap_model(dit_past),
                accelerator.unwrap_model(encoder),
                vae, normalizer, past_sampler, x_t, latent_shape,
                inject_mode, inject_scale, c_tm1, c_tp1,
            )
            cond_main = encoder(main_snapshots(x_tm1_hat, x_tp1_hat, c_t))
            z0_main = _encode_frame(vae, normalizer, x_t)          # (B,6,16,16)
            L_main = _diffusion_nll(dit_main, schedule, z0_main, cond_main)

            loss = L_past + L_main
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            if grad_clip and grad_clip > 0:
                accelerator.clip_grad_norm_(
                    list(encoder.parameters()) + list(dit_past.parameters())
                    + list(dit_main.parameters()),
                    max_norm=grad_clip,
                )
            optimizer.step()
            run_p += float(L_past.detach())
            run_m += float(L_main.detach())
            if (it + 1) % log_every == 0:
                accelerator.print(
                    f"  [joint] ep={epoch} it={it+1}/{len(train_loader)} "
                    f"L_past={run_p / (it + 1):.4f} L_main={run_m / (it + 1):.4f}"
                )

        dt = time.time() - t0
        n = max(len(train_loader), 1)
        epoch_log = {
            "phase": "joint", "epoch": epoch, "elapsed_sec": dt,
            "L_past": run_p / n, "L_main": run_m / n,
        }
        accelerator.print(
            f"[joint ep {epoch}] time={dt:.1f}s "
            f"L_past={epoch_log['L_past']:.4f} L_main={epoch_log['L_main']:.4f}"
        )
        if (epoch + 1) % train_cfg["checkpoint_every"] == 0:
            _save_bundle(out_dir / "joint_last.pt", epoch, config, accelerator,
                         optimizer, encoder=encoder, dit_past=dit_past,
                         dit_main=dit_main)
        if accelerator.is_main_process:
            with open(log_path, "a") as f:
                f.write(json.dumps(epoch_log) + "\n")

    _save_bundle(out_dir / "checkpoint_final.pt", args.joint_epochs - 1, config,
                 accelerator, optimizer, encoder=encoder, dit_past=dit_past,
                 dit_main=dit_main)
    _save_bundle(out_dir / "joint_last.pt", args.joint_epochs - 1, config,
                 accelerator, optimizer, encoder=encoder, dit_past=dit_past,
                 dit_main=dit_main)
    accelerator.print(f"[joint] done → {out_dir / 'checkpoint_final.pt'}")


# ─── 체크포인트 helper ──────────────────────────────────────────────
def _save_bundle(
    path: Path, epoch: int, config: dict, accelerator: Accelerator,
    optimizer: torch.optim.Optimizer, *,
    encoder=None, dit_past=None, dit_main=None, dit_past_state=None,
) -> None:
    """학습 중인 module(들) + (선택) 외부 state_dict 를 묶어 저장.

    dit_past_state는 prepare되지 않은(학습 안 하는) 단계용 state_dict 직접 주입.
    """
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    payload: dict = {"epoch": epoch, "optimizer": optimizer.state_dict(),
                     "config": config}
    if encoder is not None:
        payload["encoder"] = accelerator.unwrap_model(encoder).state_dict()
    if dit_past is not None:
        payload["dit_past"] = accelerator.unwrap_model(dit_past).state_dict()
    elif dit_past_state is not None:
        payload["dit_past"] = dit_past_state
    if dit_main is not None:
        payload["dit_main"] = accelerator.unwrap_model(dit_main).state_dict()
    torch.save(payload, path)


def _maybe_resume(
    args: argparse.Namespace, out_dir: Path, prefix: str,
    modules: dict[str, torch.nn.Module], optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
) -> int:
    """resume 체크포인트가 있으면 module/optimizer 로드 후 시작 epoch 반환 (prepare 전)."""
    resume = (
        args.resume_from
        if args.resume_from and args.resume_from != "None" else None
    )
    if resume is None and args.resume:
        latest = _find_latest_ckpt(out_dir, prefix)
        resume = str(latest) if latest else None
        if resume:
            accelerator.print(f"[{prefix}] --resume: {resume}")
    if resume is None:
        return 0
    ck = torch.load(resume, map_location="cpu")
    for name, mod in modules.items():
        if name in ck:
            mod.load_state_dict(ck[name])
    if "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
    return ck.get("epoch", -1) + 1


# ─── Entry ──────────────────────────────────────────────────────────
def main(config: dict, args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    {"past": run_past, "infer": run_infer,
     "main": run_main, "joint": run_joint}[args.phase](config, out_dir, args)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True,
                   choices=["past", "infer", "main", "joint"])
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    # phase별 knob (config 미수정 — 모두 CLI 기본값)
    p.add_argument("--past_epochs", type=int, default=50)
    p.add_argument("--main_epochs", type=int, default=100)
    p.add_argument("--joint_epochs", type=int, default=50)
    p.add_argument("--ensemble_size", type=int, default=100,
                   help="phase=infer/main 의 과거 조건 member 수 M")
    p.add_argument("--inject_mode", type=str, default=None,
                   choices=["all", "last", "none"],
                   help="past 샘플링 logvar 주입 schedule (infer/joint). "
                        "미지정 시 config sampling.past_inject_mode(기본 all) 사용.")
    p.add_argument("--batch_size", type=int, default=20,
                   help="phase=infer: sampler 호출당 처리할 sample 수 "
                        "(GPU batch = batch_size × ensemble_size)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg, args)
