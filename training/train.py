"""Diffusion² LDM 학습 — 4-phase 파이프라인 (past → infer → main → joint).

backbone(DiT)·encoder·VAE·sampler 는 기존 구현을 **그대로** 재사용한다. 기존의
단일 curriculum 학습을 4단계로 분리해 학습 시간을 줄이고, 마지막에 end-to-end로
미세조정한다. **validation은 수행하지 않는다.**

  phase=past : Condition Encoder + DiT_past 만 heteroscedastic NLL로 50 epoch 학습.
               (VAE는 Stage 0에서 frozen) → `<out_dir>/past.pt`.

  phase=infer: past.pt 로 **train split 전체**에 대해 sample당 ensemble_size(=M) 개의
               z_past 를 샘플링하여 fp16 memmap `(N, M, C_z, H_z, W_z)` 에 저장.
               불확실성 주입은 **마지막 denoising step에서만**(inject_mode='last')
               logvar 비례 노이즈를 준다. **train data 전용.** multi-GPU 시 각 프로세스가
               train split을 연속 구간으로 나눠 같은 memmap에 기록한다(row 순서 보존).

  phase=main : DiT_main + Encoder 를 main_epochs(=100) 만큼 학습. epoch e 는 미리
               만든 member `e % M` 을 condition 으로 사용한다(각 epoch마다 다른 과거).
               main 은 노이즈 주입을 받지 않으며 dual-head NLL을 유지한다.
               → `<out_dir>/main.pt` (encoder/dit_past/dit_main 번들).

  phase=joint: DiT_past + Encoder + DiT_main 을 joint_epochs(=50) 만큼 end-to-end로
               학습. end-to-end이므로 **매 batch마다 DiT_past가 동적으로 생성한**
               cond(mode='last')을 main에 쓴다. L = L_past + L_main.
               → `<out_dir>/checkpoint_final.pt` (inference 호환 번들).

선행 단계(반드시 먼저): Stage 0 VAE 학습 + latent 통계 계산.

실행 예:
    # 1) past 학습 (multi-GPU)
    accelerate launch --config_file config/accelerate_config.yaml \\
        -m training.train --phase past --output_dir outputs/decoupled_1 \\
        --past_epochs 50
    # 2) 조건 생성 (multi-GPU). train split을 프로세스 수로 나눠 병렬 샘플링.
    accelerate launch --config_file config/accelerate_config.yaml \\
        -m training.train --phase infer --output_dir outputs/decoupled_1 \\
        --ensemble_size 100 --inject_mode last --batch_size 20
    # 3) main 학습 (multi-GPU)
    accelerate launch --config_file config/accelerate_config.yaml \\
        -m training.train --phase main --output_dir outputs/decoupled_1 \\
        --main_epochs 100 --ensemble_size 100
    # 4) end-to-end joint 학습 (multi-GPU)
    accelerate launch --config_file config/accelerate_config.yaml \\
        -m training.train --phase joint --output_dir outputs/decoupled_1 \\
--joint_epochs 50 --inject_mode last
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
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset.era5_dataset import ERA5NormalizedDataset, collate_with_time
from inference.sampling import LatentVDMSampler
from models.dit import build_dit
from models.encoder import build_encoder
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


def _latent_shape(config: dict) -> tuple[int, int, int]:
    C_z = int(config["vae"]["latent_channels"])
    H_z, W_z = (int(s) for s in config["vae"]["latent_spatial"])
    return C_z, H_z, W_z


def _train_five_frame_loader(config: dict) -> DataLoader:
    """5시점 단위 train split DataLoader (past / joint 공용)."""
    data_cfg = config["data"]
    tc = config["training"]
    ds = ERA5NormalizedDataset(
        normalized_path=data_cfg["normalized_path"],
        mode="train", split="train", load_into_memory=True,
    )
    return DataLoader(
        ds, batch_size=tc["batch_size"], shuffle=True,
        num_workers=tc["num_workers"], pin_memory=True, drop_last=True,
        collate_fn=collate_with_time,
    )


@torch.no_grad()
def _encode_to_latent(
    vae: torch.nn.Module,
    normalizer: LatentNormalizer,
    frames: torch.Tensor,        # (B, 2, C, H, W)
) -> torch.Tensor:
    """2시점 target → VAE.encode → 결정론적 latent z=μ → 정규화 latent ẑ_0.

    VAE를 autoencoder처럼 deterministic하게 쓴다 (z=μ, 노이즈 샘플링 없음).
    **compute_latent_stats.py도 동일하게 z=μ로 통계를 내야** 정규화가 평균0/분산1에
    정합한다 — 재매개변수화 통계(Var(μ)+E[exp(logσ²)])와 섞으면 분산<1로 under-dispersed.
    """
    B, _, C, H, W = frames.shape
    pair = frames.reshape(B, 2 * C, H, W)
    mu, _ = vae.encode(pair)
    return normalizer.normalize(mu)


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
    x_tm1: torch.Tensor,
    x_t: torch.Tensor,
    latent_shape: tuple[int, int, int],
    mode: str,
) -> torch.Tensor:
    """past를 실제 샘플링·decode하여 main의 condition 재료 x̂_{t-2}를 만든다.

    teacher forcing 금지 — past **생성물**을 cond로 써 cond-distribution OOD를
    방지한다. eval 모드로 샘플링하고 sampling 전체는 no_grad (50-step 역전파 안 함).
    반환 x̂_{t-2}는 상수 텐서이며, main은 이를 (grad 흐르는) encoder로 재인코딩한다.
    """
    B = x_tm1.shape[0]
    C_z, H_z, W_z = latent_shape
    device = x_tm1.device

    was_training = raw_dit_past.training
    raw_dit_past.eval()
    cond_past = raw_encoder(torch.stack([x_tm1, x_t], dim=1))
    z_past, _ = past_sampler.sample(
        raw_dit_past, cond_past, (B, C_z, H_z, W_z), device,
        inject_uncertainty_mode=mode,
    )
    raw_dit_past.train(was_training)

    x_past = vae.decode(normalizer.denormalize(z_past))   # (B, 2C, H, W)
    C = x_tm1.shape[1]
    x_past = x_past.reshape(B, 2, C, *x_past.shape[-2:])
    return x_past[:, 1]                                   # 생성된 x̂_{t-2}


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

    encoder, dit_past = build_encoder(config), build_dit(config)
    vae, normalizer = _load_frozen_vae(config, device)
    schedule = _build_schedule(config)

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(dit_past.parameters()),
        lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]),
    )
    train_loader = _train_five_frame_loader(config)

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
            x_tm3, x_tm2 = batch["x_tm3"], batch["x_tm2"]
            x_tm1, x_t = batch["x_tm1"], batch["x_t"]
            cond = encoder(torch.stack([x_tm1, x_t], dim=1))
            z0 = _encode_to_latent(
                vae, normalizer, torch.stack([x_tm3, x_tm2], dim=1)
            )
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
    accelerator = Accelerator()
    device = accelerator.device
    rank, world = accelerator.process_index, accelerator.num_processes
    M = args.ensemble_size
    mode = args.inject_mode
    bs = max(1, args.batch_size)             # sampler 호출당 sample 수 (batch=bs×M)

    past_path = out_dir / "past.pt"
    if not past_path.exists():
        raise FileNotFoundError(f"{past_path} 없음 — 먼저 --phase past 를 실행하세요.")
    ck = torch.load(past_path, map_location="cpu")

    encoder = build_encoder(config).to(device)
    dit_past = build_dit(config).to(device)
    encoder.load_state_dict(ck["encoder"])
    dit_past.load_state_dict(ck["dit_past"])
    encoder.eval(); dit_past.eval()

    # infer는 정규화 latent ẑ_0 를 그대로 저장한다 (decode는 main/joint에서). VAE 불필요.
    schedule = _build_schedule(config)
    steps = int(config["sampling"]["past_sampling_steps_train"])
    sampler = LatentVDMSampler(schedule, num_steps=steps)
    C_z, H_z, W_z = _latent_shape(config)

    ds = ERA5NormalizedDataset(
        normalized_path=config["data"]["normalized_path"],
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
        f"[infer] world={world} N={N} M={M} steps={steps} mode={mode} "
        f"batch={bs}×M={bs * M} → {cond_path} ({mm.nbytes / 1e9:.1f} GB fp16 total)"
    )
    print(f"[infer] rank {rank}/{world} rows[{lo}:{hi}] ({hi - lo} samples)")

    for start in tqdm(range(lo, hi, bs), desc=f"infer[r{rank}]",
                      total=(hi - lo + bs - 1) // bs,
                      disable=not accelerator.is_main_process):
        S = min(bs, hi - start)
        items = [ds[start + s] for s in range(S)]
        x_tm1 = torch.stack([it["x_tm1"] for it in items]).to(device)
        x_t = torch.stack([it["x_t"] for it in items]).to(device)
        cond = encoder(torch.stack([x_tm1, x_t], dim=1))     # (S, N_tok, D)
        n_tok, d = cond.shape[1], cond.shape[2]
        cond = cond.unsqueeze(1).expand(S, M, n_tok, d).reshape(S * M, n_tok, d)
        z, _ = sampler.sample(
            dit_past, cond, (S * M, C_z, H_z, W_z), device,
            inject_uncertainty_mode=mode,
        )
        z = z.reshape(S, M, C_z, H_z, W_z).to(torch.float16).cpu().numpy()
        mm[start:start + S] = z
    mm.flush()

    # 모든 rank가 자기 구간을 다 쓴 뒤에만 meta(전역 정보)를 기록한다.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        meta = {
            "n_samples": N, "ensemble_size": M, "latent_shape": [C_z, H_z, W_z],
            "split": "train", "inject_mode": mode, "past_steps": steps,
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
        return {
            "x_tm1": s["x_tm1"], "x_t": s["x_t"], "x_tp1": s["x_tp1"],
            "z_past": torch.from_numpy(z),
        }


def _main_cond_to_target(
    batch: dict, encoder, vae, normalizer, config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """member latent → x̂_{t-2} decode → cond_main, 그리고 target z0_main 반환."""
    z_past = batch["z_past"]
    x_tm1, x_t, x_tp1 = batch["x_tm1"], batch["x_t"], batch["x_tp1"]
    B = z_past.shape[0]
    C = int(config["data"]["n_channels"])
    H, W = x_tm1.shape[-2:]
    with torch.no_grad():
        x_full = vae.decode(normalizer.denormalize(z_past))   # (B, 2C, H, W)
        x_tm2_hat = x_full.reshape(B, 2, C, H, W)[:, 1]        # 생성된 x̂_{t-2}
    cond_main = encoder(torch.stack([x_tm2_hat, x_tm1], dim=1))
    z0_main = _encode_to_latent(vae, normalizer, torch.stack([x_t, x_tp1], dim=1))
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
    dit_main = build_dit(config)
    vae, normalizer = _load_frozen_vae(config, device)
    schedule = _build_schedule(config)

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        list(dit_main.parameters()) + list(encoder.parameters()),
        lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]),
    )

    base = ERA5NormalizedDataset(
        normalized_path=config["data"]["normalized_path"],
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
                batch, encoder, vae, normalizer, config
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

    encoder, dit_past, dit_main = (
        build_encoder(config), build_dit(config), build_dit(config),
    )
    encoder.load_state_dict(ck_main["encoder"])
    dit_past.load_state_dict(ck_main["dit_past"])
    dit_main.load_state_dict(ck_main["dit_main"])

    vae, normalizer = _load_frozen_vae(config, device)
    schedule = _build_schedule(config)
    past_sampler = LatentVDMSampler(
        schedule, num_steps=int(config["sampling"]["past_sampling_steps_train"]),
    )
    latent_shape = _latent_shape(config)

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(dit_past.parameters())
        + list(dit_main.parameters()),
        lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]),
    )
    train_loader = _train_five_frame_loader(config)
    accelerator.print(
        f"[joint] device={device} procs={accelerator.num_processes} "
        f"epochs={args.joint_epochs} inject_mode={args.inject_mode}"
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
            x_tm3, x_tm2 = batch["x_tm3"], batch["x_tm2"]
            x_tm1, x_t, x_tp1 = batch["x_tm1"], batch["x_t"], batch["x_tp1"]

            # ── L_past: cond=[x_{t-1},x_t], target=[x_{t-3},x_{t-2}] ──
            cond_past = encoder(torch.stack([x_tm1, x_t], dim=1))
            z0_past = _encode_to_latent(
                vae, normalizer, torch.stack([x_tm3, x_tm2], dim=1)
            )
            L_past = _diffusion_nll(dit_past, schedule, z0_past, cond_past)

            # ── L_main: past가 동적 생성한 cond=[x̂_{t-2},x_{t-1}] ──
            x_tm2_hat = _sample_past_condition(
                accelerator.unwrap_model(dit_past),
                accelerator.unwrap_model(encoder),
                vae, normalizer, past_sampler, x_tm1, x_t, latent_shape,
                args.inject_mode,
            )
            cond_main = encoder(torch.stack([x_tm2_hat, x_tm1], dim=1))
            z0_main = _encode_to_latent(
                vae, normalizer, torch.stack([x_t, x_tp1], dim=1)
            )
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
    p.add_argument("--inject_mode", type=str, default="last",
                   choices=["all", "last", "none"],
                   help="past 샘플링 logvar 주입 schedule (infer/joint)")
    p.add_argument("--batch_size", type=int, default=20,
                   help="phase=infer: sampler 호출당 처리할 sample 수 "
                        "(GPU batch = batch_size × ensemble_size)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg, args)
