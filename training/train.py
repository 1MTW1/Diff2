"""Diffusion² 학습 메인 스크립트 — accelerate 기반 Multi-GPU.

실행 예 (multi-GPU):
    accelerate launch --config_file config/accelerate_config.yaml \\
        -m training.train --config config/default.yaml \\
        --output_dir outputs/experiment_1

단일 GPU/CPU (디버깅용):
    python -m training.train --config config/default.yaml \\
        --output_dir outputs/experiment_1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from dataset.era5_dataset import ERA5NormalizedDataset, collate_with_time
from models.dual_head_ddpm import DualHeadDDPM
from models.encoder import TemporalEncoder
from models.unet import UNet

from .curriculum import (
    CurriculumStage, get_loss_weights, get_stage, set_requires_grad,
)
from .loss import heteroscedastic_nll_loss
from .schedule import LinearNoiseSchedule


# ─── Build helpers ─────────────────────────────────────────────────
def _build_models(
    config: dict,
) -> tuple[TemporalEncoder, DualHeadDDPM, DualHeadDDPM]:
    """모델 CPU 생성. accelerator.prepare가 device 배치 처리."""
    C = config["data"]["n_channels"]
    enc_cfg = config["model"]["encoder"]
    unet_cfg = config["model"]["unet"]
    head_cfg = config["model"]["dual_head"]

    encoder = TemporalEncoder(
        in_channels=enc_cfg["in_channels"],
        hidden_channels=enc_cfg["hidden_channels"],
        num_layers=enc_cfg["num_layers"],
    )

    target_channels = 2 * C
    cond_channels = encoder.out_channels
    spatial_size = tuple(config["data"]["spatial"])
    pos_emb_channels = int(unet_cfg.get("pos_emb_channels", 0))

    def _make_ddpm() -> DualHeadDDPM:
        unet = UNet(
            in_channels=target_channels + cond_channels,
            out_channels=2 * target_channels,
            base_channels=unet_cfg["base_channels"],
            channel_mults=tuple(unet_cfg["channel_mults"]),
            num_res_blocks=unet_cfg["num_res_blocks"],
            attention_resolutions=tuple(unet_cfg["attention_resolutions"]),
            time_emb_dim=unet_cfg["time_emb_dim"],
            dropout=unet_cfg["dropout"],
            pos_emb_channels=pos_emb_channels,
            spatial_size=spatial_size if pos_emb_channels > 0 else None,
        )
        return DualHeadDDPM(
            unet,
            target_channels=target_channels,
            log_var_clip=tuple(head_cfg["log_var_clip"]),
        )

    return encoder, _make_ddpm(), _make_ddpm()


def _build_sampler(
    name: str, schedule: LinearNoiseSchedule, **kwargs,
):
    from inference.sampler import DDIMSampler, DDPMSampler
    if name == "ddpm":
        return DDPMSampler(
            schedule,
            num_inference_steps=kwargs.get("num_steps", None),
        )
    if name == "ddim":
        return DDIMSampler(
            schedule,
            num_inference_steps=kwargs.get("num_steps", 20),
            eta=kwargs.get("eta", 0.0),
        )
    raise ValueError(f"Unknown sampler: {name}")


def _build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Train/Val 모두 RAM에 사전 로드.

    각 process가 독립적으로 zarr → numpy 로드함 (멀티-GPU에서 N_GPU × dataset_size
    만큼 RAM 사용). num_workers는 in-memory라 0으로 둬도 빠름.
    """
    data_cfg = config["data"]
    bs = config["training"]["batch_size"]
    nw = config["training"]["num_workers"]

    train_ds = ERA5NormalizedDataset(
        normalized_path=data_cfg["normalized_path"],
        mode="train",
        split="train",
        load_into_memory=True,
    )
    val_ds = ERA5NormalizedDataset(
        normalized_path=data_cfg["normalized_path"],
        mode="train",          # validation도 5 시점 (loss 계산용)
        split="validation",
        load_into_memory=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        collate_fn=collate_with_time,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
        collate_fn=collate_with_time,
    )
    return train_loader, val_loader


# ─── Step / Validation ─────────────────────────────────────────────
def _train_step(
    batch: dict,
    encoder: torch.nn.Module,
    model_past: torch.nn.Module,
    model_main: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    schedule: LinearNoiseSchedule,
    aux_sampler,
    stage: int,
    accelerator: Accelerator,
    grad_clip: float | None,
) -> dict[str, float]:
    # batch는 accelerator.prepare로 자동으로 device로 옮겨짐
    x_tm3 = batch["x_tm3"]
    x_tm2 = batch["x_tm2"]
    x_tm1 = batch["x_tm1"]
    x_t = batch["x_t"]
    x_tp1 = batch["x_tp1"]
    device = x_t.device
    B, C, H, W = x_t.shape

    weights = get_loss_weights(stage)
    L_past = torch.zeros((), device=device)
    L_main = torch.zeros((), device=device)

    # ── DDPM_past loss ─────────────────────────────────────────────
    if weights["past"] > 0:
        cond_past_input = torch.stack([x_tm1, x_t], dim=1)
        cond_past = encoder(cond_past_input)

        target_past = torch.stack([x_tm3, x_tm2], dim=1)
        target_past_flat = target_past.reshape(B, 2 * C, H, W)

        m_past = torch.randint(0, schedule.M, (B,), device=device)
        noisy_past, eps_true_past = schedule.forward_noise(
            target_past_flat, m_past
        )
        eps_pred_past, log_var_past = model_past(noisy_past, m_past, cond_past)
        L_past = heteroscedastic_nll_loss(
            eps_true_past, eps_pred_past, log_var_past
        )

    # ── DDPM_main loss ─────────────────────────────────────────────
    if weights["main"] > 0:
        cond_past_input = torch.stack([x_tm1, x_t], dim=1)
        # Sampling은 unwrapped 모델로 → DDP 통신 오버헤드 회피
        raw_past = accelerator.unwrap_model(model_past)
        with torch.no_grad():
            cond_past_eval = encoder(cond_past_input)
            past_sample, _, _ = aux_sampler.sample(
                raw_past,
                cond=cond_past_eval,
                shape=(B, 2 * C, H, W),
                device=device,
            )
            past_sample = past_sample.reshape(B, 2, C, H, W)
            x_tm3_hat = past_sample[:, 0]
            x_tm2_hat = past_sample[:, 1]

        cond_main_input = torch.stack(
            [x_tm3_hat, x_tm2_hat, x_tm1], dim=1
        )
        cond_main = encoder(cond_main_input)

        target_main = torch.stack([x_t, x_tp1], dim=1)
        target_main_flat = target_main.reshape(B, 2 * C, H, W)

        m_main = torch.randint(0, schedule.M, (B,), device=device)
        noisy_main, eps_true_main = schedule.forward_noise(
            target_main_flat, m_main
        )
        eps_pred_main, log_var_main = model_main(
            noisy_main, m_main, cond_main
        )
        L_main = heteroscedastic_nll_loss(
            eps_true_main, eps_pred_main, log_var_main
        )

    L_total = weights["past"] * L_past + weights["main"] * L_main

    optimizer.zero_grad(set_to_none=True)
    accelerator.backward(L_total)
    if grad_clip is not None and grad_clip > 0:
        # 모든 학습 대상 파라미터에 대해 norm clip
        params = (
            list(encoder.parameters())
            + list(model_past.parameters())
            + list(model_main.parameters())
        )
        accelerator.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

    return {
        "L_total": float(L_total.detach()),
        "L_past": float(L_past.detach()),
        "L_main": float(L_main.detach()),
    }


@torch.no_grad()
def _validate(
    encoder: torch.nn.Module,
    model_past: torch.nn.Module,
    model_main: torch.nn.Module,
    val_loader: DataLoader,
    schedule: LinearNoiseSchedule,
    aux_sampler,
    accelerator: Accelerator,
    max_batches: int | None = None,
) -> dict[str, float]:
    encoder.eval(); model_past.eval(); model_main.eval()
    device = accelerator.device
    raw_past = accelerator.unwrap_model(model_past)

    local_sum_p = torch.zeros((), device=device)
    local_sum_m = torch.zeros((), device=device)
    local_n = torch.zeros((), device=device)

    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        x_tm3 = batch["x_tm3"]
        x_tm2 = batch["x_tm2"]
        x_tm1 = batch["x_tm1"]
        x_t = batch["x_t"]
        x_tp1 = batch["x_tp1"]
        B, C, H, W = x_t.shape

        # past loss
        cond_past = encoder(torch.stack([x_tm1, x_t], dim=1))
        target_past_flat = torch.stack(
            [x_tm3, x_tm2], dim=1
        ).reshape(B, 2 * C, H, W)
        m = torch.randint(0, schedule.M, (B,), device=device)
        noisy_past, eps_p = schedule.forward_noise(target_past_flat, m)
        eps_pp, lv_p = model_past(noisy_past, m, cond_past)
        Lp = heteroscedastic_nll_loss(eps_p, eps_pp, lv_p)

        # main loss
        past_sample, _, _ = aux_sampler.sample(
            raw_past, cond=cond_past,
            shape=(B, 2 * C, H, W), device=device,
        )
        past_sample = past_sample.reshape(B, 2, C, H, W)
        cond_main = encoder(torch.stack(
            [past_sample[:, 0], past_sample[:, 1], x_tm1], dim=1
        ))
        target_main_flat = torch.stack(
            [x_t, x_tp1], dim=1
        ).reshape(B, 2 * C, H, W)
        m = torch.randint(0, schedule.M, (B,), device=device)
        noisy_main, eps_m = schedule.forward_noise(target_main_flat, m)
        eps_mp, lv_m = model_main(noisy_main, m, cond_main)
        Lm = heteroscedastic_nll_loss(eps_m, eps_mp, lv_m)

        local_sum_p += Lp.detach() * B
        local_sum_m += Lm.detach() * B
        local_n += B

    # 전체 process 집계: reduce(sum)으로 모든 rank에 sum된 값 동일 broadcast
    stacked = torch.stack([local_sum_p, local_sum_m, local_n])  # (3,)
    totals = accelerator.reduce(stacked, reduction="sum")        # (3,) sum
    sum_p, sum_m, n = totals.tolist()
    n = max(n, 1)

    encoder.train(); model_past.train(); model_main.train()
    return {
        "val_L_past": sum_p / n,
        "val_L_main": sum_m / n,
    }


def _apply_stage(
    stage: int,
    encoder: torch.nn.Module,
    model_past: torch.nn.Module,
    model_main: torch.nn.Module,
) -> None:
    if stage == CurriculumStage.PAST_ONLY:
        set_requires_grad(encoder, True)
        set_requires_grad(model_past, True)
        set_requires_grad(model_main, False)
    elif stage == CurriculumStage.MAIN_ONLY:
        set_requires_grad(encoder, True)
        set_requires_grad(model_past, False)
        set_requires_grad(model_main, True)
    else:
        set_requires_grad(encoder, True)
        set_requires_grad(model_past, True)
        set_requires_grad(model_main, True)


def _save_ckpt(
    path: Path,
    epoch: int,
    encoder: torch.nn.Module,
    model_past: torch.nn.Module,
    model_main: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    accelerator: Accelerator,
) -> None:
    # 메인 프로세스에서만 저장. 다른 프로세스는 대기.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(
            {
                "epoch": epoch,
                "encoder": accelerator.unwrap_model(encoder).state_dict(),
                "model_past": accelerator.unwrap_model(model_past).state_dict(),
                "model_main": accelerator.unwrap_model(model_main).state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            },
            path,
        )


# ─── Main ──────────────────────────────────────────────────────────
def _find_latest_ckpt(output_dir: Path) -> Path | None:
    """output_dir에서 가장 최근 체크포인트 자동 탐색.

    우선순위:
    1. checkpoint_last.pt (학습 완료 후 저장됨)
    2. 가장 epoch 번호 큰 checkpoint_epochNNNN.pt (주기 저장)
    3. checkpoint_best.pt (validation 최고 모델)
    """
    last = output_dir / "checkpoint_last.pt"
    if last.exists():
        return last
    epoch_ckpts = sorted(output_dir.glob("checkpoint_epoch*.pt"))
    if epoch_ckpts:
        return epoch_ckpts[-1]
    best = output_dir / "checkpoint_best.pt"
    if best.exists():
        return best
    return None


def main(
    config: dict,
    output_dir: str,
    resume_from: str | None = None,
    resume: bool = False,
) -> None:
    # find_unused_parameters: curriculum stage 전환 시 일부 모델의 grad가
    # 비기 때문에 DDP 동기화에서 필요.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    set_seed(42)

    device = accelerator.device
    accelerator.print(
        f"[info] device={device} num_processes={accelerator.num_processes} "
        f"mixed_precision={accelerator.mixed_precision}"
    )

    out_dir = Path(output_dir)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    log_path = out_dir / "train_log.jsonl"

    # ── 모델 / 스케줄 / 옵티마이저 ─────────────────────────────────
    encoder, model_past, model_main = _build_models(config)

    diff_cfg = config["diffusion"]
    schedule = LinearNoiseSchedule(
        M=diff_cfg["M"],
        beta_start=float(diff_cfg["beta_start"]),
        beta_end=float(diff_cfg["beta_end"]),
        device=device,
    )

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        list(encoder.parameters())
        + list(model_past.parameters())
        + list(model_main.parameters()),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    train_loader, val_loader = _build_dataloaders(config)
    accelerator.print(
        f"[info] train_ds={len(train_loader.dataset)} "
        f"val_ds={len(val_loader.dataset)} (both in RAM)"
    )

    # ── Resume (prepare 전에 raw 모델로 load) ──────────────────────
    resolved_resume: str | None = None
    if resume_from and resume_from != "None":
        resolved_resume = resume_from
    elif resume:
        latest = _find_latest_ckpt(out_dir)
        if latest is not None:
            resolved_resume = str(latest)
            accelerator.print(
                f"[info] --resume: auto-found {resolved_resume}"
            )
        else:
            accelerator.print(
                f"[info] --resume set but no checkpoint in {out_dir} — "
                f"starting fresh"
            )

    start_epoch = 0
    if resolved_resume is not None:
        ckpt = torch.load(resolved_resume, map_location="cpu")
        encoder.load_state_dict(ckpt["encoder"])
        model_past.load_state_dict(ckpt["model_past"])
        model_main.load_state_dict(ckpt["model_main"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        accelerator.print(
            f"[info] resumed from {resolved_resume} (epoch={start_epoch})"
        )

    # ── Prepare (device 배치 + DDP wrap + DataLoader sharding) ────
    (
        encoder, model_past, model_main, optimizer, train_loader, val_loader,
    ) = accelerator.prepare(
        encoder, model_past, model_main, optimizer, train_loader, val_loader,
    )

    aux_cfg = train_cfg["aux_sampler"]
    aux_sampler = _build_sampler(
        aux_cfg["type"], schedule,
        num_steps=aux_cfg.get("num_steps", 20),
        eta=aux_cfg.get("eta", 0.0),
    )

    curr = train_cfg["curriculum"]
    log_every = config["logging"]["log_every"]
    grad_clip = train_cfg.get("grad_clip", 1.0)

    accelerator.print(
        f"[info] train batches/epoch (per process) = {len(train_loader)}"
    )

    # ── 학습 루프 ──────────────────────────────────────────────────
    best_val = float("inf")
    for epoch in range(start_epoch, train_cfg["total_epochs"]):
        stage = get_stage(
            epoch,
            curr["stage1_epochs"],
            curr["stage2_epochs"],
            curr["stage3_epochs"],
        )
        _apply_stage(stage, encoder, model_past, model_main)

        running = {"L_total": 0.0, "L_past": 0.0, "L_main": 0.0, "n": 0}
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            losses = _train_step(
                batch, encoder, model_past, model_main,
                optimizer, schedule, aux_sampler, stage,
                accelerator, grad_clip,
            )
            for k in ("L_total", "L_past", "L_main"):
                running[k] += losses[k]
            running["n"] += 1

            if (it + 1) % log_every == 0:
                avg = {k: running[k] / running["n"]
                       for k in ("L_total", "L_past", "L_main")}
                accelerator.print(
                    f"  epoch={epoch} stage={stage} "
                    f"it={it+1}/{len(train_loader)} "
                    f"L_total={avg['L_total']:.4f} "
                    f"L_past={avg['L_past']:.4f} "
                    f"L_main={avg['L_main']:.4f}"
                )

        epoch_dt = time.time() - t0
        n = max(running["n"], 1)
        epoch_log = {
            "epoch": epoch,
            "stage": stage,
            "elapsed_sec": epoch_dt,
            "L_total": running["L_total"] / n,
            "L_past": running["L_past"] / n,
            "L_main": running["L_main"] / n,
        }
        accelerator.print(
            f"[epoch {epoch}] stage={stage} time={epoch_dt:.1f}s "
            f"L_total={epoch_log['L_total']:.4f}"
        )

        # ── Validation ─────────────────────────────────────────────
        if (epoch + 1) % train_cfg["validation_every"] == 0:
            val_metrics = _validate(
                encoder, model_past, model_main, val_loader,
                schedule, aux_sampler, accelerator, max_batches=20,
            )
            epoch_log.update(val_metrics)
            accelerator.print(
                f"           val_L_past={val_metrics['val_L_past']:.4f} "
                f"val_L_main={val_metrics['val_L_main']:.4f}"
            )

            val_total = val_metrics["val_L_past"] + val_metrics["val_L_main"]
            if val_total < best_val:
                best_val = val_total
                _save_ckpt(
                    out_dir / "checkpoint_best.pt", epoch,
                    encoder, model_past, model_main, optimizer, config,
                    accelerator,
                )
                accelerator.print(
                    f"           saved best (val={val_total:.4f})"
                )

        if (epoch + 1) % train_cfg["checkpoint_every"] == 0:
            _save_ckpt(
                out_dir / f"checkpoint_epoch{epoch:04d}.pt", epoch,
                encoder, model_past, model_main, optimizer, config,
                accelerator,
            )

        if accelerator.is_main_process:
            with open(log_path, "a") as f:
                f.write(json.dumps(epoch_log) + "\n")

    _save_ckpt(
        out_dir / "checkpoint_last.pt",
        train_cfg["total_epochs"] - 1,
        encoder, model_past, model_main, optimizer, config,
        accelerator,
    )
    accelerator.print("[done] training finished")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resume_from", type=str, default=None,
                   help="명시적 체크포인트 경로로 resume")
    p.add_argument("--resume", action="store_true",
                   help="output_dir에서 가장 최근 체크포인트 자동 resume")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.output_dir, args.resume_from, args.resume)
