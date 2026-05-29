"""Stage 0 — WeatherVAE 선학습 (instruction_v2 §3 Stage 0).

VAE를 재구성 + 약한 KL로 **diffusion과 무관하게** 개별 학습한다. 학습 완료 후
freeze하고 `scripts/compute_latent_stats.py`로 latent 통계를 측정하면, 이후 모든
diffusion 학습(`training/train.py`)에서 VAE와 정규화 통계는 고정된다.

실행 예 (multi-GPU):
    accelerate launch --config_file config/accelerate_config.yaml \\
        -m training.train_vae --config config/default.yaml \\
        --output_dir outputs/vae

단일 GPU/CPU:
    python -m training.train_vae --config config/default.yaml \\
        --output_dir outputs/vae
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from dataset.era5_dataset import ERA5PairDataset
from models.vae import build_vae, weather_vae_loss


def _build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    vt = config["vae"]["training"]
    train_ds = ERA5PairDataset(
        normalized_path=data_cfg["normalized_path"],
        split="train", load_into_memory=True,
    )
    val_ds = ERA5PairDataset(
        normalized_path=data_cfg["normalized_path"],
        split="validation", load_into_memory=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=vt["batch_size"], shuffle=True,
        num_workers=vt.get("num_workers", 4), pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=vt["batch_size"], shuffle=False,
        num_workers=vt.get("num_workers", 4), pin_memory=True,
    )
    return train_loader, val_loader


@torch.no_grad()
def _validate(
    vae: torch.nn.Module,
    val_loader: DataLoader,
    kl_weight: float,
    grad_weight: float,
    accelerator: Accelerator,
    max_batches: int = 30,
) -> float:
    vae.eval()
    device = accelerator.device
    local_sum = torch.zeros((), device=device)
    local_n = torch.zeros((), device=device)
    for i, pair in enumerate(val_loader):
        if i >= max_batches:
            break
        # validation은 posterior mean으로 deterministic 재구성 평가
        recon, mu, log_var = vae(pair, sample=False)
        loss, _ = weather_vae_loss(
            recon, pair, mu, log_var, kl_weight, grad_weight
        )
        local_sum += loss.detach() * pair.shape[0]
        local_n += pair.shape[0]
    totals = accelerator.reduce(
        torch.stack([local_sum, local_n]), reduction="sum"
    )
    s, n = totals.tolist()
    vae.train()
    return s / max(n, 1)


def _save_ckpt(
    path: Path,
    epoch: int,
    vae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    accelerator: Accelerator,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(
            {
                "epoch": epoch,
                "vae": accelerator.unwrap_model(vae).state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            },
            path,
        )


def main(config: dict, output_dir: str, resume: bool = False) -> None:
    accelerator = Accelerator()
    set_seed(42)
    vae_cfg = config["vae"]
    vt = vae_cfg["training"]

    out_dir = Path(output_dir)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    log_path = out_dir / "vae_train_log.jsonl"

    vae = build_vae(vae_cfg)
    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=float(vt["lr"]),
        weight_decay=float(vt.get("weight_decay", 0.0)),
    )
    train_loader, val_loader = _build_dataloaders(config)
    accelerator.print(
        f"[info] VAE Stage 0 — train_pairs={len(train_loader.dataset)} "
        f"val_pairs={len(val_loader.dataset)}"
    )

    start_epoch = 0
    last_ckpt = out_dir / "vae_last.pt"
    if resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location="cpu")
        vae.load_state_dict(ckpt["vae"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        accelerator.print(f"[info] resumed VAE from epoch {start_epoch}")

    vae, optimizer, train_loader, val_loader = accelerator.prepare(
        vae, optimizer, train_loader, val_loader,
    )

    kl_weight = float(vae_cfg["kl_weight"])
    grad_weight = float(vae_cfg.get("grad_weight", 0.0))
    grad_clip = float(vt.get("grad_clip", 1.0))
    log_every = config["logging"]["log_every"]
    epochs = int(vt["epochs"])

    best_val = float("inf")
    for epoch in range(start_epoch, epochs):
        running = {"vae_total": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
        t0 = time.time()
        for it, pair in enumerate(train_loader):
            recon, mu, log_var = vae(pair, sample=True)
            loss, logs = weather_vae_loss(
                recon, pair, mu, log_var, kl_weight, grad_weight
            )
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            if grad_clip > 0:
                accelerator.clip_grad_norm_(vae.parameters(), grad_clip)
            optimizer.step()

            for k in ("vae_total", "recon", "kl"):
                running[k] += logs[k]
            running["n"] += 1
            if (it + 1) % log_every == 0:
                n = running["n"]
                accelerator.print(
                    f"  epoch={epoch} it={it+1}/{len(train_loader)} "
                    f"total={running['vae_total']/n:.5f} "
                    f"recon={running['recon']/n:.5f} kl={running['kl']/n:.3f}"
                )

        n = max(running["n"], 1)
        epoch_log = {
            "epoch": epoch, "elapsed_sec": time.time() - t0,
            "vae_total": running["vae_total"] / n,
            "recon": running["recon"] / n,
            "kl": running["kl"] / n,
        }

        if (epoch + 1) % vt.get("validation_every", 5) == 0:
            val = _validate(
                vae, val_loader, kl_weight, grad_weight, accelerator
            )
            epoch_log["val_total"] = val
            accelerator.print(
                f"[epoch {epoch}] recon={epoch_log['recon']:.5f} "
                f"val_total={val:.5f}"
            )
            if val < best_val:
                best_val = val
                _save_ckpt(
                    out_dir / "vae_best.pt", epoch, vae, optimizer,
                    config, accelerator,
                )
                accelerator.print(f"           saved best (val={val:.5f})")
        else:
            accelerator.print(
                f"[epoch {epoch}] recon={epoch_log['recon']:.5f}"
            )

        if (epoch + 1) % vt.get("checkpoint_every", 10) == 0:
            _save_ckpt(
                out_dir / f"vae_epoch{epoch:04d}.pt", epoch, vae,
                optimizer, config, accelerator,
            )
        _save_ckpt(out_dir / "vae_last.pt", epoch, vae, optimizer,
                   config, accelerator)

        if accelerator.is_main_process:
            with open(log_path, "a") as f:
                f.write(json.dumps(epoch_log) + "\n")

    accelerator.print(
        "[done] VAE Stage 0 finished. 다음 단계:\n"
        "  python -m scripts.compute_latent_stats "
        f"--vae_checkpoint {out_dir / 'vae_best.pt'}"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.output_dir, args.resume)
