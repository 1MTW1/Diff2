"""Ensemble 생성 (instruction.md §4.12).

추론 시:
1. Per ensemble member b마다 DDPM_past를 새로 sampling → 다양한 (x̂_{t-3}, x̂_{t-2})
2. 그 condition으로 DDPM_main을 새로 sampling → 다양한 x_t^(b)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from dataset.denormalize import denormalize, load_stats
from dataset.era5_dataset import ERA5NormalizedDataset
from models.dual_head_ddpm import DualHeadDDPM
from models.encoder import TemporalEncoder
from models.unet import UNet
from training.schedule import LinearNoiseSchedule

from .sampler import DDIMSampler, DDPMSampler


@torch.no_grad()
def generate_ensemble(
    x_tm1: torch.Tensor,
    x_t: torch.Tensor,
    encoder: TemporalEncoder,
    model_past: DualHeadDDPM,
    model_main: DualHeadDDPM,
    sampler,
    B: int = 20,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, dict]:
    """단일 (x_{t-1}, x_t) 입력에 대해 B개 ensemble member 생성 (batched).

    Instruction.md §8.5의 권장사항대로 ensemble 차원을 batch dim에 펼쳐서
    sampler를 단 2회 호출 (past 1회 + main 1회). 매 step의 noise는 batch dim에서
    자동으로 독립적이라 member별 다양성 보존.

    Args:
        x_tm1, x_t: (1, C, H, W)
        sampler:    DDPMSampler 또는 DDIMSampler
    Returns:
        ensemble:   (B, C, H, W) normalized space (x_t의 ensemble member)
        diagnostics: dict, log_vars_past/main 모두 (B, 2*C, H, W)
    """
    x_tm1 = x_tm1.to(device)
    x_t = x_t.to(device)
    _, C, H, W = x_tm1.shape

    # ── Step 1: B개 past sample을 한 번에 ─────────────────────────
    # past condition은 모든 member에서 동일 → B로 expand
    cond_past_single = encoder(
        torch.stack([x_tm1, x_t], dim=1)
    )                                                        # (1, C', H, W)
    cond_past_B = cond_past_single.expand(B, -1, -1, -1)     # (B, C', H, W)

    past_sample, _, ell_past = sampler.sample(
        model_past, cond=cond_past_B,
        shape=(B, 2 * C, H, W), device=device,
    )
    past_sample = past_sample.reshape(B, 2, C, H, W)
    x_tm3_hat = past_sample[:, 0]                            # (B, C, H, W)
    x_tm2_hat = past_sample[:, 1]

    # ── Step 2: B개 main sample을 한 번에 ──────────────────────────
    # 각 member마다 cond_main이 다름 (past sample이 달라서). x_{t-1}만 공유.
    x_tm1_B = x_tm1.expand(B, -1, -1, -1)                    # (B, C, H, W)
    cond_main = encoder(
        torch.stack([x_tm3_hat, x_tm2_hat, x_tm1_B], dim=1)
    )                                                        # (B, C', H, W)
    main_sample, _, ell_main = sampler.sample(
        model_main, cond=cond_main,
        shape=(B, 2 * C, H, W), device=device,
    )
    main_sample = main_sample.reshape(B, 2, C, H, W)
    ensemble = main_sample[:, 0]                             # (B, C, H, W)

    return ensemble, {
        "log_vars_past": ell_past.detach().cpu(),            # (B, 2*C, H, W)
        "log_vars_main": ell_main.detach().cpu(),
    }


# ─── CLI ────────────────────────────────────────────────────────────
def _build_models_from_ckpt(
    ckpt: dict, device: torch.device
) -> tuple[TemporalEncoder, DualHeadDDPM, DualHeadDDPM]:
    config = ckpt["config"]
    C = config["data"]["n_channels"]
    enc_cfg = config["model"]["encoder"]
    unet_cfg = config["model"]["unet"]
    head_cfg = config["model"]["dual_head"]

    encoder = TemporalEncoder(
        in_channels=enc_cfg["in_channels"],
        hidden_channels=enc_cfg["hidden_channels"],
        num_layers=enc_cfg["num_layers"],
    ).to(device)
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
        ).to(device)
        return DualHeadDDPM(
            unet,
            target_channels=target_channels,
            log_var_clip=tuple(head_cfg["log_var_clip"]),
        ).to(device)

    model_past = _make_ddpm()
    model_main = _make_ddpm()
    encoder.load_state_dict(ckpt["encoder"])
    model_past.load_state_dict(ckpt["model_past"])
    model_main.load_state_dict(ckpt["model_main"])
    encoder.eval(); model_past.eval(); model_main.eval()
    return encoder, model_past, model_main


def _build_sampler_from_config(config: dict, schedule: LinearNoiseSchedule):
    inf_cfg = config["inference"]["sampler"]
    if inf_cfg["type"] == "ddpm":
        return DDPMSampler(
            schedule,
            num_inference_steps=inf_cfg.get("num_steps", None),
        )
    return DDIMSampler(
        schedule,
        num_inference_steps=inf_cfg["num_steps"],
        eta=float(inf_cfg.get("eta", 0.0)),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--ensemble_size", type=int, default=20)
    p.add_argument("--n_samples", type=int, default=10,
                   help="얼마나 많은 시각에 대해 ensemble을 생성할지")
    p.add_argument("--denormalize", action="store_true",
                   help="저장 시 원본 단위로 역정규화")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    # 학습 시 config 사용 (호환성)
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
    sampler = _build_sampler_from_config(config, schedule)

    ds = ERA5NormalizedDataset(
        normalized_path=config["data"]["normalized_path"],
        mode="inference",
        split=args.split,
        load_into_memory=False,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean = std = None
    if args.denormalize:
        mean, std = load_stats(config["data"]["stats_path"], device=str(device))

    n_samples = min(args.n_samples, len(ds))
    print(f"[info] generating {args.ensemble_size}-member ensembles "
          f"for {n_samples} samples")

    results = []
    for i in tqdm(range(n_samples), desc="ensemble"):
        sample = ds[i]
        x_tm1 = sample["x_tm1"].unsqueeze(0)
        x_t = sample["x_t"].unsqueeze(0)
        time_t = sample["time_t"]

        ensemble, diag = generate_ensemble(
            x_tm1, x_t, encoder, model_past, model_main, sampler,
            B=args.ensemble_size, device=device,
        )

        out = {"ensemble": ensemble.cpu()}
        if args.denormalize and mean is not None:
            import numpy as np
            ens_denorm = denormalize(
                ensemble, np.array([time_t] * ensemble.shape[0]), mean, std,
            )
            out["ensemble_denorm"] = ens_denorm.cpu()
        out["log_vars_main"] = diag["log_vars_main"]
        out["log_vars_past"] = diag["log_vars_past"]
        out["x_t_gt"] = x_t.squeeze(0).cpu()
        out["time_t"] = str(time_t)

        torch.save(out, out_dir / f"ensemble_{i:04d}.pt")
        results.append({"idx": i, "time_t": str(time_t)})

    with open(out_dir / "index.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
