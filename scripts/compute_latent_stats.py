"""Latent 통계 측정 — instruction_v2 §2.2 / Stage 0 후처리.

frozen VAE로 학습 split의 모든 연속 2시점 쌍을 encode하여 **결정론적**
latent z = μ 의 1·2차 모멘트를 측정하고 `latent_stats.pt`로 저장한다. 이후 모든
diffusion 학습/추론은 이 통계로 latent를 평균0/분산1로 정규화한다.

VAE를 deterministic autoencoder처럼 사용하므로 z=μ 로 통계를 낸다 (노이즈 없음).
train.py `_encode_to_latent` 가 z=μ 로 인코딩하는 것과 **반드시 정합**해야 정규화가
평균0/분산1을 만족한다 (한쪽만 노이즈를 섞으면 분산<1). `--seed` 는 더 이상 통계에
영향 없음 — 무작위성이 제거됨.

mode (config["latent_norm"]["mode"] 또는 --mode):
  - channel_pixelwise: μ,σ ∈ (C_z, H_z, W_z)  — 위치별·채널별 (기본)
  - channelwise:       μ,σ ∈ (C_z, 1, 1)      — 채널별만 (데이터 적을 때 fallback)

실행 예:
    python -m scripts.compute_latent_stats \\
        --config config/default_reparam.yaml \\
        --vae_checkpoint outputs/vae/vae_last.pt
"""
from __future__ import annotations

import argparse

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.era5_dataset import ERA5PairDataset
from models.vae import build_vae


@torch.no_grad()
def compute_stats(
    vae,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    seed: int = 0,
) -> dict:
    """결정론적 latent z = μ 의 1·2차 모멘트 streaming 계산 (autoencoder 방식).

    train.py `_encode_to_latent` 와 동일하게 z=μ 만 사용한다. pixelwise stats는
    Var(μ) 만 포함한다 (재매개변수화 노이즈 항 E[exp(logσ²)] 없음).
    """
    torch.manual_seed(seed)                  # ε 재현성
    sum_chw: torch.Tensor | None = None      # Σ z          (C_z, H_z, W_z)
    sumsq_chw: torch.Tensor | None = None    # Σ z²
    count = 0                                # 누적 sample 수

    for pair in tqdm(loader, desc="encode"):
        pair = pair.to(device)               # (B, 6, 64, 64)
        mu, _ = vae.encode(pair)             # (B, C_z, H_z, W_z)
        # deterministic autoencoder: z = μ (노이즈 샘플링 없음 — train.py와 정합)
        z = mu.float()
        if sum_chw is None:
            sum_chw = torch.zeros_like(z[0])
            sumsq_chw = torch.zeros_like(z[0])
        sum_chw += z.sum(dim=0)
        sumsq_chw += (z * z).sum(dim=0)
        count += z.shape[0]

    if sum_chw is None or count == 0:
        raise RuntimeError("No data encoded — empty dataset?")

    # ── channel_pixelwise: 위치별·채널별 ───────────────────────────
    mean_chw = sum_chw / count
    var_chw = (sumsq_chw / count) - mean_chw ** 2
    var_chw = var_chw.clamp_min(0.0)

    if mode == "channel_pixelwise":
        mu_out = mean_chw                                   # (C_z, H_z, W_z)
        sigma_out = var_chw.sqrt()
    elif mode == "channelwise":
        # 공간축까지 pool → 채널별만
        n_spatial = sum_chw.shape[1] * sum_chw.shape[2]
        total = count * n_spatial
        mean_c = sum_chw.sum(dim=(1, 2)) / total            # (C_z,)
        var_c = sumsq_chw.sum(dim=(1, 2)) / total - mean_c ** 2
        var_c = var_c.clamp_min(0.0)
        mu_out = mean_c[:, None, None]                      # (C_z, 1, 1)
        sigma_out = var_c.sqrt()[:, None, None]
    else:
        raise ValueError(f"Unknown latent_norm mode: {mode}")

    return {
        "mode": mode,
        "mu": mu_out.cpu(),
        "sigma": sigma_out.cpu(),
        "count": count,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--vae_checkpoint", type=str, required=True)
    p.add_argument("--mode", type=str, default=None,
                   help="config 대신 mode 강제 지정")
    p.add_argument("--output", type=str, default=None,
                   help="저장 경로 (기본: config latent_norm.stats_path)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0,
                   help="재매개변수화 ε의 RNG 시드 (재현성).")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = args.mode or config["latent_norm"]["mode"]
    out_path = args.output or config["latent_norm"]["stats_path"]

    # ── frozen VAE 로드 ────────────────────────────────────────────
    ckpt = torch.load(args.vae_checkpoint, map_location="cpu")
    vae_cfg = ckpt.get("config", config)["vae"]
    vae = build_vae(vae_cfg).to(device)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # ── 학습 split의 모든 연속 2시점 쌍 ────────────────────────────
    ds = ERA5PairDataset(
        normalized_path=config["data"]["normalized_path"],
        split="train",
        load_into_memory=False,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    print(f"[info] encoding {len(ds)} pairs, mode={mode}")

    stats = compute_stats(vae, loader, device, mode, seed=args.seed)
    torch.save(stats, out_path)
    print(
        f"[done] latent stats saved → {out_path}\n"
        f"       mode={stats['mode']} mu{tuple(stats['mu'].shape)} "
        f"sigma{tuple(stats['sigma'].shape)} "
        f"(σ mean={stats['sigma'].mean():.4f}) count={stats['count']}"
    )


if __name__ == "__main__":
    main()
