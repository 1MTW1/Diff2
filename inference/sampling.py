"""Latent diffusion 추론 + 불확실성 주입 (instruction_v2 §4, 중심-프레임 복원형).

관측 x_t → DDPM_past가 멤버별로 다양한 이웃 [x̂_{t-1}, x̂_{t+1}]를 생성 →
DDPM_main이 그 이웃들로부터 x̂_t를 복원. 앙상블 = {x̂_t} (중심 프레임의 불확실성).
모든 diffusion은 **latent 공간**에서 수행되고, VAE decode는 trajectory가 끝난 뒤 한다.

핵심 메커니즘 (§4.1) — **DDPM_past 전용**:
    **마지막 denoising step에서만** dual-head의 log_var로부터 latent 공간에
    `exp(0.5·log_var)·η` (η~N(0,I), latent 요소별 독립) 노이즈를 추가한다.
    이 독립 노이즈가 VAE decoder의 upsampling/conv를 거치며 공간 상관이 있는
    기상장 perturbation으로 복원된다 — 이것이 LDM 변환의 핵심 목적이다.
    DDPM_main에서는 노이즈를 주입하지 않는다 (main의 log_var는 학습 NLL 가중 전용).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from dataset.era5_dataset import ERA5NormalizedDataset
from models.dit import build_dit
from models.encoder import build_encoder
from models.latent_norm import LatentNormalizer
from models.schedule import VDMSchedule
from models.vae import build_vae


class LatentVDMSampler:
    """연속시간 VDM ancestral sampler (latent 공간).

    `inject_uncertainty=True`이면 매 step에서 dual-head log_var 노이즈를 추가한다
    (DDPM_past 전용). step 수 N은 연속시간 schedule이라 자유롭게 선택 가능하며
    `sample(num_steps=...)`로 호출별 override 할 수 있다 (past/main 분리). 단 past는
    학습 시 `past_sampling_steps_train`과 일관되게 맞춰야 한다 (§3.3 OOD).
    """

    def __init__(self, schedule: VDMSchedule, num_steps: int = 50):
        if num_steps < 1:
            raise ValueError(f"num_steps must be ≥ 1, got {num_steps}")
        self.schedule = schedule
        self.num_steps = num_steps

    def _reverse_step(
        self,
        z_t: torch.Tensor,
        eps_pred: torch.Tensor,
        t_cur: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        """VDM ancestral 역步 t_cur → t_next (t_next < t_cur).

        q(z_s | z_t, z_0)의 닫힌 형식 (VP):
            α_{t|s} = α_t/α_s,  σ²_{t|s} = σ_t² − α_{t|s}²·σ_s²
            mean = (α_{t|s}·σ_s²/σ_t²)·z_t + (α_s·σ²_{t|s}/σ_t²)·ẑ_0
            var  = σ_s²·σ²_{t|s}/σ_t²

        σ²_{t|s}는 뺄셈 대신 VDM canonical 형태로 계산한다:
            σ²_{t|s} = σ_t²·(1 − e^{γ_s−γ_t}) = −σ_t²·expm1(γ_s − γ_t)
        step이 촘촘하면 γ_s ≈ γ_t라 뺄셈 형태는 catastrophic cancellation을
        일으키지만(특히 t→0에서 σ_t²↓), expm1은 인자가 0 근처여도 정확하고
        γ_s < γ_t이므로 결과가 구조적으로 양수 → clamp 불필요.
        """
        sched = self.schedule
        a_t, s_t = sched.alpha_sigma(t_cur)        # (B,1,1,1)
        a_s, s_s = sched.alpha_sigma(t_next)
        g_t = sched.gamma(t_cur)
        g_s = sched.gamma(t_next)

        z0_hat = (z_t - s_t * eps_pred) / a_t
        a_ts = a_t / a_s
        s_t2 = s_t ** 2
        var_ts = -s_t2 * torch.expm1(g_s - g_t)

        mean = (a_ts * s_s ** 2 / s_t2) * z_t + (a_s * var_ts / s_t2) * z0_hat
        var = s_s ** 2 * var_ts / s_t2
        return mean + var.sqrt() * torch.randn_like(z_t)

    @torch.no_grad()
    def sample(
        self,
        dit: torch.nn.Module,
        cond_tokens: torch.Tensor,
        shape: tuple[int, ...],
        device: torch.device | str,
        inject_uncertainty: bool = False,
        num_steps: int | None = None,
        inject_uncertainty_mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """latent에서 N-step denoising.

        Args:
            num_steps: 이 호출의 denoising step 수. None이면 `self.num_steps`.
                       past/main에 서로 다른 step 수를 줄 때 사용한다.
            inject_uncertainty_mode: dual-head log_var 노이즈 주입 schedule.
                "all"  → 매 step (= 기존 inject_uncertainty=True).
                "last" → 마지막 denoising step (i=n_steps-1, t→0) 에서만.
                "none" → 주입 없음 (= 기존 inject_uncertainty=False).
                None (기본) 이면 boolean `inject_uncertainty` 로 매핑된다.
                명시 시 boolean 보다 우선.

        Returns:
            z0:        (B, C_z, H_z, W_z) 최종 정규화 latent ẑ_0
            log_var:   마지막 step의 dual-head log_var (진단용)
        """
        if inject_uncertainty_mode is None:
            mode = "all" if inject_uncertainty else "none"
        else:
            mode = inject_uncertainty_mode
        if mode not in ("all", "last", "none"):
            raise ValueError(
                f"inject_uncertainty_mode must be 'all'|'last'|'none', got {mode!r}"
            )

        n_steps = self.num_steps if num_steps is None else int(num_steps)
        if n_steps < 1:
            raise ValueError(f"num_steps must be ≥ 1, got {n_steps}")
        B = shape[0]
        # 연속 시점 1 → 0 (t=1 순수 노이즈, t=0 거의 clean)
        ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
        z = torch.randn(shape, device=device)
        log_var = torch.zeros_like(z)

        for i in range(n_steps):
            t_cur = ts[i].expand(B)
            t_next = ts[i + 1].expand(B)
            eps_pred, log_var = dit(z, t_cur, cond_tokens)

            if ts[i + 1].item() <= 0.0:
                # 마지막 step: 모델의 깨끗한 latent 추정 ẑ_0 = (z−σ·ε̂)/α
                a_t, s_t = self.schedule.alpha_sigma(t_cur)
                z = (z - s_t * eps_pred) / a_t
            else:
                z = self._reverse_step(z, eps_pred, t_cur, t_next)

            # §4.1 불확실성 주입 schedule
            is_last = (i == n_steps - 1)
            inject_now = (mode == "all") or (mode == "last" and is_last)
            if inject_now:
                z = z + torch.exp(0.5 * log_var) * torch.randn_like(z)

        return z, log_var


def _decode_pair(
    vae: torch.nn.Module,
    normalizer: LatentNormalizer,
    z: torch.Tensor,             # (B, 2·C_z, H_z, W_z)
) -> tuple[torch.Tensor, torch.Tensor]:
    """past latent(12ch) → 채널 split → denorm → 프레임별 decode → (x̂_{t-1}, x̂_{t+1})."""
    cz = z.shape[1] // 2
    x0 = vae.decode(normalizer.denormalize(z[:, :cz]))
    x1 = vae.decode(normalizer.denormalize(z[:, cz:]))
    return x0, x1


@torch.no_grad()
def generate_future_ensemble(
    x_t: torch.Tensor,
    encoder: torch.nn.Module,
    dit_past: torch.nn.Module,
    dit_main: torch.nn.Module,
    vae: torch.nn.Module,
    normalizer: LatentNormalizer,
    sampler: LatentVDMSampler,
    ensemble_size: int = 20,
    device: torch.device | str = "cuda",
    past_num_steps: int | None = None,
    main_num_steps: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """단일 관측 x_t에 대해 중심 프레임 x_t 앙상블 생성 (batched).

    past가 멤버별로 다양한 이웃 [x̂_{t-1}, x̂_{t+1}]를 생성(마지막 step에서만 logvar 주입)
    하고, main이 각 멤버를 x̂_t로 복원한다. 앙상블 다양성은 past의 불확실성 주입에서
    발생해 main으로 전파된다. 앙상블 차원을 batch dim에 펼쳐 past 1회 + main 1회만 호출.

    Args:
        x_t:            (1, C, H, W) 관측 중심 프레임
        past_num_steps: DDPM_past denoising step 수 (None이면 sampler 기본값)
        main_num_steps: DDPM_main denoising step 수 (None이면 sampler 기본값)
    Returns:
        ensemble: (ensemble_size, 1, C, H, W) — 복원된 x̂_t 앙상블
        diag:     진단용 dict (past/main log_var, 생성된 이웃)
    """
    B = ensemble_size
    x_t = x_t.to(device)
    _, C, H, W = x_t.shape
    cz = vae.latent_channels                                    # per-frame = 6
    H_z, W_z = normalizer.mu.shape[-2:]

    # ── [1] DDPM_past: 다양한 이웃 [x̂_{t-1}, x̂_{t+1}] 생성 ─────────
    # past condition = x_t (관측) — 모든 member 동일 → expand
    cond_past = encoder(x_t.unsqueeze(1))                       # (1, 256, D), F=1
    cond_past = cond_past.expand(B, -1, -1)                     # → (B, 256, D)
    z_past, lv_past = sampler.sample(
        dit_past, cond_past, (B, 2 * cz, H_z, W_z), device,
        inject_uncertainty_mode="last",                        # past 전용, 마지막 step만
        num_steps=past_num_steps,
    )
    x_tm1_hat, x_tp1_hat = _decode_pair(vae, normalizer, z_past)   # 각 (B, C, H, W)

    # ── [2] DDPM_main: x̂_t 복원 (불확실성 주입 없음) ──────────────
    # main condition = [x̂_{t-1}, x̂_{t+1}] — 전부 past 생성물 (teacher forcing 금지)
    cond_main = encoder(torch.stack([x_tm1_hat, x_tp1_hat], dim=1))  # (B, 512, D), F=2
    z_main, lv_main = sampler.sample(
        dit_main, cond_main, (B, cz, H_z, W_z), device,
        inject_uncertainty=False,                              # main은 주입 안 함
        num_steps=main_num_steps,
    )
    x_t_hat = vae.decode(normalizer.denormalize(z_main))       # (B, C, H, W)
    ensemble = x_t_hat.reshape(B, 1, C, H, W)                  # 복원된 x̂_t

    return ensemble, {
        "log_var_past": lv_past.detach().cpu(),
        "log_var_main": lv_main.detach().cpu(),
        "x_neighbors_gen": torch.stack(
            [x_tm1_hat, x_tp1_hat], dim=1
        ).detach().cpu(),                                      # (B, 2, C, H, W)
    }


# ─── CLI ────────────────────────────────────────────────────────────
def _load_models(config: dict, ckpt: dict, device: torch.device):
    """diffusion 체크포인트 + VAE 체크포인트 + latent 통계 로드."""
    cz = int(config["vae"]["latent_channels"])                 # per-frame = 6
    encoder = build_encoder(config).to(device)
    dit_past = build_dit(config, latent_channels=2 * cz).to(device)   # 12ch
    dit_main = build_dit(config, latent_channels=cz).to(device)       # 6ch
    encoder.load_state_dict(ckpt["encoder"])
    dit_past.load_state_dict(ckpt["dit_past"])
    dit_main.load_state_dict(ckpt["dit_main"])

    vae_ckpt = torch.load(config["vae"]["checkpoint"], map_location="cpu")
    vae = build_vae(vae_ckpt.get("config", config)["vae"]).to(device)
    vae.load_state_dict(vae_ckpt["vae"])

    normalizer = LatentNormalizer.from_file(
        config["latent_norm"]["stats_path"], map_location=device,
    ).to(device)

    for m in (encoder, dit_past, dit_main, vae):
        m.eval()
    return encoder, dit_past, dit_main, vae, normalizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="diffusion 체크포인트 (encoder/dit_past/dit_main)")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--ensemble_size", type=int, default=20)
    p.add_argument("--n_samples", type=int, default=10)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "config" in ckpt:
        config = ckpt["config"]

    encoder, dit_past, dit_main, vae, normalizer = _load_models(
        config, ckpt, device,
    )
    schedule = VDMSchedule(
        gamma_min=float(config["schedule"]["gamma_min"]),
        gamma_max=float(config["schedule"]["gamma_max"]),
    )
    past_num_steps = int(config["sampling"]["past_num_steps"])
    main_num_steps = int(config["sampling"]["main_num_steps"])
    sampler = LatentVDMSampler(schedule, num_steps=past_num_steps)

    ds = ERA5NormalizedDataset(
        normalized_path=config["data"]["normalized_path"],
        mode="inference", split=args.split, load_into_memory=False,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_samples = min(args.n_samples, len(ds))
    print(f"[info] generating {args.ensemble_size}-member ensembles "
          f"for {n_samples} samples "
          f"(past_steps={past_num_steps}, main_steps={main_num_steps})")

    results = []
    for i in tqdm(range(n_samples), desc="ensemble"):
        sample = ds[i]
        # past condition = 관측 중심 프레임 x_t
        x_t = sample["x_t"].unsqueeze(0)

        ensemble, diag = generate_future_ensemble(
            x_t, encoder, dit_past, dit_main, vae, normalizer,
            sampler, ensemble_size=args.ensemble_size, device=device,
            past_num_steps=past_num_steps, main_num_steps=main_num_steps,
        )
        torch.save(
            {
                "ensemble": ensemble.cpu(),           # (M, 1, C, H, W) 복원 x̂_t
                "x_t": x_t.cpu(),                     # 관측 GT
                "log_var_past": diag["log_var_past"],
                "log_var_main": diag["log_var_main"],
                "time_t": str(sample["time_t"]),
            },
            out_dir / f"ensemble_{i:04d}.pt",
        )
        results.append({"idx": i, "time_t": str(sample["time_t"])})

    with open(out_dir / "index.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
