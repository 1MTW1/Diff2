"""x̂_{t-15} 교체 실험 — DDPM_main 조건의 x̂_{t-2} 정보 기여도 검증.

DDPM_main 의 condition `[x̂_{t-2}, x_{t-1}]` 중 x̂_{t-2} 를 13 스텝 더 과거의
생성물 x̂_{t-15} 로 교체하고, 동일 main noise seed 로 짝지어 앙상블을 만든
다. 두 앙상블의 차이가 작으면 x̂_{t-2} 가 main 의 t-시점 예측에 거의 정보를
주지 않는다는 뜻이고, 차이가 크면 의미 있는 flow-dependent condition 임을
시사한다.

x̂_{t-15} 생성:
    DDPM_past 의 target 은 [x_{t'-3}, x_{t'-2}] (cond=[x_{t'-1}, x_{t'}]) 이므로
    anchor 를 t'=t-13 으로 이동 → cond=[x_{t-14}, x_{t-13}] → target=
    [x̂_{t-16}, x̂_{t-15}] 의 두 번째 프레임을 사용한다.

Paired comparison:
    baseline / alt 모두 동일 starting noise + 동일 reverse-step noise 를 main
    DDPM 에 흘리도록 main sampling 직전에 RNG state 를 캡처/복원한다. 따라서
    paired member 간 차이는 **condition 차이로부터만** 발생한다.

산출물 (test 시점 sub-sample = Mon/Wed/Fri):
    figures/  exp_xtm15_diff_sample{k}.png   — 4×4 composite (15 member diff
              + 30-member ensemble mean diff). exp5 와 동일 panel/quiver 스타일.
    stats/    exp_xtm15_diff_sample{k}.npz   — 양쪽 픽셀 앙상블, GT, diff stats.
              exp_xtm15_diff_summary.json    — 전체 시점 평균/표준편차 요약.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from dataset.denormalize import HOUR_TO_IDX
from dataset.era5_dataset import ERA5NormalizedDataset
from inference.sampling import LatentVDMSampler, _load_models
from models.schedule import VDMSchedule

from .utils import (
    TEMP_IDX, U_IDX, V_IDX, ensure_dir, load_norm_stats, set_plot_defaults,
)


# ── plot 상수 (exp5 와 동일) ─────────────────────────────────────────
_QUIVER_STRIDE = 8
_QUIVER_SCALE = 400.0
_QUIVER_CLIP_MAG = 60.0
_QUIVER_KEY_MAGNITUDE = 20.0

_N_MEMBER_PANELS = 14           # GT(0) + 14 member diff(1–14) + mean diff(15)
_DEFAULT_N_MEMBERS = 30         # ensemble mean = 30 멤버 평균
_LAG_FRAMES = 13                # x̂_{t-15} 를 만들기 위한 past anchor shift


def _denorm_field(x_norm: np.ndarray, time_t, mean, std) -> np.ndarray:
    hour = pd.Timestamp(time_t).hour
    h_idx = HOUR_TO_IDX[int(hour)]
    mu = mean[h_idx].cpu().numpy()
    sig = std[h_idx].cpu().numpy()
    return x_norm * sig + mu


def _scale_only(x_norm: np.ndarray, time_t, std) -> np.ndarray:
    """차이 / spread 처럼 mean shift 가 무효인 양에 σ 만 곱한다."""
    hour = pd.Timestamp(time_t).hour
    h_idx = HOUR_TO_IDX[int(hour)]
    sig = std[h_idx].cpu().numpy()
    return x_norm * sig


def _clip_wind(u: np.ndarray, v: np.ndarray, max_mag: float):
    mag = np.sqrt(u * u + v * v)
    scale = np.minimum(1.0, max_mag / (mag + 1e-9))
    return u * scale, v * scale


def _plot_one_panel(ax, temp, u, v, vmin, vmax, title, cmap="RdBu_r"):
    im = ax.pcolormesh(temp, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    H, W = temp.shape
    ys, xs = np.mgrid[0:H, 0:W]
    s = _QUIVER_STRIDE
    u_c, v_c = _clip_wind(u, v, _QUIVER_CLIP_MAG)
    q = ax.quiver(
        xs[::s, ::s], ys[::s, ::s],
        u_c[::s, ::s], v_c[::s, ::s],
        color="k",
        scale=_QUIVER_SCALE, scale_units="width",
        width=0.0035, headwidth=4, headlength=5,
        pivot="middle",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return im, q


def _save_diff_composite(
    gt: np.ndarray,             # (C, H, W) — 물리 단위 GT
    baseline_pix: np.ndarray,   # (N, C, H, W) — 물리 단위
    alt_pix: np.ndarray,        # (N, C, H, W) — 물리 단위
    time_t,
    out_path: Path,
) -> None:
    """exp5 _save_composite 와 동일 layout:
        panel 0     : GT (절대값, abs colorbar)
        panel 1..14 : baseline_k − alt_k (paired member diff, diff colorbar)
        panel 15    : ensemble mean diff = mean(baseline) − mean(alt) (diff colorbar)
    """
    N = baseline_pix.shape[0]
    if N < _N_MEMBER_PANELS:
        raise ValueError(
            f"need at least {_N_MEMBER_PANELS} members, got {N}"
        )
    member_diff = baseline_pix[:_N_MEMBER_PANELS] - alt_pix[:_N_MEMBER_PANELS]
    mean_diff = baseline_pix.mean(0) - alt_pix.mean(0)

    fig, axes = plt.subplots(4, 4, figsize=(17, 16))
    flat = axes.flatten()

    # ── 색범위 (exp5 와 동일 방식) ─────────────────────────────
    abs_pool = gt[TEMP_IDX].flatten()
    abs_vmin = float(np.percentile(abs_pool, 1))
    abs_vmax = float(np.percentile(abs_pool, 99))

    diff_pool = np.concatenate(
        [member_diff[:, TEMP_IDX].flatten(), mean_diff[TEMP_IDX].flatten()]
    )
    diff_max = float(np.percentile(np.abs(diff_pool), 99))
    diff_max = max(diff_max, 1e-6)

    # ── panel 0: GT ──────────────────────────────────────────────
    im_abs, last_q = _plot_one_panel(
        flat[0],
        gt[TEMP_IDX], gt[U_IDX], gt[V_IDX],
        abs_vmin, abs_vmax, "Ground Truth",
    )
    # ── panel 1..14: baseline_k − alt_k (paired) ─────────────────
    im_diff = None
    for k in range(_N_MEMBER_PANELS):
        d = member_diff[k]
        im_diff, last_q = _plot_one_panel(
            flat[1 + k],
            d[TEMP_IDX], d[U_IDX], d[V_IDX],
            -diff_max, diff_max,
            f"Member {k + 1}: base − alt",
        )
    # ── panel 15: ensemble mean diff ─────────────────────────────
    _, last_q = _plot_one_panel(
        flat[15],
        mean_diff[TEMP_IDX], mean_diff[U_IDX], mean_diff[V_IDX],
        -diff_max, diff_max,
        f"Ensemble mean ({N}): base − alt",
    )

    fig.suptitle(
        f"x̂_{{t-2}} vs x̂_{{t-15}} composite — "
        f"{pd.Timestamp(time_t).strftime('%Y-%m-%d %H:%M')}  "
        f"(panels 1–15: baseline_k − alt_k paired by main RNG; panel 16: ensemble mean diff)",
        y=0.98, fontsize=12,
    )

    # exp5 와 동일한 수동 레이아웃 + 2 colorbar
    fig.subplots_adjust(
        left=0.03, right=0.86, top=0.94, bottom=0.07,
        hspace=0.18, wspace=0.05,
    )

    cax_abs = fig.add_axes([0.88, 0.55, 0.020, 0.34])
    cb_abs = fig.colorbar(im_abs, cax=cax_abs)
    cb_abs.set_label("absolute T (K)  — Ground Truth")

    cax_diff = fig.add_axes([0.88, 0.12, 0.020, 0.34])
    cb_diff = fig.colorbar(im_diff, cax=cax_diff)
    cb_diff.set_label("baseline − alt T diff (K)")

    if last_q is not None:
        axes[0, 0].quiverkey(
            last_q, 0.04, 0.02, _QUIVER_KEY_MAGNITUDE,
            f"{_QUIVER_KEY_MAGNITUDE:.0f} m/s  (cap {_QUIVER_CLIP_MAG:.0f})",
            labelpos="E", coordinates="figure",
        )

    fig.savefig(out_path)
    plt.close(fig)


# ── inference ───────────────────────────────────────────────────────
def _select_mon_wed_fri_indices(times: np.ndarray) -> np.ndarray:
    dow = pd.DatetimeIndex(times).dayofweek.to_numpy()
    keep = np.where((dow == 0) | (dow == 2) | (dow == 4))[0]
    return keep


@torch.no_grad()
def _decode_pixel(
    z: torch.Tensor, vae, normalizer, frame_idx: int = 0,
) -> torch.Tensor:
    """latent (B, C_z, H_z, W_z) → 픽셀 frame_idx (B, C, H, W).

    frame_idx=0 → 첫 번째 프레임 (DDPM_main 의 경우 x_t).
    """
    B = z.shape[0]
    x = vae.decode(normalizer.denormalize(z))                # (B, 2C, H, W)
    C2 = x.shape[1]
    C = C2 // 2
    x = x.reshape(B, 2, C, x.shape[-2], x.shape[-1])
    return x[:, frame_idx]


@torch.no_grad()
def _generate_paired_ensemble(
    *,
    x_tm14: torch.Tensor, x_tm13: torch.Tensor,
    x_tm1: torch.Tensor, x_t: torch.Tensor,
    encoder, dit_past, dit_main, vae, normalizer,
    sampler: LatentVDMSampler,
    n_members: int,
    past_num_steps: int | None,
    main_num_steps: int | None,
    inject_uncertainty_mode: str = "all",
    device: torch.device,
) -> dict:
    """단일 시점에 대해 baseline / alt 두 main 앙상블 생성 (paired).

    baseline:  past cond=[x_{t-1}, x_t]      → x̂_{t-2},  main cond=[x̂_{t-2}, x_{t-1}]
    alt:       past cond=[x_{t-14}, x_{t-13}] → x̂_{t-15}, main cond=[x̂_{t-15}, x_{t-1}]

    main DDPM 직전에 CUDA/CPU RNG state 를 저장/복원하여 member k 사이의 main
    noise 가 동일해지도록 한다 → 차이의 출처가 condition 으로 한정된다.
    """
    B = n_members
    x_tm14 = x_tm14.to(device)
    x_tm13 = x_tm13.to(device)
    x_tm1 = x_tm1.to(device)
    x_t = x_t.to(device)
    _, C, H, W = x_tm1.shape
    C_z = vae.latent_channels
    H_z, W_z = normalizer.mu.shape[-2:]

    # ── [1] past (baseline anchor=t) ─────────────────────────────
    cond_past_b = encoder(torch.stack([x_tm1, x_t], dim=1))
    cond_past_b = cond_past_b.expand(B, -1, -1)
    z_past_b, _ = sampler.sample(
        dit_past, cond_past_b, (B, C_z, H_z, W_z), device,
        inject_uncertainty_mode=inject_uncertainty_mode, num_steps=past_num_steps,
    )
    x_past_b = vae.decode(normalizer.denormalize(z_past_b))
    x_past_b = x_past_b.reshape(B, 2, C, H, W)
    x_tm2_hat = x_past_b[:, 1]                                  # (B, C, H, W)

    # ── [1'] past (alt anchor=t-13) ──────────────────────────────
    cond_past_a = encoder(torch.stack([x_tm14, x_tm13], dim=1))
    cond_past_a = cond_past_a.expand(B, -1, -1)
    z_past_a, _ = sampler.sample(
        dit_past, cond_past_a, (B, C_z, H_z, W_z), device,
        inject_uncertainty_mode=inject_uncertainty_mode, num_steps=past_num_steps,
    )
    x_past_a = vae.decode(normalizer.denormalize(z_past_a))
    x_past_a = x_past_a.reshape(B, 2, C, H, W)
    x_tm15_hat = x_past_a[:, 1]                                 # (B, C, H, W)

    # ── [2] main (baseline) — RNG state 저장 ─────────────────────
    cpu_state = torch.random.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state(device)
        if device.type == "cuda" else None
    )
    x_tm1_B = x_tm1.expand(B, -1, -1, -1)
    cond_main_b = encoder(torch.stack([x_tm2_hat, x_tm1_B], dim=1))
    z_main_b, _ = sampler.sample(
        dit_main, cond_main_b, (B, C_z, H_z, W_z), device,
        inject_uncertainty=False, num_steps=main_num_steps,
    )

    # ── [2'] main (alt) — 동일 RNG state 복원 후 sampling ──────────
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)
    cond_main_a = encoder(torch.stack([x_tm15_hat, x_tm1_B], dim=1))
    z_main_a, _ = sampler.sample(
        dit_main, cond_main_a, (B, C_z, H_z, W_z), device,
        inject_uncertainty=False, num_steps=main_num_steps,
    )

    # ── decode main → x_t 프레임 ──────────────────────────────────
    x_main_b = _decode_pixel(z_main_b, vae, normalizer, frame_idx=0)
    x_main_a = _decode_pixel(z_main_a, vae, normalizer, frame_idx=0)

    return {
        "baseline_pixel": x_main_b.detach().cpu(),  # (B, C, H, W) — 정규화
        "alt_pixel":      x_main_a.detach().cpu(),
        "x_tm2_hat":      x_tm2_hat.detach().cpu(),
        "x_tm15_hat":     x_tm15_hat.detach().cpu(),
    }


# ── statistics ───────────────────────────────────────────────────────
def _compute_stats(
    baseline_pix: np.ndarray,    # (N, C, H, W) 물리 단위
    alt_pix: np.ndarray,         # (N, C, H, W)
) -> dict:
    """per-variable 차이 통계 (물리 단위)."""
    var_names = ("t", "u", "v")
    mean_b = baseline_pix.mean(0)                        # (C, H, W)
    mean_a = alt_pix.mean(0)
    diff_mean = mean_b - mean_a                          # ensemble mean 차이
    member_diff = baseline_pix - alt_pix                 # (N, C, H, W) paired

    spread_b = baseline_pix.std(0, ddof=1)               # (C, H, W)
    spread_a = alt_pix.std(0, ddof=1)

    out: dict = {}
    for i, name in enumerate(var_names):
        d_mean = diff_mean[i]
        d_member = member_diff[:, i]
        out[name] = {
            # ensemble mean 간 차이
            "mean_diff_rmse": float(np.sqrt((d_mean ** 2).mean())),
            "mean_diff_mae":  float(np.abs(d_mean).mean()),
            "mean_diff_max":  float(np.abs(d_mean).max()),
            # member-wise paired diff (baseline_k − alt_k)
            "member_diff_rmse": float(np.sqrt((d_member ** 2).mean())),
            "member_diff_mae":  float(np.abs(d_member).mean()),
            # ensemble spread 변화
            "spread_baseline_mean": float(spread_b[i].mean()),
            "spread_alt_mean":      float(spread_a[i].mean()),
            "spread_ratio":         float(
                spread_a[i].mean() / (spread_b[i].mean() + 1e-12)
            ),
        }
    return out


# ── main runner ──────────────────────────────────────────────────────
def run_experiment(
    *,
    config_path: str,
    checkpoint_path: str,
    figures_dir: str,
    stats_dir: str,
    n_members: int,
    past_num_steps: int | None,
    main_num_steps: int | None,
    sub_sample: bool,
    limit: int | None,
    seed: int,
    inject_uncertainty_mode: str = "all",
) -> None:
    set_plot_defaults()
    fig_dir = ensure_dir(figures_dir)
    st_dir = ensure_dir(stats_dir)

    with open(config_path) as f:
        config = yaml.safe_load(f)

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
    samp_cfg = config["sampling"]
    default_steps = int(samp_cfg.get("num_steps", 50))
    if past_num_steps is None:
        past_num_steps = int(samp_cfg.get("past_num_steps", default_steps))
    if main_num_steps is None:
        main_num_steps = int(samp_cfg.get("main_num_steps", default_steps))
    sampler = LatentVDMSampler(schedule, num_steps=past_num_steps)
    mean_stats, std_stats = load_norm_stats(config["data"]["stats_path"])

    ds = ERA5NormalizedDataset(
        normalized_path=config["data"]["normalized_path"],
        mode="train",            # x_tp1 + 과거 frame 접근 가능
        split="test",
        load_into_memory=False,
    )

    # ── 인덱스 선택: Mon/Wed/Fri 또는 전체, 그리고 t-14 접근 가능 ───
    if sub_sample:
        keep_abs = _select_mon_wed_fri_indices(ds.times)
    else:
        keep_abs = np.arange(len(ds.times))
    # mode='train' 의 valid 범위 + abs_idx - 14 ≥ 0
    min_abs = max(ds.valid_start, _LAG_FRAMES + 1)
    keep_abs = keep_abs[(keep_abs >= min_abs) & (keep_abs < ds.valid_end)]
    keep_rel = keep_abs - ds.valid_start

    if limit is not None:
        keep_rel = keep_rel[:limit]
        keep_abs = keep_abs[:limit]

    print(f"[info] checkpoint={checkpoint_path}")
    print(f"[info] n_members={n_members}, past_steps={past_num_steps}, "
          f"main_steps={main_num_steps}, past_inject={inject_uncertainty_mode!r}")
    print(f"[info] evaluating {len(keep_rel)} timesteps")

    torch.manual_seed(seed)
    all_stats: list[dict] = []

    pbar = tqdm(zip(keep_rel.tolist(), keep_abs.tolist()),
                total=len(keep_rel), desc="x̂_{t-15} diff")
    for k, (idx, abs_idx) in enumerate(pbar):
        sample = ds[int(idx)]
        x_tm1 = sample["x_tm1"].unsqueeze(0)        # (1, C, H, W)
        x_t = sample["x_t"].unsqueeze(0)
        time_t = sample["time_t"]
        # 추가 frame: x_{t-14}, x_{t-13} (raw 정규화 격자)
        x_tm14 = torch.from_numpy(ds._get_frame(int(abs_idx) - 14)).unsqueeze(0)
        x_tm13 = torch.from_numpy(ds._get_frame(int(abs_idx) - 13)).unsqueeze(0)

        cache = _generate_paired_ensemble(
            x_tm14=x_tm14, x_tm13=x_tm13, x_tm1=x_tm1, x_t=x_t,
            encoder=encoder, dit_past=dit_past, dit_main=dit_main,
            vae=vae, normalizer=normalizer, sampler=sampler,
            n_members=n_members,
            past_num_steps=past_num_steps,
            main_num_steps=main_num_steps,
            inject_uncertainty_mode=inject_uncertainty_mode,
            device=device,
        )
        base_norm = cache["baseline_pixel"].numpy()     # (N, C, H, W)
        alt_norm  = cache["alt_pixel"].numpy()
        gt_norm   = x_t[0].numpy()                       # (C, H, W) — 정규화 GT
        # 같은 t 시점 → 동일 hour 통계로 역정규화
        base_phys = _denorm_field(base_norm, time_t, mean_stats, std_stats)
        alt_phys  = _denorm_field(alt_norm,  time_t, mean_stats, std_stats)
        gt_phys   = _denorm_field(gt_norm,   time_t, mean_stats, std_stats)

        # plot (exp5 composite layout: GT + 14 paired diff + ensemble mean diff)
        _save_diff_composite(
            gt_phys, base_phys, alt_phys, time_t,
            fig_dir / f"exp_xtm15_diff_sample{k:04d}.png",
        )

        # per-sample stats
        stats = _compute_stats(base_phys, alt_phys)
        stats_record = {
            "idx_rel": int(idx),
            "idx_abs": int(abs_idx),
            "time_t": str(time_t),
            "n_members": int(n_members),
            **{f"{v}__{m}": stats[v][m] for v in stats for m in stats[v]},
        }
        all_stats.append(stats_record)

        # per-sample npz: 양쪽 픽셀 앙상블 (정규화) + 물리 차이 stats
        np.savez(
            st_dir / f"exp_xtm15_diff_sample{k:04d}.npz",
            baseline_pixel=base_norm.astype(np.float32),
            alt_pixel=alt_norm.astype(np.float32),
            x_tm2_hat=cache["x_tm2_hat"].numpy().astype(np.float32),
            x_tm15_hat=cache["x_tm15_hat"].numpy().astype(np.float32),
            x_t_true_pixel=x_t[0].numpy().astype(np.float32),  # 정규화 GT
            time_t=np.array(str(time_t)),
            stats=np.array(json.dumps(stats)),
        )

    # ── aggregate summary ────────────────────────────────────────
    df = pd.DataFrame(all_stats)
    df.to_csv(st_dir / "exp_xtm15_diff_per_sample.csv", index=False)

    summary: dict = {
        "n_samples": int(len(df)),
        "n_members": int(n_members),
        "past_num_steps": int(past_num_steps),
        "main_num_steps": int(main_num_steps),
        "inject_uncertainty_mode": inject_uncertainty_mode,
        "lag_frames": int(_LAG_FRAMES),
        "per_variable": {},
    }
    for var in ("t", "u", "v"):
        var_block: dict = {}
        for metric in (
            "mean_diff_rmse", "mean_diff_mae", "mean_diff_max",
            "member_diff_rmse", "member_diff_mae",
            "spread_baseline_mean", "spread_alt_mean", "spread_ratio",
        ):
            col = f"{var}__{metric}"
            var_block[metric] = {
                "mean": float(df[col].mean()),
                "std":  float(df[col].std(ddof=1) if len(df) > 1 else 0.0),
                "min":  float(df[col].min()),
                "max":  float(df[col].max()),
            }
        summary["per_variable"][var] = var_block

    with open(st_dir / "exp_xtm15_diff_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] {len(df)} samples → figs={fig_dir}, stats={st_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="x̂_{t-2} vs x̂_{t-15} 교체 실험"
    )
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--checkpoint", required=True,
                   help="diffusion 체크포인트 (encoder/dit_past/dit_main)")
    p.add_argument("--figures_dir", default="outputs/figures_dit/xtm15_diff")
    p.add_argument("--stats_dir", default="outputs/metrics_dit/xtm15_diff")
    p.add_argument("--n_members", type=int, default=_DEFAULT_N_MEMBERS)
    p.add_argument("--past_num_steps", type=int, default=None)
    p.add_argument("--main_num_steps", type=int, default=None)
    p.add_argument("--no_subsample", action="store_true",
                   help="설정 시 모든 test 시점 사용 (기본: Mon/Wed/Fri)")
    p.add_argument("--limit", type=int, default=5,
                   help="처음 N 시점만 처리. 기본 5 (실험은 소규모 정성 비교)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inject_uncertainty_mode", default="all",
                   choices=["all", "last", "none"],
                   help="DDPM_past dual-head log_var 주입 schedule "
                        "('all' 매 step / 'last' 마지막 step만 / 'none').")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_experiment(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        figures_dir=args.figures_dir,
        stats_dir=args.stats_dir,
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
