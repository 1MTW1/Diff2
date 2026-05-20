"""DDPM / DDIM sampler (instruction.md §4.9).

DDPMSampler: 본 task의 dual-head reverse sampling (Eq. 30) 그대로 구현.
DDIMSampler: deterministic/η>0 stochastic 모두 지원, 마지막 step의 ℓ만 반환.
"""
from __future__ import annotations

import torch

from training.schedule import LinearNoiseSchedule


class DDPMSampler:
    """Reverse-chain DDPM sampler with dual-head learned variance.

    `num_inference_steps`:
        - None 또는 M 이상: full M-step chain (각 step에서 n = m-1).
        - K < M: 균등 sub-sampled K step. 각 step에서 일반화된 Eq. 9/10/30
          (m, n) pair로 reverse posterior를 닫힌 형식 계산.

    Sub-sampled DDPM은 단순 DDIM과 달리 매 step마다 학습된 픽셀별 log_var를
    사용해 noise를 주입함 → flow-dependent spread 보존. 학습 시 aux sampler와
    추론용 sampler를 같은 분포로 맞추기 위한 권장 모드.
    """

    def __init__(
        self,
        schedule: LinearNoiseSchedule,
        num_inference_steps: int | None = None,
        use_learned_variance: bool = True,
    ):
        """
        Args:
            use_learned_variance:
                True  → reverse step의 noise = learned ℓ + scheduler baseline.
                False → scheduler baseline noise만 사용 (학습된 ℓ는 무시).
                ℓ가 다운스트림 학습 분포에 일관되지 않을 때 (예: main DDPM의
                자기 자신 reverse chain) calibration용으로 False 권장.
                ell_final은 어느 경우든 진단용으로 반환됨.
        """
        self.schedule = schedule
        self.use_learned_variance = use_learned_variance
        M = schedule.M

        if num_inference_steps is None or num_inference_steps >= M:
            self.timesteps = list(range(M - 1, -1, -1))
        else:
            step = max(M // num_inference_steps, 1)
            ts = list(range(0, M, step))
            if ts[-1] != M - 1:
                ts.append(M - 1)
            self.timesteps = list(reversed(ts))   # 큰 → 작은

    def _step_reverse(
        self,
        x: torch.Tensor,
        eps_pred: torch.Tensor,
        step_m: int,
        step_n: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reverse step (m → n; n < m, n=-1이면 x_0). Returns (mu, sched_var).

        Eq. 9/10 일반화: q(x_n | x_m, x_0) 의 닫힌 형식.
        """
        sched = self.schedule

        sigma_m_sq = sched.sigma_sq[step_m]
        alpha_m = sched.sqrt_alphas_cumprod[step_m]
        sigma_m = torch.sqrt(sigma_m_sq)

        if step_n < 0:
            # 마지막 step: x_0 prediction, noise 안 더함
            mu = (x - sigma_m * eps_pred) / alpha_m
            return mu, None

        sigma_n_sq = sched.sigma_sq[step_n]
        sqrt_alpha_n = sched.sqrt_alphas_cumprod[step_n]
        alpha_mn = torch.sqrt(
            sched.alphas_cumprod[step_m] / sched.alphas_cumprod[step_n]
        )
        sigma_mn_sq = sigma_m_sq - alpha_mn ** 2 * sigma_n_sq

        # μ = (α_{m|n} σ_n² / σ_m²) x_m + (σ_{m|n}² √ᾱ_n / σ_m²) x_0_hat
        term1 = (alpha_mn * sigma_n_sq / sigma_m_sq) * x
        x0_hat = (x - sigma_m * eps_pred) / alpha_m
        term2 = (sigma_mn_sq * sqrt_alpha_n / sigma_m_sq) * x0_hat
        mu = term1 + term2

        # scheduler baseline variance: σ_n² σ_{m|n}² / σ_m²
        sched_var = sigma_n_sq * sigma_mn_sq / sigma_m_sq
        return mu, sched_var

    @torch.no_grad()
    def sample(
        self,
        model,
        cond: torch.Tensor,
        shape: tuple[int, ...],
        device: torch.device | str,
        return_trajectory: bool = False,
    ):
        """
        Returns:
            x_0:        (B, ...) 최종 sample
            trajectory: list[tensor] or None
            ell_final:  대표 ℓ_θ (마지막에서 두 번째 step 또는 마지막)
        """
        B = shape[0]
        x = torch.randn(shape, device=device)
        trajectory = [x.clone()] if return_trajectory else None
        ell_final = None

        T = len(self.timesteps)
        for i, step_m in enumerate(self.timesteps):
            step_n = self.timesteps[i + 1] if i + 1 < T else -1

            m_tensor = torch.full(
                (B,), step_m, device=device, dtype=torch.long,
            )
            eps_pred, log_var = model(x, m_tensor, cond)

            mu, sched_var = self._step_reverse(x, eps_pred, step_m, step_n)

            if step_n < 0:
                x = mu
                ell_final = log_var
            else:
                # Scheduler baseline noise (m → n 일반화)
                eps_s = torch.randn_like(x)
                sched_noise = torch.sqrt(sched_var) * eps_s

                if self.use_learned_variance:
                    eps_l = torch.randn_like(x)
                    learned_noise = torch.exp(0.5 * log_var) * eps_l
                    x = mu + learned_noise + sched_noise
                else:
                    x = mu + sched_noise

                if i == T - 2:
                    # 마지막에서 두 번째 step의 ℓ가 대표값 (진단용)
                    ell_final = log_var

            if return_trajectory:
                trajectory.append(x.clone())

        return x, trajectory, ell_final


class DDIMSampler:
    """DDIM sampler (Song et al., 2020). 추론 가속용.

    η=0: deterministic. η>0: stochastic (η=1이면 거의 DDPM과 동일).
    Dual-head의 ℓ는 마지막 step의 값을 반환 (diagnostics용).
    """

    def __init__(
        self,
        schedule: LinearNoiseSchedule,
        num_inference_steps: int = 20,
        eta: float = 0.0,
    ):
        self.schedule = schedule
        self.num_inference_steps = num_inference_steps
        self.eta = eta

        # 균등 sub-sampling: M-1, M-1-step, ..., 0
        M = schedule.M
        step = max(M // num_inference_steps, 1)
        ts = list(range(0, M, step))
        if ts[-1] != M - 1:
            ts.append(M - 1)
        # 큰 → 작은 순으로 reverse
        self.timesteps = list(reversed(ts))

    @torch.no_grad()
    def sample(
        self,
        model,
        cond: torch.Tensor,
        shape: tuple[int, ...],
        device: torch.device | str,
        return_trajectory: bool = False,
    ):
        B = shape[0]
        x = torch.randn(shape, device=device)
        trajectory = [x.clone()] if return_trajectory else None
        ell_final = None

        alphas_cumprod = self.schedule.alphas_cumprod   # (M,)

        for i, step in enumerate(self.timesteps):
            m = torch.full((B,), step, device=device, dtype=torch.long)
            eps_pred, log_var = model(x, m, cond)
            ell_final = log_var  # 매 step 갱신 → 마지막 step의 ℓ 유지

            a_m = alphas_cumprod[step]
            sqrt_a_m = torch.sqrt(a_m)
            sqrt_om_a_m = torch.sqrt(1.0 - a_m)

            # x_0 prediction
            x0_pred = (x - sqrt_om_a_m * eps_pred) / sqrt_a_m

            # next 인덱스
            if i + 1 < len(self.timesteps):
                next_step = self.timesteps[i + 1]
                a_n = alphas_cumprod[next_step]
            else:
                # 마지막 → x_0
                a_n = torch.ones_like(a_m)

            # DDIM σ
            # σ = η * sqrt((1-ᾱ_n)/(1-ᾱ_m)) * sqrt(1 - ᾱ_m/ᾱ_n)
            sigma_term = torch.zeros_like(a_m)
            if self.eta > 0 and (1.0 - a_n).item() > 0:
                sigma_term = (
                    self.eta
                    * torch.sqrt((1.0 - a_n) / (1.0 - a_m))
                    * torch.sqrt(1.0 - a_m / a_n)
                )

            # direction
            dir_coef = torch.sqrt(
                torch.clamp(1.0 - a_n - sigma_term ** 2, min=0.0)
            )

            noise = (
                sigma_term * torch.randn_like(x)
                if self.eta > 0 else torch.zeros_like(x)
            )

            x = torch.sqrt(a_n) * x0_pred + dir_coef * eps_pred + noise

            if return_trajectory:
                trajectory.append(x.clone())

        return x, trajectory, ell_final
