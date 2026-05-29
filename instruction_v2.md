# Instruction: Diffusion² 모델의 LDM(Latent Diffusion) 변환

이 문서는 기존 픽셀 공간 Dual-DDPM 기상 앙상블 생성 모델(`Diffusion² Flow-Dependent
Ensemble Generation`)을 **Latent Diffusion Model(LDM)** 로 변환하기 위한 구현 지시서다.
Claude Code는 이 문서를 따라 코드를 작성/수정한다.

> **이 변환의 근본 동기 (반드시 이해할 것)**
> `dual-head`가 예측하는 `log_var` ℓ은 past와 main(future)에서 **역할이 다르다**:
> - **DDPM_past (과거 생성)**: 매 denoising step에서 `exp(0.5·ℓ)`만큼 노이즈를
>   **실제로 더해** 앙상블 spread(불확실성)를 만든다. 이렇게 생성된 다양한 과거가
>   main의 condition으로 들어가 future 앙상블의 다양성으로 **전파**된다.
> - **DDPM_main (미래 생성)**: ℓ은 노이즈 주입이 아니라 **heteroscedastic NLL loss에서
>   맞추기 어려운 픽셀(latent 요소)의 가중치를 낮추는** 역할만 한다.
>
> 문제는 past의 노이즈 주입에 있다. **픽셀 공간에서 픽셀별 독립 노이즈를 더하면
> 공간적으로 uncorrelated(white) noise**가 되어, 인접 격자 간 상관이 강한 실제
> 기상장과 물리적으로 맞지 않는다.
> **Latent 공간에서 latent 요소별 독립 노이즈를 더한 뒤 VAE decoder를 통과시키면,
> decoder의 upsampling/conv 구조가 그 perturbation을 인접 픽셀로 펼쳐 공간 상관이
> 있는 기상장 perturbation으로 복원**된다. 이것이 LDM 변환의 핵심 목적이다.
> 부수적으로, 향후 0.1° 고해상도 데이터로 확장 시 latent 공간이 계산적으로 유리하다.

---

## 0. 확정 설계 요약표

| 항목 | 결정 |
|---|---|
| **VAE 입력** | 2시점 × 3변수 채널 concat = `(B, 6, 64, 64)` |
| **VAE 압축** | 공간 64×64 → 16×16 (×4), 채널 `C_z = 입력채널(6) × 2 = 12`, 총 **8배 압축** |
| **VAE 학습** | diffusion과 **분리하여 개별 선학습** 후 freeze (Stage 0) |
| **VAE KL** | **약하게** (압축기 역할, 분포를 가우시안으로 강제하지 않음) |
| **Latent 정규화** | VAE 학습 후 데이터셋 전체 통계 측정 → **channel + pixelwise** 평균0/분산1 정규화. fallback으로 channelwise-only 옵션 |
| **Condition Encoder** | 기존 3D Conv 구조 유지, 항상 **2시점** 입력. 출력을 patchify(p=4)로 64×64→16×16 토큰화. **모든 stage에서 학습** (VAE만 frozen) |
| **Backbone** | U-Net → **DiT (Transformer)**. latent를 patch_size=1로 토큰화 → **256 토큰** (각 latent 픽셀 = 1 토큰) |
| **Positional encoding** | **2D sinusoidal** (learnable pos_emb 대체, 해상도 변경에 일반화 유리) |
| **Dual-head** | **유지**. 출력 `(eps_pred, log_var)`, log_var는 latent 요소별 불확실성 |
| **Noise schedule** | **VDM 형태**: 연속 t∈[0,1] → 고정 선형 log-SNR γ(t). VP. ε-prediction 유지 |
| **학습 시 t 샘플링** | **연속 Uniform[0,1]** (추론 step 수 자유, noise-level OOD 방지) |
| **Dual-DDPM 구조** | **past→main 연쇄 의존 유지**. main의 condition은 past의 **생성 샘플** (teacher forcing 금지) |
| **불확실성 주입** | **DDPM_past 샘플링에서만**: 매 denoising step에서 latent 공간에 `exp(0.5·log_var)` 노이즈 추가 → 다양한 과거 생성. **DDPM_main의 log_var는 노이즈 주입이 아니라 NLL loss 가중 역할**. decode는 trajectory 종료 후 1회만 |

---

## 1. 데이터 / 텐서 규약

### 1.1 시점 인덱스

입력 시퀀스: `x_{t-3}, x_{t-2}, x_{t-1}, x_t, x_{t+1}` (각 `(C=3, 64, 64)`, 6시간 간격)

| | DDPM_past | DDPM_main |
|---|---|---|
| condition 입력 (2시점) | `[x_{t-1}, x_{t}]` (관측) | past의 **생성 샘플** x_{t-2}와 관측된 x_{t-1} |
| 예측 target (2시점) | `[x_{t-3}, x_{t-2}]` | `[x_t, x_{t+1}]` |

> **중요 — main의 condition 구성**: main의 condition은 past가 생성한 과거 기상장에
> 의존해야 한다(연쇄 의존). 구체적으로 main은 past를 실제 sampling하여 얻은 생성물
> `x̂`을 condition으로 받는다. **학습/추론 모두 동일하게 past 생성물을 사용**하며,
> 원본 관측값을 condition으로 넣는 teacher forcing은 **금지**한다 (cond-distribution
> OOD 방지). 정확한 시점 조합(예: `[x̂_{t-2}, x_{t-1}]` 등)은 기존 코드의 시점
> 정의를 따르되, **반드시 past 생성물을 1개 이상 포함**하도록 한다.

### 1.2 핵심 텐서 기호

| 기호 | 값 | 의미 |
|---|---|---|
| `B` | — | batch size |
| `C` | 3 | 변수 수 (t, u, v) |
| `H, W` | 64, 64 | 원본 공간 해상도 |
| `H_z, W_z` | 16, 16 | latent 공간 해상도 |
| `C_z` | 12 | latent 채널 수 (= 2C × 2) |
| `target_frames` | 2 | DDPM 1회가 예측하는 프레임 수 |
| `vae_in_channels` | 6 | = target_frames × C |
| `N_tok` | 256 | DiT 토큰 수 (= H_z × W_z, patch_size=1) |
| `t` | [0,1] | 연속 diffusion 시점 |

### 1.3 공간별 흐름 요약

```
원본 target (B,6,64,64)
  ──VAE.encode──▶ latent z (B,12,16,16)
  ──정규화──────▶ ẑ = (z-μ)/σ            (μ,σ: (12,16,16) 또는 (12,1,1))
  ──diffusion───▶ DiT가 ẑ를 256토큰으로 다룸
  ──샘플링 종료──▶ ẑ_0
  ──역정규화────▶ z_0 = ẑ_0·σ + μ
  ──VAE.decode──▶ 기상장 (B,6,64,64)      (decode는 마지막에 1회)
```

---

## 2. 모듈별 구현 지시

각 모듈은 `models/` 패키지에 둔다. 기존 모듈명을 유지하되, latent 버전으로
변경되는 부분을 명시한다.

### 2.1 VAE — `models/vae.py` (신규)

**역할**: 2시점 기상장 ↔ latent 변환. diffusion과 분리되어 개별 학습된다.

- **클래스**: `WeatherVAE`
- **encoder**: `(B, 6, 64, 64)` → posterior `(μ_z, logσ²_z)` 각 `(B, 12, 16, 16)`
  - downsampling 2회 (64→32→16), conv + GroupNorm + SiLU 기반
- **reparameterize**: `z = μ_z + exp(0.5·logσ²_z)·ε`
- **decoder**: `(B, 12, 16, 16)` → `(B, 6, 64, 64)`
  - upsampling 2회 (16→32→64)
- **손실**: `L_recon + λ_kl · L_kl`
  - `L_recon`: 재구성 손실 (MSE 또는 L1 + 필요 시 gradient/perceptual 보조항)
  - `L_kl`: 표준 VAE KL. **λ_kl는 매우 작게** (config로 노출).
    - **참고값**: LDM 원논문(Rombach et al., 2022) 및 Stable Diffusion의 KL-regularized
      autoencoder는 표준정규분포를 향한 KL 페널티를 **약 1e-6**으로 가중한다. 이보다
      크면 posterior collapse(encoder가 입력을 무시하고 상수 분포 출력)가 발생하므로
      **1e-6 이하**를 권장. 기본값 `1e-6`에서 시작.
    - 목적은 latent를 가우시안으로 만드는 게 아니라 latent가 발산하지 않도록 하는
      약한 정규화일 뿐이다.
- **주의**: latent를 N(0,I)로 만들려고 KL을 키우지 말 것. 분포 정규화는 §2.2에서
  사후 통계로 처리한다.

### 2.2 Latent 정규화 — `models/latent_norm.py` (신규)

**역할**: VAE 학습 완료 후, latent의 분산을 ≈1로 맞춰 noise schedule의 SNR 가정에
정합시킨다. **분포를 가우시안으로 바꾸는 게 아니라 1·2차 모멘트만 맞춘다.**

- **통계 측정 스크립트**: `scripts/compute_latent_stats.py`
  - frozen VAE로 **전체(또는 충분히 큰) 학습 데이터**를 encode
  - `μ, σ`를 다음 두 모드로 계산:
    - **`channel_pixelwise`** (기본): `μ, σ ∈ (12, 16, 16)` — 위치별·채널별
    - **`channelwise`** (fallback): `μ, σ ∈ (12, 1, 1)` — 채널별만
  - 결과를 `latent_stats.pt`로 저장 (mode 정보 포함)
- **정규화 클래스**: `LatentNormalizer`
  - `normalize(z) = (z - μ) / σ`
  - `denormalize(ẑ) = ẑ · σ + μ`
  - 저장된 통계를 로드, `register_buffer`로 보관 (학습 대상 아님)
- **config flag**: `latent_norm.mode ∈ {channel_pixelwise, channelwise}`.
  데이터가 적어 위치별 통계가 불안정하면 channelwise로 후퇴.

> **이론적 근거 (코드 주석에 반영)**: latent ẑ는 diffusion 입장에서 픽셀 이미지
> x와 같은 "데이터"다. 가우시안일 필요가 없다. 분산만 ≈1이면 VP noise schedule이
> 의도대로 동작하고, 유한 step에서도 z_T ≈ N(0,I)가 성립한다.

### 2.3 Condition Encoder — `models/encoder.py` (수정)

**역할**: 2시점 condition 기상장을 토큰 시퀀스로 변환. **모든 stage에서 학습**.

- 기존 `TemporalEncoder`의 **3D Conv 구조 유지**. 단 입력 시점 수는 항상 **2** (고정).
  - 입력: `(B, 2, C=3, 64, 64)`
  - Conv3d + GroupNorm + SiLU → temporal pooling → `(B, C', 64, 64)`
- **patchify 추가** (`patch_size=4`):
  - `(B, C', 64, 64)` → 4×4 패치 → `(B, 16×16=256, D)` 토큰 시퀀스
  - conv-patch(`Conv2d(C', D, kernel=4, stride=4)`) 권장
- **2D sinusoidal positional encoding** 추가 (16×16 격자 기준)
- **출력**: condition 토큰 `(B, 256, D)` — DiT의 cross-modal attention에 사용
- past/main이 **동일 인스턴스 공유**. 입력 시점 수가 2로 통일되어 가변 T 처리 불필요.

### 2.4 DiT Backbone + Dual-Head — `models/dit.py` (신규, 기존 UNet 대체)

**역할**: latent target + condition 토큰을 Transformer로 처리하여 `(eps_pred, log_var)` 출력.

- **클래스**: `DualHeadDiT`
- **입력**:
  - `z_t`: noisy latent target `(B, 12, 16, 16)`
  - `t`: 연속 시점 `(B,)` ∈ [0,1]
  - `cond_tokens`: condition 토큰 `(B, 256, D)`
- **토큰화**: `z_t`를 patch_size=1로 토큰화 → `(B, 256, D)` (각 latent 픽셀 = 1 토큰)
  - linear/conv projection `(12 → D)`
  - **2D sinusoidal positional encoding** (16×16)
- **시점 임베딩**: 연속 `t` → sinusoidal embedding → MLP → AdaLN/scale-shift로 각
  Transformer block에 주입 (DiT 방식). 기존 `SinusoidalTimeEmbedding` 재사용 가능
  (단 연속 입력 받도록 조정).
- **Transformer blocks** (SD3 MM-DiT 철학 참고):
  - target 토큰과 condition 토큰을 attention으로 결합
  - 권장: 각 block에서 self-attention(target) + cross/joint-attention(target↔cond)
  - block 수, hidden dim `D`, head 수는 config로 노출 (예: depth=12, D=384, heads=6)
- **출력 head (dual-head 유지)**: 최종 토큰 → unpatchify → `(B, 2·12, 16, 16)`
  - `eps_pred = out[:, :12]` — latent 노이즈 예측
  - `log_var  = out[:, 12:]` — latent 요소별 log 분산, `[-10, 10]` clip
  - 출력 projection은 **zero-init** (학습 초기 안정성)

### 2.5 Noise Schedule (VDM) — `models/schedule.py` (신규)

**역할**: 연속 시점 t → (α, σ) 출력. VP + 고정 선형 log-SNR.

- **클래스**: `VDMSchedule`
- **log-SNR**: `γ(t) = γ_min + t · (γ_max − γ_min)` (선형, 단조)
  - `γ_min, γ_max` config 노출. 기상 latent에 맞춰 튜닝 대상 (초기값 예: γ_min=-... ,
    γ_max=... ; ImageNet 관례값에서 시작 후 per-σ loss 보고 조정).
  - 부호 규약: γ를 log-SNR로 두되, t↑일수록 노이즈↑가 되도록 일관되게 구현하고
    주석에 명시.
- **α, σ (VP)**:
  ```
  α(t)² = sigmoid(−γ(t))
  σ(t)² = sigmoid( γ(t))
  →  α(t)² + σ(t)² = 1  (VP 보장)
  ```
- **forward (noising)**: `z_t = α(t)·ẑ_0 + σ(t)·ε`,  `ε ~ N(0,I)`
- **메서드**: `alpha(t), sigma(t), gamma(t)` 모두 `(B,)` 입력 → `(B,1,1,1)` 출력
- ε-prediction과 완벽 호환 (D(x;σ) 전처리로 갈아엎지 않음).

---

## 3. 학습 파이프라인

### Stage 0 — VAE 선학습 (별도)

1. `WeatherVAE`를 재구성 + 약한 KL로 학습 (diffusion과 무관).
2. 학습 완료 후 **freeze**.
3. `scripts/compute_latent_stats.py` 실행 → `latent_stats.pt` 저장.
4. 이후 모든 diffusion 학습에서 VAE와 정규화 통계는 **고정**.

### Stage 1~3 — Diffusion 학습 (`training/train.py` 수정)

학습되는 컴포넌트: **Condition Encoder + DiT_past + DiT_main** (VAE/정규화는 frozen).

| Stage | 학습 대상 | frozen |
|---|---|---|
| 1 | Condition Encoder + DiT_past | VAE |
| 2 | DiT_main (+ Condition Encoder 계속 학습) | VAE, DiT_past |
| 3 | 전체 joint | VAE |

> Condition encoder는 입력 시점 수가 2로 통일되어 past/main 모두에서 동일 형태로
> 동작하므로, **모든 stage에서 계속 학습**한다 (stage별 freeze 분기 불필요).

### 3.1 단일 diffusion 학습 step (각 DDPM 공통)

```
1. target 2시점 → VAE.encode → z_0 → normalize → ẑ_0
2. t ~ Uniform[0, 1]                       # 연속 샘플링 (필수)
3. ε ~ N(0, I)
4. ẑ_t = α(t)·ẑ_0 + σ(t)·ε                  # VDM forward
5. cond_tokens = Encoder(condition 2시점)
6. (eps_pred, log_var) = DiT(ẑ_t, t, cond_tokens)
7. loss = heteroscedastic_NLL(eps_pred, ε, log_var)   # latent 공간에서 계산
```

- **NLL loss는 latent 공간에서** 계산한다 (픽셀 공간 아님). target은 VAE로 인코딩한
  `ẑ_0`에서 유도된 `ε`이다.
- heteroscedastic NLL 예: `0.5·(exp(−ℓ)·(eps_pred−ε)² + ℓ)` (요소별 평균).
- 필요 시 SNR 기반 loss 가중(min-SNR 등)을 옵션으로 추가 가능 (기본 off).
- **log_var의 쓰임 (past/main 공통과 차이)**:
  - **학습 시**: past·main 모두 위 heteroscedastic NLL에 동일하게 log_var를 사용한다.
  - **추론 시**: **past만** log_var로 노이즈를 주입(§4.1)하고, **main은 추론 시
    log_var를 사용하지 않는다** (학습 시 loss 가중 역할로 끝남).

### 3.2 main 학습 시 condition 생성 (연쇄 의존 — 핵심)

main을 학습할 때 condition은 **past를 실제로 sampling한 생성물**로 만든다.

```
1. past를 §4의 sampler로 실제 샘플링 → latent ẑ_past_0
2. ẑ_past_0 → denormalize → VAE.decode → 생성 기상장 x̂_past
3. main의 condition = (x̂_past의 해당 시점) + (필요한 관측 시점)  # past 생성물 포함
4. 이 condition으로 §3.1의 main 학습 step 수행
```

- **teacher forcing 금지**: main 학습에 원본 관측 과거를 condition으로 쓰지 않는다.
- **학습 비용 완화 옵션** (config):
  - `past_sampling_steps_train`: 학습 시 past 샘플링 step 수. 기본은 추론과 동일하게
    두어 cond 분포 일치. 비용이 크면 줄이되 추론 step과 큰 차이 금지.
  - `past_sample_cache`: past 생성물 캐싱 여부. 기본 off.

### 3.3 OOD 관리 체크리스트 (반드시 준수)

| OOD 종류 | 원인 | 대응 |
|---|---|---|
| noise-level OOD | 학습/추론 노이즈 레벨 불일치 | 학습 시 t ~ Uniform[0,1] 연속 샘플링 |
| cond-distribution OOD | main이 학습 때 관측, 추론 때 past 생성물 | 학습 때도 past 생성물을 cond로 사용 |
| past-sampling-step OOD | past를 학습/추론에서 다른 step으로 샘플링 | `past_sampling_steps_train` ≈ 추론 step |

---

## 4. 추론 파이프라인 (`inference/` 또는 `sampling.py`)

future 앙상블 생성 절차. **모든 diffusion은 latent 공간에서**, decode는 마지막 1회.

```
[1] past 생성 (불확실성 주입 — 앙상블 다양성의 원천)
    - past condition(관측 2시점) → Encoder → cond_tokens
    - latent에서 DiT_past로 N step denoising → ẑ_past_0
      * 매 denoising step에서 exp(0.5·log_var) 노이즈를 latent에 추가 (§4.1)
    - ẑ_past_0 → denormalize → VAE.decode → x̂_past   # main의 cond 재료

[2] future 생성 (불확실성 주입 없음)
    - main condition(past 생성물 x̂_past 포함 2시점) → Encoder → cond_tokens
    - latent에서 DiT_main으로 N step denoising → ẑ_main_0
      * 표준 VDM 역步만 수행. log_var는 추론 시 노이즈로 더하지 않는다
        (main의 log_var는 학습 시 NLL loss 가중에만 사용됨)
    - ẑ_main_0 → denormalize → VAE.decode → 최종 future 기상장

[3] 앙상블: 서로 다른 random seed로 [1]~[2] 반복
    - 다양성은 [1]의 past 불확실성 주입에서 발생하여 [2]로 전파된다
```

### 4.1 불확실성 주입 (핵심 메커니즘) — **DDPM_past 전용**

**DDPM_past 샘플링에서만**, denoising step마다 dual-head의 `log_var` ℓ로부터
**latent 공간에서** 노이즈를 추가한다. (DDPM_main에서는 수행하지 않는다.)

```
[DDPM_past] 각 denoising step:
    (eps_pred, log_var) = DiT_past(ẑ_t, t, cond_tokens)
    ẑ_{t-Δ} = VDM_reverse_step(ẑ_t, eps_pred, t)      # 표준 역步
    ẑ_{t-Δ} = ẑ_{t-Δ} + exp(0.5·log_var) · η          # η ~ N(0,I), latent별 독립
```

- 이 latent별 독립 노이즈가 **VAE decode를 거치면 공간 상관 있는 perturbation**으로
  변환된다 (이 변환이 LDM 변환의 핵심 목적).
- **DDPM_main에서는 위 노이즈 추가를 하지 않는다.** main의 `log_var`는 학습 시
  heteroscedastic NLL loss에서 맞추기 어려운 latent 요소의 가중치를 낮추는 데만 쓰인다.
- decode는 latent trajectory가 **완전히 끝난 뒤 1회만** 수행한다. 중간 step마다
  decode하지 않는다.
- step 수 N은 자유롭게 선택 가능 (연속시간 schedule). 단 품질은 N에 따라 달라지며
  (이산화 오차) 일반적으로 큰 N이 더 정확하다. past/main의 N과 학습 시
  `past_sampling_steps_train`을 일관되게 맞춘다.

---

## 5. Config 스펙 (`config/default.yaml` 확장)

```yaml
data:
  n_channels: 3            # C (t, u, v)
  spatial: [64, 64]
  target_frames: 2

vae:
  in_channels: 6           # target_frames * C
  latent_channels: 12      # C_z = 2 * in_channels
  latent_spatial: [16, 16]
  kl_weight: 1.0e-6        # 약한 KL
  # (encoder/decoder 채널 등 세부는 구현 시 정의)

latent_norm:
  mode: channel_pixelwise  # {channel_pixelwise, channelwise}
  stats_path: latent_stats.pt

encoder:                   # condition encoder
  hidden_channels: 64      # C'
  num_layers: 2
  patch_size: 4            # 64 -> 16 토큰 격자
  token_dim: 384           # D

dit:
  token_dim: 384           # D (encoder.token_dim과 일치)
  depth: 12
  num_heads: 6
  patch_size: 1            # latent 16x16 -> 256 토큰
  dropout: 0.1

schedule:                  # VDM
  type: linear_logsnr
  gamma_min: -6.0          # 튜닝 대상 (초기값 예시)
  gamma_max:  6.0          # 튜닝 대상 (초기값 예시)

dual_head:
  log_var_clip: [-10, 10]

sampling:
  num_steps: 50            # 추론 denoising step (N)
  past_sampling_steps_train: 50   # 학습 시 past 샘플링 step (추론과 일치 권장)
  past_sample_cache: false

train:
  curriculum: [past, main, joint]  # Stage 1,2,3
  loss: heteroscedastic_nll
  min_snr_weighting: false
```

---

## 6. 구현 순서 및 산출물

1. `models/vae.py` (`WeatherVAE`) + VAE 학습 스크립트
2. `scripts/compute_latent_stats.py` + `models/latent_norm.py`
3. `models/schedule.py` (`VDMSchedule`)
4. `models/encoder.py` 수정 (2시점 고정 + patchify + sinusoidal PE)
5. `models/dit.py` (`DualHeadDiT`) — 기존 `UNet` 대체
6. `training/train.py` 수정 — Stage 0~3, latent step, past 연쇄 샘플링
7. `sampling.py` — 추론 + 불확실성 주입(§4.1)
8. `config/default.yaml` 확장
9. `models/__init__.py` export 갱신

---

## 7. 최종 체크리스트 (구현 검증)

- [ ] VAE는 개별 선학습 후 freeze. KL 가중치가 작은가(가우시안 강제 아님)?
- [ ] latent 정규화는 사후 통계로 평균0/분산1만 맞추는가(분포 가우시안화 아님)?
- [ ] 정규화 mode가 channel_pixelwise / channelwise 둘 다 지원되는가?
- [ ] condition encoder 입력 시점이 항상 2이고 모든 stage에서 학습되는가?
- [ ] DiT가 latent를 256 토큰(patch_size=1)으로 다루는가? dual-head 유지?
- [ ] 2D sinusoidal PE가 condition·target 양쪽에 적용되는가?
- [ ] VDM schedule이 t∈[0,1] → (α,σ), VP(α²+σ²=1)인가? ε-pred 유지?
- [ ] 학습 시 t를 연속 Uniform[0,1]로 샘플링하는가?
- [ ] NLL loss가 **latent 공간**에서 계산되는가?
- [ ] dual-DDPM 연쇄: main의 cond가 past **생성물**인가? teacher forcing 금지 준수?
- [ ] 불확실성 주입 exp(0.5·log_var)이 **DDPM_past 추론에서만** latent 공간 매 step 추가되는가?
- [ ] **DDPM_main 추론에서는 log_var 노이즈 주입을 하지 않는가?** (main의 log_var는 학습 NLL 가중 전용)
- [ ] KL 가중치 기본값이 ~1e-6인가 (LDM/SD 관례)?
- [ ] VAE decode는 trajectory 종료 후 **1회만** 수행되는가?
- [ ] past/main sampling step과 past_sampling_steps_train이 일관되는가?
