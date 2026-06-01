# Model Structure — Diffusion² LDM (Latent Diffusion Ensemble)

한반도 64×64 패치 기상장(변수 `t`, `u`, `v`, 6시간 간격)에 대한
**중심-프레임 복원형 Dual-DDPM** 구조. 모든 diffusion은 VAE의 **latent 공간**에서
수행된다 (Latent Diffusion Model, LDM). 픽셀 공간 U-Net 구조는 이 버전에서 제거되었다.

---

## 0. Repo 구조

```
models/
  vae.py          WeatherVAE — 프레임별 픽셀↔latent 변환
  encoder.py      ConditionEncoder — 기상장 프레임 → condition 토큰
  dit.py          DualHeadDiT — Transformer diffusion backbone
  schedule.py     VDMSchedule — 연속시간 VP noise schedule
  latent_norm.py  LatentNormalizer — 사후 latent 통계 정규화
  pos_emb.py      sinusoidal_2d_pos_emb — 2D PE 유틸 (DiT·Encoder 공용)
  time_embedding.py SinusoidalTimeEmbedding — 연속 t → 임베딩

training/
  train_vae.py    Stage 0: VAE 학습
  train.py        Stage 1~3: diffusion 학습 (4-phase CLI)
  loss.py         heteroscedastic_nll_loss

inference/
  sampling.py     LatentVDMSampler + generate_future_ensemble

dataset/
  era5_dataset.py ERA5NormalizedDataset

config/
  default.yaml    하이퍼파라미터 (shape 결정 기준점)
```

---

## 1. 텐서 기호

| 기호 | 값 | 의미 |
|---|---|---|
| `B` | — | batch size |
| `C` | 3 | 기상 변수 수 (`t`, `u`, `v`) |
| `H, W` | 64, 64 | 픽셀 공간 해상도 |
| `C_z` | 6 | per-frame VAE latent 채널 |
| `H_z, W_z` | 16, 16 | latent 공간 해상도 (64/4) |
| `D` | 384 | Transformer token hidden dim |
| `N_tok` | 256 | latent 토큰 수 (16×16) |
| `p` | 4 | condition encoder patchify 크기 (64/4=16 격자) |
| `M` | 20 | 추론 앙상블 멤버 수 |

---

## 2. 모듈별 상세

### 2.1 `WeatherVAE` — `models/vae.py`

**역할**: 단일 기상장 프레임을 latent로 압축·복원한다. Stage 0에서 단독 학습 후
diffusion 학습 시 **frozen**으로 사용된다. time embedding 없음.

```
(B, 3, 64, 64)  ──encode──▶  (μ_z, logσ²_z), 각 (B, 6, 16, 16)
(B, 6, 16, 16)  ──decode──▶  (B, 3, 64, 64)
```

**`__init__` 구조** (`models/vae.py:L78-120`):

```python
# Encoder: 64 → 32 → 16 (ch_mult=(1,2) → 2회 downsample)
encoder = [Conv2d(3→128), ResBlock(128→128), Downsample,
                           ResBlock(128→256), Downsample, ResBlock(256→256)]
enc_norm  = GroupNorm(256)
to_posterior = Conv2d(256 → 12)   # 2·C_z = 12 (μ·logσ² 동시 출력)

# Decoder: 16 → 32 → 64 (encoder 대칭)
from_latent = Conv2d(6 → 256)
decoder     = [ResBlock(256→256), Upsample, ResBlock(256→128), Upsample]
dec_norm    = GroupNorm(128)
to_output   = Conv2d(128 → 3)
```

**`forward` shape 추적**:

```
x:   (B, 3, 64, 64)
  ↓ encoder
h:   (B, 256, 16, 16)
  ↓ enc_norm + SiLU + to_posterior
h:   (B, 12, 16, 16)   ← chunk(2, dim=1)
mu:  (B, 6, 16, 16)
lv:  (B, 6, 16, 16)    ← clamp(-30, 20)
  ↓ reparameterize: z = μ + exp(0.5·lv)·ε
z:   (B, 6, 16, 16)
  ↓ decode
recon: (B, 3, 64, 64)
```

**원리**: reparameterization trick `z = μ + exp(0.5·logσ²)·ε` (`models/vae.py:L136-137`).
KL 가중치 `λ_kl = 1e-6` — LDM/SD 관례로 매우 약하게 설정 (latent 분포 정규화는
`LatentNormalizer`가 담당, KL은 발산 방지용만).

**Loss** (`models/vae.py:L182-207`):
```
L = MSE(recon, target) + 1e-6 · KL(q||N(0,I))
KL = 0.5 · mean(μ² + σ² − logσ² − 1)     # 표준 VAE KL
```

---

### 2.2 `LatentNormalizer` — `models/latent_norm.py`

**역할**: VAE 학습 완료 후 latent의 1·2차 모멘트를 평균0/분산1로 맞춘다.
학습 파라미터 없음 (buffer만). VP noise schedule이 의도대로 동작하기 위한 전처리.

```python
ẑ = (z − μ) / σ     # normalize (models/latent_norm.py:L41-42)
z = ẑ·σ + μ          # denormalize
```

- `channel_pixelwise` 모드: `μ, σ ∈ (C_z, H_z, W_z)` — 위치·채널별 개별 통계
- `channelwise` 모드: `μ, σ ∈ (C_z, 1, 1)` — 채널별만 (데이터 부족 시 fallback)
- `compute_latent_stats.py`가 z=μ(결정론적)로 통계를 내야 정합 — 재매개변수화 통계와 혼용 시 under-dispersed.

---

### 2.3 `ConditionEncoder` — `models/encoder.py`

**역할**: 픽셀 공간 기상장 프레임을 DiT가 cross-attention으로 참조할 토큰 시퀀스로 변환.
`(B, F, C, H, W) → (B, F·N_tok, D)`.

```
DDPM_past  : F=1 (x_t)              → (B, 256, 384)
DDPM_main  : F=2 ([x̂_{t-1},x̂_{t+1}]) → (B, 512, 384)
```

**`__init__` 구조** (`models/encoder.py:L57-82`):

```python
# 프레임별 2D conv
conv = [Conv2d(3→32, k=3), GroupNorm, SiLU,
        Conv2d(32→64, k=3), GroupNorm, SiLU]    # hidden_channels=64

# Patchify: 64×64 → 16×16 토큰 격자
patchify = Conv2d(64 → 384, kernel=4, stride=4)  # (B, D, 16, 16)

# 2D sinusoidal PE (고정, 학습 없음)
pos_emb: (1, 256, 384)   # register_buffer

# time embedding (F>1일 때 슬롯 구분)
time_emb = nn.Embedding(2, 384)  # max_frames=2: 0=t-1, 1=t+1
```

**`forward` shape 추적** (F=2 경우):

```
x:     (B, 2, 3, 64, 64)
  ↓ _encode_one 각 프레임
  conv: (B, 64, 64, 64)
  patchify: (B, 384, 16, 16)
  flatten+transpose: (B, 256, 384)
  + pos_emb: (B, 256, 384)
  ↓ time_emb 주입 (f=0: t-1, f=1: t+1)
  toks[0] + te[0]: (B, 256, 384)
  toks[1] + te[1]: (B, 256, 384)
  ↓ cat(dim=1)
output: (B, 512, 384)
```

**DiT와의 인터페이스**: 토큰 수가 가변 (256 또는 512). DiT cross-attention의
key/value 길이가 달라도 `nn.MultiheadAttention`은 임의 길이를 처리하므로 호환.

---

### 2.4 `DualHeadDiT` — `models/dit.py`

**역할**: latent 공간 diffusion backbone. noisy latent + condition 토큰 + 연속 시점 t를
받아 `(ε̂, log_var)`를 출력한다.

```
z_t:         (B, C_z, 16, 16)   ← 노이즈 latent
t:           (B,) ∈ [0, 1]
cond_tokens: (B, N_cond, 384)   ← ConditionEncoder 출력
     ↓
eps_pred:    (B, C_z, 16, 16)   ← 노이즈 예측 ε̂
log_var:     (B, C_z, 16, 16)   ← latent 요소별 log 분산 (clip ±10)
```

**`__init__` 구조** (`models/dit.py:L115-171`):

```python
# 토큰화 (patch_size=1: latent 픽셀 1개 = 1 토큰)
x_embed = Conv2d(C_z → 384, kernel=1)
pos_emb: (1, 256, 384)   # 2D sinusoidal, register_buffer

# 연속 t → 임베딩 (정수 sinusoidal 재사용, ×1000 스케일)
t_embed = SinusoidalTimeEmbedding(384) → Linear → SiLU → Linear   # (B, 384)

# Transformer blocks (depth=4)
blocks = ModuleList([DiTBlock(384, heads=6, mlp_ratio=4.0)] × 4)

# 출력 projection → dual-head
final = _FinalLayer(384, out_dim=2·C_z)
```

**`DiTBlock` 구조** (`models/dit.py:L28-82`):

각 블록은 3단계로 구성된다.

1. **Self-attention** (target 토큰 간): AdaLN-zero 변조 후 `MHA(x, x, x)`
2. **Cross-attention** (target query ← condition key/value): `MHA(x, cond, cond)`
3. **MLP** (GELU, hidden=`4·D`): 채널별 피드포워드

AdaLN 변조: `t_emb → Linear(384→9·384) → 9개 분할` → 각 단계마다 `(shift, scale, gate)`.

```python
# AdaLN-zero modulation (models/dit.py:L22-25)
x_modulated = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
x_out = x + gate.unsqueeze(1) * attn_or_mlp
```

**초기화 규칙** (`models/dit.py:L156-171`): AdaLN projection과 최종 linear를 **zero-init**
→ 학습 시작 시 각 block이 identity로 동작, 깊은 네트워크의 초기 안정성 확보.

**`forward` shape 추적**:

```
z_t:   (B, C_z, H_z, W_z)
  ↓ x_embed (Conv2d k=1)
       (B, D, 16, 16)
  ↓ flatten(2).transpose(1,2) + pos_emb
x:     (B, 256, 384)
t_emb: (B, 384)             ← t·1000 → SinusoidalTimeEmbedding → MLP
  ↓ for block in blocks:     x = block(x, cond_tokens, t_emb)
x:     (B, 256, 384)
  ↓ final(x, t_emb)
       (B, 256, 2·C_z)
  ↓ transpose(1,2).reshape
out:   (B, 2·C_z, 16, 16)
eps_pred = out[:, :C_z]     (B, C_z, 16, 16)
log_var  = out[:, C_z:]     (B, C_z, 16, 16), clamp(-10, 10)
```

---

### 2.5 `VDMSchedule` — `models/schedule.py`

**역할**: 연속시간 VP noise schedule. `t∈[0,1]` → `(α(t), σ(t))` 계수 제공.

```
γ(t) = γ_min + t·(γ_max − γ_min)        # 선형 −log-SNR
α(t) = sqrt(sigmoid(−γ(t)))              # 신호 계수
σ(t) = sqrt(sigmoid( γ(t)))              # 노이즈 계수
α²(t) + σ²(t) = 1                        # VP 보장 (sigmoid 항등식)
```

config: `γ_min = -6.0`, `γ_max = 6.0` (t=0 거의 clean, t=1 거의 순수 노이즈).

**forward noising** (`models/schedule.py:L64-78`):

```python
z_t = α(t)·z_0 + σ(t)·ε,   ε ~ N(0, I)
```

---

### 2.6 `SinusoidalTimeEmbedding` — `models/time_embedding.py`

표준 sinusoidal PE. `(B,) → (B, dim)`. DualHeadDiT가 연속 t에 대해
`t × time_scale(=1000)` 를 곱한 뒤 입력한다 — 정수 timestep용 embedding을 연속값에 재사용.

---

## 3. 학습 절차 (4-phase)

### Stage 0: VAE (`training/train_vae.py`)

```
단일 기상장 프레임 (B, 3, 64, 64) → WeatherVAE → recon + (μ, logσ²)
L = MSE(recon, x) + 1e-6·KL
```

완료 후 → `outputs/vae_v2/vae_best.pt` 저장, 이후 모든 phase에서 **frozen** 로드.

이어서 `scripts/compute_latent_stats.py`로 latent 통계 계산 → `latent_stats.pt`.

---

### Stage 1~3: Diffusion (`training/train.py`)

**4-phase CLI** (`--phase {past, infer, main, joint}`):

| Phase | 학습 대상 | Loss | 입출력 |
|---|---|---|---|
| `past` | Encoder + DiT_past(12ch) | L_past = NLL(ẑ_[t-1,t+1], cond=x_t) | cond F=1 → target 12ch |
| `infer` | 추론 전용 (no grad) | — | train split 전체 M개 z_past memmap 생성 |
| `main` | Encoder + DiT_main(6ch) | L_main = NLL(ẑ_t, cond=[x̂_{t-1},x̂_{t+1}]) | cond F=2 → target 6ch |
| `joint` | Encoder + DiT_past + DiT_main | L = L_past + L_main | 동적 cond 생성 |

**단일 train step 의사코드** (past phase):

```python
# batch: {x_tm1, x_t, x_tp1}  각 (B, 3, 64, 64)
cond = encoder(x_t.unsqueeze(1))              # (B, 256, 384), F=1
z0   = encode_pair_concat(vae, norm, x_tm1, x_tp1)  # (B, 12, 16, 16)

t    = torch.rand(B)                          # t ~ Uniform[0,1]
z_t, eps = schedule.forward_noise(z0, t)      # z_t = α·z0 + σ·ε
eps_pred, log_var = dit_past(z_t, t, cond)
loss = heteroscedastic_nll_loss(eps, eps_pred, log_var)

optimizer.zero_grad(); loss.backward(); clip_grad(1.0); optimizer.step()
```

**Loss 함수** (`training/loss.py:L7-29`):

```
L = 0.5·exp(-ℓ)·||ε − ε̂||² + 0.5·ℓ
    ─────────────────────────────────
    ℓ = log_var, precision = exp(-ℓ)
```

정밀도 가중 MSE + log_var 정규화. log_var가 크면 (불확실하면) precision↓ → squared error에
낮은 가중치, 대신 ℓ 항으로 패널티 부과 — 자동 불확실성 보정.

**infer phase 핵심**: train split N개 샘플에 대해 M개 z_past를 fp16 memmap `(N, M, 12, 16, 16)` 에 저장.
multi-GPU 시 각 rank가 연속 구간 담당 (row 순서 보존).

**main phase 핵심** (`MainCondDataset`): `epoch % M` 번째 member의 z_past를 decode하여
condition으로 사용 — epoch마다 다른 과거 조건 (teacher forcing 금지).

**joint phase 핵심**: 매 batch마다 DiT_past가 **동적으로** z_past를 샘플링 (`inject_mode='last'`)
→ 실시간 condition 생성 → L = L_past + L_main.

---

## 4. 추론 — 앙상블 생성

`inference/sampling.py::generate_future_ensemble` 가 single-sample 앙상블을 생성한다.

**`LatentVDMSampler`** (`inference/sampling.py:L32-145`):

VDM ancestral sampler. 연속시간 reverse step:

```
q(z_s | z_t, z_0)의 닫힌 형식 (VP):
  ẑ_0 = (z_t − σ_t·ε̂) / α_t
  mean = (α_{t|s}·σ_s²/σ_t²)·z_t + (α_s·σ²_{t|s}/σ_t²)·ẑ_0
  var  = σ_s²·σ²_{t|s}/σ_t²
  σ²_{t|s} = −σ_t²·expm1(γ_s − γ_t)    ← expm1로 catastrophic cancellation 방지
```

**불확실성 주입** (`inject_uncertainty_mode`):
- `"last"` (기본): **마지막 denoising step에서만** `z = z + exp(0.5·log_var)·η`
- `"all"`: 매 step 주입
- `"none"`: 주입 없음 (DDPM_main 전용)

```python
# 마지막 step에서 logvar 주입 (inference/sampling.py:L140-143)
is_last = (i == n_steps - 1)
if (mode == "last" and is_last) or mode == "all":
    z = z + torch.exp(0.5 * log_var) * torch.randn_like(z)
```

**end-to-end 앙상블 생성 흐름** (ensemble_size=M):

```
x_t: (1, 3, 64, 64)  ← 관측 중심 프레임
  ↓ encoder(x_t, F=1) → expand
cond_past: (M, 256, 384)
  ↓ DiT_past (50 steps, inject_mode='last')
z_past: (M, 12, 16, 16)   ← M개의 다양한 과거 이웃 latent
  ↓ VAE.decode (채널 split, denorm)
x̂_{t-1}, x̂_{t+1}: 각 (M, 3, 64, 64)
  ↓ encoder([x̂_{t-1}, x̂_{t+1}], F=2)
cond_main: (M, 512, 384)
  ↓ DiT_main (200 steps, inject 없음)
z_main: (M, 6, 16, 16)
  ↓ VAE.decode (denorm)
x̂_t: (M, 3, 64, 64)   ← 앙상블 (M개 복원 중심 프레임)
```

앙상블 다양성은 **DDPM_past의 불확실성 주입**에서 발생하며, 생성된 이웃 `[x̂_{t-1},x̂_{t+1}]`를
통해 DDPM_main으로 전파된다. VAE decoder의 upsampling/conv가 latent 공간의 독립 노이즈를
공간 상관이 있는 기상장 perturbation으로 복원한다.

---

## 5. 모듈 간 상호작용

### 5.1 end-to-end 텐서 흐름도

```
픽셀 공간                  latent 공간 (×8 압축)         픽셀 복원
─────────────────────────────────────────────────────────────
x_t (1,3,64,64)
  │
  ├─▶ ConditionEncoder ──▶ cond_past (M, 256, D)
  │         F=1                  │
  │                     DiT_past (12ch, 50 steps)
  │                              │  inject_mode='last'
  │                     z_past (M, 12, 16, 16)
  │                              │
  │                  ┌───────────┴───────────┐
  │              norm.denorm           norm.denorm
  │              VAE.decode             VAE.decode
  │         x̂_{t-1} (M,3,64,64)  x̂_{t+1} (M,3,64,64)
  │                  └───────────┬───────────┘
  │                    ConditionEncoder F=2
  │                   cond_main (M, 512, D)
  │                              │
  │                     DiT_main (6ch, 200 steps)
  │                              │
  │                     z_main (M, 6, 16, 16)
  │                         norm.denorm
  │                          VAE.decode
  └──────────────────────── x̂_t (M,3,64,64)  ← 앙상블
```

### 5.2 인스턴스 공유 관계

| 모듈 | 공유 여부 | 비고 |
|---|---|---|
| `WeatherVAE` | **공유** (frozen) | past·main 양쪽에서 encode/decode |
| `LatentNormalizer` | **공유** (frozen) | 동일 통계로 양쪽 normalize/denormalize |
| `ConditionEncoder` | **공유** (학습) | F=1 (past cond) / F=2 (main cond) 양쪽 처리 |
| `DiT_past` | 단독 | latent_channels=12 |
| `DiT_main` | 단독 | latent_channels=6 |

### 5.3 암묵적 인터페이스 계약

- VAE encode는 항상 **결정론적** (z=μ, 노이즈 샘플링 없음) → latent 통계가 일관됨
- ConditionEncoder 출력은 **정규화 안 된** 픽셀 공간 기상장을 받음 (픽셀 역정규화는 안 함)
- DiT가 받는 latent는 항상 `LatentNormalizer.normalize` 된 ẑ (분산 ≈1)
- DiT가 예측하는 ε도 ẑ 공간의 노이즈 → VAE decode 전 반드시 `denormalize`

### 5.4 학습 시 vs 추론 시 차이

| | 학습 (joint) | 추론 |
|---|---|---|
| past condition | `encoder(x_t, F=1)` — 관측 그대로 | 동일 |
| main condition | DiT_past가 **동적 샘플링**한 x̂ | 동일 |
| inject_mode | `last` (z_past 마지막 step) | `last` |
| DiT_main log_var | NLL loss에 기여 (grad 흐름) | 진단용만 (no inject) |
| 앙상블 | 단일 배치 (B개 independent) | M개 expand-batch |

---

## 6. Config ↔ Shape 대응표

| Config 키 | 값 | 결정하는 shape |
|---|---|---|
| `data.n_channels` | 3 | `C` (픽셀 채널) |
| `data.spatial` | [64, 64] | `H, W` |
| `vae.in_channels` | 3 | VAE 입출력 채널 |
| `vae.latent_channels` | 6 | `C_z` (per-frame latent) |
| `vae.latent_spatial` | [16, 16] | `H_z, W_z` |
| `vae.base_channels` | 128 | encoder/decoder base width |
| `vae.ch_mult` | [1, 2] | downsample 횟수 (=2) |
| `encoder.hidden_channels` | 64 | conv feature 채널 |
| `encoder.patch_size` | 4 | 64→16 toknizer (N_tok=256) |
| `encoder.token_dim` | 384 | `D` (Encoder/DiT 공통) |
| `dit.token_dim` | 384 | `D` |
| `dit.depth` | 4 | DiTBlock 수 |
| `dit.num_heads` | 6 | attention head 수 |
| `schedule.gamma_min` | -6.0 | t=0 근방 SNR |
| `schedule.gamma_max` | 6.0 | t=1 근방 SNR |
| `sampling.past_num_steps` | 50 | DDPM_past denoising step |
| `sampling.main_num_steps` | 200 | DDPM_main denoising step |
| `inference.ensemble_size` | 20 | 앙상블 멤버 수 M |

---

## 7. 텐서 Shape 빠른 참조표

| 텐서 | Shape | 공간 | 비고 |
|---|---|---|---|
| 기상장 프레임 `x` | `(B, 3, 64, 64)` | 픽셀 | 정규화 후 |
| VAE μ (per-frame) | `(B, 6, 16, 16)` | latent | 결정론적 latent |
| 정규화 latent ẑ | `(B, 6, 16, 16)` | latent | LatentNormalizer 후 |
| past target latent | `(B, 12, 16, 16)` | latent | [ẑ_{t-1} ‖ ẑ_{t+1}] |
| past condition 토큰 | `(B, 256, 384)` | 토큰 | F=1 |
| main condition 토큰 | `(B, 512, 384)` | 토큰 | F=2 (concat) |
| DiT_past z_t | `(B, 12, 16, 16)` | latent | noisy, 12ch |
| DiT_main z_t | `(B, 6, 16, 16)` | latent | noisy, 6ch |
| eps_pred / log_var | `(B, C_z, 16, 16)` | latent | DiT 출력 각각 |
| infer memmap | `(N, M, 12, 16, 16)` | latent | fp16, train split |
| 앙상블 출력 | `(M, 1, 3, 64, 64)` | 픽셀 | 복원 x̂_t |
