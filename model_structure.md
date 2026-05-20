# Model Structure — Diffusion² Flow-Dependent Ensemble Generation

`models/` 패키지의 신경망 구조와 이들이 학습 파이프라인에서 조합되는 방식을 정리한
문서. 모든 모듈은 한반도 64×64 패치(변수 3개: `t`, `u`, `v`, 6시간 간격)를 다룬다.

---

## 1. 전체 개요

학습 가능한 컴포넌트는 3개이며, 모두 `models/` 패키지에 정의되어 있다.

| 컴포넌트 | 클래스 | 역할 |
|---|---|---|
| Temporal Encoder | `TemporalEncoder` | 여러 시점의 기상장 → condition feature map |
| Diffusion U-Net | `UNet` | noisy target + condition → dual-head 출력 |
| Dual-Head DDPM | `DualHeadDDPM` | U-Net 출력을 `(ε̂, log σ²)` 로 분리 |

`UNet`을 받쳐 주는 비학습/보조 모듈:

| 모듈 | 클래스 | 역할 |
|---|---|---|
| Time embedding | `SinusoidalTimeEmbedding` | diffusion timestep → 벡터 임베딩 |
| Self-attention | `SpatialSelfAttention` | feature map 위 multi-head attention |

런타임에는 **하나의 `TemporalEncoder` 인스턴스를 공유**하면서 **두 개의 `DualHeadDDPM`
인스턴스**(`model_past`, `model_main`)를 두는 dual-DDPM 구조다 (§5 참조).

```
   기상장 시퀀스                Encoder (공유)        Dual-Head DDPM (×2)
 ┌──────────────┐          ┌──────────────────┐   ┌──────────────────────┐
 │ (B,T,C,H,W)  │ ───────▶ │ Conv3d ×N        │ ▶ │ UNet  ── epŝ   (B,6,··)│
 │ 과거/현재 프레임 │          │ + temporal pool  │   │       └─ log σ² (B,6,··)│
 └──────────────┘          └──────────────────┘   └──────────────────────┘
                              cond (B,C',H,W)
```

---

## 2. 텐서 기호

| 기호 | 값 | 의미 |
|---|---|---|
| `B` | — | batch size (per-process) |
| `C` | 3 | 데이터 변수 수 (`t`, `u`, `v`) |
| `H, W` | 64, 64 | 공간 해상도 (한반도 패치) |
| `T` | 2 또는 3 | encoder 입력 프레임 수 |
| `target_channels` | `2·C` = 6 | DDPM 1회가 예측하는 프레임 2개 분량 |
| `C'` (`cond_channels`) | 64 | encoder 출력 채널 = `hidden_channels` |
| `M` | 200 | diffusion timestep 수 |

---

## 3. 모듈별 상세

### 3.1 `SinusoidalTimeEmbedding` — `models/time_embedding.py`

표준 sinusoidal positional encoding (Vaswani et al., 2017).

- **입력** `t`: `(B,)` 정수 diffusion timestep
- **출력**: `(B, dim)` — `[sin(args) ‖ cos(args)]` concat
- `dim`은 짝수여야 함 (홀수면 `ValueError`).
- 학습 파라미터 없음. `UNet`의 time-embedding MLP 맨 앞에 들어간다.

### 3.2 `SpatialSelfAttention` — `models/attention.py`

`(B, C, H, W)` feature map 위에서 동작하는 multi-head self-attention.

- **구조**: pre-norm(`GroupNorm(8)`) → 1×1 conv QKV → scaled dot-product attention →
  1×1 conv projection → **residual add**.
- `num_heads=4` (기본), `channels`는 `num_heads`로 나누어떨어져야 함.
- 공간 차원 `H·W`를 토큰 시퀀스로 펼쳐 `einsum`으로 attention 계산.
- 출력 shape = 입력 shape (`x + out`).

### 3.3 `TemporalEncoder` — `models/encoder.py`

여러 시점의 기상장을 condition feature map으로 압축한다.

- **입력** `x`: `(B, T, C, H, W)`
- **출력**: `(B, C', H, W)` — 시간축이 collapse 됨
- **구조**:
  1. `(B,T,C,H,W)` → `(B,C,T,H,W)` permute (Conv3d 입력 규약)
  2. `Conv3d(3×3×3) → GroupNorm → SiLU` 를 `num_layers`회 (기본 2회)
     - 중간 레이어 채널: `max(hidden_channels//2, in_channels)`
     - 마지막 레이어 채널: `hidden_channels`
  3. `AdaptiveAvgPool3d((1,None,None))` 로 시간축 → 1 로 축소 후 `squeeze`
- `out_channels = hidden_channels`.
- **같은 인스턴스를 `T=2`(past)와 `T=3`(main) 양쪽에 재사용** — Conv3d/pooling이
  가변 `T`에 모두 동작하므로 가능.

### 3.4 `UNet` — `models/unet.py`

Diffusion U-Net backbone. dual-head용으로 출력 채널이 입력 target 채널의 2배다.

- **`forward(x, t, cond)` 입력**
  - `x`: `(B, target_channels, H, W)` — noisy target
  - `t`: `(B,)` — diffusion timestep
  - `cond`: `(B, C', H, W)` — encoder가 만든 condition
- **출력**: `(B, out_channels, H, W)` = `(B, 2·target_channels, H, W)`

**입력 합성** — forward 시작부에서 채널 방향 concat:

```
h = concat([ x , cond , pos_emb ], dim=1)
    (B, target_channels + C' + pos_emb_channels, H, W)
```

`pos_emb`는 학습 가능한 2D 절대 위치 prior(`nn.Parameter`, `(pos_emb_channels,H,W)`)로,
고정 도메인의 climatology 위치 의존성을 모델에 주입한다. `pos_emb_channels=0`이면 비활성화.

**아키텍처 (default config 기준):**

| 구성 | 값 |
|---|---|
| `base_channels` | 64 |
| `channel_mults` | `(1, 2, 4, 8)` → level 채널 `64, 128, 256, 512` |
| `num_res_blocks` | 2 (down 레벨당; up은 `+1` = 3) |
| `attention_resolutions` | `[0, 1]` — **bottom-indexed**, 가장 깊은 2개 레벨에 attention |
| `time_emb_dim` | 256 |
| `dropout` | 0.1 |

- **Time embedding MLP**: `SinusoidalTimeEmbedding → Linear → SiLU → Linear`,
  각 `ResBlock`에 timestep 정보 주입.
- **Down path**: 레벨마다 `ResBlock ×num_res_blocks` (해당 레벨이 attention 레벨이면
  각 ResBlock 뒤 `SpatialSelfAttention`), 마지막 레벨 제외하고 `Downsample`(stride-2 conv).
  공간 해상도 `64 → 32 → 16 → 8`.
- **Middle**: `ResBlock → SpatialSelfAttention → ResBlock`.
- **Up path**: 레벨마다 `ResBlock ×(num_res_blocks+1)`, 각 ResBlock 입력에서 down path의
  skip feature를 concat. 첫 레벨 제외하고 `Upsample`(nearest interpolate + conv).
- **Output**: `GroupNorm → SiLU → Conv2d(3×3)`. 마지막 conv는 **zero-init**(가중치·bias)로
  학습 초기 안정성 확보.

보조 클래스: `ResBlock`(pre-norm residual + time-emb 주입), `Downsample`, `Upsample`,
`_DownStage`, 채널 수에 안전한 GroupNorm 헬퍼 `_gn`.

### 3.5 `DualHeadDDPM` — `models/dual_head_ddpm.py`

`UNet`을 감싸 dual-head 출력을 분리하는 얇은 wrapper.

- **`forward(x, t, cond)`** → `(eps_pred, log_var)` 튜플
  - `eps_pred`: `out[:, :target_channels]` — 예측 노이즈 ε̂
  - `log_var`: `out[:, target_channels:]` — 픽셀별 log 분산 ℓ, `[-10, 10]`로 clip
- heteroscedastic NLL loss와 sub-sampled DDPM 샘플링에서 픽셀별 불확실성으로 사용된다.

---

## 4. 모듈 의존 관계

```
DualHeadDDPM
└── UNet
    ├── SinusoidalTimeEmbedding   (time_embed MLP)
    ├── SpatialSelfAttention      (하위 2 resolution + middle)
    └── ResBlock / Downsample / Upsample

TemporalEncoder                   (독립 — UNet 외부에서 cond 생성)
```

`models/__init__.py`는 `SinusoidalTimeEmbedding`, `SpatialSelfAttention`,
`TemporalEncoder`, `UNet`, `DualHeadDDPM`를 export한다.

---

## 5. 학습 시 조합: Dual-DDPM 구조

`training/train.py::_build_models`는 다음을 만든다.

- **`encoder`**: `TemporalEncoder` 1개 (past·main 공유)
- **`model_past`**, **`model_main`**: `DualHeadDDPM` 2개 (각자 독립 `UNet` 보유)

각 `UNet`의 채널 설정:

```
in_channels  = target_channels + cond_channels   = 6 + 64  = 70
              (+ pos_emb_channels 8 → input_conv 실제 입력 78)
out_channels = 2 · target_channels               = 12      (ε̂ 6 + log σ² 6)
```

**데이터 흐름** (입력 시퀀스 `x_{t-3}, x_{t-2}, x_{t-1}, x_t, x_{t+1}`):

| | DDPM_past | DDPM_main |
|---|---|---|
| condition 입력 | `encoder([x_{t-1}, x_t])` (T=2) | `encoder([x̂_{t-3}, x̂_{t-2}, x_{t-1}])` (T=3) |
| 예측 target | `[x_{t-3}, x_{t-2}]` | `[x_t, x_{t+1}]` |
| 비고 | — | `x̂_{t-3}, x̂_{t-2}` 는 DDPM_past를 aux sampler로 샘플링한 결과 |

DDPM_main의 condition은 DDPM_past의 **샘플 출력**에 의존하므로, 추론과 같은 sampler를
학습에도 써서 `cond_main`의 train/inference 분포를 일치시킨다(OOD 방지).

**Curriculum (`training/curriculum.py`)**:

1. Stage 1 — DDPM_past만 학습
2. Stage 2 — DDPM_main만 학습 (past frozen)
3. Stage 3 — joint 학습

---

## 6. Config 매핑 (`config/default.yaml`)

| Config 키 | 사용 위치 |
|---|---|
| `data.n_channels` (3) | `C` → `target_channels = 2C` |
| `data.spatial` ([64,64]) | `UNet(spatial_size=...)` (pos_emb 활성화 시) |
| `model.encoder.{in_channels,hidden_channels,num_layers}` | `TemporalEncoder` |
| `model.unet.{base_channels,channel_mults,num_res_blocks,attention_resolutions,time_emb_dim,dropout,pos_emb_channels}` | `UNet` |
| `model.dual_head.log_var_clip` ([-10,10]) | `DualHeadDDPM` |
| `diffusion.M` (200) | diffusion timestep 범위 (`t ∈ [0, M)`) |
