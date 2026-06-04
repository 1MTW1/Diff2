# Diffusion² LDM — Climatology-Anomaly 전환 구현 명세서 (v4)

> 이 문서는 Claude Code가 구현해야 할 작업을 정리한 것이다.
> 기존 코드베이스(`models/`, `training/`, `inference/`, `dataset/`, `config/`)를
> 먼저 읽고, 아래 명세에 맞춰 **기존 함수 시그니처·경로·텐서 규약에 맞물리도록** 구현한다.
> 명세에 적힌 변수명/경로는 의도를 나타내며, 실제 코드의 네이밍 컨벤션을 따른다.
>
> v4 변경점: 데이터를 daily mean 집계 대신 **매일 00 UTC 스냅샷만 사용** (SEEDS 방식).
> daily_mean 스크립트 제거, 태스크는 "00 UTC 스냅샷 anomaly 예측"으로 재정의.
> v3 변경점(유지): climatology 를 "doy별 raw 계산 → 시계열 ±7일 smoothing" 2단계로 계산;
> climatology condition 정규화는 전체 데이터의 pixelwise mean/std (위치별, doy무관).

---

## 0. 전체 목표와 핵심 아이디어

### 현재 문제
- DDPM_main 앙상블 평균 x̂_t 가 x_t 를 잘 못 따라감 (mean bias)
- 앙상블 spread 가 RMSE 대비 너무 작음 (under-dispersion)

### 해결 방향 (확정)
1. **매일 00 UTC 스냅샷만 사용 (SEEDS 방식).** 6시간 4프레임 중 00시만 추출 → 하루 1프레임.
   daily mean 집계 안 함 (일중 변동을 평균으로 뭉개지 않음). 태스크는 "00 UTC 스냅샷
   기상장의 표준화 anomaly 예측"으로 재정의됨. 이웃 t±1 도 하루 간격 00시 스냅샷.
2. **예측 대상을 표준화 anomaly 로 전환.** DDPM_past·DDPM_main 모두 climatology 를
   결정론적 anchor 로 고정하고, "평년 대비 표준화 편차"만 diffusion 으로 생성.
   climatology 가 계절·지리의 mean 을 잡아 mean bias 완화, 타깃이 작아져 학습 안정.
3. **climatology 를 condition 으로 주입.** 단 "생성 대상 시점의 climatology"를 줌
   (SEEDS 방식). 대기 상태가 연중 시점에 비정상(non-stationary)이므로 시점 맥락 제공.
4. **DDPM_past sampling 에 logvar 잔여노이즈를 0.1 scale 로 매 step 주입** → spread 공급.

### climatology 의 핵심 성질 (중요)
- climatology 는 **생성 대상이 아니라 조회(lookup) 대상**이다. doy 만 알면 결정론적으로 조회.
- 추론 시 입력은 x_t 와 그 doy 뿐. 거기서 c_t, c_{t-1}, c_{t+1} 은 전부 자동 조회.
- 이웃 climatology 를 "생성"하지 않는다. anomaly 만 생성한다.

---

## 1. Condition 구성 규칙 (가장 중요 — 먼저 확정)

**규칙: 각 diffusion 은 "자기가 생성하는 시점의 climatology"를 condition 으로 받는다.**

| | 생성 대상 (anomaly) | condition: anomaly | condition: climatology |
|---|---|---|---|
| **DDPM_past** | x̂_{t-1}, x̂_{t+1} | x̃_t (관측, F=1) | c_{t-1}, c_{t+1} (조회) |
| **DDPM_main** | x̂_t | x̂_{t-1}, x̂_{t+1} (past 생성, 멤버별) | c_t (조회) |

근거:
- climatology 의 역할 = "생성 대상 시점의 비정상성 정보 제공".
- 복원식과 시점 일치: x̂_t^pixel = x̂_t^std · c_sig_t + c_mu_t 에서 쓰는 게 c_t.
- daily 라 c_t ≈ c_{t±1} 이라 실질 차이는 작지만, 일관성·복원 정합 위해 위 규칙 준수.

**누설 금지**: DDPM_main 의 condition 에 x_t / x̃_t (중심 anomaly 정답)를 절대 넣지 않는다.
주는 climatology c_t 는 평년 통계(특정 사례 정답 아님)이므로 누설 아님.

---

## 2. 정규화 설계 (두 종류, 역할 다름)

### 2.1 Anomaly 정규화 — local (doy·위치별) 표준화
기상장을 climatology 로 표준화하여 표준화 anomaly 생성:
```
x̃_τ = (x_00utc_τ − c_mu[doy_τ]) / c_sig[doy_τ]
```
- c_mu, c_sig : (doy, 변수, 위치)별 평년 평균/표준편차 (§3 [2]의 smoothed climatology).
- 이 한 번으로 anomaly화 + 변수 스케일 통일 + 계절성 제거 동시 수행.
- 결과 x̃ 는 대략 zero-mean, unit-variance (값 ~ ±3).

### 2.2 Climatology condition 정규화 — pixelwise (위치별, doy무관) 표준화  ★v3 수정
condition 으로 줄 climatology c_mu[doy] 를 **pixelwise 통계**로 정규화:
```
c̃_τ = (c_mu[doy_τ] − P_mu) / P_sig
```
- **P_mu, P_sig : 전체 00시 스냅샷 데이터의 pixelwise mean/std (C, 64, 64), doy 무관.**
  - 각 (변수, 위치)에서 전 기간(2000~2019)·전 doy 의 00시 스냅샷을 모아 계산한 평균/표준편차.
  - 위치별 상수이며 doy 에는 의존하지 않음.
- 이유:
  - climatology 를 자기 doy·위치 통계(c_mu, c_sig)로 표준화하면 0 이 되어 정보 소멸 → 금지.
  - pixelwise(위치별, doy무관) 통계로 표준화하면, 변수·위치 스케일은 정규화하면서
    **계절(doy)에 따른 climatology 변동(절대 맥락)은 보존**됨.
- 즉 **anomaly 는 (doy·위치별) local 표준화, climatology 는 (위치별·doy무관) pixelwise 표준화.**
  둘 다 결과 스케일이 비슷해져 같은 인코더에 안정적으로 들어감.
- 주의: P_mu, P_sig 는 anomaly 표준화에 쓰는 c_mu/c_sig 와 **다른 통계**다. 혼동 금지.

### 2.3 Latent 정규화 — 단일 LatentNormalizer
VAE encode 후 latent 분산을 1로:
```
ẑ = (z − μ_z) / σ_z          # LatentNormalizer, past·main 공유 (단일 인스턴스)
```
- past·main 둘 다 같은 anomaly latent 공간 → normalizer 1개로 충분.

### 2.4 정규화 통계 계산 순서 (의존성)
```
[1]  00 UTC 스냅샷 추출 (매일 00시 1프레임)
[2]  climatology (2단계: doy별 raw 계산 → ±7일 smoothing)  → c_mu, c_sig (doy별)
[2b] pixelwise 통계 (전체 00시 스냅샷의 위치별 mean/std)    → P_mu, P_sig
[3]  표준화 anomaly x̃ 저장   (= (x − c_mu)/c_sig)
Stage0: VAE 학습 (입력 = x̃)
        → compute_latent_stats.py  (μ_z, σ_z; z=μ 결정론적으로 계산)
```

---

## 3. 데이터 전처리 (신규 스크립트)

원본 6h 데이터의 저장 형식·차원·시간 인덱싱은 **기존 `dataset/era5_dataset.py` 를 읽어 따른다.**

### [1] `scripts/compute_daily_snapshot.py`
- 매일 **00 UTC 스냅샷만 추출** → (C, 64, 64). (06/12/18 시각은 버림)
- daily mean 집계 안 함 (SEEDS 방식).
- 결측: 해당 날짜의 00시 스냅샷이 없으면 그 날 제외.
- 변수 순서·단위 원본 유지.
- (기존 dataset 이 시간 필터를 지원하면 별도 스크립트 없이 dataset 단에서 00시 선택 가능.)

### [2] `scripts/compute_climatology.py`  ★v3 수정: 2단계 방식
입력: [1] 00시 스냅샷 데이터 (2000~2019). 출력: `climatology.npz` (c_mu, c_sig; (366, C, 64, 64)).

**1단계 — doy별 raw climatology 계산 (smoothing 전):**
- 각 doy d 에 대해, 그 doy 에 해당하는 20년치 00시 스냅샷 샘플들만 모아:
  - raw_mu[d]  = 평균장 (C, 64, 64)
  - raw_sig[d] = 표준편차장 (C, 64, 64)
- 결과: doy별 raw climatology **시계열** raw_mu[0..365], raw_sig[0..365].
- 윤년: 366-slot. doy→slot 매핑 규칙을 한 곳에 정의(학습·추론 공통).

**2단계 — 그 시계열에 ±7일 sliding window smoothing:**
- raw_mu, raw_sig **시계열 각각**에 대해, doy 축으로 중심 ±7일(총 15일) 이동평균:
  ```
  c_mu[d]  = mean( raw_mu[d-7 .. d+7] )      # doy 축 평균
  c_sig[d] = mean( raw_sig[d-7 .. d+7] )
  ```
- **연말-연초 wrap**: doy 축 윈도우 순환(modulo 366). (12/31 의 윈도우가 1월 초까지)
- **윤년 2/29**: smoothing 후 c_mu/c_sig 의 2/29 슬롯 = 0.5*(2/28 + 3/1) 보간.
  (±7일 window 면 양옆이 이미 포함되므로 fallback 성격.)
- **c_sig floor**: 최종 c_sig = max(c_sig, eps) (eps=1e-6) — 0 division 방어.

> 핵심: "±7일×20년 샘플을 한꺼번에 pool" 하는 게 아니라, **doy별 climatology 를 먼저
> 만든 뒤 그 climatology 시계열을 smoothing** 한다 (SEEDS 가 timeseries 를 15일 window 로
> smoothing 한 방식과 동일). 결과(특히 std)가 pool 방식과 다르므로 반드시 이 순서로.

### [2b] `scripts/compute_pixel_stats.py`
- 입력: [1] 00시 스냅샷 데이터 전체 (2000~2019).
- **pixelwise mean/std (C, 64, 64)**: 각 (변수, 위치)에서 전 기간·전 doy 00시 스냅샷을 모아
  평균 P_mu, 표준편차 P_sig 계산. doy 에 무관(전 기간 marginalize).
- climatology condition 의 pixelwise 표준화(2.2)에 사용.
- P_sig floor 동일 적용(eps=1e-6).
- 출력: `pixel_stats.npz` (P_mu, P_sig). 추론에도 필요 → 보관.

### [3] `scripts/compute_anomaly.py`
- 입력: [1] + [2].
- 표준화 anomaly:  x̃ = (x_00utc − c_mu[doy]) / c_sig[doy]
- 출력: 날짜별 x̃ (C, 64, 64). VAE·diffusion 의 입력/타깃.
- doy 계산은 [2] 인덱싱 규칙과 일치.

> 중간 산출물 분리 저장 권장(재실행·디버깅). 기존 dataset 이 on-the-fly 처리를 선호하면
> 그 구조에 맞춰도 됨. 핵심: VAE 가 보는 데이터 = 표준화 anomaly x̃.

---

## 4. Climatology 모듈 (신규)

### `models/climatology.py` — `ClimatologyBank`
- `climatology.npz`(c_mu, c_sig), `pixel_stats.npz`(P_mu, P_sig) 로드 → buffer 보관
  (학습 파라미터 없음).
- doy 인덱싱 규칙은 [2]와 동일.
- 메서드:
  - `standardize_anomaly(x, doy) -> x̃`              # (x − c_mu[doy]) / c_sig[doy]   (local)
  - `destandardize_anomaly(x̃, doy) -> x`             # x̃ · c_sig[doy] + c_mu[doy]
  - `climatology_pixel_norm(doy) -> c̃`               # (c_mu[doy] − P_mu) / P_sig   (condition용)
  - (doy 배치 조회 지원: (B,) doy → (B, C, 64, 64))
- 픽셀 공간에서 동작.

---

## 5. Encoder — 공유 인코더 + type embedding + snapshot self-attention

### 5.1 설계 (SEEDS S2.3 를 우리 규모로 단순화)
SEEDS 는 anomaly seeds·climatology·denoise 타깃을 **공유 인코더**로 처리하고,
snapshot 종류는 **type(categorical) embedding** 으로만 구분, 마지막에 snapshot 간
self-attention(SEEDS 의 s-축 transformer)으로 결합한다. 우리는 변수 3개·단일 패치라
SEEDS 의 변수축·공간축 axial attention 은 conv/patchify 로 대체하고 **핵심(type embedding +
snapshot self-attention)만 채택**한다.

### 5.2 `ConditionEncoder` 수정 (`models/encoder.py`)
입력 snapshot 들 (각각 표준화된 픽셀장 (C,64,64)):
- DDPM_past condition: { x̃_t (anomaly), c̃_{t-1}, c̃_{t+1} (climatology, pixel-norm) }
- DDPM_main condition: { x̃_{t-1}, x̃_{t+1} (anomaly), c̃_t (climatology, pixel-norm) }

처리:
```
[단계 A] 공간 인코딩 (모든 snapshot 공유 conv+patchify, 기존 구조 재사용)
  각 snapshot (C,64,64) → conv → patchify → tokens (N_tok, D)
  + type embedding (snapshot 종류별, broadcast 해서 더함)

[단계 B] snapshot 간 self-attention (신규)
  모든 snapshot 토큰 concat → (N_snap·N_tok, D)
  → self-attention block(s) → condition tokens
  → DiT cross-attention 의 key/value
```

### 5.3 Type embedding (★ 핵심)
- `nn.Embedding(num_types, D)` 으로 snapshot 종류 구분. 기존 frame-slot embedding
  (`nn.Embedding(2, D)`: t-1, t+1)을 **확장**.
- type 종류(예): `ANOM_CENTER`(=x̃_t, past용), `ANOM_PREV`(x̃_{t-1}), `ANOM_NEXT`(x̃_{t+1}),
  `CLIMATOLOGY`(c̃). past/main 이 쓰는 type 집합이 다름.
  - past: {ANOM_CENTER, CLIMATOLOGY(prev/next)}
  - main: {ANOM_PREV, ANOM_NEXT, CLIMATOLOGY(center)}
  - 구체 enum 은 구현 시 정리하되, **anomaly vs climatology 구분 + 시점(prev/center/next)
    구분**이 표현되면 됨.
- type embedding 은 **단계 A 출력 토큰에 broadcast 하여 더함** (SEEDS 방식, 단계 B 입력).

### 5.4 climatology 입력 처리
- climatology 는 `ClimatologyBank.climatology_pixel_norm(doy)` 로 **pixelwise 표준화**된 c̃ 를
  받아 anomaly 와 **같은 conv** 로 인코딩 (스케일이 2.2 로 맞춰져 있어 안정).
- 별도 ClimatologyEncoder 불필요 — 공유 인코더 + type embedding 으로 구분.

### 5.5 DiT (`models/dit.py`)
- 구조 변경 없음. cross-attention 의 condition 토큰 수가 늘어나는 것만 반영
  (MultiheadAttention 은 임의 길이 처리 가능).

---

## 6. VAE (`models/vae.py`)
- **구조 변경 없음.** 학습 입출력이 표준화 anomaly x̃ 로 바뀜.
- `training/train_vae.py`: 데이터 로딩이 x̃ 를 공급하도록 수정 ([3] 산출물 또는
  ClimatologyBank.standardize_anomaly 를 dataset 에 끼움).
- 손실(MSE + 1e-6·KL) 그대로.

---

## 7. Diffusion 모델 (past·main 통일)
- 둘 다 **표준화 anomaly latent (ẑ) 공간**에서 동작. climatology 는 diffusion 대상 아님.
- `VDMSchedule`, `DualHeadDiT`, `heteroscedastic_nll_loss` **모두 그대로**.
  (forward noising·ε-pred·NLL 불변. 다루는 데이터가 anomaly latent 일 뿐)
- ε-prediction 유지. x0/anomaly 예측으로 바꾸지 않음.
- LatentNormalizer 단일 인스턴스 공유.

---

## 8. 학습 절차 (`training/train.py`)
기존 4-phase(past/infer/main/joint) 골격 유지. 데이터가 anomaly latent 인 점 + climatology
condition 추가 반영.

### 8.1 main phase train step (예시)
```python
# batch: {x_tm1, x_t, x_tp1, doy_tm1, doy_t, doy_tp1}  (00 UTC 스냅샷 픽셀)
# ── anomaly 표준화 (local: doy·위치별) ──
xtm1 = clim.standardize_anomaly(x_tm1, doy_tm1)   # (B,C,64,64) ~±3
xtp1 = clim.standardize_anomaly(x_tp1, doy_tp1)
xt   = clim.standardize_anomaly(x_t,   doy_t)
# ── climatology (pixel-norm, 생성 대상=t 시점) ──
c_t  = clim.climatology_pixel_norm(doy_t)         # (B,C,64,64) ~ 비슷한 스케일
# ── condition: 이웃 anomaly + 중심 climatology ──
cond = encoder(
    anomalies=[xtm1, xtp1],          # type: ANOM_PREV, ANOM_NEXT
    climatology=c_t,                 # type: CLIMATOLOGY
)
# ── 타깃: 중심 anomaly latent ──
z0 = latent_norm.normalize(vae.encode_mu(xt))     # (B,C_z,16,16)
# ── diffusion (기존과 동일) ──
t = torch.rand(B)
z_t, eps = schedule.forward_noise(z0, t)
eps_pred, log_var = dit_main(z_t, t, cond)
loss = heteroscedastic_nll_loss(eps, eps_pred, log_var)
```

### 8.2 past phase train step
- 타깃: 이웃 anomaly latent [ẑ_{t-1}, ẑ_{t+1}].
- condition: 중심 anomaly x̃_t (ANOM_CENTER) + 이웃 climatology c̃_{t-1}, c̃_{t+1}
  (생성 대상=t±1 시점이므로 이웃 climatology; pixel-norm).
- 나머지 diffusion 코드 동일.

### 8.3 infer / joint phase
- infer: 멤버별 ẑ_past memmap 생성 (기존).
- joint: L = L_past + L_main (기존). climatology 는 양 phase 에서 동일 조회 → 정합.

---

## 9. 추론 (`inference/sampling.py`)

### 9.1 logvar 0.1 scale 주입 (DDPM_past 전용)
- 기존 `inject_uncertainty_mode='all'` 경로에 **scale 0.1** 적용. DDPM_past 만.
  DDPM_main 은 주입 없음('none').
- scale 은 config 로 분리 (`sampling.past_inject_scale: 0.1`).
```python
if mode == "all":   # past 전용
    z = z + past_inject_scale * torch.exp(0.5 * log_var) * torch.randn_like(z)
```
- **주의(주석)**: step 수에 비례해 유효 spread 누적 (≈ sqrt(steps)·0.1). 0.1 은 시작값,
  spread 측정 후 튜닝 대상.

### 9.2 end-to-end 앙상블 생성 (M 멤버)
```
입력: x_t (00 UTC 스냅샷 픽셀), doy_t  →  doy_{t-1}, doy_{t+1} 자동 계산
# climatology 전부 조회 (생성 아님)
c_t   = clim.climatology_pixel_norm(doy_t)
c_tm1 = clim.climatology_pixel_norm(doy_tm1)
c_tp1 = clim.climatology_pixel_norm(doy_tp1)

# 1. 관측 anomaly
xt = clim.standardize_anomaly(x_t, doy_t)

# 2. DDPM_past: 이웃 anomaly latent 생성 (멤버별)
cond_past = encoder(anomalies=[xt(ANOM_CENTER)], climatology=[c_tm1, c_tp1]).expand(M)
ẑ_past = ddpm_past.sample(cond_past, M, inject_mode='all', scale=0.1)  # (M,12,16,16)
x̃_tm1, x̃_tp1 = vae.decode(latent_norm.denorm(ẑ_past.split))           # 표준화 anomaly(픽셀)

# 3. DDPM_main: 중심 anomaly latent 생성
cond_main = encoder(anomalies=[x̃_tm1, x̃_tp1], climatology=c_t)         # 중심 climatology
ẑ ~ N(0, I)
for step in reverse_steps:                                            # inject 없음
    eps_pred, log_var = dit_main(ẑ, t, cond_main)
    ẑ = ancestral_step(ẑ, eps_pred, ...)
x̃_t = vae.decode(latent_norm.denorm(ẑ))                               # 표준화 anomaly(픽셀)

# 4. 복원 (anomaly → 픽셀, local destandardize)
x̂_t = clim.destandardize_anomaly(x̃_t, doy_t)                          # x̃_t · c_sig[doy_t] + c_mu[doy_t]
```
- 시작점은 기존대로 N(0, I) (anomaly 예측이어도 불변).
- 복원에 쓰는 c_sig[doy_t], c_mu[doy_t] 는 local 통계(2.1). condition 의 c̃_t(pixel-norm,
  2.2)와 다른 정규화임에 주의 — 둘은 용도가 다름(condition vs 복원).

---

## 10. Config 추가/변경 (`config/default.yaml`)
```
data.use_00utc_only: true        # 매일 00 UTC 스냅샷만 사용 (daily mean 안 함)
climatology.window_radius: 7
climatology.path: <climatology.npz>
climatology.pixel_stats_path: <pixel_stats.npz>      # ★v3: P_mu, P_sig
climatology.sigma_floor: 1e-6
encoder.num_snapshot_types: <enum 크기>      # type embedding
encoder.snapshot_attn_layers: <예: 2>        # snapshot self-attention 깊이
sampling.past_inject_scale: 0.1
sampling.past_inject_mode: all
sampling.main_inject_mode: none
```
기존 shape config(C, H, W, C_z 등) 유지.

---

## 11. 구현 순서 (권장)
1. `models/climatology.py` (ClimatologyBank) + `scripts/compute_daily_snapshot.py` (00시 추출)
2. `scripts/compute_climatology.py` (2단계: doy별 raw → ±7일 smoothing, 윤년/wrap/floor)
3. `scripts/compute_pixel_stats.py` (P_mu, P_sig; pixelwise, doy무관)
4. `scripts/compute_anomaly.py` + dataset 연동 (VAE 입력 = x̃)
5. VAE 학습 경로 수정 → compute_latent_stats (anomaly latent 통계)
6. `models/encoder.py`: type embedding 확장 + climatology 입력 + snapshot self-attention
7. `training/train.py`: past/main phase 의 condition 구성(§1 규칙)대로 수정
   (diffusion 코어 코드 자체는 불변 확인)
8. `inference/sampling.py`: logvar 0.1 주입(past) + climatology 조회 + 복원(§9.2)
9. config 반영

---

## 12. 검증 체크리스트
- [ ] 데이터가 00 UTC 스냅샷만인지 (daily mean 집계 안 했는지). 06/12/18 미포함 확인
- [ ] anomaly 표준화 왕복: destandardize_anomaly(standardize_anomaly(x)) ≈ x (<1e-4)
- [ ] 표준화 anomaly x̃ 의 채널별 mean≈0, std≈1
- [ ] climatology 계산이 2단계(doy별 raw → smoothing)로 됐는지. (pool 1단계 아님)
- [ ] smoothing 후 c_mu/c_sig 가 doy 축으로 부드럽게 변하는지 (들쭉날쭉하지 않음)
- [ ] climatology pixel-norm c̃ 의 스케일이 x̃ 와 유사 범위인지 (인코더 안정성)
- [ ] climatology 가 자기 doy·위치 통계로 표준화되어 0 으로 붕괴하지 않는지
      (pixelwise·doy무관 통계 P_mu/P_sig 사용 확인)
- [ ] P_mu/P_sig 가 c_mu/c_sig 와 다른 통계인지 (혼동 금지)
- [ ] LatentNormalizer 후 latent 분산 ≈ 1
- [ ] climatology wrap (12/31 ↔ 1/1) 연속성, 2/29 보간값 확인
- [ ] condition 시점 규칙: past=이웃 climatology(c_{t±1}), main=중심 climatology(c_t)
- [ ] DDPM_main inject 없음 / DDPM_past inject=all·scale=0.1
- [ ] type embedding 이 anomaly vs climatology, 시점(prev/center/next)을 구분하는지
- [ ] 추론 시작점이 N(0, I) 인지
- [ ] 복원에 local 통계(c_mu[doy_t], c_sig[doy_t]) 사용, condition 에 pixel-norm 사용 (혼동 금지)
- [ ] (성능) 앙상블 mean 이 x_t 를 더 잘 따라가는지, spread/RMSE 비율이 1 에 근접하는지
      → 부족하면 past_inject_scale 튜닝 / 추가 spread calibration 검토

---

## 13. 설계 근거 메모 (참고)
- climatology-anomaly 분해 + climatology condition 은 SEEDS (Li et al., Sci. Adv. 2024)
  방식과 정합. SEEDS 는 standardized anomaly 입출력 + climatological mean 을 condition
  snapshot 으로 주입(type embedding "Climatology")하고, climatology 를 raw daily
  timeseries 에 15일 centered window smoothing 으로 계산함(우리 §3[2] 2단계와 동일 철학).
- 데이터도 SEEDS 와 동일하게 매일 00 UTC 스냅샷만 사용 (SEEDS: "we only retain the
  00-hour UTC time snapshots ... for each day").
- 우리 차이: (a) latent diffusion (SEEDS 는 픽셀), (b) dual diffusion (past→main),
  (c) climatology condition 을 pixelwise(위치별·doy무관) 표준화 (SEEDS 의 정확한 정규화
  수식은 SM 에 미명시; 우리는 pixelwise 표준화로 명시 결정).
- mean 은 climatology anchor + neighbor condition 이, spread 는 past 불확실성 전파 +
  (필요시) calibration 이 담당하는 분리 구조.