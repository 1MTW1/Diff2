# Dual-Head DDPM Ensemble 평가 실험 명세

## 0. 개요

학습된 Dual-Head DDPM (model_past + model_main) 의 ensemble 예측 품질과 학습된
픽셀별 분산 (`log_var`, 이하 $\ell_m$) 의 의미를 정량/정성적으로 분석한다.

핵심 가설:
- $\ell_m$ 은 **flow-dependent**한 예측 불확실성을 학습한다 (픽셀별, sample별로 변화).
- $\ell_m$ 이 큰 픽셀은 실제 ensemble spread도 크다.
- Ensemble이 well-calibrated 되어 있다면 픽셀별 spread/RMSE ratio ≈ 1.


## 1. 실험 setting

### 1.1 데이터

- **데이터**: 사전 정규화된 ERA5 (`data/era5_normalized.zarr`)
- **Split**: test (2022-01-01 ~ 2022-12-31)
- **시간 해상도**: 6시간 간격 (하루 4 시점)
- **Sub-sampling**: 일주일 중 3일만 사용 (예: 월/수/금) → 약 624 시점
- **채널**: t (온도), u (동서풍), v (남북풍), 64×64 grid

### 1.2 추론 파이프라인

학습 시와 동일한 풀 체인 구조로 추론:

```
(x_{t-1}, x_t)
     │
     │ encoder
     ▼
cond_past ──► model_past (DDPM, 50 step) ──► (x̂_{t-3}, x̂_{t-2})
                                                  │
                                                  │ ┌── x_{t-1} (관측)
                                                  ▼ ▼
                                              encoder
                                                  │
                                                  ▼
                                             cond_main
                                                  │
                                                  │ model_main (DDPM, 200 step)
                                                  ▼
                                          (x̂_t, x̂_{t+1}), ℓ_m
```

- **model_past**: 학습 시 aux_sampler와 동일한 50 step DDPM (분포 일치)
- **model_main**: 200 step full DDPM chain (품질 우선)
- **평가 target**: $x_t$ 만 (model_main 출력의 첫 번째 시점)
- **분석 채널**: **온도 (t) 만** 분석 (실험 5의 composite map만 3채널)

### 1.3 Ensemble

- **Member 수**: $N = 30$
- **생성 방식**: 같은 conditioning $(x_{t-1}, x_t)$ 에 대해 독립적 random seed로 30회 sampling
- **$\ell_m$ 정의**: DDPMSampler의 `ell_final` (마지막에서 두 번째 step의 학습된 log variance), 비교 시 $\exp(\ell_m)$ 사용

### 1.4 출력 캐시

ensemble 추론이 가장 무거우므로 시점별로 한 번 생성 후 디스크에 저장하여 실험
1-5가 재사용한다.

```
outputs/ensembles/sample_{idx:05d}.npz
  - ensemble       : (N=30, C=3, H=64, W=64)  float32  # 정규화 공간
  - log_var        : (C=3, H=64, W=64)         float32  # ℓ_m (raw log)
  - x_t_true       : (C=3, H=64, W=64)         float32  # 정규화 공간 GT
  - time_t         : datetime64 scalar
```

### 1.5 디렉토리 구조

```
experiments/
  __init__.py
  ensemble_inference.py     # ensemble 생성 + npz 캐시
  exp1_uncertainty.py       # ℓ_m 의 spatial/temporal variation
  exp2_spread_vs_logvar.py  # ℓ_m vs ensemble spread correlation
  exp3_spread_skill.py      # spread/RMSE ratio
  exp4_pixel_timeseries.py  # top/bottom point 시계열
  exp5_composite_maps.py    # quiver + shading composite
  utils.py                  # denormalization, plotting helpers

scripts/
  run_inference.py          # 전체 test set ensemble 추론
  run_experiments.py        # 실험 1-5 일괄 실행

outputs/
  ensembles/                # 시점별 npz 캐시
  figures/                  # 실험별 plot 저장
  metrics/                  # 실험 1-3 수치 결과 (json)
```


## 2. 실험 1: $\ell_m$ 의 pixelwise 다양성 & 시점 간 차이

### 2.1 목적

다음 두 성질을 검증:

1. **Pixelwise 다양성 (단일 sample 내)**: $\ell_m$ 이 픽셀마다 다른 값을 갖는다.
   즉 모델이 모든 픽셀에 균일한 불확실성을 부여하지 않고, 위치별로 다르게 학습했다.
2. **시점 간 차이 (flow-dependence)**: 서로 다른 시점 sample 두 개의 $\ell_m$ map이
   서로 다르다. 즉 $\ell_m$ 이 시점에 무관한 고정 패턴 (예: 단순 지리적 prior) 이
   아니라 입력에 따라 변하는 flow-dependent 양이다.

### 2.2 검증 1: Pixelwise 다양성

각 sample $s$ 에 대해 (온도 채널만):
- $\bar v_s = \text{mean}_{(i,j)}\,\exp(\ell_m^{(s)}(i,j))$: 공간 평균
- $\sigma_s^{\text{spatial}} = \text{std}_{(i,j)}\,\exp(\ell_m^{(s)}(i,j))$: 공간 표준편차
- $\text{CV}_s = \sigma_s^{\text{spatial}} / \bar v_s$: spatial coefficient of variation

해석:
- $\text{CV}_s \approx 0$ 이면 모든 픽셀이 같은 값 → 다양성 없음
- $\text{CV}_s$ 가 클수록 픽셀별 차이가 큼

Test set 전체에 대해 $\text{CV}_s$ 의 분포를 보고 (mean, median, std).

### 2.3 검증 2: 서로 다른 시점 간 차이

핵심 진단: **두 random 시점 sample 간 $\ell_m$ map의 spatial correlation**.

절차:
1. Test set sample들 중 무작위로 1000쌍 $(s_1, s_2)$ 추출 ($s_1 \neq s_2$)
2. 각 쌍에 대해 픽셀별로 펼친 두 $\exp(\ell_m)$ map의 Pearson correlation 계산:
   $$\rho_{s_1, s_2} = \text{corr}\big(\exp(\ell_m^{(s_1)}).\text{flatten}(), \ \exp(\ell_m^{(s_2)}).\text{flatten}()\big)$$
3. 분포 $\{\rho_{s_1, s_2}\}_{1000\text{쌍}}$ 의 통계 보고

해석:
- 분포가 1 근처에 몰림 → 시점에 무관한 고정 패턴 (flow-independent)
- 분포가 0 근처 또는 넓게 퍼짐 → 시점마다 다른 패턴 (flow-dependent) ✓

### 2.4 정성 분석 (visualization)

- 무작위로 추출한 6개 sample의 $\exp(\ell_m)$ heatmap (2×3 grid)
- Colorbar 공유, 동일 scale로 표시 → 패턴 차이를 시각적으로 직접 확인
- 각 subplot에 sample의 timestamp 표시

### 2.5 출력

- `outputs/figures/exp1_logvar_heatmaps.png`: 6 sample heatmap (2×3)
- `outputs/figures/exp1_pixelwise_cv_hist.png`: $\text{CV}_s$ 분포
- `outputs/figures/exp1_pairwise_correlation_hist.png`: $\rho_{s_1, s_2}$ 분포 (1000쌍)
- `outputs/metrics/exp1_stats.json`:
  - `cv`: {mean, median, std} of $\text{CV}_s$
  - `pairwise_correlation`: {mean, median, std, n_pairs} of $\rho_{s_1, s_2}$


## 3. 실험 2: $\ell_m$ vs ensemble spread

### 3.1 목적

$\ell_m$ 이 실제 ensemble spread를 잘 예측하는지 확인.

### 3.2 정의

- **Ensemble spread (픽셀별)**:
  $$s(i,j) = \sqrt{\frac{1}{N-1}\sum_{n=1}^{N} (\hat x_n(i,j) - \bar{\hat x}(i,j))^2}$$
- **모델 예측 spread**: $\sigma_\ell(i,j) = \sqrt{\exp(\ell_m(i,j))}$

### 3.3 분석

각 sample에 대해:
- 픽셀별 $\sigma_\ell$ vs $s$ scatter plot (모든 픽셀 점)
- 각 sample의 Pearson correlation $r_s = \text{corr}(\sigma_\ell, s)$ 계산
- Test set 전체에 대해 $r_s$ 분포 → 평균 correlation 보고

### 3.4 출력

- `outputs/figures/exp2_scatter_examples.png`: 무작위 3개 sample의 scatter (subplot)
- `outputs/figures/exp2_correlation_hist.png`: $r_s$ 분포 histogram
- `outputs/metrics/exp2_stats.json`: $r_s$ 의 mean/median/std


## 4. 실험 3: Spread / RMSE ratio (calibration)

### 4.1 목적

Ensemble이 well-calibrated 되어 있다면 픽셀별로 평균적으로
spread $\approx$ RMSE 가 성립해야 한다.

### 4.2 정의

- **픽셀별 spread**: $s(i,j)$ (실험 2와 동일)
- **픽셀별 error**: $e(i,j) = \bar{\hat x}(i,j) - x_t^{\text{true}}(i,j)$
- **픽셀별 ratio**:
  $$r(i,j) = \frac{s(i,j)}{|e(i,j)| + \varepsilon}$$
  ($\varepsilon = 10^{-6}$, 분모 안정화)

### 4.3 계산 방식

사용자 지정대로 **픽셀별 ratio를 먼저 구하고 평균**한다:
$$\bar r_s = \text{mean}_{(i,j)}\, r(i,j)$$

Test set 전체에 대해 $\bar r_s$ 의 분포 계산.

> **주의**: 이 방식은 RMSE가 0에 가까운 픽셀에서 ratio가 폭발할 수 있어
> 정규화 공간 / 역정규화 공간 모두에서 별도로 계산하여 비교한다.
> 추가로 표준 공식
> $\sqrt{\overline{s^2}} / \sqrt{\overline{e^2}}$ 도 같이 계산하여 참고치로 보고.

### 4.4 정규화 / 역정규화

- **정규화 공간**: 모델 출력 그대로 (단위 없음, 평균 0 분산 1 가정)
- **역정규화 공간**: t는 K (또는 °C), 채널별 σ로 spread/error scaling

### 4.5 출력

- `outputs/figures/exp3_ratio_hist.png`: $\bar r_s$ 분포 (정규화/역정규화 두 panel)
- `outputs/metrics/exp3_stats.json`: 정규화·역정규화 각각의 mean/median/std,
  표준 공식과의 비교


## 5. 실험 4: 픽셀 시계열 분석 (top/bottom $\ell_m$)

### 5.1 목적

$\ell_m$ 이 높은 픽셀과 낮은 픽셀에서 ensemble member 분포가 어떻게 다른지 확인.

### 5.2 픽셀 선택

전체 test set $\ell_m$ 을 누적해 픽셀 위치별 평균 $\bar\ell(i,j)$ 계산:
$$\bar\ell(i,j) = \frac{1}{S}\sum_s \exp(\ell_m^{(s)}(i,j))$$

이로부터:
- **Top 3**: $\bar\ell$ 이 가장 높은 픽셀 위치 3개
- **Bottom 3**: $\bar\ell$ 이 가장 낮은 픽셀 위치 3개

(픽셀 위치는 한 번 결정되면 모든 sample에서 같은 위치 사용.)

### 5.3 Plot 구성

각 선정된 픽셀 1개에 대해 1 figure:
- **x축**: test set 시점 인덱스 (시간 순)
- **y축**: 해당 픽셀의 온도 값 (역정규화)
- **회색 선 (30개)**: 30 ensemble members 각각의 시계열
- **파란 선**: ensemble mean
- **검은 선**: ground truth
- **제목**: 픽셀 좌표 (i, j) 와 $\bar\ell$ 값

총 6 figure (top 3 + bottom 3).

### 5.4 해석 포인트

- Top 픽셀: ensemble spread가 넓고 GT가 그 spread 안에 들어와야 함
- Bottom 픽셀: ensemble이 좁게 모여 있고 mean이 GT에 가까워야 함

### 5.5 출력

- `outputs/figures/exp4_top1_pixel_{i}_{j}.png` ... `exp4_top3_*.png`
- `outputs/figures/exp4_bottom1_pixel_{i}_{j}.png` ... `exp4_bottom3_*.png`


## 6. 실험 5: Composite Map (quiver + shading)

### 6.1 목적

무작위 시점 3개에 대해 ensemble 결과를 시각적으로 비교.

### 6.2 시점 선택

Test set에서 무작위 3개 시점 선택 (재현 가능하도록 seed 고정).

### 6.3 Plot 구성

**시점 1개당 1 figure**, 총 3 figure. 각 figure는 4×4 subplot:

| 위치 | 내용 |
|---|---|
| (0,0) | Ground Truth |
| (0,1) ~ (3,1) | Ensemble member 1 ~ 14 (4×4 grid에서 첫 칸 GT, 마지막 칸 mean 제외) |
| (3,3) | Ensemble Mean |

→ 총 16칸 = GT + 14 members + mean

(정확한 배치는 GT 좌상단, mean 우하단, 그 사이에 14 members 순서대로.)

### 6.4 각 subplot의 표현

- **온도 (t)**: shading (filled contour 또는 pcolormesh), colormap 공유
- **바람 (u, v)**: quiver vector, 4 grid마다 subsample (16×16 화살표 정도)
- **Vector key**: 각 figure에 1개, 화살표 크기 기준 (예: "10 m/s") 표시
- **모든 subplot 공유**: colorbar (온도), vector key (바람)

### 6.5 공간

**역정규화 공간**에서 plot:
- 온도: K 또는 °C
- 바람: m/s

### 6.6 Ensemble spread 보조 plot

Ensemble spread는 정규화 / 역정규화 공간에서 모두 1 panel씩 추가:
- 같은 figure 내 별도 subplot 2개 또는 별도 figure
- 권장: 별도 figure로 `outputs/figures/exp5_sample{k}_spread.png`

### 6.7 출력

- `outputs/figures/exp5_composite_sample{1,2,3}.png` (16-panel composite)
- `outputs/figures/exp5_spread_sample{1,2,3}.png` (정규화 / 역정규화 spread)


## 7. 실행 순서

1. **추론 단계** (한 번만 실행):
   ```bash
   python -m scripts.run_inference \
       --checkpoint outputs/.../checkpoint_best.pt \
       --output_dir outputs/ensembles \
       --n_members 30 \
       --past_steps 50 \
       --main_steps 200 \
       --days_per_week 3
   ```

2. **분석 단계** (캐시된 ensemble 재사용):
   ```bash
   python -m scripts.run_experiments \
       --ensemble_dir outputs/ensembles \
       --figures_dir outputs/figures \
       --metrics_dir outputs/metrics
   ```


## 8. 산출물 요약

| 실험 | 그림 수 | 수치 결과 |
|---|---|---|
| 1 | 3 | pixelwise CV 분포, 시점 쌍 spatial correlation 분포 |
| 2 | 2 | sample별 σ_ℓ–spread correlation 분포 |
| 3 | 1 (2 panel) | 정규화/역정규화 spread-RMSE ratio 분포 |
| 4 | 6 | (수치 없음, 시각적 확인) |
| 5 | 6 | (수치 없음, 시각적 확인) |

총 figure 18개, metrics JSON 3개.


## 9. 검증 체크리스트

코드 작성 시 다음을 점검:

- [ ] model_past sampling이 학습 시 aux_sampler와 동일한 50 step / DDPM 인지
- [ ] model_main sampling이 200 step full chain DDPM 인지
- [ ] `ell_final` 이 마지막에서 두 번째 step 값인지 (DDPMSampler 코드 확인)
- [ ] 정규화 통계 (mean, std) 가 학습 때와 동일한 출처에서 로드되는지
- [ ] 30 ensemble member가 독립 random seed로 생성되는지
- [ ] Test sub-sampling이 시간순 균등 (월/수/금) 인지
- [ ] 분석은 온도 채널만, 실험 5의 composite만 3채널인지
- [ ] 모든 figure가 동일한 random seed로 재현 가능한지
