# Diffusion²-based Flow-Dependent Ensemble Generation 구현 지침

본 문서는 PyTorch 기반 딥러닝 모델 구현을 위한 상세 지침서이다. 본 모델은 Diffusion² 논문 (Luo et al., 2025, arXiv:2510.04365)의 dual-head parameterization mechanism을 활용하여, 단일 시점 기상장으로부터 flow-dependent pixel-wise spread를 가지는 ensemble member를 생성하는 모델이다.

---

## 1. 프로젝트 개요

### 1.1 Task 정의

- **입력**: x_{t-1}, x_t (두 시점의 기상장 텐서)
- **출력**: x_t의 ensemble members {x_t^(b)}_{b=1}^B
- **목표**: 픽셀별로 다른 spread를 가지는 ensemble 생성 (uncertainty가 큰 픽셀에서는 sample들이 더 흩어지도록)

### 1.2 핵심 아이디어

1. **DDPM_past**: x_{t-1}, x_t를 condition으로 받아 x_{t-3}, x_{t-2} (더 먼 과거)를 생성하면서 dual-head로 픽셀별 log-variance ℓ_past를 학습한다.
2. **DDPM_main**: DDPM_past가 생성한 과거와 x_{t-1}을 condition으로 받아 x_t, x_{t+1}을 reconstruction하면서 dual-head로 ℓ_main을 학습한다.
3. **Inference**: 두 디퓨전 모두 매 ensemble member마다 새로 sampling 수행. Spread의 source는 (1) DDPM_past의 stochasticity (다양한 과거 생성), (2) DDPM_main의 reverse variance의 픽셀별 차이.

### 1.3 데이터

본 task는 **ERA5 reanalysis 데이터의 한반도 영역 64×64 patch**를 사용한다. 전처리 파이프라인(`Data_preprocess.md` 참고)이 이미 다음 결과물을 생성한 상태로 가정한다.

**전처리 산출물 (data/ 디렉토리)**:
- `data/normalization_stats.zarr`: 시각별 픽셀별 정규화 통계량
  - `mean`: (4, 3, 64, 64) - (hour, channel, lat, lon)
  - `std`: (4, 3, 64, 64)
  - Coords: hour=[0, 6, 12, 18], channel=['t', 'u', 'v'], lat (64,), lon (64,)
- `data/era5_normalized.zarr`: 사전 정규화된 데이터
  - `normalized`: (T, 3, 64, 64) - (time, channel, lat, lon)
  - T ≈ 33,600 (2000-2022, 6시간 간격)
  - Coords: time (datetime64), channel, lat, lon

**입력 텐서 shape**: (B, C, H, W) per timestep
- B: batch size
- C: 3 (temperature, u-wind, v-wind)
- H = W = 64 (한반도 중심 64×64 patch)

**시점 사용 패턴**:
- 학습 시 5개 연속 시점 동시 사용: (x_{t-3}, x_{t-2}, x_{t-1}, x_t, x_{t+1})
- 추론 시 2개 시점만 사용: (x_{t-1}, x_t)
- 모든 시점이 6시간 간격 (ERA5의 6h 데이터)

**중요: 정규화 처리**
- 학습/추론 시 데이터를 별도로 정규화하지 않는다. `data/era5_normalized.zarr`가 이미 시각별 픽셀별 z-score로 정규화되어 있다.
- 모델은 normalized space에서 학습/추론된다.
- 추론 결과를 원본 단위로 변환하려면 `data/normalization_stats.zarr`의 (μ, σ)를 사용하여 denormalize한다.
- 정규화 방식: `x_normalized = (x - μ[hour(t), c, h, w]) / σ[hour(t), c, h, w]`

**데이터 split (권장)**:
- Train: 2000-2019 (정규화 통계 계산에 사용된 기간)
- Validation: 2020-2021
- Test: 2022

---

## 2. 프로젝트 구조

다음 디렉토리 구조로 코드를 구성한다.

```
project_root/
├── instruction.md              # 본 문서
├── Data_preprocess.md          # 데이터 전처리 지침 (별도 문서)
├── README.md                   # 사용법
├── requirements.txt            # 의존성
├── config/
│   └── default.yaml            # 학습/모델 hyperparameter
├── data/                       # 전처리 결과물 (Data_preprocess.md로 생성됨)
│   ├── normalization_stats.zarr   # 정규화 통계량 (4, 3, 64, 64)
│   └── era5_normalized.zarr        # 정규화된 데이터 (T, 3, 64, 64)
├── preprocess/                 # 전처리 코드 (Data_preprocess.md 참고)
│   ├── __init__.py
│   ├── config.py
│   ├── compute_stats.py
│   ├── normalize_dataset.py
│   ├── normalize_ops.py
│   └── utils.py
├── dataset/                    # 학습/추론용 Dataset 클래스
│   ├── __init__.py
│   └── era5_dataset.py         # 5 시점 동시 로딩 Dataset
├── models/
│   ├── __init__.py
│   ├── encoder.py              # Encoder (Conv3d 기반)
│   ├── unet.py                 # U-Net backbone with self-attention
│   ├── attention.py            # Self-attention block
│   ├── dual_head_ddpm.py       # Dual-head DDPM wrapper
│   └── time_embedding.py       # Sinusoidal time embedding
├── training/
│   ├── __init__.py
│   ├── schedule.py             # Linear noise schedule
│   ├── loss.py                 # Heteroscedastic NLL
│   ├── curriculum.py           # 3-stage curriculum
│   └── train.py                # 학습 메인 스크립트
├── inference/
│   ├── __init__.py
│   ├── sampler.py              # DDPM / DDIM sampler
│   └── ensemble.py             # Ensemble 생성
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # Spread-skill, calibration
│   └── visualize.py            # 시각화 함수
└── scripts/
    ├── train.sh
    └── inference.sh
```

**주의**: `data/` 디렉토리의 두 zarr 파일은 본 instruction에서 다루는 학습/추론 코드의 입력이다. 이 파일들은 `Data_preprocess.md`에 명시된 전처리 코드(`preprocess/`)로 별도 생성되어야 한다.

---

## 3. 수식 및 알고리즘 상세

### 3.1 Notation

- x_τ ∈ R^(C×H×W): 시점 τ의 기상장
- m ∈ {1, 2, ..., M}: diffusion timestep (M은 총 diffusion step 수)
- n = m-1
- α_m, σ_m: noise schedule parameters
- ᾱ_m = ∏_{s=1}^m α_s
- ε ~ N(0, I): standard Gaussian noise
- ε̂: 모델이 예측한 noise
- ℓ: 모델이 예측한 log-variance
- ⊙: element-wise multiplication
- diag(v): 벡터 v를 대각원소로 갖는 대각행렬

### 3.2 Linear Noise Schedule

```
β_1 = 1e-4
β_M = 0.05
β_m = β_1 + (β_M - β_1) * (m - 1) / (M - 1)   # linear interpolation
α_m = 1 - β_m
ᾱ_m = ∏_{s=1}^m α_s

σ_m^2 = 1 - ᾱ_m                                  # forward variance at step m
α(m) = sqrt(ᾱ_m)                                  # forward scale at step m
```

도출 공식 (논문 표기에 맞춤):

```
α_{m|n} = α(m) / α(n) = sqrt(ᾱ_m / ᾱ_n)
σ_{m|n}^2 = σ_m^2 - (α_{m|n})^2 * σ_n^2
```

이들은 reverse process의 mean과 variance 계산에 사용된다.

### 3.3 Forward Process

데이터를 점진적으로 noise로 손상시키는 과정:

```
q(x_m | x_0) = N(x_m; α(m) * x_0, σ_m^2 * I)
```

Reparameterization:

```
x_m = α(m) * x_0 + σ_m * ε,   ε ~ N(0, I)
```

### 3.4 Reverse Process (Dual-head, Eq. 8-10)

**네트워크 출력 (Dual-head)**:

```
(ε̂_θ(x_m, m, c), ℓ_θ(x_m, m, c))
```

c는 condition (Encoder 출력).

**Reverse mean μ_θ (Eq. 9)**:

```
μ_θ = (α_{m|n} * σ_n^2 / σ_m^2) * x_m 
    + (σ_{m|n}^2 * sqrt(ᾱ_n) / σ_m^2) * (x_m - σ_m * ε̂_θ) / α(m)
```

**Reverse covariance Σ_θ (Eq. 10)**:

```
Σ_θ = diag(exp(ℓ_θ)) + (σ_n^2 * σ_{m|n}^2 / σ_m^2) * I
```

첫째 항은 학습된 픽셀별 분산, 둘째 항은 스케줄러 baseline.

### 3.5 Reverse Sampling Step (Eq. 30)

한 step의 reverse sampling:

```
x_{m-1} = μ_θ + exp(0.5 * ℓ_θ) ⊙ ε + (σ_n * σ_{m|n} / σ_m) * ε'
```

여기서 ε, ε' ~ N(0, I)는 서로 독립적인 Gaussian noise.

**Note**: m = 1, n = 0인 마지막 step에서는 σ_0 ≈ 0이므로 둘째 noise 항은 거의 0. 일반적으로 마지막 step에서는 noise를 더하지 않고 μ_θ만 사용하는 경우도 있다. 본 구현에서는 일관성을 위해 위 식을 그대로 사용하되, n = 0이면 σ_n = 0으로 처리한다.

### 3.6 Heteroscedastic NLL Loss (Eq. 20)

본 논문의 핵심 손실 함수. ELBO에서 유도된 형태:

```
L = (1/2) * exp(-ℓ_θ) * ||ε - ε̂_θ||^2 + (1/2) * ℓ_θ
```

여기서:
- 첫째 항: precision-weighted regression. ℓ_θ가 크면 (분산 큼) 오차 페널티가 감소.
- 둘째 항: log-determinant 정규화. ℓ_θ가 무한정 커지는 것을 방지.

**Pixel-wise 적용**: ε, ε̂, ℓ가 모두 (B, T, C, H, W) shape이라면, element-wise 연산 후 평균.

```python
def heteroscedastic_nll_loss(eps_true, eps_pred, log_var):
    # eps_true: (B, T, C, H, W) - 진짜 noise
    # eps_pred: (B, T, C, H, W) - 예측 noise
    # log_var: (B, T, C, H, W) - 예측 log-variance (ℓ)
    
    precision = torch.exp(-log_var)  # (B, T, C, H, W)
    squared_error = (eps_true - eps_pred) ** 2  # (B, T, C, H, W)
    
    nll = 0.5 * precision * squared_error + 0.5 * log_var
    return nll.mean()
```

### 3.7 Numerical Stability 고려

ℓ_θ는 log-variance이므로 unbounded이지만 실용적 안정성을 위해 clipping 권장:

```python
log_var = torch.clamp(log_var, min=-10.0, max=10.0)
```

이는 exp(-10) ≈ 4.5e-5에서 exp(10) ≈ 22026 범위로 분산을 제한한다.

---

## 4. 모듈 구현 상세

### 4.1 Data Loading (`dataset/era5_dataset.py`)

#### 요구사항

- **사전 정규화된 데이터**(`data/era5_normalized.zarr`)를 직접 로드
- 학습 시: 5개 연속 시점 (x_{t-3}, x_{t-2}, x_{t-1}, x_t, x_{t+1})을 동시에 반환
- 추론 시: 2개 시점 (x_{t-1}, x_t)만 반환
- 데이터셋 내부에서는 추가 정규화 연산을 수행하지 않음 (이미 정규화된 데이터 사용)
- Train/val/test split 지원

#### 인터페이스

```python
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset


class ERA5NormalizedDataset(Dataset):
    """사전 정규화된 ERA5 데이터를 5 시점 / 2 시점 단위로 로딩.
    
    데이터 구조:
        data/era5_normalized.zarr: 
            normalized: (T, C=3, H=64, W=64), float32
            Coords: time (datetime64), channel=['t','u','v'], lat (64,), lon (64,)
    """
    
    SPLIT_RANGES = {
        'train':      ('2000-01-01', '2019-12-31'),
        'validation': ('2020-01-01', '2021-12-31'),
        'test':       ('2022-01-01', '2022-12-31'),
    }
    
    def __init__(
        self,
        normalized_path: str = 'data/era5_normalized.zarr',
        mode: str = 'train',           # 'train' or 'inference'
        split: str = 'train',          # 'train' / 'validation' / 'test'
        load_into_memory: bool = False,
    ):
        """
        Args:
            normalized_path: 사전 정규화된 zarr 파일 경로
            mode: 
              - 'train': 5 시점 반환 (학습용)
              - 'inference': 2 시점 반환 (추론용)
            split: 사용할 시간 범위 ('train' / 'validation' / 'test')
            load_into_memory: True면 zarr 전체를 numpy로 메모리에 로드 (빠르지만 RAM 사용 큼)
        """
        super().__init__()
        self.mode = mode
        self.split = split
        
        # zarr 열기
        ds = xr.open_zarr(normalized_path)
        
        # split에 해당하는 시간 범위로 자르기
        start, end = self.SPLIT_RANGES[split]
        ds_split = ds.sel(time=slice(start, end))
        
        # 시간 배열 저장 (인덱싱용)
        self.times = ds_split.time.values
        self.n_times = len(self.times)
        
        # 데이터 핸들
        if load_into_memory:
            # (T, C, H, W) 전체를 numpy로 로드 (빠른 학습용, RAM 충분할 때)
            self.data = ds_split['normalized'].values.astype(np.float32)
            self.zarr_handle = None
        else:
            # zarr를 lazy하게 사용 (메모리 절약)
            self.data = None
            self.zarr_handle = ds_split['normalized']
        
        # 유효 인덱스 범위 (5 시점/2 시점에 필요한 padding 고려)
        # mode='train': idx-3 ~ idx+1이 모두 유효해야 함 → idx ∈ [3, n_times-2]
        # mode='inference': idx-1, idx → idx ∈ [1, n_times-1]
        if mode == 'train':
            self.valid_start = 3
            self.valid_end = self.n_times - 2  # exclusive
        elif mode == 'inference':
            self.valid_start = 1
            self.valid_end = self.n_times
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def __len__(self):
        return self.valid_end - self.valid_start
    
    def _get_frame(self, abs_idx: int) -> np.ndarray:
        """Absolute index의 (C, H, W) frame 반환."""
        if self.data is not None:
            return self.data[abs_idx]
        else:
            # zarr lazy load
            return self.zarr_handle.isel(time=abs_idx).values.astype(np.float32)
    
    def __getitem__(self, idx):
        """
        Args:
            idx: dataset 내부 인덱스 (0 ~ len-1)
        
        Returns:
            mode='train':
                dict {
                    'x_tm3': (C, H, W),
                    'x_tm2': (C, H, W),
                    'x_tm1': (C, H, W),
                    'x_t':   (C, H, W),
                    'x_tp1': (C, H, W),
                    'time_t': datetime64,  # x_t의 시각 (denormalize 시 사용)
                }
            mode='inference':
                dict {
                    'x_tm1':  (C, H, W),
                    'x_t':    (C, H, W),
                    'time_t': datetime64,
                }
        """
        abs_idx = idx + self.valid_start
        
        if self.mode == 'train':
            # 5 시점 슬라이싱
            if self.data is not None:
                # In-memory: 한 번에 슬라이스
                chunk = self.data[abs_idx - 3 : abs_idx + 2]  # (5, C, H, W)
            else:
                chunk = self.zarr_handle.isel(
                    time=slice(abs_idx - 3, abs_idx + 2)
                ).values.astype(np.float32)
            
            return {
                'x_tm3':  torch.from_numpy(chunk[0]),
                'x_tm2':  torch.from_numpy(chunk[1]),
                'x_tm1':  torch.from_numpy(chunk[2]),
                'x_t':    torch.from_numpy(chunk[3]),
                'x_tp1':  torch.from_numpy(chunk[4]),
                'time_t': self.times[abs_idx],
            }
        
        else:  # inference
            x_tm1 = self._get_frame(abs_idx - 1)
            x_t = self._get_frame(abs_idx)
            return {
                'x_tm1':  torch.from_numpy(x_tm1),
                'x_t':    torch.from_numpy(x_t),
                'time_t': self.times[abs_idx],
            }
```

#### Denormalization Utility (`dataset/denormalize.py`)

추론 결과를 원본 단위로 복원할 때 사용. 별도 모듈로 분리.

```python
"""정규화된 텐서를 원본 단위로 역변환."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import xarray as xr


HOUR_TO_IDX = {0: 0, 6: 1, 12: 2, 18: 3}


def load_stats(
    stats_path: str = 'data/normalization_stats.zarr',
    device: str = 'cpu',
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        mean: (4, C=3, H=64, W=64) torch.Tensor
        std:  (4, C=3, H=64, W=64) torch.Tensor
    """
    stats = xr.open_zarr(stats_path).load()
    mean = torch.from_numpy(stats['mean'].values.astype(np.float32)).to(device)
    std = torch.from_numpy(stats['std'].values.astype(np.float32)).to(device)
    return mean, std


def denormalize(
    x_norm: torch.Tensor,
    times: np.ndarray,  # datetime64 array, shape (B,) or scalar
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """역정규화: normalized → 원본 단위.
    
    Args:
        x_norm: (B, C, H, W) or (C, H, W) normalized tensor
        times: 각 sample의 datetime64 (시각 추출에 사용)
        mean, std: (4, C, H, W) stats tensors
    
    Returns:
        x_orig: same shape as x_norm
    """
    if x_norm.ndim == 3:
        # Single sample
        hour = pd.Timestamp(times).hour
        h_idx = HOUR_TO_IDX[hour]
        return x_norm * std[h_idx] + mean[h_idx]
    
    # Batch
    hours = pd.DatetimeIndex(times).hour.to_numpy()
    h_indices = torch.tensor(
        [HOUR_TO_IDX[h] for h in hours],
        dtype=torch.long,
        device=x_norm.device,
    )
    mu_batch = mean[h_indices]    # (B, C, H, W)
    sigma_batch = std[h_indices]
    return x_norm * sigma_batch + mu_batch
```

#### 구현 노트

- **사전 정규화된 데이터 사용**: Dataset 내부에서 정규화 연산 없음. 학습 속도 향상.
- **시간 간격**: 인접 시점 간 6시간 고정 (ERA5 6h 데이터).
- **경계 처리**: `valid_start`, `valid_end`로 padding 영역을 자동으로 제외.
- **In-memory vs lazy**: 
  - 데이터 크기가 작으면 (예: train 20년 × 1460 timesteps × 3 × 64 × 64 × 4 bytes ≈ 4.5 GB) `load_into_memory=True` 권장.
  - 더 큰 데이터셋이거나 RAM 부족 시 False로 두고 zarr를 lazy하게 사용.
- **시각 정보 보존**: `time_t`를 함께 반환하여 추론 후 denormalize 시 사용.
- **추가 정규화 안 함**: 5 시점 모두 같은 zarr에서 가져오므로 자동으로 동일 정규화 적용된 상태.

#### 사용 예

```python
# 학습용 dataset
train_ds = ERA5NormalizedDataset(
    normalized_path='data/era5_normalized.zarr',
    mode='train',
    split='train',
    load_into_memory=True,
)
print(f"Train size: {len(train_ds)}")
sample = train_ds[0]
print(f"x_t shape: {sample['x_t'].shape}")  # (3, 64, 64)

# 추론용 dataset (test split)
test_ds = ERA5NormalizedDataset(
    normalized_path='data/era5_normalized.zarr',
    mode='inference',
    split='test',
    load_into_memory=False,
)

# Denormalize 예시
from dataset.denormalize import load_stats, denormalize
mean, std = load_stats('data/normalization_stats.zarr', device='cuda')
ensemble_normalized = ...  # 모델 추론 결과 (B, C, H, W) on cuda
times = ...  # 해당 sample의 datetime64
ensemble_orig = denormalize(ensemble_normalized, times, mean, std)  # 원본 단위
```

### 4.2 Time Embedding (`models/time_embedding.py`)

표준 sinusoidal positional encoding (Vaswani et al., 2017):

```python
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        """
        Args:
            dim: embedding dimension (보통 128 or 256)
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        """
        Args:
            t: (B,) diffusion timestep (정수)
        Returns:
            (B, dim) embedding
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # (B, dim)
```

이 embedding을 U-Net의 각 block에 AdaGN (Adaptive Group Normalization) 또는 단순 channel-wise addition으로 주입.

### 4.3 Encoder (`models/encoder.py`)

다른 시점의 기상장을 condition feature로 변환.

#### 인터페이스

```python
class TemporalEncoder(nn.Module):
    """시간적으로 stacked된 기상장을 single feature map으로 변환"""
    
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        """
        Args:
            in_channels: 기상 변수 수 C
            hidden_channels: feature dimension C'
            num_layers: Conv3d layer 수
        """
        super().__init__()
        
        layers = []
        c_in = in_channels
        for i in range(num_layers):
            c_out = hidden_channels if i == num_layers - 1 else hidden_channels // 2
            layers.append(nn.Conv3d(
                c_in, c_out,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1)
            ))
            layers.append(nn.GroupNorm(8, c_out))
            layers.append(nn.SiLU())
            c_in = c_out
        
        self.conv = nn.Sequential(*layers)
        
        # 시간 차원 collapse
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - T frames stacked
        Returns:
            (B, C', H, W) - temporal context feature
        """
        # rearrange to (B, C, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        
        # Apply 3D conv
        x = self.conv(x)  # (B, C', T, H, W)
        
        # Pool over time
        x = self.temporal_pool(x)  # (B, C', 1, H, W)
        x = x.squeeze(2)  # (B, C', H, W)
        
        return x
```

#### 사용 예

- DDPM_past condition: T=2 (x_{t-1}, x_t) → (B, C', H, W)
- DDPM_main condition: T=3 (x̂_{t-3}, x̂_{t-2}, x_{t-1}) → (B, C', H, W)

같은 Encoder 인스턴스 사용 (parameter sharing). T가 달라도 AdaptiveAvgPool3d로 동일한 출력 shape 보장.

### 4.4 Self-Attention Block (`models/attention.py`)

U-Net의 하위 layer에서 사용할 spatial self-attention.

```python
class SpatialSelfAttention(nn.Module):
    """Spatial self-attention for U-Net feature maps"""
    
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # QKV projection
        qkv = self.qkv(h)  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)  # (B, C, H, W) each
        
        # Reshape for multi-head attention
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)
        
        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * scale  # (B, heads, HW, HW)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhij,bhdj->bhdi', attn, v)  # (B, heads, head_dim, HW)
        
        # Reshape back
        out = out.reshape(B, C, H, W)
        out = self.proj(out)
        
        return x + out  # residual
```

### 4.5 U-Net Backbone (`models/unet.py`)

Diffusion model의 main backbone. 4 단계 downsampling/upsampling, 하위 2 단계에 self-attention.

#### 인터페이스

```python
class UNet(nn.Module):
    """
    U-Net backbone for diffusion model.
    
    Input/Output shape: (B, T*C_data, H, W)
        - T 시점이 channel dimension에 펼쳐짐
        - 예: target이 (x_t, x_{t+1})이고 C_data 변수라면 T*C_data = 2*C_data
    """
    
    def __init__(self,
                 in_channels,           # T * C_data + condition channels
                 out_channels,          # T * C_data * 2 (dual-head: ε와 ℓ)
                 base_channels=64,
                 channel_mults=(1, 2, 4, 8),
                 num_res_blocks=2,
                 attention_resolutions=(0, 1),  # 0=largest, increasing=smaller
                 time_emb_dim=256,
                 dropout=0.1):
        super().__init__()
        # ...
    
    def forward(self, x, t, cond):
        """
        Args:
            x: (B, T*C_data, H, W) - noisy target (channel-concat across time)
            t: (B,) - diffusion timestep
            cond: (B, C', H, W) - condition from Encoder
        
        Returns:
            (B, out_channels, H, W) - dual-head output
        """
        pass
```

#### ResNet Block 구조

```python
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.skip = (nn.Conv2d(in_channels, out_channels, 1) 
                     if in_channels != out_channels else nn.Identity())
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)
```

#### Condition Injection

Condition `cond`은 (B, C', H, W) shape. 입력 단계에서 channel concatenation으로 결합:

```python
def forward(self, x, t, cond):
    # x: (B, T*C_data, H, W)
    # cond: (B, C', H, W)
    
    # Channel-concat condition with input
    x_in = torch.cat([x, cond], dim=1)  # (B, T*C_data + C', H, W)
    
    # Time embedding
    t_emb = self.time_embedding(t)
    
    # ... U-Net 본체 ...
```

#### Attention 위치

`attention_resolutions=(0, 1)`은 인덱싱 컨벤션 명확히 정의:

- 0: 가장 깊은 layer (가장 작은 resolution)
- 1: 두 번째로 깊은 layer
- 이게 "하위 2 layer에 self-attention"의 정확한 의미

총 4단계 downsampling이라면, attention은 가장 작은 2 resolution에서 적용 (예: H/8 × W/8, H/16 × W/16).

### 4.6 Dual-Head DDPM Wrapper (`models/dual_head_ddpm.py`)

U-Net 출력을 ε̂와 ℓ로 분리하는 wrapper.

```python
class DualHeadDDPM(nn.Module):
    """
    Dual-head diffusion model wrapper.
    U-Net 출력 channel을 ε̂와 ℓ로 분리.
    """
    
    def __init__(self, unet, target_channels, log_var_clip=(-10.0, 10.0)):
        """
        Args:
            unet: UNet 인스턴스
            target_channels: T * C_data (출력의 절반)
            log_var_clip: ℓ clipping range
        """
        super().__init__()
        self.unet = unet
        self.target_channels = target_channels
        self.log_var_clip = log_var_clip
    
    def forward(self, x, t, cond):
        """
        Args:
            x: (B, T*C_data, H, W) noisy target
            t: (B,) timestep
            cond: (B, C', H, W) condition
        
        Returns:
            eps_pred: (B, T*C_data, H, W)
            log_var: (B, T*C_data, H, W) - clipped
        """
        out = self.unet(x, t, cond)  # (B, 2*T*C_data, H, W)
        
        eps_pred = out[:, :self.target_channels]
        log_var = out[:, self.target_channels:]
        log_var = torch.clamp(log_var, *self.log_var_clip)
        
        return eps_pred, log_var
```

### 4.7 Noise Schedule (`training/schedule.py`)

Linear schedule + reverse process 관련 계산.

```python
class LinearNoiseSchedule:
    def __init__(self, M=200, beta_start=1e-4, beta_end=0.05, device='cuda'):
        self.M = M
        
        # Linear betas
        self.betas = torch.linspace(beta_start, beta_end, M, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute commonly used quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sigma_sq = 1.0 - self.alphas_cumprod  # σ_m^2
    
    def get_alpha_bar(self, m):
        """ᾱ_m"""
        return self.alphas_cumprod[m]
    
    def get_sigma_sq(self, m):
        """σ_m^2 = 1 - ᾱ_m"""
        return self.sigma_sq[m]
    
    def get_alpha_mn(self, m, n):
        """α_{m|n} = sqrt(ᾱ_m / ᾱ_n)"""
        if n < 0:
            return torch.sqrt(self.alphas_cumprod[m])
        return torch.sqrt(self.alphas_cumprod[m] / self.alphas_cumprod[n])
    
    def get_sigma_mn_sq(self, m, n):
        """σ_{m|n}^2 = σ_m^2 - α_{m|n}^2 * σ_n^2"""
        if n < 0:
            return self.sigma_sq[m]
        alpha_mn = self.get_alpha_mn(m, n)
        return self.sigma_sq[m] - alpha_mn ** 2 * self.sigma_sq[n]
    
    def forward_noise(self, x_0, m, noise=None):
        """
        Forward diffusion: x_m = sqrt(ᾱ_m) * x_0 + sqrt(1 - ᾱ_m) * ε
        
        Args:
            x_0: (B, ...) clean data
            m: (B,) timestep
            noise: optional, sampled if None
        
        Returns:
            x_m: noisy data
            noise: used noise (for loss computation)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Broadcast m-indexed values to match x_0 shape
        sqrt_ab = self.sqrt_alphas_cumprod[m].view(-1, *([1] * (x_0.dim() - 1)))
        sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[m].view(-1, *([1] * (x_0.dim() - 1)))
        
        x_m = sqrt_ab * x_0 + sqrt_one_minus_ab * noise
        return x_m, noise
    
    def reverse_mean(self, x_m, eps_pred, m):
        """
        μ_θ 계산 (Eq. 9)
        
        Args:
            x_m: (B, ...) at step m
            eps_pred: (B, ...) predicted noise
            m: (B,) or scalar timestep
        
        Returns:
            mu: (B, ...) reverse mean
        """
        n = m - 1
        
        sigma_m_sq = self.get_sigma_sq(m).view(-1, *([1] * (x_m.dim() - 1)))
        alpha_m = self.sqrt_alphas_cumprod[m].view(-1, *([1] * (x_m.dim() - 1)))
        
        if torch.any(n < 0):
            # m == 0 case: x_0 = (x_m - σ_m * eps_pred) / α(m)
            sigma_m = torch.sqrt(sigma_m_sq)
            return (x_m - sigma_m * eps_pred) / alpha_m
        
        sigma_n_sq = self.get_sigma_sq(n).view(-1, *([1] * (x_m.dim() - 1)))
        alpha_mn = self.get_alpha_mn(m, n).view(-1, *([1] * (x_m.dim() - 1)))
        sigma_mn_sq = self.get_sigma_mn_sq(m, n).view(-1, *([1] * (x_m.dim() - 1)))
        sqrt_alpha_n = self.sqrt_alphas_cumprod[n].view(-1, *([1] * (x_m.dim() - 1)))
        sigma_m = torch.sqrt(sigma_m_sq)
        
        # μ_θ = (α_{m|n} σ_n^2 / σ_m^2) x_m + (σ_{m|n}^2 √ᾱ_n / σ_m^2) (x_m - σ_m ε̂) / α(m)
        term1 = (alpha_mn * sigma_n_sq / sigma_m_sq) * x_m
        term2_inner = (x_m - sigma_m * eps_pred) / alpha_m
        term2 = (sigma_mn_sq * sqrt_alpha_n / sigma_m_sq) * term2_inner
        
        return term1 + term2
    
    def reverse_variance_scheduler_term(self, m):
        """
        Scheduler baseline variance: σ_n^2 σ_{m|n}^2 / σ_m^2
        """
        n = m - 1
        if torch.any(n < 0):
            return torch.zeros_like(self.sigma_sq[m])
        
        sigma_m_sq = self.get_sigma_sq(m)
        sigma_n_sq = self.get_sigma_sq(n)
        sigma_mn_sq = self.get_sigma_mn_sq(m, n)
        
        return sigma_n_sq * sigma_mn_sq / sigma_m_sq
```

### 4.8 Loss Function (`training/loss.py`)

```python
def heteroscedastic_nll_loss(eps_true, eps_pred, log_var, reduction='mean'):
    """
    Heteroscedastic Gaussian NLL loss (Eq. 20)
    
    L = 0.5 * exp(-ℓ) * ||ε - ε̂||^2 + 0.5 * ℓ
    
    Args:
        eps_true: (B, ...) true noise
        eps_pred: (B, ...) predicted noise
        log_var: (B, ...) predicted log-variance (ℓ)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        loss: scalar (if reduction='mean' or 'sum') or (B, ...) tensor
    """
    precision = torch.exp(-log_var)
    squared_error = (eps_true - eps_pred) ** 2
    
    loss = 0.5 * precision * squared_error + 0.5 * log_var
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
```

### 4.9 Sampler (`inference/sampler.py`)

#### DDPM Sampler (학습용 + 추론용)

```python
class DDPMSampler:
    def __init__(self, schedule):
        self.schedule = schedule
    
    @torch.no_grad()
    def sample(self, model, cond, shape, device, return_trajectory=False):
        """
        Full reverse chain sampling.
        
        Args:
            model: DualHeadDDPM 인스턴스
            cond: (B, C', H, W) condition
            shape: 생성할 sample의 shape (예: (B, T*C_data, H, W))
            device: 'cuda' or 'cpu'
            return_trajectory: True면 모든 step의 sample 반환
        
        Returns:
            x_0: 최종 sample
            (optional) trajectory: list of intermediate samples
            ell_final: 마지막 step의 ℓ_θ (u 계산에 사용)
        """
        M = self.schedule.M
        B = cond.shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        trajectory = [x.clone()] if return_trajectory else None
        ell_final = None
        
        for step in range(M - 1, -1, -1):
            m = torch.full((B,), step, device=device, dtype=torch.long)
            
            eps_pred, log_var = model(x, m, cond)
            
            if step == 0:
                # 마지막 step: noise 안 더함
                mu = self.schedule.reverse_mean(x, eps_pred, m)
                x = mu
                ell_final = log_var
            else:
                # Reverse sampling step (Eq. 30)
                mu = self.schedule.reverse_mean(x, eps_pred, m)
                
                # Learned variance term
                eps_learned = torch.randn_like(x)
                learned_noise = torch.exp(0.5 * log_var) * eps_learned
                
                # Scheduler baseline term
                sched_var = self.schedule.reverse_variance_scheduler_term(m)
                sched_var = sched_var.view(-1, *([1] * (x.dim() - 1)))
                eps_sched = torch.randn_like(x)
                sched_noise = torch.sqrt(sched_var) * eps_sched
                
                x = mu + learned_noise + sched_noise
                
                if step == 1:
                    # m = 1에서의 ℓ를 final로 사용 (가장 신호 강한 영역)
                    ell_final = log_var
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        return x, trajectory, ell_final
```

#### DDIM Sampler (선택적, 추론 가속용)

```python
class DDIMSampler:
    def __init__(self, schedule, num_inference_steps=20, eta=0.0):
        """
        Args:
            schedule: LinearNoiseSchedule (M=200 학습된 schedule)
            num_inference_steps: DDIM step 수 (M보다 작게)
            eta: 0이면 deterministic, 1이면 DDPM과 동일
        """
        self.schedule = schedule
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        
        # Sub-sampled timesteps
        step_size = schedule.M // num_inference_steps
        self.timesteps = list(range(0, schedule.M, step_size))[::-1]
    
    @torch.no_grad()
    def sample(self, model, cond, shape, device):
        """
        DDIM sampling. 표준 DDIM update rule 사용.
        Dual-head ℓ은 마지막 step에서 추출하여 반환.
        """
        # 표준 DDIM 구현 (Song et al., 2020)
        # ε̂_θ는 매 step 호출, ℓ_θ는 마지막 step만 저장
        # eta=0 deterministic의 경우 learned variance 사용 안 함
        # eta>0이면 stochastic, learned variance를 weighted noise로 활용 가능
        pass  # 자세한 구현은 표준 DDIM 참고
```

### 4.10 Curriculum Training (`training/curriculum.py`)

```python
class CurriculumStage:
    """학습 단계 정의"""
    PAST_ONLY = 1   # DDPM_past만 학습
    MAIN_ONLY = 2   # DDPM_past freeze, DDPM_main 학습
    JOINT = 3       # 모두 학습

def get_stage(epoch, stage1_epochs, stage2_epochs, stage3_epochs):
    if epoch < stage1_epochs:
        return CurriculumStage.PAST_ONLY
    elif epoch < stage1_epochs + stage2_epochs:
        return CurriculumStage.MAIN_ONLY
    else:
        return CurriculumStage.JOINT

def get_loss_weights(stage):
    if stage == CurriculumStage.PAST_ONLY:
        return {'past': 1.0, 'main': 0.0}
    elif stage == CurriculumStage.MAIN_ONLY:
        return {'past': 0.0, 'main': 1.0}  # past는 frozen이라 학습 안 됨
    else:  # JOINT
        return {'past': 1.0, 'main': 1.0}

def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad
```

### 4.11 Training Loop (`training/train.py`)

전체 학습 루프의 가장 중요한 부분.

```python
def train_step(batch, models, optimizer, schedule, encoder, 
               sampler_for_aux, stage, device):
    """
    한 step의 학습.
    
    Args:
        batch: dict {x_tm3, x_tm2, x_tm1, x_t, x_tp1} - 각 (B, C, H, W)
        models: dict {'past': DDPM_past, 'main': DDPM_main}
        optimizer: AdamW
        schedule: LinearNoiseSchedule
        encoder: TemporalEncoder
        sampler_for_aux: DDPMSampler (학습 중 DDPM_past sampling용)
        stage: CurriculumStage
        device: 'cuda'
    """
    x_tm3 = batch['x_tm3'].to(device)  # (B, C, H, W)
    x_tm2 = batch['x_tm2'].to(device)
    x_tm1 = batch['x_tm1'].to(device)
    x_t   = batch['x_t'].to(device)
    x_tp1 = batch['x_tp1'].to(device)
    
    B, C, H, W = x_t.shape
    
    weights = get_loss_weights(stage)
    L_past = torch.tensor(0.0, device=device)
    L_main = torch.tensor(0.0, device=device)
    
    # ========== DDPM_past 학습 ==========
    if weights['past'] > 0:
        # Condition: stack(x_{t-1}, x_t)
        cond_past_input = torch.stack([x_tm1, x_t], dim=1)  # (B, 2, C, H, W)
        cond_past = encoder(cond_past_input)  # (B, C', H, W)
        
        # Target: stack(x_{t-3}, x_{t-2})
        target_past = torch.stack([x_tm3, x_tm2], dim=1)  # (B, 2, C, H, W)
        target_past_flat = target_past.reshape(B, 2 * C, H, W)  # channel concat
        
        # Sample random timestep
        m = torch.randint(0, schedule.M, (B,), device=device)
        
        # Forward noise
        noisy_past, eps_true_past = schedule.forward_noise(target_past_flat, m)
        
        # Predict
        eps_pred_past, log_var_past = models['past'](noisy_past, m, cond_past)
        
        # Heteroscedastic NLL
        L_past = heteroscedastic_nll_loss(eps_true_past, eps_pred_past, log_var_past)
    
    # ========== DDPM_past sampling (학습 중 condition 만들기) ==========
    if weights['main'] > 0:
        # Past sampling은 inference처럼 진행
        # 학습 효율을 위해 num_inference_steps를 줄일 수 있음 (예: DDIM 20 step)
        cond_past_input = torch.stack([x_tm1, x_t], dim=1)
        cond_past = encoder(cond_past_input)
        
        with torch.no_grad():  # Sampling chain에는 gradient 차단
            past_sample, _, _ = sampler_for_aux.sample(
                models['past'],
                cond=cond_past,
                shape=(B, 2 * C, H, W),
                device=device
            )
            # past_sample: (B, 2C, H, W)
            past_sample = past_sample.reshape(B, 2, C, H, W)
            x_tm3_hat = past_sample[:, 0]
            x_tm2_hat = past_sample[:, 1]
        
        # ========== DDPM_main 학습 ==========
        # Condition: stack(x̂_{t-3}, x̂_{t-2}, x_{t-1})
        cond_main_input = torch.stack([x_tm3_hat, x_tm2_hat, x_tm1], dim=1)  # (B, 3, C, H, W)
        cond_main = encoder(cond_main_input)  # (B, C', H, W)
        
        # Target: stack(x_t, x_{t+1})
        target_main = torch.stack([x_t, x_tp1], dim=1)  # (B, 2, C, H, W)
        target_main_flat = target_main.reshape(B, 2 * C, H, W)
        
        m = torch.randint(0, schedule.M, (B,), device=device)
        noisy_main, eps_true_main = schedule.forward_noise(target_main_flat, m)
        
        eps_pred_main, log_var_main = models['main'](noisy_main, m, cond_main)
        
        L_main = heteroscedastic_nll_loss(eps_true_main, eps_pred_main, log_var_main)
    
    # ========== Total Loss & Backward ==========
    L_total = weights['past'] * L_past + weights['main'] * L_main
    
    optimizer.zero_grad()
    L_total.backward()
    optimizer.step()
    
    return {
        'L_total': L_total.item(),
        'L_past': L_past.item(),
        'L_main': L_main.item(),
    }


def main_training(config):
    """전체 학습 loop"""
    device = 'cuda'
    
    # Build models
    encoder = TemporalEncoder(...).to(device)
    unet_past = UNet(...).to(device)
    unet_main = UNet(...).to(device)
    
    models = {
        'past': DualHeadDDPM(unet_past, target_channels=2*C).to(device),
        'main': DualHeadDDPM(unet_main, target_channels=2*C).to(device),
    }
    
    schedule = LinearNoiseSchedule(M=config.M, device=device)
    sampler = DDPMSampler(schedule)  # or DDIMSampler with fewer steps for efficiency
    
    optimizer = torch.optim.AdamW(
        list(encoder.parameters())
        + list(models['past'].parameters())
        + list(models['main'].parameters()),
        lr=config.lr
    )
    
    # Dataset (사전 정규화된 zarr 사용)
    from dataset.era5_dataset import ERA5NormalizedDataset
    from torch.utils.data import DataLoader
    
    train_dataset = ERA5NormalizedDataset(
        normalized_path=config.data.normalized_path,
        mode='train',
        split='train',
        load_into_memory=config.data.load_into_memory,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_dataset = ERA5NormalizedDataset(
        normalized_path=config.data.normalized_path,
        mode='train',          # 학습 셋과 같은 형식 (5 시점)
        split='validation',
        load_into_memory=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Curriculum stages
    for epoch in range(config.total_epochs):
        stage = get_stage(epoch, 
                         config.stage1_epochs,
                         config.stage2_epochs,
                         config.stage3_epochs)
        
        # Set requires_grad based on stage
        if stage == CurriculumStage.MAIN_ONLY:
            set_requires_grad(models['past'], False)
            set_requires_grad(encoder, True)  # Encoder는 main과 같이 학습
            set_requires_grad(models['main'], True)
        elif stage == CurriculumStage.PAST_ONLY:
            set_requires_grad(models['past'], True)
            set_requires_grad(encoder, True)
            set_requires_grad(models['main'], False)
        else:  # JOINT
            set_requires_grad(models['past'], True)
            set_requires_grad(encoder, True)
            set_requires_grad(models['main'], True)
        
        for batch in train_loader:
            losses = train_step(batch, models, optimizer, schedule,
                              encoder, sampler, stage, device)
            # ... log losses ...
        
        # Save checkpoint, validation, etc.
```

### 4.12 Ensemble Inference (`inference/ensemble.py`)

```python
@torch.no_grad()
def generate_ensemble(x_tm1, x_t, models, encoder, sampler, schedule, B=20, device='cuda'):
    """
    x_t의 ensemble member 생성.
    
    Args:
        x_tm1: (1, C, H, W) input at t-1
        x_t: (1, C, H, W) input at t
        models: dict {'past': DDPM_past, 'main': DDPM_main}
        encoder: TemporalEncoder
        sampler: DDPMSampler 또는 DDIMSampler
        schedule: LinearNoiseSchedule
        B: ensemble size
        device: 'cuda'
    
    Returns:
        ensemble: (B, C, H, W) - x_t의 ensemble members
        diagnostics: dict with ℓ_past, ℓ_main, etc.
    """
    x_tm1 = x_tm1.to(device)
    x_t = x_t.to(device)
    _, C, H, W = x_tm1.shape
    
    # Condition for DDPM_past (모든 b에 공통)
    cond_past_input = torch.stack([x_tm1, x_t], dim=1)  # (1, 2, C, H, W)
    cond_past = encoder(cond_past_input)  # (1, C', H, W)
    
    ensemble = []
    log_vars_past_collected = []
    log_vars_main_collected = []
    
    for b in range(B):
        # ========== Step 1: DDPM_past sampling (b마다 새 random seed) ==========
        past_sample, _, ell_past = sampler.sample(
            models['past'],
            cond=cond_past,
            shape=(1, 2 * C, H, W),
            device=device
        )
        past_sample = past_sample.reshape(1, 2, C, H, W)
        x_tm3_hat = past_sample[:, 0]
        x_tm2_hat = past_sample[:, 1]
        
        log_vars_past_collected.append(ell_past.cpu())
        
        # ========== Step 2: DDPM_main sampling ==========
        cond_main_input = torch.stack([x_tm3_hat, x_tm2_hat, x_tm1], dim=1)  # (1, 3, C, H, W)
        cond_main = encoder(cond_main_input)  # (1, C', H, W)
        
        main_sample, _, ell_main = sampler.sample(
            models['main'],
            cond=cond_main,
            shape=(1, 2 * C, H, W),
            device=device
        )
        main_sample = main_sample.reshape(1, 2, C, H, W)
        x_t_hat_b = main_sample[:, 0]  # x_t 위치만 추출
        
        ensemble.append(x_t_hat_b.squeeze(0))
        log_vars_main_collected.append(ell_main.cpu())
    
    ensemble = torch.stack(ensemble, dim=0)  # (B, C, H, W)
    
    diagnostics = {
        'log_vars_past': torch.stack(log_vars_past_collected),
        'log_vars_main': torch.stack(log_vars_main_collected),
    }
    
    return ensemble, diagnostics
```

### 4.13 Evaluation Metrics (`evaluation/metrics.py`)

```python
def compute_spread(ensemble):
    """
    Pixel-wise ensemble spread.
    
    Args:
        ensemble: (B, C, H, W)
    
    Returns:
        spread: (C, H, W) - per-pixel std across ensemble members
    """
    return ensemble.std(dim=0)


def compute_skill(ensemble_mean, ground_truth):
    """
    Pixel-wise forecast skill (absolute error of ensemble mean).
    
    Args:
        ensemble_mean: (C, H, W)
        ground_truth: (C, H, W)
    
    Returns:
        skill: (C, H, W) - per-pixel absolute error
    """
    return (ensemble_mean - ground_truth).abs()


def spread_skill_ratio(ensemble, ground_truth):
    """
    Domain-averaged spread-skill ratio. Ideally close to 1.
    
    < 1: underdispersive (spread too small)
    > 1: overdispersive (spread too large)
    """
    spread = compute_spread(ensemble)
    skill = compute_skill(ensemble.mean(dim=0), ground_truth)
    return spread.mean() / (skill.mean() + 1e-8)


def spatial_correlation(map1, map2):
    """
    Pearson correlation between two spatial maps.
    
    Args:
        map1, map2: (..., H, W) - flatten 후 비교
    
    Returns:
        correlation: scalar
    """
    flat1 = map1.flatten()
    flat2 = map2.flatten()
    
    flat1_c = flat1 - flat1.mean()
    flat2_c = flat2 - flat2.mean()
    
    num = (flat1_c * flat2_c).sum()
    denom = torch.sqrt((flat1_c ** 2).sum() * (flat2_c ** 2).sum())
    
    return (num / (denom + 1e-8)).item()


def crps(ensemble, ground_truth):
    """
    Continuous Ranked Probability Score.
    
    Args:
        ensemble: (B, C, H, W) - B ensemble members
        ground_truth: (C, H, W)
    
    Returns:
        crps: (C, H, W) - per-pixel CRPS
    """
    B = ensemble.shape[0]
    
    # Term 1: E|X - y|
    term1 = (ensemble - ground_truth.unsqueeze(0)).abs().mean(dim=0)
    
    # Term 2: 0.5 * E|X - X'|
    # Note: B^2 항을 모두 계산하므로 메모리 주의 (필요시 batch로 처리)
    diffs = (ensemble.unsqueeze(0) - ensemble.unsqueeze(1)).abs()  # (B, B, C, H, W)
    term2 = 0.5 * diffs.mean(dim=(0, 1))
    
    return term1 - term2  # lower is better


def rank_histogram(ensembles, ground_truths, num_bins=None):
    """
    Rank histogram for ensemble calibration.
    
    Args:
        ensembles: list of (B, C, H, W) - multiple time samples
        ground_truths: list of (C, H, W)
        num_bins: if None, B+1
    
    Returns:
        histogram: (num_bins,) - flat distribution = well-calibrated
    """
    ranks = []
    for ens, gt in zip(ensembles, ground_truths):
        B = ens.shape[0]
        combined = torch.cat([gt.unsqueeze(0), ens], dim=0)  # (B+1, C, H, W)
        rank = torch.argsort(combined, dim=0).argsort(dim=0)[0]  # GT의 rank
        ranks.append(rank.flatten())
    
    all_ranks = torch.cat(ranks).cpu().numpy()
    bins = num_bins if num_bins else B + 1
    return np.histogram(all_ranks, bins=bins)[0]
```

### 4.14 Visualization (`evaluation/visualize.py`)

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_ensemble_member(ensemble, variable_names, save_path=None, n_show=16):
    """
    Ensemble member들을 grid로 시각화.
    
    Args:
        ensemble: (B, C, H, W)
        variable_names: list of variable names
        save_path: 저장 경로
        n_show: 보여줄 member 수
    """
    n_show = min(n_show, ensemble.shape[0])
    n_vars = len(variable_names)
    
    for var_idx, var_name in enumerate(variable_names):
        fig, axes = plt.subplots(4, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        vmin = ensemble[:n_show, var_idx].min()
        vmax = ensemble[:n_show, var_idx].max()
        
        for i in range(n_show):
            ax = axes[i]
            im = ax.imshow(ensemble[i, var_idx].cpu().numpy(),
                          vmin=vmin, vmax=vmax, cmap='RdBu_r')
            ax.set_title(f'Member {i+1}')
            ax.axis('off')
        
        plt.suptitle(f'Ensemble Members: {var_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}_{var_name}.png', dpi=150)
        plt.close()


def visualize_spread_vs_uncertainty(ensemble, log_var_main, variable_names, save_path=None):
    """
    Ensemble spread와 학습된 ℓ_main 비교.
    
    Args:
        ensemble: (B, C, H, W)
        log_var_main: (C, H, W) - ℓ_main 평균 (또는 첫 ensemble member의 값)
        variable_names: list
    """
    spread = ensemble.std(dim=0).cpu().numpy()  # (C, H, W)
    uncertainty = torch.exp(log_var_main).sqrt().cpu().numpy()  # (C, H, W)
    
    n_vars = len(variable_names)
    fig, axes = plt.subplots(n_vars, 2, figsize=(12, 4 * n_vars))
    
    if n_vars == 1:
        axes = axes[None, :]
    
    for var_idx, var_name in enumerate(variable_names):
        ax = axes[var_idx, 0]
        im = ax.imshow(spread[var_idx], cmap='viridis')
        ax.set_title(f'{var_name}: Ensemble Spread')
        plt.colorbar(im, ax=ax)
        
        ax = axes[var_idx, 1]
        im = ax.imshow(uncertainty[var_idx], cmap='viridis')
        ax.set_title(f'{var_name}: Predicted Uncertainty (√exp(ℓ_main))')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
```

---

## 5. Configuration

### 5.1 Default config (`config/default.yaml`)

```yaml
# Data
data:
  normalized_path: 'data/era5_normalized.zarr'
  stats_path: 'data/normalization_stats.zarr'
  variables: ['t', 'u', 'v']     # temperature, u-wind, v-wind
  n_channels: 3
  spatial: [64, 64]               # H x W (한반도 patch)
  time_step_hours: 6
  
  splits:
    train:      ['2000-01-01', '2019-12-31']
    validation: ['2020-01-01', '2021-12-31']
    test:       ['2022-01-01', '2022-12-31']
  
  load_into_memory: true   # train data를 RAM에 전체 로드 (속도 향상, ~4.5GB)

# Model
model:
  encoder:
    in_channels: 3        # 변수 수와 일치
    hidden_channels: 64
    num_layers: 2
  
  unet:
    base_channels: 64
    channel_mults: [1, 2, 4, 8]
    num_res_blocks: 2
    attention_resolutions: [0, 1]  # lower 2 resolutions
    time_emb_dim: 256
    dropout: 0.1
  
  dual_head:
    log_var_clip: [-10.0, 10.0]

# Diffusion
diffusion:
  M: 200
  beta_start: 1.0e-4
  beta_end: 0.05

# Training
training:
  batch_size: 16
  lr: 1.0e-4
  weight_decay: 1.0e-5
  total_epochs: 200
  
  curriculum:
    stage1_epochs: 50   # DDPM_past only
    stage2_epochs: 100  # DDPM_main only (past frozen)
    stage3_epochs: 50   # Joint
  
  # Auxiliary sampling during training (DDPM_past chain)
  aux_sampler:
    type: 'ddim'      # 'ddpm' or 'ddim'
    num_steps: 20     # for ddim
  
  checkpoint_every: 10
  validation_every: 5

# Inference
inference:
  sampler:
    type: 'ddim'
    num_steps: 20
    eta: 0.5
  
  ensemble_size: 20

# Logging
logging:
  output_dir: 'outputs/'
  save_visualizations: true
  log_every: 100
```

---

## 6. 학습 및 추론 실행

### 6.1 전처리 (먼저 실행 - 한 번만)

`Data_preprocess.md`의 지침에 따라:

```bash
# 1. 정규화 통계량 계산 (2000-2019 train split 기준)
python -m preprocess.compute_stats

# 2. 정규화된 데이터 zarr 저장 (2000-2022 전체)
python -m preprocess.normalize_dataset
```

위 두 단계가 완료되면 `data/normalization_stats.zarr`와 `data/era5_normalized.zarr`가 생성된다.

### 6.2 학습 스크립트 (`scripts/train.sh`)

```bash
#!/bin/bash
python -m training.train \
    --config config/default.yaml \
    --output_dir outputs/experiment_1 \
    --resume_from None
```

### 6.3 추론 스크립트 (`scripts/inference.sh`)

```bash
#!/bin/bash
python -m inference.ensemble \
    --config config/default.yaml \
    --checkpoint outputs/experiment_1/checkpoint_best.pt \
    --split test \
    --output_dir outputs/inference_1 \
    --ensemble_size 20
```

추론은 `data/era5_normalized.zarr`의 test split (2022)을 자동으로 사용한다.

---

## 7. 검증 체크리스트

구현 완료 후 다음 사항을 확인한다.

### 7.0 데이터 로딩 검증

- [ ] `data/era5_normalized.zarr`가 존재하고 shape이 (T, 3, 64, 64)인가
- [ ] `data/normalization_stats.zarr`가 존재하고 shape이 (4, 3, 64, 64)인가
- [ ] `ERA5NormalizedDataset(mode='train', split='train')[0]`의 출력이 5 시점을 모두 포함하는가
- [ ] 각 sample의 텐서가 (3, 64, 64) shape, float32 dtype인가
- [ ] Train split의 normalized data 전체 평균 ≈ 0, std ≈ 1인가
- [ ] 시각별로 데이터를 분리했을 때 픽셀별 평균 ≈ 0, std ≈ 1인가

### 7.1 학습 동작 확인

- [ ] L_past가 감소하는가
- [ ] L_main이 감소하는가
- [ ] ℓ_past, ℓ_main이 학습 시 합리적 범위에 있는가 (-5 ~ 5 정도)
- [ ] ℓ_past, ℓ_main의 spatial pattern이 의미 있는가 (모든 픽셀에서 같은 값이 아님)

### 7.2 추론 동작 확인

- [ ] DDPM_past가 매 ensemble member마다 다른 sample을 생성하는가
- [ ] DDPM_main이 매번 다른 x_t를 생성하는가
- [ ] Ensemble member들의 spread가 픽셀별로 다른가
- [ ] Spread의 spatial pattern이 ℓ_main의 spatial pattern과 correlate 되는가

### 7.3 정량적 검증

- [ ] Spread-skill ratio 측정 (이상적으로 ≈ 1)
- [ ] Spread와 ℓ_main의 spatial correlation > 0.5
- [ ] CRPS 측정 및 단일 deterministic baseline과 비교
- [ ] Rank histogram이 flat에 가까운가

### 7.4 시각적 검증

- [ ] 16개 ensemble member를 grid로 plot
- [ ] Spread map과 uncertainty map을 나란히 plot
- [ ] 기상학적으로 변동성이 큰 영역 (전선, 산악, 해안)에서 spread가 큰지 확인

---

## 8. 잠재적 문제 및 디버깅

### 8.1 ℓ가 너무 작게 학습됨

증상: ℓ_main이 거의 모든 픽셀에서 작은 값. Ensemble spread가 거의 없음.

원인: x_{t-1}이 너무 informative해서 x_t 예측이 정확.

대응:
1. Scaling factor β 도입: `Σ = diag(β * exp(ℓ_main)) + scheduler_term` (β > 1)
2. 학습 시 x_{t-1}에 작은 noise augmentation: `x_tm1 + 0.01 * randn_like(x_tm1)`
3. Log_var의 clipping 하한을 풀어줌 (-10 → -5 등)

### 8.2 ℓ가 너무 크게 학습됨

증상: ℓ_main이 매우 큼. Sample이 noise처럼 보임.

원인: ε 예측이 부정확. 모델이 수렴하지 못함.

대응:
1. 학습률 감소
2. Log_var clipping 상한 강화 (10 → 3)
3. 학습 epoch 늘리기
4. ε prediction loss (precision 항)에 가중치 더 줘서 정확한 ε 예측 강제

### 8.3 학습이 불안정함

증상: Loss가 발산하거나 NaN

원인: 
- exp(-ℓ)이 매우 큰 값이 되어 gradient explosion
- AdaGN의 instability

대응:
1. Gradient clipping: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`
2. Log_var clipping 범위 좁히기
3. Warmup learning rate

### 8.4 Stage 2 학습에서 DDPM_main이 학습 안 됨

증상: Stage 2에서 L_main이 감소하지 않음

원인: DDPM_past의 sampling이 부정확해서 c_main이 noise처럼 됨

대응:
1. Stage 1 더 오래
2. Stage 1 후 ℓ_past의 spatial pattern 확인 (의미 있는 학습 검증)
3. Sampling 시 더 많은 step 사용

### 8.5 추론 속도 문제

증상: B=20 ensemble 생성에 너무 오래 걸림

대응:
1. DDIM 사용 (M=200 → num_steps=20)
2. Batch processing of ensemble members (병렬화)
3. Mixed precision (fp16)

---

## 9. 의존성 (`requirements.txt`)

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
pyyaml>=6.0
matplotlib>=3.7.0
xarray>=2023.0.0
zarr>=2.16.0          # for zarr I/O
dask>=2023.0.0        # for lazy zarr operations
netCDF4>=1.6.0
tqdm>=4.65.0
einops>=0.6.0
```

---

## 10. 참고 사항

### 10.1 본 구현이 따라야 하는 원칙

1. **Diffusion² 논문의 Eq. 9, 10, 20을 정확히 따라야 한다.** LANS는 사용하지 않는다.
2. **Dual-head는 ε와 ℓ 모두 출력해야 한다.** 단일 head로 줄이지 않는다.
3. **학습 시 DDPM_past sampling은 1 sample만, 추론 시 매 ensemble member마다 새로 sampling한다.**
4. **DDPM_main의 condition에 x_t를 직접 포함하지 않는다** (h_t 의도적 제외). 시간적 이웃을 통한 간접 전달만 허용.
5. **사전 정규화된 데이터를 사용한다.** 모델 내부에서 정규화 연산을 수행하지 않는다.

### 10.2 데이터 흐름 요약

전체 파이프라인:

```
[전처리 - Data_preprocess.md] (한 번만 실행)
ERA5 raw zarr files
    ↓ (compute_stats.py)
data/normalization_stats.zarr  (4, 3, 64, 64) - hour×channel×lat×lon
    ↓ (normalize_dataset.py)
data/era5_normalized.zarr      (T, 3, 64, 64) - 정규화된 시계열

[학습 - 본 instruction] 
data/era5_normalized.zarr
    ↓ (ERA5NormalizedDataset)
5 시점 batch (B, 3, 64, 64) × 5
    ↓ (training/train.py)
학습된 모델 (encoder + DDPM_past + DDPM_main)

[추론 - 본 instruction]
data/era5_normalized.zarr (test split)
    ↓ (ERA5NormalizedDataset, mode='inference')
(x_{t-1}, x_t) (B, 3, 64, 64) × 2
    ↓ (inference/ensemble.py)
Ensemble {x_t^(b)}_{b=1}^B in normalized space
    ↓ (denormalize via stats)
원본 단위 ensemble (선택적)
```

### 10.3 본 모델이 노리는 학습 신호

- ℓ_past: x_{t-1}, x_t로부터 더 먼 과거를 예측할 때의 픽셀별 불확실성. 시간적 변동성이 큰 영역에서 큼.
- ℓ_main: 시간적 이웃으로부터 x_t를 재구성할 때의 픽셀별 불확실성. Forecasting의 어려움이 반영됨.
- Ensemble spread: 두 source (DDPM_past stochasticity + DDPM_main variance)의 결합으로 자연스럽게 flow-dependent.

### 10.4 향후 확장 가능성

- LANS 추가 (MuLAN style adaptive noise schedule)
- Multi-variable conditioning (다른 기상 변수를 condition으로)
- Multi-timestep ensemble (x_t뿐 아니라 x_{t+1}, ..., x_{t+k}까지)
- Latent diffusion으로 확장 (VAE로 압축 후 diffusion)
- 도메인 확장 (한반도 → 동아시아 → 전 지구)

---

## 11. 마무리

본 instruction.md는 Diffusion²의 dual-head parameterization을 기상장 ensemble 생성에 응용하는 모델의 전체 설계와 구현 방향을 담고 있다. 코드 작성 시 본 문서의 수식과 알고리즘을 정확히 따라야 하며, 특히 손실 함수 (Eq. 20)와 reverse sampling (Eq. 30)의 구현은 수치적 안정성을 고려하여 신중히 작성해야 한다.

구현 중 의문이 생기면 다음 항목을 확인:
1. Diffusion² 논문 (arXiv:2510.04365)의 해당 수식
2. 본 문서 Section 3 (수식 및 알고리즘 상세)
3. 본 문서 Section 8 (잠재적 문제 및 디버깅)

