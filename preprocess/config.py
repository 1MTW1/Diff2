"""Data preprocessing 공통 설정."""
from __future__ import annotations

# ── 입력 ─────────────────────────────────────────────────────────
DATASET_GLOB = "/geodata2/MuLANWR/era5_6h_1lv_*.zarr"

# ── 도메인 ───────────────────────────────────────────────────────
# 한반도 중심 64×64 patch
KOREA_LAT_CENTER = 37.0
KOREA_LON_CENTER = 127.5
PATCH = 64

# ── 변수 ─────────────────────────────────────────────────────────
VARIABLES = ["temperature", "u_component_of_wind", "v_component_of_wind"]
CHANNEL_NAMES = ["t", "u", "v"]

# ── 연도 ─────────────────────────────────────────────────────────
YEAR_MIN = 2000     # 데이터 가용 시작
YEAR_MAX = 2022     # 데이터 가용 끝

# 통계량 계산은 train split만 사용 (leakage 방지)
STATS_YEAR_MIN = 2000
STATS_YEAR_MAX = 2019

# ── 시각 ─────────────────────────────────────────────────────────
HOURS = [0, 6, 12, 18]   # 6h 데이터의 4개 시각
HOUR_TO_IDX = {0: 0, 6: 1, 12: 2, 18: 3}

# ── Leap day (DoY 60 = Feb 29) ────────────────────────────────────
# 시각별 통계에서도 윤년 처리: Feb 29는 통계 계산에서 제외
LEAP_DATE = (2, 29)

# ── 출력 ─────────────────────────────────────────────────────────
STATS_PATH = "data/normalization_stats.zarr"
NORMALIZED_PATH = "data/era5_normalized.zarr"

# ── I/O ──────────────────────────────────────────────────────────
LOAD_CHUNK = 512       # 시간축 청크 크기 (RAM 부하 조절)
WRITE_CHUNK = 1024     # zarr 저장 청크 크기

# ── Numerical ────────────────────────────────────────────────────
STD_EPS = 1e-6         # σ가 0에 가까울 때 안정화
