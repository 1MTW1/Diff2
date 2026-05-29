"""v2 LDM/DiT Ensemble 평가 실험 패키지 (experiment.md의 LDM 변환판).

experiments/ (v1 픽셀 공간 U-Net DDPM)의 exp1~exp6를 v2 LDM/DiT 모델에
맞춰 옮긴 것이다. 분석은 모두 **diffusion latent 공간** (C_z=12, 16×16)에서
수행되며, exp5의 4×4 composite map만 VAE decode 후 픽셀 공간(C=3, 64×64)에서
그려진다. DDPM_main의 x_t 예측을 평가한다.
"""
