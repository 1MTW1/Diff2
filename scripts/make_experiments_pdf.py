"""experiments_dit + (experiments에서 import한) 코드 요약 PDF 생성기.

핵심 발췌 위주의 한국어 PDF를 outputs/ 아래에 만든다.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPO = Path("/home/popcodh/Diff2")


def register_fonts() -> tuple[str, str]:
    # CFF/OTF 인 NotoSansCJK는 reportlab의 TTFont가 임베드하지 못한다.
    # 대신 PDF 표준 CIDFont 참조 (HYGothic-Medium / HYSMyeongJo-Medium) 를
    # 사용 — Acrobat, 브라우저 pdf.js, macOS Preview 등 주요 viewer 가
    # 시스템 CJK 폰트로 렌더링한다.
    pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
    pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
    return "HYSMyeongJo-Medium", "HYGothic-Medium"


def make_styles(body_font: str, code_font: str) -> dict:
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontName=body_font,
            fontSize=22, leading=28, spaceAfter=6, textColor=colors.HexColor("#1f3b73"),
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName=body_font,
            fontSize=12, leading=16, spaceAfter=18,
            textColor=colors.HexColor("#444"),
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName=body_font,
            fontSize=15, leading=20, spaceBefore=10, spaceAfter=6,
            textColor=colors.HexColor("#1f3b73"),
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName=body_font,
            fontSize=12, leading=16, spaceBefore=8, spaceAfter=4,
            textColor=colors.HexColor("#244"),
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName=body_font,
            fontSize=10, leading=14.5, alignment=TA_LEFT, spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName=body_font,
            fontSize=9, leading=12, textColor=colors.HexColor("#555"),
        ),
        "code": ParagraphStyle(
            "Code", parent=base["Code"], fontName=code_font,
            fontSize=8.0, leading=10.5, leftIndent=4, rightIndent=4,
            backColor=colors.HexColor("#f4f6fa"),
            borderColor=colors.HexColor("#cdd5e3"),
            borderPadding=4, borderWidth=0.5,
            textColor=colors.HexColor("#11203a"),
            spaceBefore=2, spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["Italic"], fontName=body_font,
            fontSize=8.5, leading=11, textColor=colors.HexColor("#555"),
            spaceAfter=4,
        ),
    }
    return styles


def code(text: str, style) -> Preformatted:
    return Preformatted(text.rstrip("\n"), style)


# ───────────────────────── content builders ─────────────────────────

def section_header(title: str, path: str, styles: dict) -> list:
    return [
        Paragraph(title, styles["h1"]),
        Paragraph(f"<font color='#666'>{path}</font>", styles["small"]),
        Spacer(1, 4),
    ]


def role(text: str, styles: dict) -> Paragraph:
    return Paragraph(f"<b>역할.</b> {text}", styles["body"])


def label(text: str, styles: dict) -> Paragraph:
    return Paragraph(f"<b>{text}</b>", styles["h2"])


def para(text: str, styles: dict) -> Paragraph:
    return Paragraph(text, styles["body"])


def caption(text: str, styles: dict) -> Paragraph:
    return Paragraph(text, styles["caption"])


def cover(styles: dict) -> list:
    items = [
        Paragraph("experiments_dit 코드 정리", styles["title"]),
        Paragraph(
            "v2 LDM/DiT 앙상블 평가 실험 패키지 — 파일별 역할과 핵심 구현 발췌",
            styles["subtitle"],
        ),
        Paragraph("문서 구성", styles["h2"]),
        Paragraph(
            "본 문서는 experiments_dit 패키지의 모든 .py 파일과, "
            "그 안에서 thin-wrapper로 재사용되는 experiments/ 의 두 파일 "
            "(exp1_uncertainty.py, exp2_spread_vs_logvar.py) 을 정리한다. "
            "각 파일마다 (1) 한 줄 요약, (2) 패키지 안에서의 역할, "
            "(3) 핵심 구현을 코드 발췌와 함께 설명한다.",
            styles["body"],
        ),
        Spacer(1, 6),
        Paragraph("전체 데이터 흐름", styles["h2"]),
        Paragraph(
            "DDPM_past → VAE decode → encoder → DDPM_main 의 두-단계 latent "
            "diffusion 파이프라인으로 (B, C_z=12, 16, 16) 짜리 N-멤버 latent "
            "앙상블 z_main 과 dual-head 의 log_var, GT 의 posterior μ 를 "
            "캐시 npz 로 떨군다 (ensemble_inference.py). 이후 exp1~exp6 가 "
            "캐시를 읽어 다양한 calibration·flow-dependence 통계와 그림을 "
            "생산한다 (exp5만 픽셀 공간, 나머지는 latent 공간).",
            styles["body"],
        ),
        Spacer(1, 6),
        Paragraph("문서 내 표기", styles["h2"]),
        Paragraph(
            "C_z=12, H_z=W_z=16 (latent shape); C=3, H=W=64 (픽셀 shape, "
            "변수 t/u/v); N = ensemble member 수; ℓ_m = main DDPM 의 log_var "
            "head 출력; σ_ℓ = √exp(ℓ_m).",
            styles["small"],
        ),
        Spacer(1, 18),
        Paragraph("목차", styles["h2"]),
    ]
    toc_data = [
        ["#", "파일", "한 줄 설명"],
        ["1", "experiments_dit/__init__.py", "패키지 문서화 docstring"],
        ["2", "experiments_dit/__main__.py", "exp1~exp6 일괄 실행 진입점"],
        ["3", "experiments_dit/utils.py", "latent 캐시 I/O · denorm · plot helper"],
        ["4", "experiments_dit/ensemble_inference.py", "DDPM_main latent 앙상블 N-member 생성·캐시"],
        ["5", "experiments_dit/exp1_uncertainty.py", "exp1 thin wrapper (v1 재사용)"],
        ["6", "experiments_dit/exp2_spread_vs_logvar.py", "exp2 thin wrapper (v1 재사용)"],
        ["7", "experiments_dit/exp3_spread_skill.py", "latent spread/RMSE ratio (calibration)"],
        ["8", "experiments_dit/exp4_pixel_timeseries.py", "low/high log_var pool spread 비교"],
        ["9", "experiments_dit/exp5_composite_maps.py", "픽셀 composite 16-panel + spread map"],
        ["10", "experiments_dit/exp6_pixel_ratio.py", "latent pixel-wise spread/RMSE"],
        ["11", "experiments/exp1_uncertainty.py", "ℓ_m diversity + flow-dependence 본체"],
        ["12", "experiments/exp2_spread_vs_logvar.py", "σ_ℓ ↔ spread per-sample 상관 본체"],
    ]
    tbl = Table(toc_data, colWidths=[10*mm, 78*mm, 90*mm], hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "HYSMyeongJo-Medium", 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b73")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#1f3b73")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f0f3fa")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    items.append(tbl)
    items.append(PageBreak())
    return items


# ─────────────────────── per-file sections ───────────────────────

def s_init(styles) -> list:
    out = section_header("1. experiments_dit/__init__.py", "패키지 docstring", styles)
    out += [
        role(
            "experiments_dit 패키지의 최상위 docstring. "
            "분석 공간 (latent C_z=12, 16×16) 과 픽셀 공간 (C=3, 64×64) 사용 "
            "지점을 한 줄로 명시한다. 코드 로직은 없다.",
            styles,
        ),
        label("핵심 발췌", styles),
        code(
            '"""v2 LDM/DiT Ensemble 평가 실험 패키지 (experiment.md의 LDM 변환판).\n'
            '...\n'
            '맞춰 옮긴 것이다. 분석은 모두 **diffusion latent 공간** (C_z=12, 16×16)에서\n'
            '수행되며, exp5의 4×4 composite map만 VAE decode 후 픽셀 공간(C=3, 64×64)에서\n'
            '그려진다. DDPM_main의 x_t 예측을 평가한다.\n'
            '"""',
            styles["code"],
        ),
    ]
    return out


def s_main(styles) -> list:
    out = section_header("2. experiments_dit/__main__.py", "일괄 실행 진입점", styles)
    out += [
        role(
            "캐시된 앙상블 (outputs/ensembles_dit) 을 입력으로 받아 exp1~exp6 를 "
            "독립 subprocess 로 순차 실행한다. exp4·exp5 는 metrics 출력이 없어 "
            "(모듈명, has_metrics) 튜플 리스트로 인자 차이를 표시한다. 어느 하나가 "
            "실패해도 나머지는 계속 돌고, 마지막에 실패 모듈을 모아 1로 종료한다.",
            styles,
        ),
        label("실행 대상 정의 — 모듈명과 metrics_dir 지원 여부", styles),
        code(
            '_EXPERIMENTS: list[tuple[str, bool]] = [\n'
            '    ("exp1_uncertainty", True),\n'
            '    ("exp2_spread_vs_logvar", True),\n'
            '    ("exp3_spread_skill", True),\n'
            '    ("exp4_pixel_timeseries", False),\n'
            '    ("exp5_composite_maps", False),\n'
            '    ("exp6_pixel_ratio", True),\n'
            ']',
            styles["code"],
        ),
        label("subprocess 디스패치 루프 — 실패 누적 후 exit code 반영", styles),
        code(
            'for mod, has_metrics in _EXPERIMENTS:\n'
            '    cmd = [sys.executable, "-m", f"experiments_dit.{mod}",\n'
            '           "--ensemble_dir", args.ensemble_dir,\n'
            '           "--figures_dir", args.figures_dir]\n'
            '    if has_metrics:\n'
            '        cmd += ["--metrics_dir", args.metrics_dir]\n'
            '    if subprocess.run(cmd).returncode != 0:\n'
            '        failed.append(mod)\n'
            '...\n'
            'sys.exit(1 if failed else 0)',
            styles["code"],
        ),
        caption(
            "사용 예: python -m experiments_dit --ensemble_dir outputs/ensembles_dit "
            "--figures_dir outputs/figures_dit --metrics_dir outputs/metrics_dit",
            styles,
        ),
    ]
    return out


def s_utils(styles) -> list:
    out = section_header("3. experiments_dit/utils.py", "공용 유틸 (캐시 I/O · denorm · plot)", styles)
    out += [
        role(
            "experiments_dit 의 모든 exp* 모듈이 공유하는 helper 모음. "
            "(1) sample_*.npz 캐시 스키마를 EnsembleSample dataclass 로 매핑, "
            "(2) hour-of-day 별 mean/std 로 (역)정규화, "
            "(3) 디렉토리·matplotlib 기본값 helper 를 제공한다. "
            "픽셀 키 (ensemble_pixel, x_t_true_pixel) 는 Optional 로 두어 "
            "exp5 용 캐시가 없으면 None 으로 채운다.",
            styles,
        ),
        label("캐시 데이터 dataclass — 모든 exp* 가 이 객체를 받는다", styles),
        code(
            '@dataclass\n'
            'class EnsembleSample:\n'
            '    ensemble: np.ndarray        # (N, C_z=12, 16, 16) 정규화 latent ẑ_0\n'
            '    log_var: np.ndarray         # (C_z, 16, 16) main DDPM dual-head log_var (멤버 평균)\n'
            '    x_t_true: np.ndarray        # (C_z, 16, 16) GT frame-pair latent posterior μ\n'
            '    time_t: np.datetime64\n'
            '    path: Path\n'
            '    ensemble_pixel: Optional[np.ndarray] = None    # (N, 3, 64, 64) exp5 전용\n'
            '    x_t_true_pixel: Optional[np.ndarray] = None    # (3, 64, 64)    exp5 전용',
            styles["code"],
        ),
        label("npz 로더 — 픽셀 키 부재시 None 으로 graceful fallback", styles),
        code(
            'def load_ensemble_npz(path: Path | str) -> EnsembleSample:\n'
            '    data = np.load(path, allow_pickle=False)\n'
            '    keys = set(data.files)\n'
            '    ensemble_pixel = (\n'
            '        data["ensemble_pixel"].astype(np.float32)\n'
            '        if "ensemble_pixel" in keys else None\n'
            '    )\n'
            '    ...\n'
            '    return EnsembleSample(\n'
            '        ensemble=data["ensemble"].astype(np.float32),\n'
            '        log_var=data["log_var"].astype(np.float32),\n'
            '        x_t_true=data["x_t_true"].astype(np.float32),\n'
            '        time_t=np.datetime64(str(data["time_t"])),\n'
            '        path=path, ensemble_pixel=ensemble_pixel, x_t_true_pixel=x_t_true_pixel,\n'
            '    )',
            styles["code"],
        ),
        label("시간대별 역정규화 — HOUR_TO_IDX 로 적절한 (μ, σ) 슬라이스 선택", styles),
        code(
            'def denorm_array(x_norm, time_t, mean, std):\n'
            '    """(C,H,W) or (N,C,H,W) 입력. mean/std는 (4,C,H,W) torch 텐서."""\n'
            '    hour = pd.Timestamp(time_t).hour\n'
            '    h_idx = HOUR_TO_IDX[int(hour)]\n'
            '    mu = mean[h_idx].cpu().numpy()      # (C, H, W)\n'
            '    sigma = std[h_idx].cpu().numpy()\n'
            '    return x_norm * sigma + mu\n\n'
            'def denorm_spread(spread_norm, time_t, std):\n'
            '    """Spread/std는 σ만 곱함 — additive shift 무효."""\n'
            '    ...\n'
            '    return spread_norm * sigma',
            styles["code"],
        ),
        label("상수 — exp들이 참조할 채널 인덱스", styles),
        code(
            'VARIABLES = ("t", "u", "v")\n'
            'TEMP_IDX = 0   # 픽셀 공간 온도 채널 (exp5)\n'
            'U_IDX = 1; V_IDX = 2\n'
            'LATENT_CH = 0  # latent 공간에서 대표로 분석할 채널 (exp3/4/6)',
            styles["code"],
        ),
    ]
    return out


def s_ensemble_inference(styles) -> list:
    out = section_header(
        "4. experiments_dit/ensemble_inference.py",
        "DDPM_main latent 앙상블 N-member 생성 & 캐시 (선행 단계)",
        styles,
    )
    out += [
        role(
            "test set 중 월/수/금 시점에 대해 두-단계 latent diffusion 파이프라인 "
            "( inference/sampling 의 generate_future_ensemble 과 동일 ) 을 돌려 "
            "DDPM_main 이 예측한 latent ẑ_0 의 N-member 앙상블을 만든다. "
            "각 시점마다 ensemble · log_var · x_t_true (+ 픽셀 캐시) 를 묶어 "
            "sample_{idx:05d}.npz 로 저장한다. accelerate launch 멀티 프로세스 "
            "환경에서는 timestep 차원을 rank::world 로 sharding 한다.",
            styles,
        ),
        label("test 시점 sub-sample — Mon/Wed/Fri 만 추리는 룰", styles),
        code(
            'def _select_mon_wed_fri_indices(times):\n'
            '    dow = pd.DatetimeIndex(times).dayofweek.to_numpy()\n'
            '    keep = np.where((dow == 0) | (dow == 2) | (dow == 4))[0]\n'
            '    return keep',
            styles["code"],
        ),
        label(
            "두-단계 latent diffusion: DDPM_past → decode → encoder → DDPM_main",
            styles,
        ),
        code(
            '@torch.no_grad()\n'
            'def _generate_main_ensemble(x_tm1, x_t, x_tp1, encoder, dit_past, dit_main,\n'
            '                            vae, normalizer, sampler, n_members, device, ...):\n'
            '    B = n_members\n'
            '    C_z = vae.latent_channels;  H_z, W_z = normalizer.mu.shape[-2:]\n\n'
            '    # [1] DDPM_past — 다양한 과거 후보 (inject_uncertainty=True)\n'
            '    cond_past = encoder(torch.stack([x_tm1, x_t], dim=1)).expand(B, -1, -1)\n'
            '    z_past, _ = sampler.sample(\n'
            '        dit_past, cond_past, (B, C_z, H_z, W_z), device,\n'
            '        inject_uncertainty=True, num_steps=past_num_steps,\n'
            '    )\n'
            '    x_past = vae.decode(normalizer.denormalize(z_past)).reshape(B, 2, C, H, W)\n'
            '    x_tm2_hat = x_past[:, 1]  # past pair 의 두번째 frame = x̂_{t-2}\n\n'
            '    # [2] DDPM_main — future 생성 (inject_uncertainty=False, 평가 대상)\n'
            '    x_tm1_B = x_tm1.expand(B, -1, -1, -1)\n'
            '    cond_main = encoder(torch.stack([x_tm2_hat, x_tm1_B], dim=1))\n'
            '    z_main, lv_main = sampler.sample(\n'
            '        dit_main, cond_main, (B, C_z, H_z, W_z), device,\n'
            '        inject_uncertainty=False, num_steps=main_num_steps,\n'
            '    )',
            styles["code"],
        ),
        label("픽셀 앙상블·GT latent·log_var 멤버 reduce", styles),
        code(
            '    # exp5용 픽셀 앙상블 — main latent 를 디코딩한 (B, 3, 64, 64) 중 frame 0\n'
            '    x_main = vae.decode(normalizer.denormalize(z_main)).reshape(B, 2, C, H, W)\n'
            '    ensemble_pixel = x_main[:, 0]\n\n'
            '    # log_var 멤버 평균 (spec상 single map → 멤버 평균으로 대표값 저장)\n'
            '    log_var = lv_main.mean(dim=0) if _LOG_VAR_MEMBER_REDUCE == "mean" else lv_main[0]\n\n'
            '    # GT [x_t, x_{t+1}] 블록을 VAE encode → posterior μ → normalize\n'
            '    gt_pair = torch.cat([x_t, x_tp1], dim=1)\n'
            '    mu_gt, _ = vae.encode(gt_pair)\n'
            '    x_t_true = normalizer.normalize(mu_gt)[0]',
            styles["code"],
        ),
        label("multi-GPU sharding & cross-process seed 분리", styles),
        code(
            'all_positions = np.arange(len(keep_rel))\n'
            'my_positions  = all_positions[rank::world]\n'
            'my_indices    = keep_rel[rank::world]\n'
            '...\n'
            'torch.manual_seed(seed + rank)   # process 간 중복 noise 방지',
            styles["code"],
        ),
        label("npz 캐시 저장 스키마", styles),
        code(
            'np.savez(out_path,\n'
            '    ensemble        = cache["ensemble"].numpy().astype(np.float32),   # (N,12,16,16)\n'
            '    log_var         = cache["log_var"].numpy().astype(np.float32),    # (12,16,16)\n'
            '    x_t_true        = cache["x_t_true"].numpy().astype(np.float32),   # (12,16,16)\n'
            '    ensemble_pixel  = cache["ensemble_pixel"].numpy().astype(np.float32),  # (N,3,64,64) — exp5\n'
            '    x_t_true_pixel  = cache["x_t_true_pixel"].numpy().astype(np.float32),  # (3,64,64)   — exp5\n'
            '    time_t          = np.array(str(time_t)),\n'
            ')',
            styles["code"],
        ),
    ]
    return out


def s_exp1_dit(styles) -> list:
    out = section_header(
        "5. experiments_dit/exp1_uncertainty.py", "v1 exp1의 thin wrapper", styles,
    )
    out += [
        role(
            "experiments/exp1_uncertainty.run_exp1 을 그대로 재export 한다. "
            "v1 캐시의 채널 0 (TEMP_IDX) 분석 로직이 latent 캐시 채널 0 "
            "(LATENT_CH) 과 의미가 일치하기 때문에 코드 복제 없이 동일 함수로 "
            "재사용한다. CLI 만 ensemble_dir 기본값 outputs/ensembles_dit 로 "
            "바꿔준다.",
            styles,
        ),
        label("핵심 발췌", styles),
        code(
            'from experiments.exp1_uncertainty import run_exp1  # noqa: F401\n\n'
            'if __name__ == "__main__":\n'
            '    p = argparse.ArgumentParser()\n'
            '    p.add_argument("--ensemble_dir", default="outputs/ensembles_dit")\n'
            '    p.add_argument("--figures_dir", default="outputs/figures_dit")\n'
            '    p.add_argument("--metrics_dir", default="outputs/metrics_dit")\n'
            '    p.add_argument("--n_pairs", type=int, default=1000)\n'
            '    ...\n'
            '    run_exp1(args.ensemble_dir, args.figures_dir, args.metrics_dir, ...)',
            styles["code"],
        ),
        caption(
            "실제 알고리즘은 §11 experiments/exp1_uncertainty.py 절을 참조.",
            styles,
        ),
    ]
    return out


def s_exp2_dit(styles) -> list:
    out = section_header(
        "6. experiments_dit/exp2_spread_vs_logvar.py",
        "v1 exp2의 thin wrapper", styles,
    )
    out += [
        role(
            "exp1 과 같은 패턴 — experiments/exp2_spread_vs_logvar.run_exp2 를 "
            "그대로 재export. 채널 0 분석이 latent 0 채널과 의미가 일치하므로 "
            "코드 복제 없음.",
            styles,
        ),
        label("핵심 발췌", styles),
        code(
            'from experiments.exp2_spread_vs_logvar import run_exp2  # noqa: F401',
            styles["code"],
        ),
        caption(
            "실제 알고리즘은 §12 experiments/exp2_spread_vs_logvar.py 절을 참조.",
            styles,
        ),
    ]
    return out


def s_exp3(styles) -> list:
    out = section_header(
        "7. experiments_dit/exp3_spread_skill.py",
        "latent 공간 sample별 Spread/RMSE ratio (calibration)",
        styles,
    )
    out += [
        role(
            "각 sample 의 latent 채널 0 에서 픽셀별 ratio (spread/|err|) 를 먼저 "
            "구하고, sample 내에서 평균을 내 sample 별 한 개 ratio 분포 (S,) 를 "
            "얻는다. 동시에 √mean(s²)/√mean(e²) 형태의 aggregate ratio 도 함께 "
            "보고하여 픽셀 분포의 극단값 영향을 줄인 비교를 제공한다. v1 은 "
            "정규화/K-space 두 가지를 보고했으나 v2 latent 분석에서는 latent "
            "공간만 남긴다.",
            styles,
        ),
        label("두 ratio 정의 — per-pixel mean vs aggregate RMS ratio", styles),
        code(
            '_EPS = 1e-6\n\n'
            'def _pixel_ratio_mean(spread, err):\n'
            '    return float((spread / (np.abs(err) + _EPS)).mean())\n\n'
            'def _aggregate_ratio(spread, err):\n'
            '    num = float(np.sqrt((spread ** 2).mean()))\n'
            '    den = float(np.sqrt((err ** 2).mean())) + _EPS\n'
            '    return num / den',
            styles["code"],
        ),
        label("sample 루프 — spread / err 계산은 latent 채널 0 에서", styles),
        code(
            'for i, f in enumerate(files):\n'
            '    es = load_ensemble_npz(f)\n'
            '    ens_t = es.ensemble[:, LATENT_CH]                  # (N, H_z, W_z)\n'
            '    spread = ens_t.std(axis=0, ddof=1)                 # (H_z, W_z)\n'
            '    err    = ens_t.mean(axis=0) - es.x_t_true[LATENT_CH]\n\n'
            '    norm_ratios[i] = _pixel_ratio_mean(spread, err)\n'
            '    norm_agg[i]    = _aggregate_ratio(spread, err)',
            styles["code"],
        ),
        label("산출물", styles),
        code(
            'fig_dir/exp3_ratio_hist.png       # per-sample pixel-mean ratio 분포\n'
            'met_dir/exp3_stats.json           # mean/median/std (+ epsilon)',
            styles["code"],
        ),
    ]
    return out


def s_exp4(styles) -> list:
    out = section_header(
        "8. experiments_dit/exp4_pixel_timeseries.py",
        "low/high log_var pool 에서 spread 비교 (단일 시점 + aggregate)",
        styles,
    )
    out += [
        role(
            "한 시점의 latent log_var map 을 ranking 기준으로 사용해 interior "
            "픽셀을 하위/상위 percentile pool 로 나눈다. 각 pool 에서 "
            "σ_ℓ = √exp(log_var) 와 실제 ensemble spread 를 픽셀 단위로 모아 "
            "막대그래프 (mean ± std + jitter scatter) 로 비교한다. "
            "run_exp4 는 단일 sample, run_exp4_aggregate 는 모든 sample 의 "
            "pool 평균을 다시 sample 차원으로 통계낸다. v1 의 K-space 패널은 "
            "latent 분석으로 단순화되며 16×16 grid 라 boundary margin 기본값을 "
            "2 로 축소했다.",
            styles,
        ),
        label("pool 선택 — interior 픽셀에서 하위/상위 percentile flatten 인덱스", styles),
        code(
            'def _pick_low_high_pools(rank_map, percentile, margin):\n'
            '    H, W = rank_map.shape\n'
            '    interior = np.zeros((H, W), dtype=bool)\n'
            '    if margin > 0:\n'
            '        interior[margin:H - margin, margin:W - margin] = True\n'
            '    else:\n'
            '        interior[:, :] = True\n'
            '    valid_idx  = np.where(interior.flatten())[0]\n'
            '    valid_vals = rank_map.flatten()[valid_idx]\n'
            '    cutoff = max(int(round(valid_idx.size * percentile)), 1)\n'
            '    order  = np.argsort(valid_vals)\n'
            '    return valid_idx[order[:cutoff]], valid_idx[order[-cutoff:]]',
            styles["code"],
        ),
        label("pool 별 σ_ℓ 와 ensemble spread 동시 추출", styles),
        code(
            'def _pool_values(flat_indices, lv_map, ensemble):\n'
            '    H, W = lv_map.shape\n'
            '    rows = (flat_indices // W).astype(int)\n'
            '    cols = (flat_indices %  W).astype(int)\n'
            '    sigma_ell = np.sqrt(lv_map[rows, cols])\n'
            '    spread    = ensemble[:, LATENT_CH, rows, cols].std(axis=0, ddof=1)\n'
            '    return {"sigma_ell": sigma_ell, "ens_spread": spread, ...}',
            styles["code"],
        ),
        label("단일 sample 런 — sample_idx None 이면 seed 로 무작위", styles),
        code(
            'def run_exp4(..., percentile=0.1, margin=2, sample_idx=None, seed=42):\n'
            '    ...\n'
            '    if sample_idx is None:\n'
            '        sample_idx = int(rng.integers(0, len(files)))\n'
            '    es = load_ensemble_npz(files[sample_idx])\n'
            '    lv_map = np.exp(es.log_var[LATENT_CH])\n'
            '    low_idx, high_idx = _pick_low_high_pools(np.sqrt(lv_map), percentile, margin)\n'
            '    low  = _pool_values(low_idx,  lv_map, es.ensemble)\n'
            '    high = _pool_values(high_idx, lv_map, es.ensemble)',
            styles["code"],
        ),
        label("aggregate 런 — sample별 pool 평균을 다시 sample 차원으로 모은다", styles),
        code(
            'for f in files:\n'
            '    es = load_ensemble_npz(f); lv_map = np.exp(es.log_var[LATENT_CH])\n'
            '    low_idx, high_idx = _pick_low_high_pools(np.sqrt(lv_map), percentile, margin)\n'
            '    low  = _pool_values(low_idx,  lv_map, es.ensemble)\n'
            '    high = _pool_values(high_idx, lv_map, es.ensemble)\n'
            '    per_sample["low_sigma" ].append(float(low ["sigma_ell" ].mean()))\n'
            '    per_sample["low_spread"].append(float(low ["ens_spread"].mean()))\n'
            '    per_sample["high_sigma"].append(float(high["sigma_ell" ].mean()))\n'
            '    per_sample["high_spread"].append(float(high["ens_spread"].mean()))',
            styles["code"],
        ),
        caption(
            "Plot helper _draw_panel_from_arrays 는 4개 array (LOW σ_ℓ / LOW spread / "
            "HIGH σ_ℓ / HIGH spread) 를 받아 mean ± std bar + jitter scatter + "
            "median/μ annotation 을 한 axis 에 그린다.",
            styles,
        ),
    ]
    return out


def s_exp5(styles) -> list:
    out = section_header(
        "9. experiments_dit/exp5_composite_maps.py",
        "픽셀 공간 composite (4×4 panel) + spread map",
        styles,
    )
    out += [
        role(
            "6 개 exp 중 유일하게 픽셀 공간에서 그려진다. ensemble_pixel · "
            "x_t_true_pixel 캐시 키를 hour-of-day 별 (μ, σ) 로 K 단위 역정규화한 "
            "뒤, GT (절대), 14 member 의 diff = member − mean (편차), "
            "ensemble mean (절대) 의 16-panel composite 와 normalized/denormalized "
            "spread map 두 종을 저장한다. 바람 quiver 는 magnitude cap (60 m/s) 후 "
            "stride 8 로 표시한다.",
            styles,
        ),
        label("픽셀 캐시 부재 시 명시적 에러", styles),
        code(
            'es = load_ensemble_npz(files[idx])\n'
            'if es.ensemble_pixel is None or es.x_t_true_pixel is None:\n'
            '    raise KeyError(\n'
            '        f"{files[idx]} 에 픽셀 캐시 ... 가 없습니다 — "\n'
            '        f"experiments_dit.ensemble_inference 로 생성하세요."\n'
            '    )',
            styles["code"],
        ),
        label("역정규화 — 절대장은 μ·σ, spread 는 σ 만", styles),
        code(
            'def _denorm_field(x_norm, time_t, mean, std):\n'
            '    hour = pd.Timestamp(time_t).hour\n'
            '    h_idx = HOUR_TO_IDX[int(hour)]\n'
            '    return x_norm * std[h_idx].cpu().numpy() + mean[h_idx].cpu().numpy()\n\n'
            'def _denorm_spread_field(s_norm, time_t, std):\n'
            '    ...\n'
            '    return s_norm * std[h_idx].cpu().numpy()    # mean shift 없음',
            styles["code"],
        ),
        label("바람 magnitude cap — 방향은 보존하면서 길이를 잘라낸다", styles),
        code(
            '_QUIVER_CLIP_MAG = 60.0\n\n'
            'def _clip_wind(u, v, max_mag):\n'
            '    mag   = np.sqrt(u * u + v * v)\n'
            '    scale = np.minimum(1.0, max_mag / (mag + 1e-9))\n'
            '    return u * scale, v * scale',
            styles["code"],
        ),
        label("16-panel composite 의 색범위 결정 — 절대 vs diff 분리", styles),
        code(
            'abs_pool  = np.concatenate([gt[TEMP_IDX].flatten(), mean_field[TEMP_IDX].flatten()])\n'
            'abs_vmin  = float(np.percentile(abs_pool, 1))\n'
            'abs_vmax  = float(np.percentile(abs_pool, 99))\n\n'
            'diff_pool = (members[:, TEMP_IDX] - mean_field[TEMP_IDX]).flatten()\n'
            'diff_max  = max(float(np.percentile(np.abs(diff_pool), 99)), 1e-6)\n\n'
            '# panel 0 = GT(abs), panel 1..14 = member - mean (diff), panel 15 = mean(abs)\n'
            'for k in range(14):\n'
            '    m = members[k]\n'
            '    diff_t = m[TEMP_IDX] - mean_field[TEMP_IDX]\n'
            '    _plot_one_panel(flat[1 + k], diff_t, m[U_IDX], m[V_IDX], -diff_max, diff_max, ...)',
            styles["code"],
        ),
        label("산출물", styles),
        code(
            'fig_dir/exp5_composite_sample{k}.png    # 4×4 panel composite (K 단위)\n'
            'fig_dir/exp5_spread_sample{k}.png        # normalized + denormalized spread 두 panel',
            styles["code"],
        ),
    ]
    return out


def s_exp6(styles) -> list:
    out = section_header(
        "10. experiments_dit/exp6_pixel_ratio.py",
        "latent pixel-wise Spread/RMSE (per-pixel summary)",
        styles,
    )
    out += [
        role(
            "exp3 와 대칭. exp3 가 sample 단위로 ratio 를 요약했다면, exp6 는 "
            "latent pixel (i,j) 단위로 시간 차원에 대해 RMS 누적해서 "
            "ratio(i,j) = √mean_s[spread²(s,i,j)] / √mean_s[err²(s,i,j)] 을 "
            "구하고 (H_z·W_z=256) 픽셀 분포를 histogram 으로, 추가로 "
            "spread/rmse/ratio 의 spatial map 도 함께 그린다. ratio map 은 "
            "1 중심 diverging colormap 으로, 데이터 dev 가 작아도 0.1 floor 를 "
            "둬서 시각 비교가 가능하다.",
            styles,
        ),
        label("샘플 차원으로 RMS 사전 누적 — 메모리 효율적", styles),
        code(
            'def _accumulate(files):\n'
            '    n = len(files); sp_n2 = er_n2 = None\n'
            '    for f in files:\n'
            '        es = load_ensemble_npz(f)\n'
            '        ens_t  = es.ensemble[:, LATENT_CH]                # (N, H_z, W_z)\n'
            '        spread = ens_t.std(axis=0, ddof=1)                # (H_z, W_z)\n'
            '        err    = ens_t.mean(axis=0) - es.x_t_true[LATENT_CH]\n\n'
            '        if sp_n2 is None:\n'
            '            sp_n2 = spread ** 2;  er_n2 = err ** 2\n'
            '        else:\n'
            '            sp_n2 += spread ** 2; er_n2 += err ** 2\n'
            '    return sp_n2 / n, er_n2 / n',
            styles["code"],
        ),
        label("픽셀 ratio + spatial map", styles),
        code(
            'sp_n2, er_n2 = _accumulate(files)\n'
            'spread_n = np.sqrt(sp_n2)\n'
            'rmse_n   = np.sqrt(er_n2)\n'
            'ratio_n  = spread_n / (rmse_n + _EPS)\n\n'
            '_save_hist(ratio_n, fig_dir / "exp6_pixel_ratio_hist.png")\n'
            '_save_maps(spread_n, rmse_n, ratio_n, fig_dir / "exp6_pixel_maps.png")',
            styles["code"],
        ),
        label("ratio map 의 색범위 — 1 중심, dev 의 최소 floor 0.1", styles),
        code(
            'dev = float(max(abs(ratio_n.max() - 1.0), abs(ratio_n.min() - 1.0)))\n'
            'dev = max(dev, 0.1)\n'
            'ax.imshow(ratio_n, cmap="RdBu_r", vmin=1 - dev, vmax=1 + dev)',
            styles["code"],
        ),
    ]
    return out


def s_v1_exp1(styles) -> list:
    out = section_header(
        "11. experiments/exp1_uncertainty.py",
        "ℓ_m 의 pixelwise 다양성 & 시점 간 차이 (실제 구현)",
        styles,
    )
    out += [
        role(
            "experiments_dit/exp1_uncertainty.py 가 wrapping 하는 본체. "
            "(1) sample 별 spatial CV — 한 map 내에서 ℓ_m 이 얼마나 inhomogeneous "
            "한지, (2) random sample 쌍의 spatial Pearson 상관 — 시점이 바뀌면 "
            "패턴도 바뀌는가 (flow-dependence) 의 두 통계를 모두 구해 hist 와 "
            "heatmap (6 random samples) 으로 저장한다.",
            styles,
        ),
        label("시점 stack — exp(ℓ_m_t) 만 채널 0 에서 추출", styles),
        code(
            'def _load_temp_logvar_stack(ensemble_dir):\n'
            '    files = list_ensemble_files(ensemble_dir)\n'
            '    arrs, times = [], []\n'
            '    for f in files:\n'
            '        es = load_ensemble_npz(f)\n'
            '        arrs.append(np.exp(es.log_var[TEMP_IDX]).astype(np.float32))\n'
            '        times.append(es.time_t)\n'
            '    return np.stack(arrs, axis=0), times    # (S, H, W)',
            styles["code"],
        ),
        label("spatial CV — sample 한 장 내부의 픽셀 std / |mean|", styles),
        code(
            'def _pixelwise_cv(stack):\n'
            '    flat = stack.reshape(stack.shape[0], -1)\n'
            '    m = flat.mean(axis=1); s = flat.std(axis=1)\n'
            '    return s / (np.abs(m) + 1e-12)         # (S,)',
            styles["code"],
        ),
        label("flow-dependence — 무작위 sample 쌍의 spatial Pearson", styles),
        code(
            'def _pairwise_spatial_corr(stack, n_pairs, rng):\n'
            '    S, H, W = stack.shape\n'
            '    flat = stack.reshape(S, -1).astype(np.float64)\n'
            '    centered = flat - flat.mean(axis=1, keepdims=True)\n'
            '    norms = np.sqrt((centered ** 2).sum(axis=1))\n'
            '    rhos = np.empty(n_pairs)\n'
            '    cnt = 0\n'
            '    while cnt < n_pairs:\n'
            '        i, j = int(rng.integers(0, S)), int(rng.integers(0, S))\n'
            '        if i == j: continue\n'
            '        num = float((centered[i] * centered[j]).sum())\n'
            '        den = float(norms[i] * norms[j]) + 1e-12\n'
            '        rhos[cnt] = num / den; cnt += 1\n'
            '    return rhos',
            styles["code"],
        ),
        label("플롯 — heatmap 6장 + CV hist + 쌍별 corr hist", styles),
        code(
            '_save_heatmaps(stack, times, fig_dir / "exp1_logvar_heatmaps.png", rng)\n'
            '_save_cv_hist (cvs,           fig_dir / "exp1_pixelwise_cv_hist.png")\n'
            '_save_pair_hist(rhos,         fig_dir / "exp1_pairwise_correlation_hist.png")',
            styles["code"],
        ),
        label("run_exp1 — 전체 파이프라인 묶기", styles),
        code(
            'def run_exp1(ensemble_dir, figures_dir, metrics_dir, n_pairs=1000, seed=42):\n'
            '    rng = np.random.default_rng(seed)\n'
            '    stack, times = _load_temp_logvar_stack(Path(ensemble_dir))\n'
            '    cvs  = _pixelwise_cv(stack)\n'
            '    rhos = _pairwise_spatial_corr(stack, n_pairs=n_pairs, rng=rng)\n'
            '    ... # heatmap, hist 저장\n'
            '    stats = {"cv": {...}, "pairwise_correlation": {...}}\n'
            '    json.dump(stats, open(met_dir / "exp1_stats.json", "w"), indent=2)\n'
            '    return stats',
            styles["code"],
        ),
    ]
    return out


def s_v1_exp2(styles) -> list:
    out = section_header(
        "12. experiments/exp2_spread_vs_logvar.py",
        "σ_ℓ ↔ ensemble spread per-sample 상관 (실제 구현)",
        styles,
    )
    out += [
        role(
            "각 sample 한 장에서 σ_ℓ = √exp(ℓ_m) 와 멤버 std (ensemble spread) "
            "두 map 을 픽셀 단위로 flatten 해서 Pearson r 을 계산한다. "
            "결과 (S,) 분포의 hist 와 무작위 3개 sample 의 scatter, 그리고 GT "
            "시점간 spatial corr baseline 과 overlay 한 비교 hist 까지 만든다. "
            "baseline 이 0 근처로 분리되면 σ_ℓ↔spread 상관이 trivial 한 시점 간 "
            "통계 일치가 아니라는 근거가 된다.",
            styles,
        ),
        label("flatten 상관 + ensemble std helper", styles),
        code(
            'def _per_sample_corr(sigma_ell_flat, spread_flat):\n'
            '    a = sigma_ell_flat - sigma_ell_flat.mean()\n'
            '    b = spread_flat    - spread_flat.mean()\n'
            '    num = float((a * b).sum())\n'
            '    den = float(np.sqrt((a * a).sum() * (b * b).sum())) + 1e-12\n'
            '    return num / den\n\n'
            'def _ensemble_spread(ens):\n'
            '    """ens: (N, C, H, W) → (C, H, W) std (ddof=1)."""\n'
            '    return ens.std(axis=0, ddof=1)',
            styles["code"],
        ),
        label("sample 루프 — σ_ℓ, spread 두 map 을 flatten → Pearson", styles),
        code(
            'for i, f in enumerate(files):\n'
            '    es = load_ensemble_npz(f)\n'
            '    sigma_ell = np.sqrt(np.exp(es.log_var[TEMP_IDX]))   # (H, W)\n'
            '    spread    = _ensemble_spread(es.ensemble)[TEMP_IDX] # (H, W)\n'
            '    x, y = sigma_ell.flatten(), spread.flatten()\n'
            '    corrs[i] = _per_sample_corr(x, y)\n'
            '    cache.append((i, x, y))',
            styles["code"],
        ),
        label("GT baseline — sample 간 spatial corr 분포", styles),
        code(
            'def _load_gt_temp_stack(files):\n'
            '    arrs = [load_ensemble_npz(f).x_t_true[TEMP_IDX].astype(np.float32)\n'
            '            for f in files]\n'
            '    return np.stack(arrs, axis=0)                       # (S, H, W)\n\n'
            'gt_stack    = _load_gt_temp_stack(files)\n'
            'gt_pair_rhos = _pairwise_spatial_corr(gt_stack, n_gt_pairs, rng)',
            styles["code"],
        ),
        label("플롯 산출물", styles),
        code(
            'fig_dir/exp2_scatter_examples.png   # 3 random sample scatter (σ_ℓ vs spread)\n'
            'fig_dir/exp2_correlation_hist.png   # per-sample r 분포 (S,)\n'
            'fig_dir/exp2_vs_gt_baseline.png     # σ_ℓ↔spread vs GT pairwise corr overlay',
            styles["code"],
        ),
    ]
    return out


def build_story(styles: dict) -> list:
    story: list = []
    story += cover(styles)
    builders = [
        s_init, s_main, s_utils, s_ensemble_inference,
        s_exp1_dit, s_exp2_dit,
        s_exp3, s_exp4, s_exp5, s_exp6,
        s_v1_exp1, s_v1_exp2,
    ]
    for b in builders:
        story += b(styles)
        story.append(PageBreak())
    if story and isinstance(story[-1], PageBreak):
        story.pop()
    return story


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("HYSMyeongJo-Medium", 8)
    canvas.setFillColor(colors.HexColor("#888"))
    canvas.drawRightString(
        A4[0] - 15 * mm, 10 * mm, f"— {doc.page} —"
    )
    canvas.drawString(
        15 * mm, 10 * mm,
        "experiments_dit 코드 정리",
    )
    canvas.restoreState()


def main():
    body_font, code_font = register_fonts()
    styles = make_styles(body_font, code_font)
    out_path = REPO / "outputs" / "experiments_dit_code_summary.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=15 * mm, bottomMargin=18 * mm,
        title="experiments_dit 코드 정리",
        author="Diff2 / experiments_dit",
    )
    story = build_story(styles)
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"[ok] wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
