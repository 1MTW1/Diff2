"""시각화 함수 (instruction.md §4.14)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def visualize_ensemble_member(
    ensemble: torch.Tensor,
    variable_names: list[str],
    save_path: str | None = None,
    n_show: int = 16,
) -> None:
    """Ensemble member들을 (4x4) grid로 저장.

    Args:
        ensemble: (B, C, H, W)
        save_path: prefix. 변수별로 '{save_path}_{var}.png' 저장.
    """
    n_show = min(n_show, ensemble.shape[0])

    for var_idx, var_name in enumerate(variable_names):
        fig, axes = plt.subplots(4, 4, figsize=(16, 8))
        axes = axes.flatten()

        vmin = float(ensemble[:n_show, var_idx].min())
        vmax = float(ensemble[:n_show, var_idx].max())

        for i in range(n_show):
            ax = axes[i]
            ax.imshow(
                ensemble[i, var_idx].cpu().numpy(),
                vmin=vmin, vmax=vmax, cmap="RdBu_r",
            )
            ax.set_title(f"Member {i + 1}")
            ax.axis("off")
        for i in range(n_show, len(axes)):
            axes[i].axis("off")

        plt.suptitle(f"Ensemble Members: {var_name}")
        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}_{var_name}.png", dpi=150)
        plt.close(fig)


def visualize_spread_vs_uncertainty(
    ensemble: torch.Tensor,
    log_var_main: torch.Tensor,
    variable_names: list[str],
    save_path: str | None = None,
) -> None:
    """Ensemble spread와 학습된 ℓ_main 비교 plot.

    Args:
        ensemble:     (B, C, H, W)
        log_var_main: (C, H, W)
    """
    spread = ensemble.std(dim=0).cpu().numpy()
    uncertainty = torch.exp(log_var_main).sqrt().cpu().numpy()

    n_vars = len(variable_names)
    fig, axes = plt.subplots(n_vars, 2, figsize=(12, 4 * n_vars))
    if n_vars == 1:
        axes = axes[None, :]

    for var_idx, var_name in enumerate(variable_names):
        ax = axes[var_idx, 0]
        im = ax.imshow(spread[var_idx], cmap="viridis")
        ax.set_title(f"{var_name}: Ensemble Spread")
        plt.colorbar(im, ax=ax)

        ax = axes[var_idx, 1]
        im = ax.imshow(uncertainty[var_idx], cmap="viridis")
        ax.set_title(f"{var_name}: Predicted Uncertainty √exp(ℓ_main)")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)
