"""Heteroscedastic Gaussian NLL loss (instruction.md Eq. 20)."""
from __future__ import annotations

import torch


def heteroscedastic_nll_loss(
    eps_true: torch.Tensor,
    eps_pred: torch.Tensor,
    log_var: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """L = 0.5 * exp(-ℓ) * ||ε - ε̂||^2 + 0.5 * ℓ.

    Args:
        eps_true, eps_pred, log_var: 같은 shape (B, ...)
        reduction: 'mean', 'sum', 'none'
    """
    precision = torch.exp(-log_var)
    squared_error = (eps_true - eps_pred) ** 2
    loss = 0.5 * precision * squared_error + 0.5 * log_var

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unknown reduction: {reduction}")
