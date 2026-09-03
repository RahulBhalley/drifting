"""Learning-rate and optimizer construction matching the JAX release."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor


def learning_rate(
    step: int,
    *,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    schedule: str,
) -> float:
    if warmup_steps < 0 or total_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and total_steps positive")
    if step < warmup_steps:
        return 1e-6 + (base_lr - 1e-6) * step / max(warmup_steps, 1)
    if schedule == "const":
        return float(base_lr)
    if schedule not in {"cos", "cosine"}:
        raise ValueError(f"unsupported learning-rate schedule: {schedule!r}")
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return float(
        base_lr
        * (1e-6 + (1 - 1e-6) * 0.5 * (1 + math.cos(math.pi * progress)))
    )


def build_adamw(
    parameters: Iterable[Tensor],
    *,
    base_lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float = 0.0,
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        parameters,
        lr=base_lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )


def assign_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


__all__ = ["assign_learning_rate", "build_adamw", "learning_rate"]
