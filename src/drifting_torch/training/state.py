"""Mutable runtime state for exact PyTorch generator training resume."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class GeneratorTrainState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    ema: dict[str, Tensor]
    ema_decay: float
    generator: torch.Generator
    completed_steps: int = 0
    scaler: object | None = None

    @classmethod
    def create(
        cls,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        ema_decay: float,
        seed: int,
        scaler: object | None = None,
    ) -> "GeneratorTrainState":
        if not 0 <= ema_decay <= 1:
            raise ValueError("ema_decay must be in [0, 1]")
        ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return cls(model, optimizer, ema, float(ema_decay), generator, scaler=scaler)

    @torch.no_grad()
    def update_ema(self) -> None:
        for name, value in self.model.state_dict().items():
            source = value.detach()
            if self.ema[name].is_floating_point():
                self.ema[name].mul_(self.ema_decay).add_(
                    source, alpha=1.0 - self.ema_decay
                )
            else:
                self.ema[name].copy_(source)


class MAETrainState(GeneratorTrainState):
    """The same exact-resume state contract specialized by name for MAE."""


__all__ = ["GeneratorTrainState", "MAETrainState"]
