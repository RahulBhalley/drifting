import math

import pytest
import torch

from drifting_torch.training.schedules import build_adamw, learning_rate


def test_warmup_constant_and_cosine_schedule_values():
    assert learning_rate(0, base_lr=1e-3, warmup_steps=10, total_steps=100, schedule="const") == 1e-6
    assert learning_rate(10, base_lr=1e-3, warmup_steps=10, total_steps=100, schedule="const") == 1e-3
    expected = 1e-3 * (1e-6 + (1 - 1e-6) * 0.5 * (1 + math.cos(math.pi * 0.5)))
    assert learning_rate(55, base_lr=1e-3, warmup_steps=10, total_steps=100, schedule="cosine") == pytest.approx(expected)
    with pytest.raises(ValueError, match="schedule"):
        learning_rate(0, base_lr=1e-3, warmup_steps=0, total_steps=1, schedule="bad")


def test_adamw_uses_scientific_hyperparameters():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = build_adamw([parameter], base_lr=2e-4, beta1=0.9, beta2=0.95, weight_decay=0.01)
    group = optimizer.param_groups[0]
    assert group["lr"] == 2e-4
    assert group["betas"] == (0.9, 0.95)
    assert group["eps"] == 1e-8
    assert group["weight_decay"] == 0.01

