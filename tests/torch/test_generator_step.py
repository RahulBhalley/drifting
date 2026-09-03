import pytest
import torch

from drifting_torch.data.datasets import DataBatch
from drifting_torch.memory_bank import ClassMemoryBank
from drifting_torch.models.features import FrozenFeatureExtractor
from drifting_torch.models.generator import DitGen
from drifting_torch.training.generator import GeneratorStepOptions, generator_train_step
from drifting_torch.training.schedules import build_adamw
from drifting_torch.training.state import GeneratorTrainState


def tiny_state(seed: int = 3) -> GeneratorTrainState:
    torch.manual_seed(2)
    model = DitGen(
        cond_dim=16,
        num_classes=3,
        input_size=4,
        in_channels=1,
        patch_size=2,
        hidden_size=16,
        depth=1,
        num_heads=2,
        mlp_ratio=2,
        out_channels=1,
    )
    optimizer = build_adamw(model.parameters(), base_lr=1e-3, beta1=0.9, beta2=0.95)
    return GeneratorTrainState.create(model, optimizer, ema_decay=0.9, seed=seed)


def banks():
    positive = ClassMemoryBank(num_classes=3, capacity=4, seed=5)
    negative = ClassMemoryBank(num_classes=1, capacity=8, seed=7)
    images = torch.linspace(-1, 1, 6 * 1 * 4 * 4).reshape(6, 1, 4, 4)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    positive.add(images, labels)
    negative.add(images, torch.zeros_like(labels))
    return positive, negative


def test_generator_step_updates_model_then_ema_with_finite_metrics():
    state = tiny_state()
    before = {name: value.clone() for name, value in state.model.state_dict().items()}
    ema_before = {name: value.clone() for name, value in state.ema.items()}
    batch = DataBatch(torch.zeros(2, 1, 4, 4), torch.tensor([0, 1]))
    options = GeneratorStepOptions(
        gen_per_label=2,
        pos_per_sample=2,
        neg_per_sample=2,
        cfg_min=1.0,
        cfg_max=2.0,
        radii=(0.2,),
        max_grad_norm=2.0,
        base_lr=1e-3,
        warmup_steps=0,
        total_steps=3,
    )
    result = generator_train_step(
        state, batch, banks(), FrozenFeatureExtractor(), options
    )
    assert state.completed_steps == 1
    assert all(torch.isfinite(value) for value in result.metrics.values())
    changed = [name for name, value in state.model.state_dict().items() if not torch.equal(value, before[name])]
    assert changed
    for name in changed:
        expected = ema_before[name] * 0.9 + state.model.state_dict()[name] * 0.1
        torch.testing.assert_close(state.ema[name], expected)


def test_step_can_replay_explicit_cfg_noise_and_noise_labels():
    state = tiny_state()
    batch = DataBatch(torch.zeros(2, 1, 4, 4), torch.tensor([0, 1]))
    options = GeneratorStepOptions(
        gen_per_label=2, pos_per_sample=1, neg_per_sample=1, radii=(0.2,),
        cfg=torch.tensor([1.1, 1.7]),
        noise=torch.zeros(4, 1, 4, 4),
        noise_labels=torch.zeros(4, 1, dtype=torch.long),
        base_lr=1e-3, warmup_steps=0, total_steps=1,
    )
    result = generator_train_step(state, batch, banks(), FrozenFeatureExtractor(), options)
    assert result.metrics["cfg_mean"].item() == pytest.approx(1.4)
