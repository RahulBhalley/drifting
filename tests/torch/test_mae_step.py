import pytest
import torch

from drifting_torch.data import DataBatch
from drifting_torch.models.mae import MAEResNet
from drifting_torch.training.mae import MAEStepOptions, classifier_weight, mae_train_step
from drifting_torch.training.schedules import build_adamw
from drifting_torch.training.state import MAETrainState


def tiny_mae_state(seed=13):
    torch.manual_seed(12)
    model = MAEResNet(
        num_classes=3, in_channels=3, base_channels=8, patch_size=4,
        layers=(1, 1, 1, 1),
    )
    optimizer = build_adamw(model.parameters(), base_lr=1e-3, beta1=0.9, beta2=0.95)
    return MAETrainState.create(model, optimizer, ema_decay=0.9, seed=seed)


def fixed_batch():
    images = torch.linspace(-1, 1, 2 * 3 * 16 * 16).reshape(2, 3, 16, 16)
    return DataBatch(images, torch.tensor([0, 2]))


def fixed_mask():
    mask = torch.zeros(2, 1, 16, 16)
    mask[:, :, ::2] = 1
    return mask


def test_classifier_weight_ramp():
    assert classifier_weight(95, total=100, finetune_steps=4, warmup=2, target=0.1) == 0
    assert classifier_weight(97, total=100, finetune_steps=4, warmup=2, target=0.1) == pytest.approx(0.05)
    assert classifier_weight(99, total=100, finetune_steps=4, warmup=2, target=0.1) == pytest.approx(0.1)


def test_mae_step_updates_parameters_ema_and_metrics():
    state = tiny_mae_state()
    before = {name: value.clone() for name, value in state.model.state_dict().items()}
    result = mae_train_step(
        state,
        fixed_batch(),
        fixed_mask(),
        MAEStepOptions(lambda_cls=0.25, base_lr=1e-3, total_steps=3),
    )
    assert state.completed_steps == 1
    assert {"loss", "cls_loss", "recon_loss", "accuracy", "mask_ratio", "g_norm", "lr"} <= result.metrics.keys()
    assert all(torch.isfinite(value) for value in result.metrics.values())
    assert any(not torch.equal(value, before[name]) for name, value in state.model.state_dict().items())


def test_mae_step_samples_mask_in_requested_range():
    state = tiny_mae_state()
    result = mae_train_step(
        state,
        fixed_batch(),
        None,
        MAEStepOptions(mask_ratio_min=0.5, mask_ratio_max=0.5, base_lr=1e-3, total_steps=1),
    )
    assert result.metrics["mask_ratio"].item() == pytest.approx(0.5, abs=0.2)

