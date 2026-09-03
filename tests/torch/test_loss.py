import pytest
import torch

from drifting_torch.loss import drift_loss


def fixture(dtype=torch.float32):
    generator = torch.Generator().manual_seed(7)
    gen = torch.randn(2, 3, 5, generator=generator, dtype=dtype, requires_grad=True)
    pos = torch.randn(2, 4, 5, generator=generator, dtype=dtype, requires_grad=True)
    neg = torch.randn(2, 2, 5, generator=generator, dtype=dtype, requires_grad=True)
    return gen, pos, neg


def test_drift_loss_shape_metrics_and_target_is_stopped():
    gen, pos, neg = fixture()
    loss, info = drift_loss(gen, pos, neg, R_list=(0.2, 0.05))
    assert loss.shape == (2,)
    assert set(info) == {"scale", "loss_0.2", "loss_0.05"}
    assert all(torch.isfinite(value).all() for value in (loss, *info.values()))
    loss.mean().backward()
    assert gen.grad is not None and torch.isfinite(gen.grad).all()
    assert pos.grad is None
    assert neg.grad is None


def test_drift_loss_accepts_no_negatives_and_default_weights():
    gen, pos, _ = fixture()
    loss, info = drift_loss(gen, pos, R_list=(0.1,))
    assert loss.shape == (2,)
    assert info["scale"].ndim == 0


def test_drift_loss_rejects_invalid_inputs():
    gen, pos, neg = fixture()
    with pytest.raises(ValueError, match="rank 3"):
        drift_loss(gen[0], pos, neg)
    with pytest.raises(ValueError, match="positive"):
        drift_loss(gen, pos[:, :0], neg)
    with pytest.raises(ValueError, match="radius"):
        drift_loss(gen, pos, neg, R_list=(0.0,))

