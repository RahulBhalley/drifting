import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from drifting_jax.loss import drift_loss as jax_drift_loss
from drifting_torch.loss import drift_loss as torch_drift_loss


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_drift_loss_value_metric_and_gradient_parity(dtype):
    if dtype is np.float64 and not jax.config.x64_enabled:
        pytest.skip("set JAX_ENABLE_X64=1 for the float64 diagnostic")
    rng = np.random.default_rng(31)
    gen = rng.normal(size=(2, 3, 7)).astype(dtype)
    pos = rng.normal(size=(2, 5, 7)).astype(dtype)
    neg = rng.normal(size=(2, 2, 7)).astype(dtype)
    weight_gen = rng.uniform(0.2, 1.7, size=(2, 3)).astype(dtype)
    weight_pos = rng.uniform(0.2, 1.7, size=(2, 5)).astype(dtype)
    weight_neg = rng.uniform(0.2, 1.7, size=(2, 2)).astype(dtype)
    radii = (0.2, 0.05)

    def jax_objective(value):
        loss, _ = jax_drift_loss(
            value, pos, neg, weight_gen, weight_pos, weight_neg, radii
        )
        return loss.mean()

    jax_loss, jax_info = jax_drift_loss(
        gen, pos, neg, weight_gen, weight_pos, weight_neg, radii
    )
    jax_grad = jax.grad(jax_objective)(jnp.asarray(gen))

    torch_gen = torch.tensor(gen, requires_grad=True)
    torch_loss, torch_info = torch_drift_loss(
        torch_gen,
        torch.tensor(pos),
        torch.tensor(neg),
        torch.tensor(weight_gen),
        torch.tensor(weight_pos),
        torch.tensor(weight_neg),
        radii,
    )
    torch_loss.mean().backward()

    # The released loss casts both float32 and float64 inputs to float32. This
    # therefore verifies coercion parity as well as the production path.
    tolerance = 2e-5
    np.testing.assert_allclose(torch_loss.detach().numpy(), np.asarray(jax_loss), atol=tolerance, rtol=tolerance)
    for name in jax_info:
        np.testing.assert_allclose(torch_info[name].detach().numpy(), np.asarray(jax_info[name]), atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(torch_gen.grad.numpy(), np.asarray(jax_grad), atol=tolerance, rtol=tolerance)
