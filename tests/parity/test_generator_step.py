import jax.numpy as jnp
import numpy as np
import optax
import torch

from drifting_torch.training.schedules import build_adamw


def test_adamw_single_transition_matches_optax():
    initial = np.array([1.25, -0.75], dtype=np.float32)
    gradient = np.array([0.4, -0.2], dtype=np.float32)
    learning_rate = 2e-4
    weight_decay = 0.01
    tx = optax.adamw(
        learning_rate=learning_rate, b1=0.9, b2=0.95,
        eps=1e-8, weight_decay=weight_decay,
    )
    jax_parameter = jnp.asarray(initial)
    opt_state = tx.init(jax_parameter)
    updates, _ = tx.update(jnp.asarray(gradient), opt_state, jax_parameter)
    expected = np.asarray(optax.apply_updates(jax_parameter, updates))

    parameter = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    optimizer = build_adamw(
        [parameter], base_lr=learning_rate, beta1=0.9, beta2=0.95,
        weight_decay=weight_decay,
    )
    parameter.grad = torch.from_numpy(gradient.copy())
    optimizer.step()
    np.testing.assert_allclose(parameter.detach().numpy(), expected, atol=2e-7, rtol=2e-7)
