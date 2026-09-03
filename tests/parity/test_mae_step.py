from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

from drifting_torch.training.schedules import build_adamw
from tests.parity.test_mae import build_models


def test_mae_selected_gradient_update_and_ema_match_jax():
    jax_model, jax_params, torch_model, images, labels, mask = build_models()

    def objective(params):
        loss, _ = jax_model.apply(
            {"params": params}, images, labels, mask=mask,
            lambda_cls=0.25, train=False,
        )
        return loss.mean()

    jax_grad = jax.grad(objective)(jax_params)
    tx = optax.adamw(learning_rate=1e-4, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.01)
    optimizer_state = tx.init(jax_params)
    updates, _ = tx.update(jax_grad, optimizer_state, jax_params)
    jax_updated = optax.apply_updates(jax_params, updates)
    flat_grad = traverse_util.flatten_dict(jax_grad, sep="/")
    flat_updated = traverse_util.flatten_dict(jax_updated, sep="/")

    optimizer = build_adamw(
        torch_model.parameters(), base_lr=1e-4, beta1=0.9, beta2=0.95,
        weight_decay=0.01,
    )
    output = torch_model(
        torch.from_numpy(np.array(images)).permute(0, 3, 1, 2),
        torch.from_numpy(np.array(labels)).long(),
        mask=torch.from_numpy(np.array(mask)).permute(0, 3, 1, 2),
        lambda_cls=0.25,
        train=False,
    )
    output.loss.mean().backward()
    np.testing.assert_allclose(
        torch_model.fc.weight.grad.numpy(), np.asarray(flat_grad["fc/kernel"]).T,
        atol=3e-5, rtol=3e-5,
    )
    initial_ema = torch_model.fc.weight.detach().clone()
    optimizer.step()
    np.testing.assert_allclose(
        torch_model.fc.weight.detach().numpy(), np.asarray(flat_updated["fc/kernel"]).T,
        atol=3e-5, rtol=3e-5,
    )
    torch_ema = initial_ema * 0.9 + torch_model.fc.weight.detach() * 0.1
    jax_ema = np.asarray(jax_params["fc"]["kernel"]).T * 0.9 + np.asarray(flat_updated["fc/kernel"]).T * 0.1
    np.testing.assert_allclose(torch_ema.numpy(), jax_ema, atol=3e-5, rtol=3e-5)
