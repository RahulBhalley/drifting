from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from drifting_jax.models.generator import DitGen
from drifting_jax.models.mae import MAEResNetJAX, patch_input
from drifting_jax.distributed import set_global_mesh


def _tiny_generator() -> DitGen:
    return DitGen(
        cond_dim=8,
        num_classes=4,
        noise_classes=3,
        noise_coords=2,
        input_size=4,
        in_channels=2,
        n_cls_tokens=1,
        patch_size=2,
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        out_channels=2,
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        use_bf16=False,
        attn_fp32=True,
    )


def test_generator_external_noise_replays_native_forward() -> None:
    """Catches an oracle hook that changes scaling or discrete-noise conditioning."""
    set_global_mesh(1)
    model = _tiny_generator()
    labels = jnp.array([1, 3], dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "noise": jax.random.PRNGKey(1)},
        c=labels,
        cfg_scale=jnp.array([1.0, 1.5]),
        temp=1.25,
    )
    native = model.apply(
        variables,
        c=labels,
        cfg_scale=jnp.array([1.0, 1.5]),
        temp=1.25,
        rngs={"noise": jax.random.PRNGKey(2)},
    )
    replay = model.apply(
        variables,
        c=labels,
        cfg_scale=jnp.array([1.0, 1.5]),
        temp=1.25,
        noise=native["noise"]["x"] / 1.25,
        noise_labels=native["noise"]["noise_labels"],
    )

    np.testing.assert_array_equal(replay["noise"]["x"], native["noise"]["x"])
    np.testing.assert_array_equal(
        replay["noise"]["noise_labels"], native["noise"]["noise_labels"]
    )
    np.testing.assert_allclose(replay["samples"], native["samples"], rtol=0, atol=0)


def test_mae_explicit_mask_bypasses_masking_rng() -> None:
    """Catches MAE parity runs accidentally consuming backend-native randomness."""
    model = MAEResNetJAX(
        num_classes=4,
        in_channels=2,
        base_channels=32,
        patch_size=2,
        dropout_prob=0.0,
        layers=(1, 1, 1, 1),
        use_bf16=False,
        input_patch_size=1,
    )
    images = jnp.arange(2 * 16 * 16 * 2, dtype=jnp.float32).reshape(2, 16, 16, 2) / 100
    labels = jnp.array([0, 1], dtype=jnp.int32)
    mask = jnp.zeros((*patch_input(images, 1).shape[:3], 1), dtype=jnp.float32)
    variables = model.init(
        {"params": jax.random.PRNGKey(3), "masking": jax.random.PRNGKey(4)},
        images,
        labels,
    )

    loss1, metrics1 = model.apply(variables, images, labels, mask=mask, train=False)
    loss2, metrics2 = model.apply(variables, images, labels, mask=mask, train=False)

    np.testing.assert_array_equal(loss1, loss2)
    for key in metrics1:
        np.testing.assert_array_equal(metrics1[key], metrics2[key])
    np.testing.assert_array_equal(metrics1["mask_ratio"], np.zeros((2,), dtype=np.float32))
