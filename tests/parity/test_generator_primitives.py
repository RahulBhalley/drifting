import jax
import jax.numpy as jnp
import numpy as np
import torch

from drifting_jax.models.generator import (
    RMSNorm as JaxRMSNorm,
    apply_rope as jax_apply_rope,
    get_1d_sincos_pos_embed_from_grid as jax_sincos_1d,
    get_2d_sincos_pos_embed as jax_sincos_2d,
    modulate as jax_modulate,
)
from drifting_torch.models.primitives import (
    RMSNorm,
    TimestepEmbedder,
    apply_rope,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    modulate,
    patchify_nchw,
    unpatchify_nchw,
)


def assert_fp32_close(actual, expected, *, name: str, rtol=1e-5, atol=1e-6):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    max_error = float(np.max(np.abs(actual - expected)))
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=f"{name} max_error={max_error}")


def test_sincos_position_embedding_parity():
    positions = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert_fp32_close(
        get_1d_sincos_pos_embed_from_grid(16, positions),
        jax_sincos_1d(16, positions),
        name="sincos_1d",
    )
    assert_fp32_close(
        get_2d_sincos_pos_embed(32, 4),
        jax_sincos_2d(32, 4),
        name="sincos_2d",
    )


def test_modulation_parity():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(2, 5, 8)).astype(np.float32)
    shift = rng.normal(size=(2, 8)).astype(np.float32)
    scale = rng.normal(size=(2, 8)).astype(np.float32)
    torch_value = modulate(torch.from_numpy(x), torch.from_numpy(shift), torch.from_numpy(scale))
    jax_value = jax_modulate(jnp.asarray(x), jnp.asarray(shift), jnp.asarray(scale))
    assert_fp32_close(torch_value.detach().numpy(), jax_value, name="modulate")


def test_rms_norm_parity():
    x = np.random.default_rng(12).normal(size=(2, 5, 8)).astype(np.float32)
    torch_norm = RMSNorm(8)
    jax_norm = JaxRMSNorm(8)
    variables = jax_norm.init(jax.random.PRNGKey(0), jnp.asarray(x))

    torch_value = torch_norm(torch.from_numpy(x)).detach().numpy()
    jax_value = jax_norm.apply(variables, jnp.asarray(x))
    assert_fp32_close(torch_value, jax_value, name="rms_norm")


def test_rope_parity():
    rng = np.random.default_rng(13)
    q = rng.normal(size=(2, 7, 3, 8)).astype(np.float32)
    k = rng.normal(size=(2, 7, 3, 8)).astype(np.float32)
    torch_q, torch_k = apply_rope(torch.from_numpy(q), torch.from_numpy(k))
    jax_q, jax_k = jax_apply_rope(jnp.asarray(q), jnp.asarray(k))
    assert_fp32_close(torch_q.numpy(), jax_q, name="rope_q")
    assert_fp32_close(torch_k.numpy(), jax_k, name="rope_k")


def test_patch_layout_parity_after_nhwc_nchw_conversion():
    nhwc = np.arange(2 * 8 * 8 * 3, dtype=np.float32).reshape(2, 8, 8, 3)
    jax_patches = jnp.asarray(nhwc).reshape(2, 4, 2, 4, 2, 3)
    jax_patches = jnp.transpose(jax_patches, (0, 1, 3, 2, 4, 5)).reshape(2, 16, 12)
    nchw = torch.from_numpy(nhwc).permute(0, 3, 1, 2)
    torch_patches = patchify_nchw(nchw, grid_size=4)
    assert_fp32_close(torch_patches.numpy(), jax_patches, name="patchify")

    restored = unpatchify_nchw(torch_patches, grid_size=4, patch_size=2, channels=3)
    assert_fp32_close(restored.permute(0, 2, 3, 1).numpy(), nhwc, name="unpatchify")


def test_timestep_frequency_embedding_matches_jax_formula():
    values = np.array([0.0, 1.0, 3.5], dtype=np.float32)
    embedder = TimestepEmbedder(hidden_size=8, frequency_embedding_size=6)
    actual = embedder.frequency_embedding(torch.from_numpy(values)).numpy()
    half = 3
    frequencies = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
    args = jnp.asarray(values)[:, None] * frequencies[None]
    expected = jnp.concatenate((jnp.cos(args), jnp.sin(args)), axis=-1)
    assert_fp32_close(actual, expected, name="timestep_frequency")
