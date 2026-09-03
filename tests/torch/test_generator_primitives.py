import numpy as np
import pytest
import torch

from drifting_torch.models.primitives import (
    RMSNorm,
    apply_rope,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    modulate,
    patchify_nchw,
    unpatchify_nchw,
)


def test_modulate_broadcasts_over_tokens():
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    shift = torch.ones(2, 4)
    scale = torch.full((2, 4), 0.5)
    torch.testing.assert_close(modulate(x, shift, scale), x * 1.5 + 1)


def test_rms_norm_matches_fp32_definition_and_has_gradients():
    x = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]], requires_grad=True)
    norm = RMSNorm(4, eps=1e-6)
    expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)

    actual = norm(x)

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(norm.weight.grad).all()


def test_rope_preserves_shape_and_pair_norms():
    q = torch.randn(2, 7, 3, 8)
    k = torch.randn(2, 7, 3, 8)

    q_rot, k_rot = apply_rope(q, k)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    torch.testing.assert_close(q_rot.square().sum(-1), q.square().sum(-1))
    torch.testing.assert_close(k_rot.square().sum(-1), k.square().sum(-1))


def test_rope_rejects_odd_head_dimension():
    q = torch.randn(1, 2, 1, 3)
    with pytest.raises(ValueError, match="even"):
        apply_rope(q, q)


def test_patchify_unpatchify_round_trip_nchw():
    x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    patches = patchify_nchw(x, grid_size=4)
    assert patches.shape == (2, 16, 12)
    restored = unpatchify_nchw(patches, grid_size=4, patch_size=2, channels=3)
    torch.testing.assert_close(restored, x)


def test_sincos_embeddings_follow_reference_axis_order():
    pos = np.array([[0.0, 1.0]], dtype=np.float32)
    one_d = get_1d_sincos_pos_embed_from_grid(4, pos)
    np.testing.assert_allclose(one_d[0], np.array([0.0, 0.0, 1.0, 1.0]))

    two_d = get_2d_sincos_pos_embed(8, 2)
    assert two_d.shape == (4, 8)
    np.testing.assert_allclose(two_d[0], np.array([0.0, 0.0, 1.0, 1.0] * 2))
