import pytest
import torch

from drifting_torch.models.mae import MAEResNet, patch_input, safe_std


@pytest.fixture
def tiny_mae() -> MAEResNet:
    torch.manual_seed(5)
    return MAEResNet(
        num_classes=7,
        in_channels=3,
        base_channels=8,
        patch_size=4,
        layers=(1, 1, 1, 1),
    )


def test_patch_input_moves_pixels_into_channels():
    x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    patched = patch_input(x, 2)
    assert patched.shape == (2, 12, 4, 4)
    torch.testing.assert_close(patched[:, :, 0, 0], x[:, :, :2, :2].permute(0, 2, 3, 1).reshape(2, 12))


def test_safe_std_uses_population_variance_and_epsilon():
    x = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
    actual = safe_std(x, dim=(2, 3))
    expected = torch.sqrt(torch.tensor([[5.0 + 1e-6]]))
    torch.testing.assert_close(actual, expected)


def test_mae_forward_returns_reconstruction_logits_and_per_example_losses(tiny_mae):
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([1, 4])
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, ::2, ::2] = 1

    output = tiny_mae(images, labels, mask=mask, lambda_cls=0.25, train=False)

    assert output.loss.shape == (2,)
    assert output.reconstruction.shape == images.shape
    assert output.logits.shape == (2, 7)
    assert output.mask.shape == mask.shape
    assert torch.isfinite(output.loss).all()
    torch.testing.assert_close(output.mask_ratio, mask.mean(dim=(1, 2, 3)))


def test_mae_explicit_mask_is_deterministic(tiny_mae):
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([1, 4])
    mask = torch.ones(2, 1, 32, 32)
    first = tiny_mae(images, labels, mask=mask, train=False)
    second = tiny_mae(images, labels, mask=mask, train=False)
    torch.testing.assert_close(first.loss, second.loss, rtol=0, atol=0)
    torch.testing.assert_close(first.reconstruction, second.reconstruction, rtol=0, atol=0)


def test_mae_random_mask_can_use_explicit_generator(tiny_mae):
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([1, 4])
    first = tiny_mae(images, labels, generator=torch.Generator().manual_seed(19), train=False)
    second = tiny_mae(images, labels, generator=torch.Generator().manual_seed(19), train=False)
    torch.testing.assert_close(first.mask, second.mask, rtol=0, atol=0)


def test_mae_rejects_mask_in_nhwc_layout(tiny_mae):
    with pytest.raises(ValueError, match="NCHW"):
        tiny_mae(
            torch.randn(2, 3, 32, 32),
            torch.tensor([1, 4]),
            mask=torch.ones(2, 32, 32, 1),
        )


def test_mae_activation_contract(tiny_mae):
    output = tiny_mae.get_activations(torch.randn(2, 3, 32, 32), every_k_block=1)
    assert {"norm_x", "conv1", "layer1", "layer4", "layer1_blk1"} <= output.keys()
    assert all(value.ndim == 3 for value in output.values())
    assert all(value.shape[0] == 2 for value in output.values())
    assert {"conv1_mean", "conv1_std", "conv1_mean_2", "conv1_std_2"} <= output.keys()


def test_mae_gradients_are_finite(tiny_mae):
    output = tiny_mae(
        torch.randn(2, 3, 32, 32),
        torch.tensor([1, 4]),
        mask=torch.ones(2, 1, 32, 32),
    )
    output.loss.mean().backward()
    gradients = [parameter.grad for parameter in tiny_mae.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_mae_precision_modes_are_explicit_and_exclusive():
    assert MAEResNet(base_channels=8, layers=(1, 1, 1, 1), use_fp16=True).compute_dtype == torch.float16
    with pytest.raises(ValueError, match="mutually exclusive"):
        MAEResNet(use_bf16=True, use_fp16=True)
