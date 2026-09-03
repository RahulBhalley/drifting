import pytest
import torch

from drifting_torch.models.generator import DitGen, GenerationOutput, build_generator


@pytest.fixture
def tiny_generator() -> DitGen:
    torch.manual_seed(7)
    return DitGen(
        cond_dim=16,
        num_classes=11,
        noise_classes=5,
        noise_coords=2,
        input_size=8,
        in_channels=3,
        patch_size=2,
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        out_channels=3,
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
    )


def explicit_inputs():
    labels = torch.tensor([1, 3], dtype=torch.long)
    noise = torch.linspace(-1, 1, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    noise_labels = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    return labels, noise, noise_labels


def test_generator_explicit_noise_is_repeatable(tiny_generator):
    labels, noise, noise_labels = explicit_inputs()
    out1 = tiny_generator(labels, noise=noise, noise_labels=noise_labels)
    out2 = tiny_generator(labels, noise=noise, noise_labels=noise_labels)

    assert isinstance(out1, GenerationOutput)
    torch.testing.assert_close(out1.samples, out2.samples, rtol=0, atol=0)
    torch.testing.assert_close(out1.noise.x, noise)
    torch.testing.assert_close(out1.noise.noise_labels, noise_labels)


def test_generator_returns_nchw_output_and_finite_gradients(tiny_generator):
    labels, noise, noise_labels = explicit_inputs()
    output = tiny_generator(labels, noise=noise, noise_labels=noise_labels)

    assert output.samples.shape == (2, 3, 8, 8)
    assert torch.isfinite(output.samples).all()
    loss = output.samples.square().mean()
    loss.backward()
    grads = [parameter.grad for parameter in tiny_generator.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_generator_temperature_scales_recorded_noise(tiny_generator):
    labels, noise, noise_labels = explicit_inputs()
    output = tiny_generator(labels, temp=0.25, noise=noise, noise_labels=noise_labels)
    torch.testing.assert_close(output.noise.x, noise * 0.25)


def test_generator_rejects_wrong_noise_layout(tiny_generator):
    labels, noise, noise_labels = explicit_inputs()
    with pytest.raises(ValueError, match="NCHW"):
        tiny_generator(labels, noise=noise.permute(0, 2, 3, 1), noise_labels=noise_labels)


def test_build_generator_accepts_scientific_model_mapping():
    model = build_generator(
        {
            "cond_dim": 8,
            "num_classes": 4,
            "input_size": 4,
            "in_channels": 2,
            "out_channels": 2,
            "patch_size": 2,
            "hidden_size": 16,
            "depth": 1,
            "num_heads": 2,
        }
    )
    assert isinstance(model, DitGen)
    assert model.model.input_size == 4


def test_random_noise_can_use_explicit_torch_generator(tiny_generator):
    labels = torch.tensor([1, 3], dtype=torch.long)
    first_rng = torch.Generator().manual_seed(123)
    second_rng = torch.Generator().manual_seed(123)

    first = tiny_generator(labels, generator=first_rng)
    second = tiny_generator(labels, generator=second_rng)

    torch.testing.assert_close(first.noise.x, second.noise.x, rtol=0, atol=0)
    torch.testing.assert_close(
        first.noise.noise_labels, second.noise.noise_labels, rtol=0, atol=0
    )


def test_class_tokens_standard_mlp_and_checkpointing_execute():
    model = DitGen(
        cond_dim=8,
        num_classes=4,
        input_size=4,
        in_channels=2,
        out_channels=2,
        patch_size=2,
        hidden_size=16,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        n_cls_tokens=2,
        use_remat=True,
    )
    labels = torch.tensor([1, 2])
    noise = torch.randn(2, 2, 4, 4)
    noise_labels = torch.zeros(2, 1, dtype=torch.long)

    output = model(labels, noise=noise, noise_labels=noise_labels)
    assert output.samples.shape == noise.shape
    output.samples.sum().backward()
    assert model.model.class_token_proj.weight.grad is not None


def test_bfloat16_compute_keeps_fp32_parameters():
    model = DitGen(
        cond_dim=8,
        num_classes=4,
        input_size=4,
        in_channels=2,
        out_channels=2,
        patch_size=2,
        hidden_size=16,
        depth=1,
        num_heads=2,
        use_bf16=True,
    )
    output = model(
        torch.tensor([1]),
        noise=torch.randn(1, 2, 4, 4),
        noise_labels=torch.zeros(1, 1, dtype=torch.long),
    )
    assert output.samples.dtype == torch.bfloat16
    assert output.noise.x.dtype == torch.bfloat16
    assert next(model.parameters()).dtype == torch.float32
    assert torch.isfinite(output.samples).all()


def test_generator_precision_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        DitGen(cond_dim=8, use_bf16=True, use_fp16=True)


def test_generator_fp16_mode_is_explicit():
    model = DitGen(
        cond_dim=8, num_classes=4, input_size=4, in_channels=2, out_channels=2,
        patch_size=2, hidden_size=16, depth=1, num_heads=2, use_fp16=True,
    )
    assert model.compute_dtype == torch.float16
