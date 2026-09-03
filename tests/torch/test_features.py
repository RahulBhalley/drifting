import torch
from transformers import ConvNextV2Config, ConvNextV2ForImageClassification

from drifting_torch.models.convnext import ConvNeXtV2FeatureExtractor
from drifting_torch.models.features import FrozenFeatureExtractor
from drifting_torch.models.mae import MAEResNet


def tiny_convnext() -> ConvNeXtV2FeatureExtractor:
    config = ConvNextV2Config(
        num_channels=3,
        patch_size=4,
        hidden_sizes=[8, 16, 32, 64],
        depths=[1, 1, 1, 1],
        num_labels=5,
        drop_path_rate=0.0,
    )
    torch.manual_seed(41)
    return ConvNeXtV2FeatureExtractor(ConvNextV2ForImageClassification(config))


def test_convnext_activation_contract():
    extractor = tiny_convnext()
    outputs = extractor(torch.randn(2, 3, 32, 32))
    assert {
        "convenxt_stage_0_mean",
        "convenxt_stage_1",
        "convenxt_stage_3_std",
        "global_mean",
        "global_std",
    } <= outputs.keys()
    assert all(value.ndim == 3 for value in outputs.values())
    assert all(value.shape[0] == 2 for value in outputs.values())


def test_feature_weights_frozen_but_input_gradient_flows():
    extractor = FrozenFeatureExtractor(convnext=tiny_convnext())
    images = torch.randn(2, 3, 32, 32, requires_grad=True)
    sum(value.sum() for value in extractor(images).values()).backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()
    assert all(parameter.grad is None for parameter in extractor.parameters())
    assert all(not parameter.requires_grad for parameter in extractor.parameters())


def test_reference_activations_detach_only_when_explicitly_requested():
    extractor = FrozenFeatureExtractor(convnext=tiny_convnext())
    generated = torch.randn(1, 3, 32, 32, requires_grad=True)
    live = extractor(generated)
    detached = extractor(generated, detach=True)
    assert any(value.requires_grad for value in live.values())
    assert all(not value.requires_grad for value in detached.values())


def test_mae_and_convnext_composition_namespaces_outputs():
    mae = MAEResNet(
        num_classes=5,
        base_channels=8,
        layers=(1, 1, 1, 1),
    )
    extractor = FrozenFeatureExtractor(mae=mae, convnext=tiny_convnext())
    outputs = extractor(torch.randn(1, 3, 32, 32))
    assert "global" in outputs
    assert "conv1" in outputs
    assert "convnext/global_mean" in outputs
    assert "convnext/convenxt_stage_2" in outputs
