"""Composition of frozen feature extractors used by the drifting loss."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .convnext import ConvNeXtV2FeatureExtractor
from .mae import MAEResNet


class FrozenFeatureExtractor(nn.Module):
    def __init__(
        self,
        *,
        mae: MAEResNet | None = None,
        convnext: ConvNeXtV2FeatureExtractor | None = None,
        postprocess_fn: Callable[[Tensor], Tensor] | None = None,
        mae_options: Mapping[str, Any] | None = None,
        convnext_options: Mapping[str, Any] | None = None,
        include_global: bool = True,
        has_scale: bool = False,
    ):
        super().__init__()
        self.mae = mae
        self.convnext = convnext
        self.postprocess_fn = postprocess_fn or (lambda value: value)
        self.mae_options = dict(mae_options or {})
        self.convnext_options = dict(convnext_options or {})
        self.include_global = include_global
        self.has_scale = has_scale
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        if self.mae is not None:
            self.mae.eval()
        if self.convnext is not None:
            self.convnext.eval()
        return self

    def forward(self, images: Tensor, *, detach: bool = False) -> dict[str, Tensor]:
        output: dict[str, Tensor] = {}
        if self.include_global:
            output["global"] = images.reshape(images.shape[0], 1, -1)
        if self.has_scale:
            output["norm_x"] = torch.sqrt((images**2).mean(dim=(2, 3)) + 1e-6).unsqueeze(1)
        if self.mae is not None:
            output.update(self.mae.get_activations(images, **self.mae_options))
        if self.convnext is not None:
            pixels = self.postprocess_fn(images)
            mean = torch.tensor(
                [0.485, 0.456, 0.406], device=pixels.device, dtype=pixels.dtype
            )[None, :, None, None]
            std = torch.tensor(
                [0.229, 0.224, 0.225], device=pixels.device, dtype=pixels.dtype
            )[None, :, None, None]
            convnext_output = self.convnext((pixels - mean) / std, **self.convnext_options)
            output.update({f"convnext/{name}": value for name, value in convnext_output.items()})
        if detach:
            output = {name: value.detach() for name, value in output.items()}
        return output


def build_activation_function(
    config: Mapping[str, Any],
    postprocess_fn: Callable[[Tensor], Tensor],
    *,
    mae: MAEResNet | None = None,
    convnext: ConvNeXtV2FeatureExtractor | None = None,
) -> FrozenFeatureExtractor:
    """Build an explicit, frozen feature composition from scientific options."""
    values = dict(config)
    if values.get("use_mae", True) and mae is None:
        raise ValueError("a loaded PyTorch MAE is required when use_mae=true")
    if values.get("use_convnext", False) and convnext is None:
        convnext = ConvNeXtV2FeatureExtractor.from_pretrained(
            values.get("convnext_model", "base"),
            use_bf16=values.get("convnext_bf16", False),
        )
    return FrozenFeatureExtractor(
        mae=mae if values.get("use_mae", True) else None,
        convnext=convnext if values.get("use_convnext", False) else None,
        postprocess_fn=postprocess_fn,
        mae_options=values.get("mae_options"),
        convnext_options=values.get("convnext_options"),
        has_scale=values.get("has_scale", False),
    )
