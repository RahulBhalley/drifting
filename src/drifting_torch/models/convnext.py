"""Native PyTorch ConvNeXtV2 feature activation adapter."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mae import safe_std


MODEL_IDS = {
    "base": "facebook/convnextv2-base-22k-224",
    "tiny": "facebook/convnextv2-tiny-22k-224",
}


class ConvNeXtV2FeatureExtractor(nn.Module):
    """Expose the activation contract used by the released JAX ConvNeXt path."""

    def __init__(self, model: nn.Module, *, use_bf16: bool = False):
        super().__init__()
        self.model = model
        self.use_bf16 = use_bf16
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "base",
        *,
        use_bf16: bool = False,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> "ConvNeXtV2FeatureExtractor":
        from transformers import ConvNextV2ForImageClassification

        try:
            model_id = MODEL_IDS[model_name]
        except KeyError as error:
            raise ValueError(f"Unsupported ConvNeXtV2 model: {model_name!r}") from error
        model = ConvNextV2ForImageClassification.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return cls(model, use_bf16=use_bf16)

    def train(self, mode: bool = True):
        # Frozen feature statistics must not change with the parent training mode.
        super().train(False)
        self.model.eval()
        return self

    def _core(self):
        return self.model.convnextv2 if hasattr(self.model, "convnextv2") else self.model

    @staticmethod
    def _normalize_channels(feature: Tensor) -> Tensor:
        original_dtype = feature.dtype
        value = feature.float()
        mean = value.mean(dim=1, keepdim=True)
        std = value.std(dim=1, keepdim=True, unbiased=False)
        return ((value - mean) / (std + 1e-3)).to(original_dtype)

    def forward(self, images: Tensor, **_: Any) -> dict[str, Tensor]:
        core = self._core()
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        device_type = images.device.type
        autocast = (
            torch.autocast(device_type=device_type, dtype=torch.bfloat16)
            if self.use_bf16 and device_type in {"cpu", "cuda"}
            else nullcontext()
        )
        with autocast:
            embedded = core.embeddings(images)
            encoder_output = core.encoder(embedded, output_hidden_states=True)
            stages = encoder_output.hidden_states[1:]
            output: dict[str, Tensor] = {}
            for index, feature in enumerate(stages):
                normalized = self._normalize_channels(feature)
                batch, channels, height, width = normalized.shape
                tokens = normalized.permute(0, 2, 3, 1).reshape(
                    batch, height * width, channels
                )
                if index > 0:
                    output[f"convenxt_stage_{index}"] = tokens
                output[f"convenxt_stage_{index}_mean"] = normalized.mean(
                    dim=(2, 3)
                ).unsqueeze(1)
                output[f"convenxt_stage_{index}_std"] = safe_std(
                    tokens, dim=1
                ).unsqueeze(1)
            final = stages[-1]
            output["global_mean"] = core.layernorm(final.mean(dim=(2, 3))).unsqueeze(1)
            final_normalized = self._normalize_channels(final)
            final_tokens = final_normalized.permute(0, 2, 3, 1).reshape(
                final.shape[0], -1, final.shape[1]
            )
            output["global_std"] = safe_std(final_tokens, dim=1).unsqueeze(1)
        return output
