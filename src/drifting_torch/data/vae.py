"""Native Diffusers VAE adapter with explicit posterior-noise replay."""

from __future__ import annotations

import torch
from torch import Tensor, nn


DEFAULT_MODEL_ID = "stabilityai/sd-vae-ft-mse"
DEFAULT_SCALING_FACTOR = 0.18215


class VAECodec(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = None,
        scaling_factor: float = DEFAULT_SCALING_FACTOR,
    ):
        super().__init__()
        self.model = model.eval()
        self.model_id = model_id
        self.revision = revision
        self.scaling_factor = float(scaling_factor)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "VAECodec":
        from diffusers import AutoencoderKL

        model = AutoencoderKL.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(device)
        resolved_revision = revision or getattr(model.config, "_commit_hash", None)
        scaling = float(getattr(model.config, "scaling_factor", DEFAULT_SCALING_FACTOR))
        return cls(
            model,
            model_id=model_id,
            revision=resolved_revision,
            scaling_factor=scaling,
        )

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def encode(
        self,
        images: Tensor,
        noise: Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        distribution = self.model.encode(images).latent_dist
        if noise is None:
            latent = distribution.sample(generator=generator)
        else:
            if noise.shape != distribution.mean.shape:
                raise ValueError(
                    f"posterior noise shape {tuple(noise.shape)} does not match {tuple(distribution.mean.shape)}"
                )
            latent = distribution.mean + distribution.std * noise.to(
                device=distribution.mean.device, dtype=distribution.mean.dtype
            )
        return latent * self.scaling_factor

    def decode(self, latents: Tensor) -> Tensor:
        return self.model.decode(latents / self.scaling_factor).sample.clamp(-1, 1)


__all__ = ["DEFAULT_MODEL_ID", "DEFAULT_SCALING_FACTOR", "VAECodec"]
