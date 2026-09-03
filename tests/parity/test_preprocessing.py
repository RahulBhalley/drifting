import numpy as np
from PIL import Image
import torch

from drifting_jax.data.datasets import center_crop_arr as jax_center_crop
from drifting_torch.data.transforms import adm_center_crop
from drifting_torch.data.vae import VAECodec


def test_adm_center_crop_pixels_match_jax_path():
    rng = np.random.default_rng(101)
    image = Image.fromarray(rng.integers(0, 256, size=(513, 777, 3), dtype=np.uint8))
    assert np.array_equal(
        np.asarray(adm_center_crop(image, 64)), np.asarray(jax_center_crop(image, 64))
    )


class Distribution:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self, generator=None):
        return self.mean + self.std * torch.randn(
            self.mean.shape, generator=generator, device=self.mean.device, dtype=self.mean.dtype
        )


class Encoded:
    def __init__(self, distribution):
        self.latent_dist = distribution


class Decoded:
    def __init__(self, sample):
        self.sample = sample


class FakeVAE(torch.nn.Module):
    def encode(self, images):
        return Encoded(Distribution(images[:, :2] * 0.5, images[:, 1:3].abs() + 0.1))

    def decode(self, latents):
        return Decoded(torch.cat((latents, latents[:, :1]), dim=1))


def test_vae_explicit_noise_sampling_and_decode_formula():
    codec = VAECodec(FakeVAE(), scaling_factor=0.25)
    images = torch.linspace(-1, 1, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    noise = torch.linspace(-0.5, 0.5, 2 * 2 * 4 * 4).reshape(2, 2, 4, 4)
    expected = (images[:, :2] * 0.5 + (images[:, 1:3].abs() + 0.1) * noise) * 0.25
    latent = codec.encode(images, noise=noise)
    torch.testing.assert_close(latent, expected)
    expected_decode = torch.cat((latent / 0.25, (latent / 0.25)[:, :1]), dim=1).clamp(-1, 1)
    torch.testing.assert_close(codec.decode(latent), expected_decode)
    live_latent = latent.detach().requires_grad_(True)
    codec.decode(live_latent).sum().backward()
    assert live_latent.grad is not None
