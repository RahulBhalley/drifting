"""Backend-transparent mathematical primitives used by the PyTorch models."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2:
        raise ValueError("embed_dim must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", np.asarray(pos).reshape(-1), omega)
    return np.concatenate((np.sin(out), np.cos(out)), axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    if embed_dim % 2:
        raise ValueError("embed_dim must be even")
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)
    half = embed_dim // 2
    emb_h = get_1d_sincos_pos_embed_from_grid(half, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(half, grid[1])
    return np.concatenate((emb_h, emb_w), axis=1)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        variance = x.float().square().mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(input_dtype)
        if self.weight is not None:
            normalized = normalized * self.weight.to(input_dtype)
        return normalized.to(input_dtype)


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: Tensor, k: Tensor, *, dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
    """Apply RoPE to tensors laid out as ``[batch, tokens, heads, head_dim]``."""
    if q.shape != k.shape:
        raise ValueError(f"q and k must have equal shapes, got {q.shape} and {k.shape}")
    head_dim = q.shape[-1]
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    half = head_dim // 2
    frequencies = 1.0 / (10000 ** (torch.arange(half, device=q.device, dtype=dtype) / half))
    positions = torch.arange(q.shape[1], device=q.device, dtype=dtype)
    phase = torch.outer(positions, frequencies)
    phase = torch.cat((phase, phase), dim=-1)[None, :, None, :]
    cosine = phase.cos().to(q.dtype)
    sine = phase.sin().to(q.dtype)
    return q * cosine + _rotate_half(q) * sine, k * cosine + _rotate_half(k) * sine


def patchify_nchw(x: Tensor, *, grid_size: int) -> Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW input, got shape {tuple(x.shape)}")
    batch, channels, height, width = x.shape
    if height != width or height % grid_size:
        raise ValueError(f"Input spatial shape {(height, width)} is incompatible with grid {grid_size}")
    patch = height // grid_size
    x = x.reshape(batch, channels, grid_size, patch, grid_size, patch)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(batch, grid_size * grid_size, patch * patch * channels)


def unpatchify_nchw(
    patches: Tensor,
    *,
    grid_size: int,
    patch_size: int,
    channels: int,
) -> Tensor:
    if patches.ndim != 3:
        raise ValueError(f"Expected BND patches, got shape {tuple(patches.shape)}")
    batch, tokens, width = patches.shape
    if tokens != grid_size * grid_size or width != patch_size * patch_size * channels:
        raise ValueError(
            f"Patch shape {tuple(patches.shape)} does not match grid={grid_size}, "
            f"patch_size={patch_size}, channels={channels}"
        )
    x = patches.reshape(batch, grid_size, grid_size, patch_size, patch_size, channels)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(batch, channels, grid_size * patch_size, grid_size * patch_size)


class ComputeLinear(nn.Linear):
    """Linear layer with FP32 parameters and explicit input-precision compute."""

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(dtype=x.dtype),
            None if self.bias is None else self.bias.to(dtype=x.dtype),
        )


def initialize_linear(linear: nn.Linear, kind: str = "xavier_uniform") -> None:
    if kind == "xavier_uniform":
        nn.init.xavier_uniform_(linear.weight)
    elif kind == "normal":
        nn.init.normal_(linear.weight, std=0.02)
    elif kind == "zeros":
        nn.init.zeros_(linear.weight)
    else:
        raise ValueError(f"Unknown linear initializer: {kind}")
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.linear_1 = ComputeLinear(frequency_embedding_size, hidden_size)
        self.linear_2 = ComputeLinear(hidden_size, hidden_size)
        initialize_linear(self.linear_1, "normal")
        initialize_linear(self.linear_2, "normal")

    def frequency_embedding(self, t: Tensor) -> Tensor:
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float()[:, None] * frequencies[None]
        embedding = torch.cat((args.cos(), args.sin()), dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
        return embedding

    def forward(self, t: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
        embedding = self.frequency_embedding(t)
        if dtype is not None:
            embedding = embedding.to(dtype)
        return self.linear_2(F.silu(self.linear_1(embedding)))
