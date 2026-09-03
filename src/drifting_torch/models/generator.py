"""PyTorch implementation of the Drifting LightningDiT generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .primitives import (
    ComputeLinear,
    RMSNorm,
    TimestepEmbedder,
    apply_rope,
    get_2d_sincos_pos_embed,
    initialize_linear,
    modulate,
    patchify_nchw,
    unpatchify_nchw,
)


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = ComputeLinear(hidden_size, intermediate_size)
        self.w3 = ComputeLinear(hidden_size, intermediate_size)
        self.w2 = ComputeLinear(intermediate_size, hidden_size)
        for layer in (self.w1, self.w3, self.w2):
            initialize_linear(layer)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class StandardMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = ComputeLinear(hidden_size, intermediate_size)
        self.fc2 = ComputeLinear(intermediate_size, hidden_size)
        initialize_linear(self.fc1)
        initialize_linear(self.fc2)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="none"))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        *,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_rmsnorm: bool = False,
        use_rope: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_fp32: bool = True,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.attn_fp32 = attn_fp32
        self.qkv = ComputeLinear(dim, dim * 3, bias=qkv_bias)
        self.proj = ComputeLinear(dim, dim)
        initialize_linear(self.qkv)
        initialize_linear(self.proj)
        if qk_norm:
            norm = RMSNorm if use_rmsnorm else lambda size: nn.LayerNorm(size, eps=1e-6)
            self.q_norm = norm(self.head_dim)
            self.k_norm = norm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        x: Tensor,
        *,
        deterministic: bool = True,
        return_qk: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.use_rope:
            q, k = apply_rope(q, k, dtype=torch.float32 if self.attn_fp32 else x.dtype)
        returned_qk = (q, k) if return_qk else None

        attention_dtype = torch.float32 if self.attn_fp32 else x.dtype
        q = q.to(attention_dtype) * self.head_dim**-0.5
        k = k.to(attention_dtype)
        v = v.to(attention_dtype)
        q, k, v = (value.permute(0, 2, 1, 3) for value in (q, k, v))
        weights = torch.matmul(q, k.transpose(-1, -2)).softmax(dim=-1)
        weights = F.dropout(weights, p=self.attn_drop, training=not deterministic)
        output = torch.matmul(weights, v)
        output = output.permute(0, 2, 1, 3).reshape(batch, tokens, channels).to(x.dtype)
        output = self.proj(output)
        output = F.dropout(output, p=self.proj_drop, training=not deterministic)
        return output, returned_qk


class LightningDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = False,
        use_swiglu: bool = False,
        use_rmsnorm: bool = False,
        use_rope: bool = False,
        attn_fp32: bool = True,
        cond_dim: int | None = None,
    ):
        super().__init__()
        if use_rmsnorm:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
            self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = Attention(
            hidden_size,
            num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            use_rope=use_rope,
            attn_fp32=attn_fp32,
        )
        mlp_hidden = int(hidden_size * mlp_ratio)
        if use_swiglu:
            swiglu_hidden = ((int(2 / 3 * mlp_hidden) + 31) // 32) * 32
            self.mlp = SwiGLUFFN(hidden_size, swiglu_hidden)
        else:
            self.mlp = StandardMLP(hidden_size, mlp_hidden)
        self.ada_ln = ComputeLinear(cond_dim or hidden_size, 6 * hidden_size)
        initialize_linear(self.ada_ln, "zeros")

    def forward(self, x: Tensor, c: Tensor, deterministic: bool = True) -> Tensor:
        chunks = self.ada_ln(F.silu(c.float())).to(x.dtype).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks
        attended, _ = self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            deterministic=deterministic,
        )
        x = x + gate_msa.unsqueeze(1) * attended
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        *,
        use_rmsnorm: bool = False,
        cond_dim: int | None = None,
    ):
        super().__init__()
        self.norm = (
            RMSNorm(hidden_size)
            if use_rmsnorm
            else nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        )
        self.ada_ln = ComputeLinear(cond_dim or hidden_size, 2 * hidden_size)
        self.linear = ComputeLinear(hidden_size, patch_size * patch_size * out_channels)
        initialize_linear(self.ada_ln, "zeros")
        initialize_linear(self.linear, "zeros")

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.ada_ln(F.silu(c.float())).to(x.dtype).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class LightningDiT(nn.Module):
    def __init__(
        self,
        *,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 32,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        out_channels: int = 32,
        use_qknorm: bool = False,
        use_swiglu: bool = False,
        use_rope: bool = False,
        use_rmsnorm: bool = False,
        cond_dim: int | None = None,
        n_cls_tokens: int = 0,
        attn_fp32: bool = True,
        use_remat: bool = False,
        compute_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if input_size % patch_size:
            raise ValueError("input_size must be divisible by patch_size")
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.n_cls_tokens = n_cls_tokens
        self.use_remat = use_remat
        self.compute_dtype = compute_dtype
        grid_size = input_size // patch_size
        self.grid_size = grid_size
        self.patch_embed = ComputeLinear(patch_size * patch_size * in_channels, hidden_size)
        initialize_linear(self.patch_embed)
        position = get_2d_sincos_pos_embed(hidden_size, grid_size)
        self.pos_embed = nn.Parameter(torch.from_numpy(position).float().unsqueeze(0))
        if n_cls_tokens:
            if cond_dim is None:
                raise ValueError("cond_dim is required when class tokens are enabled")
            self.class_token_proj = ComputeLinear(cond_dim, hidden_size)
            initialize_linear(self.class_token_proj)
            self.cls_embed = nn.Parameter(torch.empty(1, n_cls_tokens, hidden_size))
            nn.init.normal_(self.cls_embed, std=0.02)
        else:
            self.class_token_proj = None
            self.register_parameter("cls_embed", None)
        self.blocks = nn.ModuleList(
            [
                LightningDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_swiglu=use_swiglu,
                    use_rmsnorm=use_rmsnorm,
                    use_rope=use_rope,
                    attn_fp32=attn_fp32,
                    cond_dim=cond_dim,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size,
            patch_size,
            out_channels,
            use_rmsnorm=use_rmsnorm,
            cond_dim=cond_dim,
        )

    def forward(self, x: Tensor, c: Tensor, *, deterministic: bool = True) -> Tensor:
        x = patchify_nchw(x, grid_size=self.grid_size)
        if x.shape[-1] != self.patch_embed.in_features:
            raise ValueError(
                "Input resolution changes the effective patch width; this PyTorch "
                "model requires the configured input_size"
            )
        x = self.patch_embed(x.to(self.compute_dtype))
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)
        c = c.to(self.compute_dtype)
        if self.n_cls_tokens:
            class_tokens = self.class_token_proj(c).unsqueeze(1)
            class_tokens = class_tokens.expand(-1, self.n_cls_tokens, -1)
            class_tokens = class_tokens + self.cls_embed.to(class_tokens.dtype)
            x = torch.cat((class_tokens, x), dim=1)
        for block in self.blocks:
            if self.use_remat and torch.is_grad_enabled():
                x = checkpoint(block, x, c, deterministic, use_reentrant=False)
            else:
                x = block(x, c, deterministic)
        x = self.final_layer(x, c)
        if self.n_cls_tokens:
            x = x[:, self.n_cls_tokens :]
        return unpatchify_nchw(
            x,
            grid_size=self.grid_size,
            patch_size=self.patch_size,
            channels=self.out_channels,
        )


@dataclass(frozen=True)
class GenerationNoise:
    x: Tensor
    noise_labels: Tensor


@dataclass(frozen=True)
class GenerationOutput:
    samples: Tensor
    noise: GenerationNoise


class DitGen(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        num_classes: int = 1001,
        noise_classes: int = 0,
        noise_coords: int = 1,
        input_size: int = 32,
        in_channels: int = 3,
        n_cls_tokens: int = 0,
        patch_size: int = 2,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        out_channels: int = 3,
        use_qknorm: bool = False,
        use_swiglu: bool = False,
        use_rope: bool = False,
        use_rmsnorm: bool = False,
        use_bf16: bool = False,
        attn_fp32: bool = True,
        use_remat: bool = False,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.noise_classes = noise_classes
        self.noise_coords = noise_coords
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.compute_dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.class_embed = nn.Embedding(num_classes, cond_dim)
        nn.init.normal_(self.class_embed.weight, std=0.02)
        self.noise_embeds = nn.ModuleList(
            nn.Embedding(noise_classes, cond_dim) for _ in range(noise_coords)
        ) if noise_classes > 0 else nn.ModuleList()
        for embedding in self.noise_embeds:
            nn.init.normal_(embedding.weight, std=0.02)
        self.cfg_embedder = TimestepEmbedder(cond_dim)
        self.cfg_norm = RMSNorm(cond_dim)
        self.model = LightningDiT(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_channels=out_channels,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rope=use_rope,
            use_rmsnorm=use_rmsnorm,
            cond_dim=cond_dim,
            n_cls_tokens=n_cls_tokens,
            attn_fp32=attn_fp32,
            use_remat=use_remat,
            compute_dtype=self.compute_dtype,
        )

    def conditioning(self, labels: Tensor, cfg_scale: float | Tensor, noise_labels: Tensor) -> Tensor:
        cond = self.class_embed(labels).to(self.compute_dtype)
        for index, embedding in enumerate(self.noise_embeds):
            cond = cond + embedding(noise_labels[:, index]).to(self.compute_dtype)
        if isinstance(cfg_scale, (float, int)):
            scale = torch.full(
                (labels.shape[0],), float(cfg_scale), device=labels.device, dtype=torch.float32
            )
        else:
            scale = torch.as_tensor(cfg_scale, device=labels.device)
            if scale.ndim == 0:
                scale = scale.expand(labels.shape[0])
            if scale.shape != labels.shape:
                raise ValueError(f"cfg_scale must be scalar or shape {tuple(labels.shape)}")
        cfg = self.cfg_embedder(scale, dtype=self.compute_dtype)
        cond = cond + self.cfg_norm(cfg) * 0.02
        return cond.to(self.compute_dtype)

    def forward(
        self,
        labels: Tensor,
        cfg_scale: float | Tensor = 1.0,
        temp: float = 1.0,
        deterministic: bool = True,
        *,
        noise: Tensor | None = None,
        noise_labels: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> GenerationOutput:
        if labels.ndim != 1:
            raise ValueError(f"labels must be rank one, got shape {tuple(labels.shape)}")
        batch = labels.shape[0]
        expected_noise = (batch, self.in_channels, self.input_size, self.input_size)
        if noise is None:
            noise = torch.randn(
                expected_noise,
                device=labels.device,
                dtype=torch.float32,
                generator=generator,
            )
        elif tuple(noise.shape) != expected_noise:
            raise ValueError(f"noise must use NCHW shape {expected_noise}, got {tuple(noise.shape)}")
        else:
            noise = noise.to(device=labels.device)
        x = (noise * temp).to(self.compute_dtype)

        expected_labels = (batch, max(1, self.noise_coords))
        if noise_labels is None:
            noise_labels = torch.randint(
                0,
                max(1, self.noise_classes),
                expected_labels,
                device=labels.device,
                generator=generator,
            )
        elif tuple(noise_labels.shape) != expected_labels:
            raise ValueError(
                f"noise_labels must have shape {expected_labels}, got {tuple(noise_labels.shape)}"
            )
        else:
            noise_labels = noise_labels.to(device=labels.device, dtype=torch.long)
        cond = self.conditioning(labels, cfg_scale, noise_labels)
        samples = self.model(x, cond, deterministic=deterministic)
        return GenerationOutput(samples=samples, noise=GenerationNoise(x=x, noise_labels=noise_labels))


def build_generator(model_config: Mapping[str, Any]) -> DitGen:
    return DitGen(**dict(model_config))
