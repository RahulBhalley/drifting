"""Native PyTorch MAE-ResNet feature encoder and U-Net decoder."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .primitives import ComputeLinear


def choose_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups:
        groups -= 1
    return max(groups, 1)


class ComputeConv2d(nn.Conv2d):
    """Convolution with FP32 parameters and explicit input-precision compute."""

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(choose_gn_groups(channels, max_groups), channels, eps=1e-6)


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        *,
        stride: int = 1,
        gn_max_groups: int = 32,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.conv1 = ComputeConv2d(
            in_channels, filters, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.gn1 = group_norm(filters, gn_max_groups)
        self.conv2 = ComputeConv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.gn2 = group_norm(filters, gn_max_groups)
        if stride != 1 or in_channels != filters:
            self.proj_conv = ComputeConv2d(
                in_channels, filters, kernel_size=1, stride=stride, bias=False
            )
            self.proj_gn = group_norm(filters, gn_max_groups)
        else:
            self.proj_conv = None
            self.proj_gn = None

    def forward(self, x: Tensor, *, train: bool) -> Tensor:
        residual = x
        value = F.relu(self.gn1(self.conv1(x)))
        value = F.dropout(value, p=self.dropout_prob, training=train)
        value = self.gn2(self.conv2(value))
        if self.proj_conv is not None:
            residual = self.proj_gn(self.proj_conv(residual))
        return F.relu(residual + value)


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        base_channels: int = 64,
        layers: tuple[int, int, int, int] = (2, 2, 2, 2),
        dropout_prob: float = 0.0,
        gn_max_groups: int = 32,
    ):
        super().__init__()
        self.conv1 = ComputeConv2d(in_channels, base_channels, 3, padding=1, bias=False)
        self.gn1 = group_norm(base_channels, gn_max_groups)
        stages = []
        layer_norms = []
        current_channels = base_channels
        for stage_index, block_count in enumerate(layers):
            out_channels = base_channels * (2**stage_index)
            blocks = []
            for block_index in range(block_count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(
                    BasicBlock(
                        current_channels,
                        out_channels,
                        stride=stride,
                        gn_max_groups=gn_max_groups,
                        dropout_prob=dropout_prob,
                    )
                )
                current_channels = out_channels
            stages.append(nn.ModuleList(blocks))
            layer_norms.append(group_norm(out_channels, gn_max_groups))
        self.stages = nn.ModuleList(stages)
        self.layer_norms = nn.ModuleList(layer_norms)

    def forward(
        self,
        x: Tensor,
        *,
        train: bool,
        return_block_outputs: bool = False,
    ):
        features: dict[str, Tensor] = {}
        block_outputs: dict[str, list[Tensor]] = {}
        x = F.relu(self.gn1(self.conv1(x)))
        features["conv1"] = x
        for index, (stage, norm) in enumerate(zip(self.stages, self.layer_norms), start=1):
            outputs = []
            for block in stage:
                x = block(x, train=train)
                outputs.append(x)
            name = f"layer{index}"
            block_outputs[name] = outputs
            x = norm(x)
            features[name] = x
        if return_block_outputs:
            return features, block_outputs
        return features


class ConvGNReLU(nn.Module):
    def __init__(self, in_channels: int, channels: int, kernel: int = 3):
        super().__init__()
        self.conv = ComputeConv2d(
            in_channels, channels, kernel, padding=kernel // 2, bias=False
        )
        self.gn = group_norm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.gn(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        concatenated = in_channels + skip_channels
        self.concat_norm = group_norm(concatenated)
        self.proj = ConvGNReLU(concatenated, out_channels)
        self.refine = ConvGNReLU(out_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.concat_norm(torch.cat((x, skip), dim=1))
        return self.refine(self.proj(x))


class UNetDecoder(nn.Module):
    def __init__(self, base_channels: int, out_channels: int):
        super().__init__()
        c1, c2 = base_channels, base_channels
        c3, c4, c5 = base_channels * 2, base_channels * 4, base_channels * 8
        self.bridge = ConvGNReLU(c5, c5)
        self.up43 = UpBlock(c5, c4, c4)
        self.up32 = UpBlock(c4, c3, c3)
        self.up21 = UpBlock(c3, c2, c2)
        self.up10 = UpBlock(c2, c1, c1)
        self.head = ComputeConv2d(c1, out_channels, kernel_size=1)

    def forward(self, features: dict[str, Tensor]) -> Tensor:
        x = self.bridge(features["layer4"])
        x = self.up43(x, features["layer3"])
        x = self.up32(x, features["layer2"])
        x = self.up21(x, features["layer1"])
        x = self.up10(x, features["conv1"])
        return self.head(x)


def patch_input(x: Tensor, input_patch_size: int) -> Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW input, got {tuple(x.shape)}")
    batch, channels, height, width = x.shape
    if height % input_patch_size or width % input_patch_size:
        raise ValueError("image dimensions must be divisible by input_patch_size")
    new_height, new_width = height // input_patch_size, width // input_patch_size
    x = x.reshape(
        batch,
        channels,
        new_height,
        input_patch_size,
        new_width,
        input_patch_size,
    )
    return x.permute(0, 3, 5, 1, 2, 4).reshape(
        batch, channels * input_patch_size**2, new_height, new_width
    )


def make_patch_mask(
    x: Tensor,
    mask_ratio: Tensor,
    *,
    patch_size: int = 4,
    generator: torch.Generator | None = None,
) -> Tensor:
    batch, _, height, width = x.shape
    coarse_h, coarse_w = height // patch_size, width // patch_size
    noise = torch.rand(
        (batch, 1, coarse_h, coarse_w),
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    )
    mask = (noise < mask_ratio[:, None, None, None]).to(x.dtype)
    return mask.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)


def safe_std(
    x: Tensor,
    dim: int | tuple[int, ...],
    eps: float = 1e-6,
    keepdim: bool = False,
) -> Tensor:
    value = x.float()
    mean = value.mean(dim=dim, keepdim=True)
    variance = ((value - mean) ** 2).mean(dim=dim, keepdim=keepdim)
    return torch.sqrt(torch.clamp_min(variance, 0.0) + eps)


@dataclass(frozen=True)
class MAEOutput:
    loss: Tensor
    cls_loss: Tensor
    recon_loss: Tensor
    accuracy: Tensor
    mask_ratio: Tensor
    reconstruction: Tensor
    logits: Tensor
    mask: Tensor


class MAEResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        base_channels: int = 64,
        patch_size: int = 4,
        dropout_prob: float = 0.0,
        layers: Iterable[int] = (2, 2, 2, 2),
        use_bf16: bool = False,
        use_fp16: bool = False,
        input_patch_size: int = 1,
    ):
        super().__init__()
        layers = tuple(layers)
        if len(layers) != 4 or any(value < 1 for value in layers):
            raise ValueError("layers must contain four positive block counts")
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.patch_size = patch_size
        self.dropout_prob = dropout_prob
        self.layers = layers
        if use_bf16 and use_fp16:
            raise ValueError("use_bf16 and use_fp16 are mutually exclusive")
        self.use_bf16 = use_bf16
        self.use_fp16 = use_fp16
        self.input_patch_size = input_patch_size
        self.compute_dtype = (
            torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
        )
        patched_channels = in_channels * input_patch_size**2
        self.encoder = ResNetEncoder(
            patched_channels,
            base_channels=base_channels,
            layers=layers,
            dropout_prob=dropout_prob,
        )
        self.decoder = UNetDecoder(base_channels, patched_channels)
        self.fc = ComputeLinear(base_channels * 8, num_classes)

    def _prepare_mask(
        self,
        x: Tensor,
        mask: Tensor | None,
        mask_ratio_min: float,
        mask_ratio_max: float,
        generator: torch.Generator | None,
    ) -> Tensor:
        batch, _, height, width = x.shape
        expected = (batch, 1, height, width)
        if mask is not None:
            if tuple(mask.shape) != expected:
                raise ValueError(f"mask must use NCHW shape {expected}, got {tuple(mask.shape)}")
            return mask.to(device=x.device, dtype=self.compute_dtype)
        ratios = torch.rand(
            (batch,), device=x.device, dtype=self.compute_dtype, generator=generator
        )
        ratios = ratios * (mask_ratio_max - mask_ratio_min) + mask_ratio_min
        return make_patch_mask(x, ratios, patch_size=self.patch_size, generator=generator)

    def forward(
        self,
        images: Tensor,
        labels: Tensor,
        *,
        mask: Tensor | None = None,
        lambda_cls: float = 0.0,
        mask_ratio_min: float = 0.75,
        mask_ratio_max: float = 0.75,
        train: bool = True,
        generator: torch.Generator | None = None,
    ) -> MAEOutput:
        x = patch_input(images.to(self.compute_dtype), self.input_patch_size)
        mask = self._prepare_mask(
            x, mask, mask_ratio_min, mask_ratio_max, generator
        )
        features = self.encoder(x * (1 - mask), train=train)
        pooled = features["layer4"].mean(dim=(2, 3))
        logits = self.fc(pooled)
        reconstruction = self.decoder(features)
        cls_loss = F.cross_entropy(logits.float(), labels, reduction="none").to(self.compute_dtype)
        mse = (reconstruction - x) ** 2
        recon_loss = (mse * mask).sum(dim=(1, 2, 3)) / (
            mask.sum(dim=(1, 2, 3)) + 1e-8
        )
        loss = lambda_cls * cls_loss + (1 - lambda_cls) * recon_loss
        accuracy = (logits.argmax(dim=-1) == labels).to(self.compute_dtype)
        return MAEOutput(
            loss=loss,
            cls_loss=cls_loss,
            recon_loss=recon_loss,
            accuracy=accuracy,
            mask_ratio=mask.mean(dim=(1, 2, 3)),
            reconstruction=reconstruction,
            logits=logits,
            mask=mask,
        )

    def get_activations(
        self,
        images: Tensor,
        *,
        patch_mean_size: list[int] | None = None,
        patch_std_size: list[int] | None = None,
        use_std: bool = True,
        use_mean: bool = True,
        every_k_block: float = 2,
    ) -> dict[str, Tensor]:
        patch_mean_size = [2, 4] if patch_mean_size is None else patch_mean_size
        patch_std_size = [2, 4] if patch_std_size is None else patch_std_size
        x = patch_input(images.to(self.compute_dtype), self.input_patch_size)
        need_blocks = (
            isinstance(every_k_block, (int, float))
            and not math.isinf(float(every_k_block))
            and every_k_block >= 1
        )
        if need_blocks:
            features, block_outputs = self.encoder(
                x, train=False, return_block_outputs=True
            )
        else:
            features = self.encoder(x, train=False)
            block_outputs = {}
        output: dict[str, Tensor] = {
            "norm_x": torch.sqrt((x**2).mean(dim=(2, 3)) + 1e-6).unsqueeze(1)
        }

        def process_feature(name: str, feature: Tensor) -> None:
            batch, channels, height, width = feature.shape
            output[name] = feature.permute(0, 2, 3, 1).reshape(
                batch, height * width, channels
            )
            if use_mean:
                output[f"{name}_mean"] = feature.mean(dim=(2, 3)).unsqueeze(1)
            if use_std:
                output[f"{name}_std"] = safe_std(feature, dim=(2, 3)).unsqueeze(1)
            for size in patch_mean_size:
                if height % size == 0 and width % size == 0:
                    patches = _feature_patches(feature, size)
                    output[f"{name}_mean_{size}"] = patches.mean(dim=2)
            for size in patch_std_size:
                if height % size == 0 and width % size == 0:
                    patches = _feature_patches(feature, size)
                    output[f"{name}_std_{size}"] = safe_std(patches, dim=2)

        for name, feature in features.items():
            process_feature(name, feature)
        if need_blocks:
            step = int(every_k_block)
            for layer_index in range(1, 5):
                layer_name = f"layer{layer_index}"
                for block_index, feature in enumerate(block_outputs[layer_name], start=1):
                    if block_index % step == 0:
                        process_feature(f"{layer_name}_blk{block_index}", feature)
        return output


def _feature_patches(feature: Tensor, size: int) -> Tensor:
    batch, channels, height, width = feature.shape
    blocks_h, blocks_w = height // size, width // size
    value = feature.reshape(batch, channels, blocks_h, size, blocks_w, size)
    value = value.permute(0, 2, 4, 3, 5, 1)
    return value.reshape(batch, blocks_h * blocks_w, size * size, channels)
