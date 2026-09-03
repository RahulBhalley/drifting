"""Torchvision execution of the exact released FID Inception parameter set."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.inception import BasicConv2d, inception_v3


def resize_for_inception(images: Tensor) -> Tensor:
    """Match the release's torchscript affine-grid resize and normalization."""
    batch, channels, _, _ = images.shape
    theta = torch.eye(2, 3, dtype=images.dtype, device=images.device)
    # The released torchscript helper calculates half-pixel translations but
    # does not assign them back to theta; its effective matrix is identity.
    grid = F.affine_grid(
        theta.unsqueeze(0).repeat(batch, 1, 1),
        [batch, channels, 299, 299],
        align_corners=False,
    )
    resized = F.grid_sample(
        images, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return (resized - 128) / 128


def _lookup(tree: dict[str, Any], dotted: str) -> dict[str, Any]:
    value: Any = tree
    for part in dotted.split("."):
        value = value[part]
    return value


def _conv_weight(value: Any, expected: tuple[int, ...]) -> Tensor:
    array = np.asarray(value)
    if tuple(array.shape) == expected:
        return torch.from_numpy(array.copy())
    converted = array.transpose(3, 2, 0, 1)
    if tuple(converted.shape) != expected:
        raise ValueError(f"Inception convolution shape {array.shape} cannot map to {expected}")
    return torch.from_numpy(np.array(converted, copy=True))


class ReleasedInception(nn.Module):
    """InceptionV3 with the release's max-pool Mixed_7c and 1008 logits."""

    def __init__(self):
        super().__init__()
        self.model = inception_v3(
            weights=None, aux_logits=False, init_weights=False, num_classes=1008
        )
        self.eval()

    @classmethod
    def from_pickle(cls, path: str | Path) -> "ReleasedInception":
        with Path(path).open("rb") as handle:
            parameters = pickle.load(handle)
        model = cls()
        consumed = 0
        with torch.no_grad():
            for name, module in model.model.named_modules():
                if not isinstance(module, BasicConv2d):
                    continue
                source = _lookup(parameters, name)
                module.conv.weight.copy_(
                    _conv_weight(source["conv"]["kernel"], tuple(module.conv.weight.shape))
                )
                module.bn.weight.copy_(torch.as_tensor(source["bn"]["scale"]))
                module.bn.bias.copy_(torch.as_tensor(source["bn"]["bias"]))
                module.bn.running_mean.copy_(torch.as_tensor(source["bn"]["mean"]))
                module.bn.running_var.copy_(torch.as_tensor(source["bn"]["var"]))
                consumed += 1
            kernel = np.asarray(parameters["fc"]["kernel"])
            if tuple(kernel.shape) == tuple(model.model.fc.weight.shape):
                kernel = kernel.copy()
            else:
                kernel = kernel.T.copy()
            model.model.fc.weight.copy_(torch.from_numpy(kernel))
            model.model.fc.bias.copy_(torch.as_tensor(parameters["fc"]["bias"]))
        if consumed != 94:
            raise ValueError(f"expected 94 Inception convolution blocks, mapped {consumed}")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = resize_for_inception(images.float())
        m = self.model
        x = m.Conv2d_1a_3x3(x)
        x = m.Conv2d_2a_3x3(x)
        x = m.Conv2d_2b_3x3(x)
        x = m.maxpool1(x)
        x = m.Conv2d_3b_1x1(x)
        x = m.Conv2d_4a_3x3(x)
        x = m.maxpool2(x)
        x = self._inception_a(m.Mixed_5b, x)
        x = self._inception_a(m.Mixed_5c, x)
        x = self._inception_a(m.Mixed_5d, x)
        x = m.Mixed_6a(x)
        x = self._inception_c(m.Mixed_6b, x)
        x = self._inception_c(m.Mixed_6c, x)
        x, spatial = self._inception_c(m.Mixed_6d, x, return_spatial=True)
        x = self._inception_c(m.Mixed_6e, x)
        x = m.Mixed_7a(x)
        x = self._inception_e(m.Mixed_7b, x, max_pool=False)
        x = self._inception_e(m.Mixed_7c, x, max_pool=True)
        pooled = x.mean(dim=(2, 3))
        # The released JAX classifier passes ``unbiased=True`` to its Dense
        # wrapper, which disables the serialized bias during evaluation.
        logits = F.linear(pooled, m.fc.weight, None)
        return pooled, spatial, logits

    @staticmethod
    def _average_pool(x: Tensor) -> Tensor:
        # The release divides by the number of real pixels in boundary windows.
        return F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )

    @classmethod
    def _inception_a(cls, block: nn.Module, x: Tensor) -> Tensor:
        branch1x1 = block.branch1x1(x)
        branch5x5 = block.branch5x5_2(block.branch5x5_1(x))
        branch3x3dbl = block.branch3x3dbl_1(x)
        branch3x3dbl = block.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = block.branch3x3dbl_3(branch3x3dbl)
        branch_pool = block.branch_pool(cls._average_pool(x))
        return torch.cat((branch1x1, branch5x5, branch3x3dbl, branch_pool), dim=1)

    @classmethod
    def _inception_c(
        cls, block: nn.Module, x: Tensor, *, return_spatial: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        branch1x1 = block.branch1x1(x)
        spatial = branch1x1[:, :7]
        branch7x7 = block.branch7x7_3(block.branch7x7_2(block.branch7x7_1(x)))
        branch7x7dbl = block.branch7x7dbl_1(x)
        branch7x7dbl = block.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = block.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = block.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = block.branch7x7dbl_5(branch7x7dbl)
        branch_pool = block.branch_pool(cls._average_pool(x))
        output = torch.cat((branch1x1, branch7x7, branch7x7dbl, branch_pool), dim=1)
        return (output, spatial) if return_spatial else output

    @classmethod
    def _inception_e(cls, block: nn.Module, x: Tensor, *, max_pool: bool) -> Tensor:
        branch1x1 = block.branch1x1(x)
        branch3x3 = block.branch3x3_1(x)
        branch3x3 = torch.cat(
            (block.branch3x3_2a(branch3x3), block.branch3x3_2b(branch3x3)), dim=1
        )
        branch3x3dbl = block.branch3x3dbl_2(block.branch3x3dbl_1(x))
        branch3x3dbl = torch.cat(
            (
                block.branch3x3dbl_3a(branch3x3dbl),
                block.branch3x3dbl_3b(branch3x3dbl),
            ),
            dim=1,
        )
        pooled = (
            F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
            if max_pool
            else cls._average_pool(x)
        )
        branch_pool = block.branch_pool(pooled)
        return torch.cat((branch1x1, branch3x3, branch3x3dbl, branch_pool), dim=1)


__all__ = ["ReleasedInception", "resize_for_inception"]
