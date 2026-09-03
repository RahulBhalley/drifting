"""Validity-aware tensor collectives."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.distributed as dist

from .runtime import DistributedContext


def gather_valid(
    tensor: Tensor, valid_mask: Tensor, context: DistributedContext
) -> tuple[Tensor, Tensor]:
    if tensor.shape[0] != valid_mask.shape[0] or valid_mask.ndim != 1:
        raise ValueError("valid_mask must match the tensor's leading dimension")
    mask = valid_mask.to(device=tensor.device, dtype=torch.bool)
    if not context.distributed:
        selected = tensor[mask]
        return selected, torch.ones(len(selected), dtype=torch.bool, device=tensor.device)

    local_size = torch.tensor([len(tensor)], device=tensor.device, dtype=torch.int64)
    sizes = [torch.zeros_like(local_size) for _ in range(context.world_size)]
    dist.all_gather(sizes, local_size)
    max_size = max(int(size.item()) for size in sizes)
    padded_shape = (max_size, *tensor.shape[1:])
    padded = torch.zeros(padded_shape, dtype=tensor.dtype, device=tensor.device)
    padded[: len(tensor)] = tensor
    padded_mask = torch.zeros(max_size, dtype=torch.bool, device=tensor.device)
    padded_mask[: len(mask)] = mask
    values = [torch.empty_like(padded) for _ in range(context.world_size)]
    masks = [torch.empty_like(padded_mask) for _ in range(context.world_size)]
    dist.all_gather(values, padded)
    dist.all_gather(masks, padded_mask)
    merged_values, merged_masks = [], []
    for value, gathered_mask, size in zip(values, masks, sizes):
        length = int(size.item())
        merged_values.append(value[:length])
        merged_masks.append(gathered_mask[:length])
    all_values = torch.cat(merged_values)
    all_masks = torch.cat(merged_masks)
    selected = all_values[all_masks]
    return selected, torch.ones(len(selected), dtype=torch.bool, device=tensor.device)


__all__ = ["gather_valid"]
