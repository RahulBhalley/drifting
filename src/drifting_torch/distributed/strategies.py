"""Native model wrappers for DDP, FSDP, and hybrid sharding."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .runtime import DistributedContext


def _mixed_precision(name: str):
    if name == "fp32":
        return None
    from torch.distributed.fsdp import MixedPrecision

    dtype = torch.bfloat16 if name == "bf16" else torch.float16
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def wrap_model(model: nn.Module, context: DistributedContext) -> nn.Module:
    model = model.to(context.device)
    if context.strategy == "single":
        return model
    if context.strategy == "ddp":
        device_ids = [context.local_rank] if context.device.type == "cuda" else None
        return DistributedDataParallel(model, device_ids=device_ids)

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

    shape, names, sharding = strategy_topology(context)
    mesh = init_device_mesh(context.device.type, shape, mesh_dim_names=names)
    return FullyShardedDataParallel(
        model,
        sharding_strategy=sharding,
        mixed_precision=_mixed_precision(context.precision),
        device_id=context.device,
        device_mesh=mesh,
        use_orig_params=True,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    while True:
        if hasattr(model, "module"):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


def strategy_topology(context: DistributedContext):
    from torch.distributed.fsdp import ShardingStrategy

    if context.strategy == "fsdp":
        return (context.world_size,), ("shard",), ShardingStrategy.FULL_SHARD
    if context.strategy == "hsdp":
        return (
            (context.replicate_size, context.shard_size),
            ("replicate", "shard"),
            ShardingStrategy.HYBRID_SHARD,
        )
    raise ValueError(f"strategy topology is only defined for FSDP/HSDP, got {context.strategy!r}")


__all__ = ["strategy_topology", "unwrap_model", "wrap_model"]
