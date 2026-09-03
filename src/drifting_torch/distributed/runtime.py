"""Owned process-group lifecycle for explicit PyTorch strategies."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

import torch
import torch.distributed as dist


def _mapping(runtime: Any) -> dict[str, Any]:
    if hasattr(runtime, "to_dict"):
        return runtime.to_dict()
    return dict(runtime or {})


@dataclass
class DistributedContext:
    strategy: str
    device: torch.device
    precision: str
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None
    replicate_size: int = 1
    shard_size: int = 1
    owns_process_group: bool = False

    @classmethod
    def single(cls, device: str | torch.device = "cpu", precision: str = "fp32"):
        return cls("single", torch.device(device), precision)

    @staticmethod
    def resolve(runtime: Any) -> dict[str, Any]:
        values = _mapping(runtime)
        values["rank"] = int(values.get("rank", os.environ.get("RANK", 0)))
        values["local_rank"] = int(
            values.get("local_rank", os.environ.get("LOCAL_RANK", values["rank"]))
        )
        values["world_size"] = int(
            values.get("world_size", os.environ.get("WORLD_SIZE", 1))
        )
        return values

    @classmethod
    def initialize(cls, runtime: Any) -> "DistributedContext":
        values = cls.resolve(runtime)
        strategy = str(values.get("strategy", "single"))
        precision = str(values.get("precision", "fp32"))
        if strategy == "single":
            requested = str(values.get("device", "cpu"))
            if requested == "auto":
                requested = "cuda" if torch.cuda.is_available() else (
                    "mps" if torch.backends.mps.is_available() else "cpu"
                )
            return cls.single(requested, precision)

        world_size = int(values["world_size"])
        rank, local_rank = int(values["rank"]), int(values["local_rank"])
        if world_size <= 1:
            raise ValueError(f"strategy={strategy} requires world_size greater than one")
        if not 0 <= rank < world_size or local_rank < 0:
            raise ValueError("distributed rank topology is invalid")
        replicate_size = int(values.get("replicate_size", 1))
        shard_size = int(values.get("shard_size", world_size))
        if strategy == "hsdp" and replicate_size * shard_size != world_size:
            raise ValueError(
                "replicate_size * shard_size must equal world_size for HSDP"
            )
        if strategy not in {"ddp", "fsdp", "hsdp"}:
            raise ValueError(f"unsupported distributed strategy {strategy!r}")

        requested = str(values.get("device", "cuda"))
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(requested, local_rank) if requested == "cuda" else torch.device(requested)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA distributed runtime requested but CUDA is unavailable")
            torch.cuda.set_device(device)
        backend = str(values.get("distributed_backend") or ("nccl" if device.type == "cuda" else "gloo"))
        owns = False
        if not dist.is_initialized():
            kwargs: dict[str, Any] = {
                "backend": backend,
                "rank": rank,
                "world_size": world_size,
            }
            if values.get("init_method"):
                kwargs["init_method"] = str(values["init_method"])
            dist.init_process_group(**kwargs)
            owns = True
        elif dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise ValueError("existing process group does not match requested topology")
        return cls(
            strategy,
            device,
            precision,
            rank,
            local_rank,
            world_size,
            backend,
            replicate_size,
            shard_size,
            owns,
        )

    @property
    def distributed(self) -> bool:
        return self.strategy != "single" and self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def close(self) -> None:
        if self.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
            self.owns_process_group = False

    def __enter__(self) -> "DistributedContext":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
