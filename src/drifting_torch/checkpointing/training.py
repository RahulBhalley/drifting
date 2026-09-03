"""Atomic, exact-resume PyTorch training checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from drifting_torch.distributed.runtime import DistributedContext
from drifting_torch.memory_bank import ClassMemoryBank
from drifting_torch.training.state import GeneratorTrainState


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _plain(getattr(value, field.name)) for field in fields(value)
        }
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def trajectory_hash(config: Any) -> str:
    encoded = json.dumps(
        _plain(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rng_state() -> dict[str, Any]:
    result = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        try:
            result["torch_mps"] = torch.mps.get_rng_state()
        except RuntimeError:
            pass
    return result


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "torch_mps" in state and hasattr(torch.mps, "set_rng_state"):
        torch.mps.set_rng_state(state["torch_mps"])


def save_training_state(
    destination: str | Path,
    state: GeneratorTrainState,
    *,
    sampler,
    banks: tuple[ClassMemoryBank, ClassMemoryBank] | None = None,
    config: Any,
    context: DistributedContext | None = None,
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    context = context or DistributedContext.single()
    payload = {
        "schema_version": 1,
        "completed_steps": state.completed_steps,
        "trajectory_hash": trajectory_hash(config),
        "model": None if context.distributed else state.model.state_dict(),
        "ema": state.ema,
        "optimizer": None if context.distributed else state.optimizer.state_dict(),
        "scaler": None if state.scaler is None else state.scaler.state_dict(),
        "generator_state": state.generator.get_state(),
        "rng": _rng_state(),
        "sampler": sampler.state_dict(),
        "positive_bank": None if banks is None else banks[0].state_dict(),
        "negative_bank": None if banks is None else banks[1].state_dict(),
        "distributed": {
            "strategy": context.strategy,
            "world_size": context.world_size,
            "precision": context.precision,
            "rank": context.rank,
        },
    }
    if context.distributed:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

        staging = path.with_name(f".{path.name}.staging")
        published = path.with_name(f"{path.name}.shards")
        if context.is_main:
            staging.mkdir(parents=False, exist_ok=False)
        context.barrier()
        options = StateDictOptions(full_state_dict=False, cpu_offload=True)
        model_state, optimizer_state = get_state_dict(
            state.model, state.optimizer, options=options
        )
        dcp.save(
            {"model": model_state, "optimizer": optimizer_state},
            checkpoint_id=staging / "dcp",
        )
        shard = staging / f"rank-{context.rank:05d}.pt"
        temporary_shard = staging / f".rank-{context.rank:05d}.tmp-{os.getpid()}"
        torch.save(payload, temporary_shard)
        os.replace(temporary_shard, shard)
        context.barrier()
        if context.is_main:
            expected = [staging / f"rank-{rank:05d}.pt" for rank in range(context.world_size)]
            if not all(item.is_file() and item.stat().st_size > 0 for item in expected):
                raise RuntimeError("not all distributed checkpoint shards were written")
            if not (staging / "dcp" / ".metadata").is_file():
                raise RuntimeError("PyTorch distributed checkpoint metadata was not written")
            os.replace(staging, published)
            manifest = {
                "schema_version": 2,
                "kind": "distributed_training_state",
                "strategy": context.strategy,
                "world_size": context.world_size,
                "precision": context.precision,
                "trajectory_hash": trajectory_hash(config),
                "shards": published.name,
                "dcp": "dcp",
            }
            temporary_manifest = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            temporary_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temporary_manifest, path)
        context.barrier()
        return path
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_training_state(
    source: str | Path,
    state: GeneratorTrainState,
    *,
    sampler,
    banks: tuple[ClassMemoryBank, ClassMemoryBank] | None = None,
    config: Any,
    context: DistributedContext | None = None,
) -> GeneratorTrainState:
    path = Path(source)
    context = context or DistributedContext.single()
    with path.open("rb") as handle:
        prefix = handle.read(1)
    if prefix == b"{":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 2:
            raise ValueError("unsupported distributed training-state schema version")
        if not context.distributed:
            raise ValueError("distributed checkpoint requires a distributed runtime context")
        expected = (context.strategy, context.world_size, context.precision)
        actual = (
            manifest.get("strategy"),
            int(manifest.get("world_size", 0)),
            manifest.get("precision"),
        )
        if actual != expected:
            raise ValueError(
                "distributed checkpoint is incompatible: expected "
                f"strategy/world_size/precision={expected}, found {actual}"
            )
        if manifest["trajectory_hash"] != trajectory_hash(config):
            raise ValueError("training trajectory configuration hash does not match")
        context.barrier()
        shard_path = path.with_name(manifest["shards"]) / f"rank-{context.rank:05d}.pt"
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_state_dict,
            set_state_dict,
        )

        options = StateDictOptions(full_state_dict=False, cpu_offload=True)
        model_state, optimizer_state = get_state_dict(
            state.model, state.optimizer, options=options
        )
        dcp.load(
            {"model": model_state, "optimizer": optimizer_state},
            checkpoint_id=path.with_name(manifest["shards"]) / manifest["dcp"],
        )
        set_state_dict(
            state.model,
            state.optimizer,
            model_state_dict=model_state,
            optim_state_dict=optimizer_state,
            options=options,
        )
        loaded_distributed_state = True
    else:
        if context.distributed:
            raise ValueError("single-process checkpoint cannot resume a distributed run")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        loaded_distributed_state = False
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported training-state schema version")
    if payload["trajectory_hash"] != trajectory_hash(config):
        raise ValueError("training trajectory configuration hash does not match")
    if not loaded_distributed_state:
        state.model.load_state_dict(payload["model"], strict=True)
        state.optimizer.load_state_dict(payload["optimizer"])
    state.ema = {
        name: value.to(next(state.model.parameters()).device)
        for name, value in payload["ema"].items()
    }
    state.completed_steps = int(payload["completed_steps"])
    state.generator.set_state(payload["generator_state"])
    if state.scaler is not None and payload["scaler"] is not None:
        state.scaler.load_state_dict(payload["scaler"])
    sampler.load_state_dict(payload["sampler"])
    if banks is not None:
        if payload["positive_bank"] is None or payload["negative_bank"] is None:
            raise ValueError("checkpoint does not contain generator memory banks")
        banks[0].load_state_dict(payload["positive_bank"])
        banks[1].load_state_dict(payload["negative_bank"])
    _restore_rng(payload["rng"])
    context.barrier()
    return state


__all__ = ["load_training_state", "save_training_state", "trajectory_hash"]
