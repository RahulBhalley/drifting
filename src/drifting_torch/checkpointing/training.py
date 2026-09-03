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
    banks: tuple[ClassMemoryBank, ClassMemoryBank],
    config: Any,
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "completed_steps": state.completed_steps,
        "trajectory_hash": trajectory_hash(config),
        "model": state.model.state_dict(),
        "ema": state.ema,
        "optimizer": state.optimizer.state_dict(),
        "scaler": None if state.scaler is None else state.scaler.state_dict(),
        "generator_state": state.generator.get_state(),
        "rng": _rng_state(),
        "sampler": sampler.state_dict(),
        "positive_bank": banks[0].state_dict(),
        "negative_bank": banks[1].state_dict(),
    }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_training_state(
    source: str | Path,
    state: GeneratorTrainState,
    *,
    sampler,
    banks: tuple[ClassMemoryBank, ClassMemoryBank],
    config: Any,
) -> GeneratorTrainState:
    payload = torch.load(Path(source), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported training-state schema version")
    if payload["trajectory_hash"] != trajectory_hash(config):
        raise ValueError("training trajectory configuration hash does not match")
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
    banks[0].load_state_dict(payload["positive_bank"])
    banks[1].load_state_dict(payload["negative_bank"])
    _restore_rng(payload["rng"])
    return state


__all__ = ["load_training_state", "save_training_state", "trajectory_hash"]
