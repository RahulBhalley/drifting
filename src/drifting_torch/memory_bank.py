"""Serializable class-wise CPU memory banks for generator training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


class ClassMemoryBank:
    """A deterministic class-wise ring buffer backed by CPU tensors."""

    def __init__(
        self,
        num_classes: int = 1000,
        capacity: int = 64,
        *,
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ):
        if num_classes <= 0 or capacity <= 0:
            raise ValueError("num_classes and capacity must be positive")
        self.num_classes = int(num_classes)
        self.capacity = int(capacity)
        self.dtype = dtype
        self.storage: Tensor | None = None
        self.feature_shape: tuple[int, ...] | None = None
        self.pointer = torch.zeros(self.num_classes, dtype=torch.int64)
        self.count = torch.zeros(self.num_classes, dtype=torch.int64)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))

    @staticmethod
    def _labels(labels: Tensor | Any) -> Tensor:
        result = torch.as_tensor(labels, device="cpu")
        if result.dtype not in _INTEGER_DTYPES:
            raise ValueError("labels must have an integer dtype")
        if result.ndim != 1:
            raise ValueError("labels must have shape (batch,)")
        return result.to(torch.int64)

    def _validate_range(self, labels: Tensor) -> None:
        if labels.numel() and (
            int(labels.min()) < 0 or int(labels.max()) >= self.num_classes
        ):
            raise ValueError(f"labels must be in range [0, {self.num_classes})")

    def add(self, samples: Tensor | Any, labels: Tensor | Any) -> None:
        samples = torch.as_tensor(samples).detach().to(device="cpu", dtype=self.dtype)
        labels = self._labels(labels)
        self._validate_range(labels)
        if samples.ndim < 1 or samples.shape[0] != labels.shape[0]:
            raise ValueError("samples and labels must have the same batch dimension")
        sample_shape = tuple(samples.shape[1:])
        if self.storage is None:
            self.feature_shape = sample_shape
            self.storage = torch.zeros(
                (self.num_classes, self.capacity, *sample_shape), dtype=self.dtype
            )
        elif sample_shape != self.feature_shape:
            raise ValueError(
                f"sample shape {sample_shape} does not match {self.feature_shape}"
            )
        assert self.storage is not None
        for sample, label_tensor in zip(samples, labels):
            label = int(label_tensor)
            index = int(self.pointer[label])
            self.storage[label, index].copy_(sample)
            self.pointer[label] = (index + 1) % self.capacity
            self.count[label] = min(int(self.count[label]) + 1, self.capacity)

    def sample(self, labels: Tensor | Any, n_samples: int) -> Tensor:
        if self.storage is None or self.feature_shape is None:
            raise RuntimeError("memory bank is empty; call add() before sample()")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        labels = self._labels(labels)
        self._validate_range(labels)
        rows = []
        for label_tensor in labels:
            label = int(label_tensor)
            valid = int(self.count[label])
            if valid <= 0:
                raise RuntimeError(f"memory bank class {label} is empty")
            if valid >= n_samples:
                indices = torch.randperm(valid, generator=self.generator)[:n_samples]
            else:
                indices = torch.randint(valid, (n_samples,), generator=self.generator)
            rows.append(self.storage[label, indices])
        return torch.stack(rows, dim=0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "num_classes": self.num_classes,
            "capacity": self.capacity,
            "dtype": str(self.dtype).removeprefix("torch."),
            "feature_shape": self.feature_shape,
            "storage": None if self.storage is None else self.storage.clone(),
            "pointer": self.pointer.clone(),
            "count": self.count.clone(),
            "generator_state": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("unsupported memory-bank schema version")
        if int(state["num_classes"]) != self.num_classes or int(state["capacity"]) != self.capacity:
            raise ValueError("memory-bank dimensions do not match")
        if state["dtype"] != str(self.dtype).removeprefix("torch."):
            raise ValueError("memory-bank dtype does not match")
        self.feature_shape = (
            None if state["feature_shape"] is None else tuple(state["feature_shape"])
        )
        storage = state["storage"]
        self.storage = None if storage is None else storage.detach().cpu().clone()
        self.pointer.copy_(state["pointer"].detach().cpu())
        self.count.copy_(state["count"].detach().cpu())
        self.generator.set_state(state["generator_state"].detach().cpu())

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ClassMemoryBank":
        dtype_name = str(state["dtype"])
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unsupported memory-bank dtype: {dtype_name}")
        bank = cls(
            num_classes=int(state["num_classes"]),
            capacity=int(state["capacity"]),
            dtype=dtype,
        )
        bank.load_state_dict(state)
        return bank

    def state_dict_equal(self, other: "ClassMemoryBank") -> bool:
        left, right = self.state_dict(), other.state_dict()
        if left.keys() != right.keys():
            return False
        for name in left:
            if isinstance(left[name], Tensor):
                if not torch.equal(left[name], right[name]):
                    return False
            elif left[name] != right[name]:
                return False
        return True


__all__ = ["ClassMemoryBank"]
