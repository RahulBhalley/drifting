"""Validated scalar metric results."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Literal


@dataclass(frozen=True)
class MetricResult:
    metrics: Mapping[str, float]
    num_samples: int
    backend: Literal["jax", "torch"]

    def __post_init__(self) -> None:
        if self.backend not in {"jax", "torch"}:
            raise ValueError("backend must be jax or torch")
        if isinstance(self.num_samples, bool) or self.num_samples < 0:
            raise ValueError("num_samples must be non-negative")
        for name, value in self.metrics.items():
            if not name:
                raise ValueError("metric names must be non-empty")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"metric {name!r} must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "num_samples": self.num_samples,
            "metrics": {name: float(value) for name, value in self.metrics.items()},
        }

