"""Dataset metadata that does not depend on tensor frameworks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DatasetContract:
    source: str
    split: Literal["train", "val"]
    resolution: int
    num_classes: int
    channels: int
    layout: Literal["NCHW", "NHWC"]
    value_range: tuple[float, float]

    def __post_init__(self) -> None:
        if self.layout not in {"NCHW", "NHWC"}:
            raise ValueError("layout must identify exactly one channel axis as NCHW or NHWC")
        if self.split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        if not self.source:
            raise ValueError("source must be non-empty")
        if self.resolution <= 0 or self.num_classes <= 0 or self.channels <= 0:
            raise ValueError("resolution, num_classes, and channels must be positive")
        if len(self.value_range) != 2 or self.value_range[0] >= self.value_range[1]:
            raise ValueError("value_range must be an increasing pair")

