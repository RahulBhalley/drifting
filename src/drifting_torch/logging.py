"""Rank-gated local JSONL logging used by the PyTorch trainers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if is_rank_zero():
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        if not is_rank_zero():
            return
        values = {
            name: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else value
            for name, value in metrics.items()
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": int(step), **values}, sort_keys=True) + "\n")


__all__ = ["JsonlLogger", "is_rank_zero"]
