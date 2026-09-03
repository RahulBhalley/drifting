from __future__ import annotations

import math

import pytest

from drifting_common.data_contracts import DatasetContract
from drifting_common.metrics import MetricResult


def test_dataset_contract_rejects_layout_channel_mismatch() -> None:
    """Catches tensor contracts with no channel axis in the declared layout."""
    with pytest.raises(ValueError, match="channel axis"):
        DatasetContract(
            source="fake",
            split="train",
            resolution=16,
            num_classes=10,
            channels=3,
            layout="NHW",
            value_range=(-1.0, 1.0),
        )


def test_metric_result_rejects_non_finite_values() -> None:
    """Catches NaN metrics being serialized as successful evaluation output."""
    with pytest.raises(ValueError, match="finite"):
        MetricResult(metrics={"fid": math.nan}, num_samples=16, backend="torch")


def test_metric_result_serializes_plain_json_values() -> None:
    """Catches dataclass or mapping-proxy objects leaking into JSON output."""
    result = MetricResult(metrics={"fid": 1.25}, num_samples=16, backend="jax")

    assert result.to_dict() == {
        "backend": "jax",
        "num_samples": 16,
        "metrics": {"fid": 1.25},
    }

