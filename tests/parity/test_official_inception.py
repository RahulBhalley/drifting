import json
import os
from pathlib import Path

import pytest

from tools.compare_inception_backends import main as compare_main


WEIGHTS_ENV = "DRIFTING_INCEPTION_WEIGHTS"


def official_weights() -> Path:
    value = os.environ.get(WEIGHTS_ENV)
    if not value:
        pytest.skip(f"set {WEIGHTS_ENV} to run official Inception parity")
    path = Path(value).resolve()
    if not path.is_file():
        pytest.fail(f"Inception weights do not exist: {path}")
    return path


def test_official_inception_activations_match_jax(tmp_path: Path):
    report = tmp_path / "inception.json"
    policy = Path(__file__).with_name("policies") / "inception_fp32_cpu_v1.json"
    compare_main([
        "--weights",
        str(official_weights()),
        "--policy",
        str(policy),
        "--report",
        str(report),
    ])
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["passed"]
    assert set(result["comparisons"]) == {"pooled", "spatial", "logits"}
