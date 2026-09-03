import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


CACHE_ENV = "DRIFTING_CONVNEXT_CACHE"
EXPECTED_REVISION = "758ff0922dc09136abb55774e7f8b1e1bd0dc344"


def official_cache() -> Path:
    value = os.environ.get(CACHE_ENV)
    if not value:
        pytest.skip(f"set {CACHE_ENV} to run official ConvNeXtV2 parity")
    path = Path(value).resolve()
    if not path.is_dir():
        pytest.fail(f"ConvNeXtV2 cache does not exist: {path}")
    return path


def test_official_convnext_base_activations_match_jax(tmp_path: Path):
    cache = official_cache()
    report = tmp_path / "convnext-base.json"
    policy = Path(__file__).with_name("policies") / "convnext_fp32_cpu_v1.json"
    command = [
        sys.executable,
        "tools/compare_convnext_backends.py",
        "--model",
        "base",
        "--cache-dir",
        str(cache),
        "--local-files-only",
        "--policy",
        str(policy),
        "--report",
        str(report),
    ]
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONPATH"] = "src"
    subprocess.run(command, check=True, env=environment, capture_output=True, text=True)
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["revision"] == EXPECTED_REVISION
    assert result["passed"]
    assert len(result["comparisons"]) == 13
    assert all(item["passed"] for item in result["comparisons"].values())
