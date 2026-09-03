import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from drifting_common.artifacts import ArtifactManifest


ARTIFACT_ENV = "DRIFTING_OFFICIAL_MAE_ARTIFACT"


def official_artifact() -> Path:
    value = os.environ.get(ARTIFACT_ENV)
    if not value:
        pytest.skip(f"set {ARTIFACT_ENV} to run the official MAE parity test")
    path = Path(value).resolve()
    if not path.is_dir():
        pytest.fail(f"official MAE artifact does not exist: {path}")
    return path


def test_official_mae_artifact_is_complete_and_hash_validated():
    artifact = official_artifact()
    manifest = ArtifactManifest.from_json(artifact / "manifest.json")
    manifest.verify_files(artifact)
    assert manifest.kind == "mae"
    assert manifest.backend == "torch"
    assert manifest.conversion.validated
    report = json.loads((artifact / manifest.conversion.report_path).read_text())
    assert report["tensor_count"] == 155
    assert report["parameter_count"] == 431_354_348
    assert not report["missing_keys"]
    assert not report["unexpected_keys"]


@pytest.mark.parametrize(
    ("precision_args", "policy_name"),
    [
        (("--force-fp32",), "mae_fp32_cpu_v1.json"),
        ((), "mae_bf16_cpu_v1.json"),
    ],
)
def test_official_mae_outputs_and_activations_match_jax(
    tmp_path: Path, precision_args: tuple[str, ...], policy_name: str
):
    artifact = official_artifact()
    report = tmp_path / f"{policy_name}.report.json"
    policy = Path(__file__).with_name("policies") / policy_name
    command = [
        sys.executable,
        "tools/compare_mae_backends.py",
        "--artifact",
        str(artifact),
        "--seed",
        "0",
        "--label",
        "95",
        "--resolution",
        "32",
        *precision_args,
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
    assert result["passed"]
    assert len(result["activations"]) == 85
    assert all(item["passed"] for item in result["comparisons"].values())
    assert all(item["passed"] for item in result["activations"].values())
