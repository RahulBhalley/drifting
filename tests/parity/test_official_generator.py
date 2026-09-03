import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from drifting_common.artifacts import ArtifactManifest


ARTIFACTS = {
    "pixel": ("DRIFTING_OFFICIAL_PIXEL_ARTIFACT", 133_864_704),
    "latent": ("DRIFTING_OFFICIAL_LATENT_ARTIFACT", 132_708_880),
}


def official_artifact(kind: str) -> Path:
    variable, _ = ARTIFACTS[kind]
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"set {variable} to run the official checkpoint parity test")
    path = Path(value).resolve()
    if not path.is_dir():
        pytest.fail(f"official artifact does not exist: {path}")
    return path


@pytest.mark.parametrize("kind", ("pixel", "latent"))
def test_official_artifact_is_complete_and_hash_validated(kind: str):
    artifact = official_artifact(kind)
    manifest = ArtifactManifest.from_json(artifact / "manifest.json")
    manifest.verify_files(artifact)
    assert manifest.kind == "generator"
    assert manifest.backend == "torch"
    assert manifest.conversion.validated
    report = json.loads((artifact / manifest.conversion.report_path).read_text())
    assert report["tensor_count"] == 241
    assert report["parameter_count"] == ARTIFACTS[kind][1]
    assert not report["missing_keys"]
    assert not report["unexpected_keys"]


@pytest.mark.parametrize(
    ("precision_args", "policy_name"),
    [
        (("--force-fp32",), "generator_fp32_cpu_v1.json"),
        ((), "generator_bf16_cpu_v1.json"),
    ],
)
@pytest.mark.parametrize("kind", ("pixel", "latent"))
def test_official_generator_matches_jax(
    tmp_path: Path, precision_args: tuple[str, ...], policy_name: str, kind: str
):
    artifact = official_artifact(kind)
    report = tmp_path / f"{kind}-{policy_name}.report.json"
    policy = Path(__file__).with_name("policies") / policy_name
    command = [
        sys.executable,
        "tools/compare_backends.py",
        "--artifact",
        str(artifact),
        "--labels",
        "95",
        "--seed",
        "0",
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
    assert result["raw"]["passed"]
    assert result["image"]["passed"]
