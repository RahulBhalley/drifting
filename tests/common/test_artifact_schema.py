from __future__ import annotations

import json
from pathlib import Path

import pytest

from drifting_common.artifacts import (
    ArtifactFile,
    ArtifactManifest,
    ArtifactSource,
    ArtifactValidationError,
    ConversionRecord,
    sha256_file,
)


def test_artifact_manifest_round_trip_and_file_verification(tmp_path: Path) -> None:
    """Catches metadata loss and artifact files changing after publication."""
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"deterministic-weights")
    manifest = ArtifactManifest(
        schema_version=1,
        kind="generator",
        backend="torch",
        model_config={"hidden_size": 64, "use_rope": True},
        step=12,
        ema_decay=0.999,
        files={
            "weights": ArtifactFile(
                path="model.safetensors",
                sha256=sha256_file(weights),
                size_bytes=weights.stat().st_size,
            )
        },
        source=ArtifactSource(backend="jax", identifier="hf://pixel_B_sota", revision="abc123"),
        conversion=ConversionRecord(mapping_version=1, validated=True, report_path="conversion.json"),
    )

    manifest_path = tmp_path / "metadata.json"
    manifest.write(manifest_path)
    restored = ArtifactManifest.from_json(manifest_path)
    restored.verify_files(tmp_path)

    assert restored == manifest
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["kind"] == "generator"

    weights.write_bytes(b"changed")
    with pytest.raises(ArtifactValidationError, match="sha256"):
        restored.verify_files(tmp_path)


def test_artifact_manifest_rejects_parent_path_escape() -> None:
    """Catches artifact manifests reading files outside their directory."""
    with pytest.raises(ArtifactValidationError, match="relative artifact path"):
        ArtifactFile(path="../outside.bin", sha256="0" * 64, size_bytes=1)


def test_artifact_manifest_rejects_unvalidated_conversion() -> None:
    """Catches publication of a conversion that did not pass mapping validation."""
    with pytest.raises(ArtifactValidationError, match="validated"):
        ArtifactManifest(
            schema_version=1,
            kind="mae",
            backend="torch",
            model_config={},
            step=0,
            ema_decay=None,
            files={},
            conversion=ConversionRecord(mapping_version=1, validated=False, report_path="report.json"),
        )

