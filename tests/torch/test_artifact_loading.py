from pathlib import Path

import pytest

from drifting_common.artifacts import ArtifactValidationError
from drifting_torch.checkpointing import load_torch_generator

from .test_inference import artifact


def test_native_artifact_hash_is_validated_before_model_construction(tmp_path: Path):
    source = artifact(tmp_path)
    weights = source / "model.safetensors"
    with weights.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ArtifactValidationError, match="sha256|size"):
        load_torch_generator(source)


def test_hf_name_resolves_from_explicit_torch_artifact_root(tmp_path: Path, monkeypatch):
    root = tmp_path / "registry"
    source = artifact(root)
    source.rename(root / "named-model")
    monkeypatch.setenv("DRIFTING_TORCH_ARTIFACT_ROOT", str(root))
    loaded = load_torch_generator("hf://named-model")
    assert loaded.manifest.step == 7
