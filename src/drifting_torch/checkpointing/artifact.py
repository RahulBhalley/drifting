"""Load validated native PyTorch artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from drifting_common.artifacts import ArtifactManifest
from drifting_torch.models.generator import DitGen, build_generator
from drifting_torch.models.mae import MAEResNet


@dataclass(frozen=True)
class LoadedGenerator:
    model: DitGen
    manifest: ArtifactManifest
    root: Path


@dataclass(frozen=True)
class LoadedMAE:
    model: MAEResNet
    manifest: ArtifactManifest
    root: Path


def load_torch_generator(
    source: str | Path,
    device: torch.device | str = torch.device("cpu"),
) -> LoadedGenerator:
    root = Path(source).expanduser().resolve()
    manifest = ArtifactManifest.from_json(root / "manifest.json")
    if manifest.kind != "generator" or manifest.backend != "torch":
        raise ValueError(
            f"expected a torch generator artifact, got {manifest.backend}/{manifest.kind}"
        )
    manifest.verify_files(root)
    model = build_generator(manifest.model_config)
    weights_record = manifest.files["weights"]
    state = load_file(root / weights_record.path, device=str(device))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return LoadedGenerator(model=model, manifest=manifest, root=root)


def load_torch_mae(
    source: str | Path,
    device: torch.device | str = torch.device("cpu"),
) -> LoadedMAE:
    root = Path(source).expanduser().resolve()
    manifest = ArtifactManifest.from_json(root / "manifest.json")
    if manifest.kind != "mae" or manifest.backend != "torch":
        raise ValueError(f"expected a torch MAE artifact, got {manifest.backend}/{manifest.kind}")
    manifest.verify_files(root)
    model = MAEResNet(**dict(manifest.model_config))
    weights_record = manifest.files["weights"]
    state = load_file(root / weights_record.path, device=str(device))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return LoadedMAE(model=model, manifest=manifest, root=root)
