"""Load validated native PyTorch artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file

from drifting_common.artifacts import ArtifactFile, ArtifactManifest, sha256_file
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


def resolve_torch_artifact(source: str | Path) -> Path:
    value = str(source)
    if not value.startswith("hf://"):
        return Path(value).expanduser().resolve()
    name = value[len("hf://") :].strip()
    roots = [
        Path(os.environ.get("DRIFTING_TORCH_ARTIFACT_ROOT", "artifacts/torch")),
        Path("work/converted"),
    ]
    for root in roots:
        candidate = (root / name).resolve()
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError(
        f"converted PyTorch artifact {name!r} was not found; set "
        "DRIFTING_TORCH_ARTIFACT_ROOT or run tools/convert_checkpoint.py"
    )


def save_torch_generator_artifact(
    destination: str | Path,
    *,
    state_dict: dict[str, torch.Tensor],
    model_config: dict,
    step: int,
    ema_decay: float,
) -> Path:
    """Publish a native generator artifact using safetensors plus a manifest."""
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        weights = temporary / "model.safetensors"
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in state_dict.items()
            },
            weights,
        )
        manifest = ArtifactManifest(
            schema_version=1,
            kind="generator",
            backend="torch",
            model_config=model_config,
            step=int(step),
            ema_decay=float(ema_decay),
            files={
                "weights": ArtifactFile(
                    path=weights.name,
                    sha256=sha256_file(weights),
                    size_bytes=weights.stat().st_size,
                )
            },
        )
        manifest.write(temporary / "manifest.json")
        if target.exists():
            raise FileExistsError(f"artifact destination already exists: {target}")
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def save_torch_mae_artifact(
    destination: str | Path,
    *,
    state_dict: dict[str, torch.Tensor],
    model_config: dict,
    step: int,
    ema_decay: float,
) -> Path:
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        weights = temporary / "model.safetensors"
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in state_dict.items()},
            weights,
        )
        ArtifactManifest(
            schema_version=1,
            kind="mae",
            backend="torch",
            model_config=model_config,
            step=int(step),
            ema_decay=float(ema_decay),
            files={
                "weights": ArtifactFile(
                    path=weights.name,
                    sha256=sha256_file(weights),
                    size_bytes=weights.stat().st_size,
                )
            },
        ).write(temporary / "manifest.json")
        if target.exists():
            raise FileExistsError(f"artifact destination already exists: {target}")
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def load_torch_generator(
    source: str | Path,
    device: torch.device | str = torch.device("cpu"),
) -> LoadedGenerator:
    root = resolve_torch_artifact(source)
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
    root = resolve_torch_artifact(source)
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
