"""Validated conversion of JAX generator artifacts into PyTorch artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from safetensors.torch import load_file, save_file

from drifting_common.artifacts import (
    ArtifactFile,
    ArtifactManifest,
    ArtifactSource,
    ConversionRecord,
    sha256_file,
)
from drifting_torch.models.generator import build_generator

from .mapping import ConversionError, map_generator_state


MAPPING_VERSION = 1


@dataclass(frozen=True)
class ConversionReport:
    mapping_version: int
    source: str
    source_sha256: str
    destination: str
    parameter_count: int
    tensor_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["missing_keys"] = list(self.missing_keys)
        value["unexpected_keys"] = list(self.unexpected_keys)
        return value


def resolve_jax_artifact(source: str | Path) -> Path:
    root = Path(source).expanduser().resolve()
    if (root / "metadata.json").is_file() and (root / "ema_params.msgpack").is_file():
        return root
    candidates = sorted(
        path.parent
        for path in root.glob("**/ema_params.msgpack")
        if (path.parent / "metadata.json").is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ConversionError(f"no JAX artifact found under {root}")
    raise ConversionError(f"multiple JAX artifacts found under {root}: {candidates}")


def _flatten_jax_params(path: Path) -> dict[str, np.ndarray]:
    from flax import serialization, traverse_util

    tree = serialization.msgpack_restore(path.read_bytes())
    if isinstance(tree, dict) and "params" in tree:
        tree = tree["params"]
    return {
        str(name): np.asarray(value)
        for name, value in traverse_util.flatten_dict(tree, sep="/").items()
    }


def _file_record(path: Path, *, relative_to: Path) -> ArtifactFile:
    return ArtifactFile(
        path=path.relative_to(relative_to).as_posix(),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def convert_jax_generator(source: str | Path, destination: str | Path) -> ConversionReport:
    """Convert one JAX EMA artifact and publish only after strict reload validation."""
    source_dir = resolve_jax_artifact(source)
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise ConversionError(f"conversion destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    model_config = dict(metadata.get("model_config") or {})
    if not model_config:
        raise ConversionError("source metadata is missing model_config")
    model = build_generator(model_config)
    source_state = _flatten_jax_params(source_dir / "ema_params.msgpack")
    mapped = map_generator_state(source_state, model.state_dict())
    incompatible = model.load_state_dict(mapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ConversionError(
            f"strict mapped load failed: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        weights_path = temporary / "model.safetensors"
        save_file(mapped, weights_path)
        reloaded = load_file(weights_path, device="cpu")
        verification = build_generator(model_config).load_state_dict(reloaded, strict=True)
        if verification.missing_keys or verification.unexpected_keys:
            raise ConversionError("serialized artifact failed strict reload validation")

        report = ConversionReport(
            mapping_version=MAPPING_VERSION,
            source=str(source_dir),
            source_sha256=sha256_file(source_dir / "ema_params.msgpack"),
            destination=str(destination),
            parameter_count=sum(value.numel() for value in mapped.values()),
            tensor_count=len(mapped),
            missing_keys=tuple(verification.missing_keys),
            unexpected_keys=tuple(verification.unexpected_keys),
        )
        report_path = temporary / "conversion_report.json"
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        source_metadata = metadata.get("source", {})
        step = int(source_metadata.get("step_loaded", metadata.get("step", 0)))
        selected_ema = source_metadata.get("ema_selected", metadata.get("ema_decay"))
        ema_decay = float(selected_ema) if selected_ema is not None else None
        manifest = ArtifactManifest(
            schema_version=1,
            kind="generator",
            backend="torch",
            model_config=model_config,
            step=step,
            ema_decay=ema_decay,
            files={
                "weights": _file_record(weights_path, relative_to=temporary),
                "conversion_report": _file_record(report_path, relative_to=temporary),
            },
            source=ArtifactSource(backend="jax", identifier=str(source_dir)),
            conversion=ConversionRecord(
                mapping_version=MAPPING_VERSION,
                validated=True,
                report_path="conversion_report.json",
            ),
        )
        manifest.write(temporary / "manifest.json")
        manifest.verify_files(temporary)
        os.replace(temporary, destination)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
