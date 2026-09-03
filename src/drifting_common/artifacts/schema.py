"""Artifact manifest schema shared by JAX exports and PyTorch conversions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

from .hashing import sha256_file


class ArtifactValidationError(ValueError):
    """Raised for malformed or corrupted artifact metadata."""


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ArtifactValidationError(f"invalid relative artifact path: {value!r}")


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ArtifactValidationError("sha256 must contain exactly 64 hexadecimal characters")


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _validate_relative_path(self.path)
        _validate_sha256(self.sha256)
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ArtifactValidationError("size_bytes must be non-negative")


@dataclass(frozen=True)
class ArtifactSource:
    backend: Literal["jax", "torch"]
    identifier: str
    revision: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"jax", "torch"}:
            raise ArtifactValidationError("source backend must be jax or torch")
        if not self.identifier:
            raise ArtifactValidationError("source identifier must be non-empty")


@dataclass(frozen=True)
class ConversionRecord:
    mapping_version: int
    validated: bool
    report_path: str

    def __post_init__(self) -> None:
        if isinstance(self.mapping_version, bool) or self.mapping_version <= 0:
            raise ArtifactValidationError("mapping_version must be positive")
        _validate_relative_path(self.report_path)


@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: int
    kind: Literal["generator", "mae", "training_state"]
    backend: Literal["jax", "torch"]
    model_config: Mapping[str, Any]
    step: int
    ema_decay: float | None
    files: Mapping[str, ArtifactFile]
    source: ArtifactSource | None = None
    conversion: ConversionRecord | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ArtifactValidationError(f"unsupported schema_version: {self.schema_version}")
        if self.kind not in {"generator", "mae", "training_state"}:
            raise ArtifactValidationError(f"unsupported artifact kind: {self.kind}")
        if self.backend not in {"jax", "torch"}:
            raise ArtifactValidationError(f"unsupported backend: {self.backend}")
        if isinstance(self.step, bool) or self.step < 0:
            raise ArtifactValidationError("step must be non-negative")
        if self.ema_decay is not None and not 0.0 <= self.ema_decay <= 1.0:
            raise ArtifactValidationError("ema_decay must be in [0, 1]")
        if self.conversion is not None and not self.conversion.validated:
            raise ArtifactValidationError("conversion must be validated before publication")
        try:
            json.dumps(dict(self.model_config), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ArtifactValidationError("model_config must be finite JSON data") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, destination)

    @classmethod
    def from_json(cls, path: str | Path) -> "ArtifactManifest":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactValidationError(f"cannot read artifact manifest: {exc}") from exc
        try:
            return cls(
                schema_version=raw["schema_version"],
                kind=raw["kind"],
                backend=raw["backend"],
                model_config=raw["model_config"],
                step=raw["step"],
                ema_decay=raw.get("ema_decay"),
                files={name: ArtifactFile(**value) for name, value in raw["files"].items()},
                source=ArtifactSource(**raw["source"]) if raw.get("source") is not None else None,
                conversion=ConversionRecord(**raw["conversion"]) if raw.get("conversion") is not None else None,
            )
        except (KeyError, TypeError) as exc:
            raise ArtifactValidationError(f"invalid artifact manifest structure: {exc}") from exc

    def verify_files(self, root: str | Path) -> None:
        artifact_root = Path(root).resolve()
        for name, record in self.files.items():
            path = (artifact_root / record.path).resolve()
            if not path.is_relative_to(artifact_root):
                raise ArtifactValidationError(f"file {name!r} escapes artifact root")
            if not path.is_file():
                raise ArtifactValidationError(f"file {name!r} is missing: {record.path}")
            digest = sha256_file(path)
            if digest != record.sha256:
                raise ArtifactValidationError(f"file {name!r} sha256 mismatch")
            if path.stat().st_size != record.size_bytes:
                raise ArtifactValidationError(f"file {name!r} size mismatch")

