"""Load and compose scientific, runtime, and command-line configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .schema import ConfigError, ExperimentConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot load configuration {path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"configuration {path} must contain a mapping")
    return dict(loaded)


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _parse_override(override: str) -> tuple[list[str], Any]:
    if "=" not in override:
        raise ConfigError(f"override must use dotted.path=value syntax: {override!r}")
    dotted, raw_value = override.split("=", 1)
    keys = dotted.split(".")
    if not dotted or any(not key for key in keys):
        raise ConfigError(f"invalid override path: {dotted!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid override value for {dotted}: {exc}") from exc
    return keys, value


def _apply_override(config: dict[str, Any], override: str) -> None:
    keys, value = _parse_override(override)
    cursor = config
    for key in keys[:-1]:
        child = cursor.get(key)
        if child is None:
            child = {}
            cursor[key] = child
        if not isinstance(child, dict):
            raise ConfigError(f"override path crosses non-mapping value at {key!r}")
        cursor = child
    cursor[keys[-1]] = value


def compose_config(
    scientific_path: str | Path,
    runtime_path: str | Path | None = None,
    overrides: Sequence[str] = (),
) -> ExperimentConfig:
    """Compose scientific YAML, optional runtime YAML, then CLI overrides."""
    raw = _read_yaml(Path(scientific_path))
    if runtime_path is not None:
        runtime = _read_yaml(Path(runtime_path))
        raw = _deep_merge(raw, {"runtime": runtime})
    for override in overrides:
        _apply_override(raw, override)
    return ExperimentConfig.from_mapping(raw)

