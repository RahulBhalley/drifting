from __future__ import annotations

from pathlib import Path

import pytest

from drifting_common.config import ConfigError, compose_config


SCIENTIFIC_YAML = """
hsdp_dim: 8
logging:
  project: test-project
  entity: null
  use_wandb: false
  log_every_k: 5
dataset:
  source: fake
  resolution: 16
  use_aug: false
  use_latent: false
  use_cache: false
  num_classes: 10
  batch_size: 8
  eval_batch_size: 8
  kwargs:
    fake_train_size: 32
model:
  input_size: 16
  in_channels: 3
  hidden_size: 64
optimizer:
  lr_schedule:
    learning_rate: 0.0002
    warmup_steps: 1
    lr_schedule: const
    total_steps: 10
  weight_decay: 0.01
  adam_b1: 0.9
  adam_b2: 0.95
train:
  seed: 42
  total_steps: 10
  save_per_step: 2
  eval_per_step: 5
feature:
  use_mae: false
  use_convnext: false
"""

RUNTIME_YAML = """
backend: torch
device: cpu
strategy: single
precision: fp32
compile: false
replicate_size: 1
shard_size: 1
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_compose_config_applies_scientific_runtime_then_cli_precedence(tmp_path: Path) -> None:
    """Catches runtime overlays or CLI overrides being applied out of order."""
    scientific = _write(tmp_path / "scientific.yaml", SCIENTIFIC_YAML)
    runtime = _write(tmp_path / "runtime.yaml", RUNTIME_YAML)

    config = compose_config(
        scientific,
        runtime,
        ("train.total_steps=3", "runtime.precision=bf16", "logging.use_wandb=true"),
    )

    assert config.train.total_steps == 3
    assert config.runtime is not None
    assert config.runtime.backend == "torch"
    assert config.runtime.device == "cpu"
    assert config.runtime.precision == "bf16"
    assert config.logging.use_wandb is True
    assert config.legacy_hsdp_dim == 8
    assert config.model["hidden_size"] == 64
    assert config.feature["use_mae"] is False


def test_compose_config_without_runtime_does_not_choose_backend(tmp_path: Path) -> None:
    """Catches an implicit default backend being introduced."""
    scientific = _write(tmp_path / "scientific.yaml", SCIENTIFIC_YAML)

    config = compose_config(scientific)

    assert config.runtime is None


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("  unexpected_key: 1\n", "unknown train keys"),
        ("  total_steps: 0\n", "train.total_steps"),
    ],
)
def test_compose_config_rejects_invalid_training_config(
    tmp_path: Path,
    replacement: str,
    message: str,
) -> None:
    """Catches misspelled training keys and non-positive step counts."""
    text = SCIENTIFIC_YAML.replace("  total_steps: 10\n  save_per_step", replacement + "  save_per_step", 1)
    scientific = _write(tmp_path / "scientific.yaml", text)

    with pytest.raises(ConfigError, match=message):
        compose_config(scientific)


def test_compose_config_rejects_unknown_runtime_key(tmp_path: Path) -> None:
    """Catches silently ignored machine/runtime settings."""
    scientific = _write(tmp_path / "scientific.yaml", SCIENTIFIC_YAML)
    runtime = _write(tmp_path / "runtime.yaml", RUNTIME_YAML + "mystery: true\n")

    with pytest.raises(ConfigError, match="unknown runtime keys"):
        compose_config(scientific, runtime)


def test_compose_config_rejects_torch_hsdp_with_invalid_mesh(tmp_path: Path) -> None:
    """Catches HSDP runtime meshes whose dimensions cannot describe the world."""
    scientific = _write(tmp_path / "scientific.yaml", SCIENTIFIC_YAML)
    runtime = _write(
        tmp_path / "runtime.yaml",
        RUNTIME_YAML.replace("strategy: single", "strategy: hsdp").replace("shard_size: 1", "shard_size: 0"),
    )

    with pytest.raises(ConfigError, match="shard_size"):
        compose_config(scientific, runtime)

