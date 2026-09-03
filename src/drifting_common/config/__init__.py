"""Validated scientific and runtime configuration."""

from .loader import compose_config
from .schema import (
    ConfigError,
    DatasetConfig,
    ExperimentConfig,
    LoggingConfig,
    OptimizerConfig,
    RuntimeConfig,
    TrainingConfig,
)

__all__ = [
    "ConfigError",
    "DatasetConfig",
    "ExperimentConfig",
    "LoggingConfig",
    "OptimizerConfig",
    "RuntimeConfig",
    "TrainingConfig",
    "compose_config",
]

