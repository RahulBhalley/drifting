"""Configuration value objects with strict public-section validation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar


class ConfigError(ValueError):
    """Raised when a scientific or runtime configuration is invalid."""


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    return dict(value)


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{path} must be a positive integer")
    return value


class ConfigSection(Mapping[str, Any]):
    """Read-only-at-the-boundary mapping with convenient attribute access."""

    allowed_keys: ClassVar[frozenset[str]] = frozenset()
    required_keys: ClassVar[frozenset[str]] = frozenset()
    section_name: ClassVar[str] = "section"

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        data = dict(values or {})
        unknown = sorted(set(data) - self.allowed_keys)
        if unknown:
            raise ConfigError(f"unknown {self.section_name} keys: {', '.join(unknown)}")
        missing = sorted(self.required_keys - set(data))
        if missing:
            raise ConfigError(f"missing {self.section_name} keys: {', '.join(missing)}")
        self._values = data
        self._validate()

    def _validate(self) -> None:
        pass

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return dict(self._values)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._values!r})"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self._values == other._values  # type: ignore[attr-defined]


class DatasetConfig(ConfigSection):
    section_name = "dataset"
    allowed_keys = frozenset(
        {
            "source",
            "resolution",
            "use_aug",
            "use_latent",
            "use_cache",
            "num_classes",
            "batch_size",
            "eval_batch_size",
            "kwargs",
        }
    )
    required_keys = frozenset({"resolution", "num_classes", "batch_size", "eval_batch_size"})

    def _validate(self) -> None:
        _positive_int(self._values["resolution"], "dataset.resolution")
        _positive_int(self._values["num_classes"], "dataset.num_classes")
        _positive_int(self._values["batch_size"], "dataset.batch_size")
        _positive_int(self._values["eval_batch_size"], "dataset.eval_batch_size")
        source = self._values.get("source", "imagenet")
        if not isinstance(source, str) or not source.strip():
            raise ConfigError("dataset.source must be a non-empty string")
        for key in ("use_aug", "use_latent", "use_cache"):
            if key in self._values and not isinstance(self._values[key], bool):
                raise ConfigError(f"dataset.{key} must be a boolean")
        _require_mapping(self._values.get("kwargs", {}), "dataset.kwargs")


class LoggingConfig(ConfigSection):
    section_name = "logging"
    allowed_keys = frozenset({"project", "entity", "use_wandb", "log_every_k", "name"})

    def _validate(self) -> None:
        if "use_wandb" in self._values and not isinstance(self._values["use_wandb"], bool):
            raise ConfigError("logging.use_wandb must be a boolean")
        if "log_every_k" in self._values:
            _positive_int(self._values["log_every_k"], "logging.log_every_k")


class OptimizerConfig(ConfigSection):
    section_name = "optimizer"
    allowed_keys = frozenset({"lr_schedule", "weight_decay", "adam_b1", "adam_b2"})
    required_keys = frozenset({"lr_schedule", "adam_b1", "adam_b2"})

    def _validate(self) -> None:
        schedule = _require_mapping(self._values["lr_schedule"], "optimizer.lr_schedule")
        allowed = {"learning_rate", "warmup_steps", "lr_schedule", "total_steps"}
        unknown = sorted(set(schedule) - allowed)
        if unknown:
            raise ConfigError(f"unknown optimizer.lr_schedule keys: {', '.join(unknown)}")
        missing = sorted(allowed - set(schedule))
        if missing:
            raise ConfigError(f"missing optimizer.lr_schedule keys: {', '.join(missing)}")
        if not isinstance(schedule["learning_rate"], (int, float)) or schedule["learning_rate"] <= 0:
            raise ConfigError("optimizer.lr_schedule.learning_rate must be positive")
        if isinstance(schedule["warmup_steps"], bool) or not isinstance(schedule["warmup_steps"], int) or schedule["warmup_steps"] < 0:
            raise ConfigError("optimizer.lr_schedule.warmup_steps must be a non-negative integer")
        _positive_int(schedule["total_steps"], "optimizer.lr_schedule.total_steps")
        if schedule["lr_schedule"] not in {"const", "cos", "cosine"}:
            raise ConfigError("optimizer.lr_schedule.lr_schedule must be const, cos, or cosine")
        for key in ("adam_b1", "adam_b2"):
            value = self._values[key]
            if not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ConfigError(f"optimizer.{key} must be in [0, 1)")


class TrainingConfig(ConfigSection):
    section_name = "train"
    allowed_keys = frozenset(
        {
            "activation_kwargs",
            "cfg_list",
            "ema_decay",
            "enable_eval",
            "eval_forward_dict",
            "eval_per_step",
            "eval_samples",
            "finetune_cls",
            "finetune_last_steps",
            "forward_dict",
            "keep_every",
            "keep_last",
            "loss_kwargs",
            "neg_per_sample",
            "negative_bank_size",
            "pos_per_sample",
            "positive_bank_size",
            "push_at_resume",
            "push_per_step",
            "save_per_step",
            "seed",
            "total_steps",
            "train_batch_size",
            "warmup_finetune",
            "max_grad_norm",
        }
    )
    required_keys = frozenset({"seed", "total_steps", "save_per_step", "eval_per_step"})

    def _validate(self) -> None:
        for key in ("total_steps", "save_per_step", "eval_per_step"):
            _positive_int(self._values[key], f"train.{key}")
        for key in (
            "eval_samples",
            "finetune_last_steps",
            "keep_every",
            "keep_last",
            "neg_per_sample",
            "negative_bank_size",
            "pos_per_sample",
            "positive_bank_size",
            "push_at_resume",
            "push_per_step",
            "train_batch_size",
            "warmup_finetune",
        ):
            if key in self._values:
                value = self._values[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ConfigError(f"train.{key} must be a non-negative integer")
        if "enable_eval" in self._values and not isinstance(self._values["enable_eval"], bool):
            raise ConfigError("train.enable_eval must be a boolean")
        for key in ("activation_kwargs", "eval_forward_dict", "forward_dict", "loss_kwargs"):
            if key in self._values:
                _require_mapping(self._values[key], f"train.{key}")


class RuntimeConfig(ConfigSection):
    section_name = "runtime"
    allowed_keys = frozenset(
        {
            "backend",
            "device",
            "strategy",
            "precision",
            "compile",
            "replicate_size",
            "shard_size",
            "distributed_backend",
            "num_workers",
            "pin_memory",
        }
    )
    required_keys = frozenset({"backend", "device", "strategy", "precision"})

    def _validate(self) -> None:
        if self._values["backend"] not in {"jax", "torch"}:
            raise ConfigError("runtime.backend must be jax or torch")
        if self._values["device"] not in {"auto", "cpu", "mps", "cuda", "tpu"}:
            raise ConfigError("runtime.device must be auto, cpu, mps, cuda, or tpu")
        if self._values["strategy"] not in {"single", "ddp", "fsdp", "hsdp"}:
            raise ConfigError("runtime.strategy must be single, ddp, fsdp, or hsdp")
        if self._values["precision"] not in {"fp32", "bf16", "fp16"}:
            raise ConfigError("runtime.precision must be fp32, bf16, or fp16")
        if "compile" in self._values and not isinstance(self._values["compile"], bool):
            raise ConfigError("runtime.compile must be a boolean")
        for key in ("replicate_size", "shard_size"):
            if key in self._values:
                _positive_int(self._values[key], f"runtime.{key}")
        if self._values["strategy"] == "hsdp":
            for key in ("replicate_size", "shard_size"):
                if key not in self._values:
                    raise ConfigError(f"runtime.{key} is required for hsdp")


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    model: Mapping[str, Any]
    optimizer: OptimizerConfig
    train: TrainingConfig
    logging: LoggingConfig
    feature: Mapping[str, Any]
    runtime: RuntimeConfig | None = None
    legacy_hsdp_dim: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        raw = dict(value)
        allowed = {"dataset", "model", "optimizer", "train", "logging", "feature", "runtime", "hsdp_dim"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ConfigError(f"unknown top-level keys: {', '.join(unknown)}")
        required = {"dataset", "model", "optimizer", "train"}
        missing = sorted(required - set(raw))
        if missing:
            raise ConfigError(f"missing top-level keys: {', '.join(missing)}")
        model = _require_mapping(raw["model"], "model")
        feature = _require_mapping(raw.get("feature", {}), "feature")
        legacy_hsdp_dim = raw.get("hsdp_dim")
        if legacy_hsdp_dim is not None:
            legacy_hsdp_dim = _positive_int(legacy_hsdp_dim, "hsdp_dim")
        runtime_raw = raw.get("runtime")
        return cls(
            dataset=DatasetConfig(_require_mapping(raw["dataset"], "dataset")),
            model=model,
            optimizer=OptimizerConfig(_require_mapping(raw["optimizer"], "optimizer")),
            train=TrainingConfig(_require_mapping(raw["train"], "train")),
            logging=LoggingConfig(_require_mapping(raw.get("logging", {}), "logging")),
            feature=feature,
            runtime=RuntimeConfig(_require_mapping(runtime_raw, "runtime")) if runtime_raw is not None else None,
            legacy_hsdp_dim=legacy_hsdp_dim,
        )

