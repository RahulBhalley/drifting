"""Explicit command-line entry points for the JAX backend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from drifting_common.config import ExperimentConfig, compose_config


def _scientific_mapping(config: ExperimentConfig) -> dict:
    value = {
        "dataset": config.dataset.to_dict(),
        "model": dict(config.model),
        "optimizer": config.optimizer.to_dict(),
        "train": config.train.to_dict(),
        "logging": config.logging.to_dict(),
        "feature": dict(config.feature),
    }
    if config.legacy_hsdp_dim is not None:
        value["hsdp_dim"] = config.legacy_hsdp_dim
    return value


def _configure_jax_platform(config: ExperimentConfig) -> None:
    runtime = config.runtime
    if runtime is None:
        return
    if runtime.backend != "jax":
        raise SystemExit(f"JAX command requires runtime.backend=jax, got {runtime.backend!r}")
    platform = {"tpu": "tpu,cpu", "cpu": "cpu", "auto": ""}.get(runtime.device)
    if platform is None:
        raise SystemExit(f"JAX command does not support runtime.device={runtime.device!r}")
    if platform:
        os.environ.setdefault("JAX_PLATFORMS", platform)


def _train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Drifting model with the JAX backend.")
    parser.add_argument("--kind", choices=("generator", "mae"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    parser.add_argument("--workdir", type=Path, default=Path("runs"))
    return parser


def train_main(argv: Sequence[str] | None = None) -> None:
    args = _train_parser().parse_args(argv)
    config = compose_config(args.config, args.runtime, args.set)
    _configure_jax_platform(config)

    from drifting_jax.runtime import EasyDict, _dict_to_easydict, run_init

    run_init()
    legacy = _dict_to_easydict(_scientific_mapping(config))
    if args.kind == "generator":
        from drifting_jax.training.generator import main_gen

        main_gen(legacy, output_dir=str(args.workdir))
    else:
        from drifting_jax.training.mae import main_mae

        main_mae(legacy, output_dir=str(args.workdir))


def infer_main(argv: Sequence[str] | None = None) -> None:
    from drifting_jax.inference import main

    main(argv)


def cache_main(argv: Sequence[str] | None = None) -> None:
    from drifting_jax.data.latent import main

    main(argv)
