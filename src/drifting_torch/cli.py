"""Explicit PyTorch command-line entry points."""

from __future__ import annotations

import argparse

from drifting_common.config import compose_config

from .training.engine import train_generator


def train_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train Drifting with native PyTorch")
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args(argv)
    config = compose_config(args.config, args.runtime, args.overrides)
    summary = train_generator(config, config.runtime, args.workdir)
    print(
        f"completed_steps={summary.completed_steps} "
        f"checkpoint={summary.checkpoint} ema_artifact={summary.ema_artifact}"
    )


__all__ = ["train_main"]
