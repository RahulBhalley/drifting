"""Explicit PyTorch command-line entry points."""

from __future__ import annotations

import argparse
import json

from drifting_common.config import compose_config

from .training.engine import train_generator
from .training.mae import train_mae
from .inference import InferenceRequest, generate


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


def mae_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the Drifting MAE with native PyTorch")
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args(argv)
    config = compose_config(args.config, args.runtime, args.overrides)
    summary = train_mae(config, config.runtime, args.workdir)
    print(
        f"completed_steps={summary.completed_steps} "
        f"checkpoint={summary.checkpoint} ema_artifact={summary.ema_artifact}"
    )


def infer_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate samples with native PyTorch")
    parser.add_argument("--source", required=True)
    parser.add_argument("--class-ids", type=int, nargs="+", required=True)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("artifact", "fp32", "bf16", "fp16"), default="artifact")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    result = generate(
        InferenceRequest(
            source=args.source,
            class_ids=tuple(args.class_ids),
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            seed=args.seed,
            device=args.device,
            precision=args.precision,
            output_dir=args.output_dir,
            hf_cache_dir=args.hf_cache_dir,
            local_files_only=args.local_files_only,
        )
    )
    print(json.dumps(result.metadata, indent=2, sort_keys=True))


__all__ = ["infer_main", "mae_main", "train_main"]
