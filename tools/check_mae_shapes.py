#!/usr/bin/env python3
"""Validate JAX/PyTorch MAE state shapes without allocating model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flax import traverse_util
import jax
import torch
import yaml

from drifting_jax.models.mae import MAEResNetJAX
from drifting_torch.checkpointing import validate_mae_state_shapes
from drifting_torch.models.mae import MAEResNet


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = dict(raw["model"])
    config.setdefault("num_classes", int(raw.get("dataset", {}).get("num_classes", 1000)))

    jax_model = MAEResNetJAX(**config)
    dummy = jax_model.dummy_input()
    variables = jax.eval_shape(
        lambda: jax_model.init(
            {
                "params": jax.random.PRNGKey(0),
                "masking": jax.random.PRNGKey(1),
                "dropout": jax.random.PRNGKey(2),
            },
            **dummy,
        )
    )
    source_shapes = {
        name: tuple(value.shape)
        for name, value in traverse_util.flatten_dict(
            variables["params"], sep="/"
        ).items()
    }
    with torch.device("meta"):
        torch_model = MAEResNet(**config)
    target_shapes = {name: tuple(value.shape) for name, value in torch_model.state_dict().items()}
    validate_mae_state_shapes(source_shapes, target_shapes)
    result = {
        "config": str(args.config.resolve()),
        "tensor_count": len(source_shapes),
        "parameter_count": sum(
            int(value.numel()) for value in torch_model.state_dict().values()
        ),
        "validated": True,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
