#!/usr/bin/env python3
"""Convert a Drifting JAX artifact to a validated PyTorch artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from drifting_jax.env import HF_REPO_ID
from drifting_torch.checkpointing.converter import (
    ConversionError,
    convert_jax_generator,
    convert_jax_mae,
    resolve_jax_artifact,
)


def _materialize_source(source: str, cache: Path, kind: str) -> Path:
    if not source.startswith("hf://"):
        return Path(source)
    model_id = source.removeprefix("hf://").strip()
    if not model_id:
        raise ConversionError("expected hf://<model-id>")
    artifact_kind = "gen" if kind == "generator" else "mae"
    local_root = cache / "models" / artifact_kind / "jax" / model_id
    try:
        return resolve_jax_artifact(local_root)
    except ConversionError:
        from drifting_jax.models.hf import _download_artifact

        return _download_artifact(
            repo_id=HF_REPO_ID,
            kind=artifact_kind,
            backend="jax",
            model_id=model_id,
            output_root=str(cache),
            prefix=None,
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("generator", "mae"), default="generator")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=Path(os.environ.get("HF_ROOT", "artifacts/hf-cache")),
    )
    args = parser.parse_args(argv)
    source = _materialize_source(args.source, args.hf_cache.resolve(), args.kind)
    converter = convert_jax_generator if args.kind == "generator" else convert_jax_mae
    report = converter(source, args.output)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
