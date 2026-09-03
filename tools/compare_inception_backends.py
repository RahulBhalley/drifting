#!/usr/bin/env python3
"""Compare the released Inception weights through JAX and PyTorch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle

import numpy as np
import torch

from drifting_jax.evaluation import inception as jax_inception
from drifting_jax.evaluation import weights as jax_weights
from drifting_torch.evaluation.inception import ReleasedInception, resize_for_inception
from drifting_torch.parity import ParityPolicy, compare_tensors


MEASUREMENT_POLICY = ParityPolicy(
    max_abs_error=1e9,
    mean_abs_error=1e9,
    nonfinite_count=0,
    cosine_similarity=-1.0,
    psnr=-1e9,
    ssim=-1.0,
    uint8_mismatch_rate=1.0,
)


def _policy(path: Path | None) -> ParityPolicy:
    if path is None:
        return MEASUREMENT_POLICY
    return ParityPolicy.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    with args.weights.open("rb") as handle:
        serialized = pickle.load(handle)
    # Keep the oracle fully offline and guarantee both backends consume the
    # exact same deserialized official parameter tree.
    jax_weights.download = lambda *_args, **_kwargs: serialized
    variables = jax_weights.load_all()
    jax_model = jax_inception.InceptionV3(
        pretrained=True, include_head=True, transform_input=False
    )
    torch_model = ReleasedInception.from_pickle(args.weights)

    rng = np.random.default_rng(args.seed)
    images = rng.integers(
        0,
        256,
        size=(args.batch_size, 3, args.image_size, args.image_size),
        dtype=np.uint8,
    ).astype(np.float32)
    with torch.inference_mode():
        torch_pooled, torch_spatial, torch_logits = torch_model(torch.from_numpy(images))
    resized = resize_for_inception(torch.from_numpy(images)).numpy().transpose(0, 2, 3, 1)
    jax_pooled, jax_spatial, jax_logits = jax_model.apply(variables, resized, train=False)

    policy = _policy(args.policy)
    pairs = {
        "pooled": (np.asarray(jax_pooled), torch_pooled.numpy(), (0.0, 16.0)),
        "spatial": (
            np.asarray(jax_spatial),
            torch_spatial.numpy().transpose(0, 2, 3, 1),
            (0.0, 16.0),
        ),
        "logits": (np.asarray(jax_logits), torch_logits.numpy(), (-32.0, 32.0)),
    }
    comparisons = {}
    for name, (reference, candidate, value_range) in pairs.items():
        result = compare_tensors(reference, candidate, policy, value_range=value_range)
        comparisons[name] = {
            "shape": list(reference.shape),
            "passed": result.passed,
            "metrics": dict(result.metrics),
            "failures": result.failures,
        }
    report = {
        "weights": str(args.weights.resolve()),
        "seed": args.seed,
        "image_shape": list(images.shape),
        "comparisons": comparisons,
        "passed": all(item["passed"] for item in comparisons.values()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
