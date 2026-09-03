#!/usr/bin/env python3
"""Compare native PyTorch ConvNeXtV2 activations with the JAX conversion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from drifting_jax.distributed import set_global_mesh
from drifting_jax.models.convnext import ConvNextV2, convert_weights_to_jax
from drifting_torch.models.convnext import ConvNeXtV2FeatureExtractor, MODEL_IDS
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_IDS), default="base")
    parser.add_argument("--cache-dir", type=Path, default=Path("work/hf-cache"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    torch_extractor = ConvNeXtV2FeatureExtractor.from_pretrained(
        args.model,
        cache_dir=str(args.cache_dir.resolve()),
        local_files_only=args.local_files_only,
    )
    native = torch_extractor.model
    config = native.config
    jax_model = ConvNextV2(
        in_chans=config.num_channels,
        num_classes=config.num_labels,
        depths=tuple(config.depths),
        dims=tuple(config.hidden_sizes),
    )
    variables = jax_model.init(
        jax.random.PRNGKey(0), jnp.ones((1, 224, 224, 3), dtype=jnp.float32)
    )
    variables = convert_weights_to_jax(variables, native.state_dict(), hf=True)
    set_global_mesh(1)
    images = np.random.default_rng(args.seed).normal(size=(1, 3, 64, 64)).astype(
        np.float32
    )
    torch_outputs = torch_extractor(torch.from_numpy(images))
    jax_outputs = jax_model.apply(
        variables,
        jnp.asarray(images.transpose(0, 2, 3, 1)),
        method=jax_model.get_activations,
    )
    policy = MEASUREMENT_POLICY
    if args.policy:
        policy = ParityPolicy.from_mapping(
            json.loads(args.policy.read_text(encoding="utf-8"))
        )
    comparisons = {}
    for name in sorted(jax_outputs):
        result = compare_tensors(
            jax_outputs[name], torch_outputs[name].detach().float().numpy(), policy
        )
        comparisons[name] = {
            "passed": result.passed,
            "metrics": dict(result.metrics),
            "failures": list(result.failures),
        }
    passed = all(value["passed"] for value in comparisons.values())
    report = {
        "model_id": MODEL_IDS[args.model],
        "revision": getattr(config, "_commit_hash", None),
        "seed": args.seed,
        "passed": passed,
        "comparisons": comparisons,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
