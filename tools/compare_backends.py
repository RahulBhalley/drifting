#!/usr/bin/env python3
"""Compare a converted PyTorch generator with its JAX source artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch

from drifting_jax.checkpointing.initialize import load_generator_model_and_params
from drifting_jax.distributed import set_global_mesh
from drifting_jax.models.generator import build_generator_from_config as build_jax_generator
from drifting_torch.checkpointing import load_torch_generator
from drifting_torch.models.generator import build_generator as build_torch_generator
from drifting_torch.parity import ParityPolicy, compare_tensors
from safetensors.torch import load_file


MEASUREMENT_POLICY = ParityPolicy(
    max_abs_error=1e9,
    mean_abs_error=1e9,
    nonfinite_count=0,
    cosine_similarity=-1.0,
    psnr=-1e9,
    ssim=-1.0,
    uint8_mismatch_rate=1.0,
)


def _load_policy(path: Path | None) -> ParityPolicy:
    if path is None:
        return MEASUREMENT_POLICY
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ParityPolicy.from_mapping(raw)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--labels", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--force-fp32", action="store_true")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    loaded = load_torch_generator(args.artifact, device="cpu")
    source_dir = loaded.manifest.source.identifier
    jax_model, jax_params, _ = load_generator_model_and_params(source_dir)
    set_global_mesh(1)
    config = dict(loaded.manifest.model_config)
    if args.force_fp32:
        config["use_bf16"] = False
        jax_model = build_jax_generator(config)
        torch_model = build_torch_generator(config)
        weights_path = loaded.root / loaded.manifest.files["weights"].path
        torch_model.load_state_dict(load_file(weights_path, device="cpu"), strict=True)
        torch_model.eval()
    else:
        torch_model = loaded.model
    batch = len(args.labels)
    rng = np.random.default_rng(args.seed)
    noise_nchw = rng.standard_normal(
        (batch, config["in_channels"], config["input_size"], config["input_size"]),
        dtype=np.float32,
    )
    noise_labels = rng.integers(
        0,
        max(1, config.get("noise_classes", 0)),
        size=(batch, max(1, config.get("noise_coords", 1))),
        dtype=np.int32,
    )
    labels = np.asarray(args.labels, dtype=np.int32)
    jax_samples = jax_model.apply(
        {"params": jax_params},
        jnp.asarray(labels),
        cfg_scale=args.cfg_scale,
        noise=jnp.asarray(noise_nchw.transpose(0, 2, 3, 1)),
        noise_labels=jnp.asarray(noise_labels),
    )["samples"]
    with torch.inference_mode():
        torch_samples = torch_model(
            torch.from_numpy(labels).long(),
            cfg_scale=args.cfg_scale,
            noise=torch.from_numpy(noise_nchw),
            noise_labels=torch.from_numpy(noise_labels).long(),
        ).samples
    reference = np.asarray(jax_samples, dtype=np.float32).transpose(0, 3, 1, 2)
    candidate = torch_samples.float().numpy()
    policy = _load_policy(args.policy)
    raw = compare_tensors(reference, candidate, policy)
    image_reference = np.clip((reference + 1) / 2, 0, 1)
    image_candidate = np.clip((candidate + 1) / 2, 0, 1)
    image = compare_tensors(
        image_reference, image_candidate, policy, value_range=(0.0, 1.0)
    )
    report = {
        "artifact": str(args.artifact.resolve()),
        "source": source_dir,
        "labels": args.labels,
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "precision": "fp32" if args.force_fp32 else (
            "bf16" if config.get("use_bf16") else "fp32"
        ),
        "shape": list(reference.shape),
        "jax": {
            "mean": float(reference.mean()),
            "std": float(reference.std()),
            "min": float(reference.min()),
            "max": float(reference.max()),
        },
        "torch": {
            "mean": float(candidate.mean()),
            "std": float(candidate.std()),
            "min": float(candidate.min()),
            "max": float(candidate.max()),
        },
        "raw": {"passed": raw.passed, "metrics": dict(raw.metrics), "failures": raw.failures},
        "image": {
            "passed": image.passed,
            "metrics": dict(image.metrics),
            "failures": image.failures,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not raw.passed or not image.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
