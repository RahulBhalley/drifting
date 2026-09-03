#!/usr/bin/env python3
"""Compare a converted PyTorch MAE artifact with its JAX source."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from safetensors.torch import load_file
import torch

from drifting_common.artifacts import ArtifactManifest
from drifting_jax.checkpointing.initialize import load_init_entry
from drifting_jax.models.mae import MAEResNetJAX
from drifting_torch.models.mae import MAEResNet
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


def _comparison(reference, candidate, policy: ParityPolicy) -> dict:
    result = compare_tensors(reference, candidate, policy)
    return {
        "passed": result.passed,
        "metrics": dict(result.metrics),
        "failures": list(result.failures),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", type=int, default=95)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--force-fp32", action="store_true")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    root = args.artifact.resolve()
    manifest = ArtifactManifest.from_json(root / "manifest.json")
    manifest.verify_files(root)
    if manifest.kind != "mae" or manifest.backend != "torch":
        raise ValueError(f"expected torch/mae artifact, got {manifest.backend}/{manifest.kind}")
    config = dict(manifest.model_config)
    if args.force_fp32:
        config["use_bf16"] = False
    torch_model = MAEResNet(**config)
    state = load_file(root / manifest.files["weights"].path, device="cpu")
    torch_model.load_state_dict(state, strict=True)
    del state
    torch_model.eval()
    jax_params, _ = load_init_entry("mae", manifest.source.identifier)
    jax_model = MAEResNetJAX(**config)

    rng = np.random.default_rng(args.seed)
    channels = config.get("in_channels", 3)
    images_nchw = rng.standard_normal(
        (1, channels, args.resolution, args.resolution), dtype=np.float32
    )
    mask_nchw = (
        rng.uniform(size=(1, 1, args.resolution, args.resolution)) < 0.75
    ).astype(np.float32)
    labels = np.asarray([args.label], dtype=np.int32)
    images_nhwc = images_nchw.transpose(0, 2, 3, 1)
    mask_nhwc = mask_nchw.transpose(0, 2, 3, 1)

    (jax_loss, jax_metrics), captures = jax_model.apply(
        {"params": jax_params},
        jnp.asarray(images_nhwc),
        jnp.asarray(labels),
        mask=jnp.asarray(mask_nhwc),
        lambda_cls=0.25,
        train=False,
        capture_intermediates=lambda module, _: module.name in {"decoder", "fc"},
        mutable=["intermediates"],
    )
    with torch.inference_mode():
        torch_output = torch_model(
            torch.from_numpy(images_nchw),
            torch.from_numpy(labels).long(),
            mask=torch.from_numpy(mask_nchw),
            lambda_cls=0.25,
            train=False,
        )
    jax_reconstruction = captures["intermediates"]["decoder"]["__call__"][0]
    jax_logits = captures["intermediates"]["fc"]["__call__"][0]
    policy = _policy(args.policy)
    comparisons = {
        "loss": _comparison(jax_loss, torch_output.loss.float().numpy(), policy),
        "logits": _comparison(jax_logits, torch_output.logits.float().numpy(), policy),
        "reconstruction": _comparison(
            np.asarray(jax_reconstruction, dtype=np.float32).transpose(0, 3, 1, 2),
            torch_output.reconstruction.float().numpy(),
            policy,
        ),
    }
    for name in ("cls_loss", "recon_loss", "accuracy", "mask_ratio"):
        comparisons[name] = _comparison(
            jax_metrics[name], getattr(torch_output, name).float().numpy(), policy
        )

    jax_activations = jax_model.apply(
        {"params": jax_params},
        jnp.asarray(images_nhwc),
        every_k_block=2,
        method=jax_model.get_activations,
    )
    with torch.inference_mode():
        torch_activations = torch_model.get_activations(
            torch.from_numpy(images_nchw), every_k_block=2
        )
    if set(jax_activations) != set(torch_activations):
        raise ValueError(
            f"activation key mismatch: JAX-only={set(jax_activations) - set(torch_activations)}, "
            f"Torch-only={set(torch_activations) - set(jax_activations)}"
        )
    activation_comparisons = {
        name: _comparison(
            jax_activations[name], torch_activations[name].float().numpy(), policy
        )
        for name in sorted(jax_activations)
    }
    passed = all(item["passed"] for item in comparisons.values()) and all(
        item["passed"] for item in activation_comparisons.values()
    )
    report = {
        "artifact": str(root),
        "source": manifest.source.identifier,
        "precision": "fp32" if args.force_fp32 else (
            "bf16" if config.get("use_bf16") else "fp32"
        ),
        "seed": args.seed,
        "resolution": args.resolution,
        "passed": passed,
        "comparisons": comparisons,
        "activations": activation_comparisons,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
