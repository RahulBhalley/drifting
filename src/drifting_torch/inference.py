"""Reproducible native PyTorch generation and artifact output."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor

from drifting_torch.checkpointing import convert_jax_generator, load_torch_generator
from drifting_torch.data.vae import VAECodec
from drifting_torch.runtime import resolve_device


@dataclass(frozen=True)
class InferenceRequest:
    source: str | Path
    class_ids: tuple[int, ...]
    cfg_scale: float = 1.0
    temperature: float = 1.0
    seed: int = 0
    device: str = "auto"
    precision: str = "artifact"
    output_dir: str | Path | None = None
    noise: Tensor | None = None
    noise_labels: Tensor | None = None
    vae: VAECodec | None = None
    hf_cache_dir: str | Path | None = None
    local_files_only: bool = False


@dataclass(frozen=True)
class InferenceResult:
    raw_samples: Tensor
    images: Tensor
    metadata: dict[str, Any]
    raw_path: Path | None
    metadata_path: Path | None
    image_paths: tuple[Path, ...]


def _load_source(request: InferenceRequest):
    try:
        return load_torch_generator(request.source, device="cpu")
    except (FileNotFoundError, ValueError):
        source = Path(request.source).expanduser()
        if not (
            (source / "metadata.json").is_file()
            and (source / "ema_params.msgpack").is_file()
        ):
            raise
        if request.output_dir is None:
            raise ValueError("output_dir is required when converting a local JAX artifact")
        converted = Path(request.output_dir).resolve() / "converted-artifact"
        if not converted.exists():
            convert_jax_generator(source, converted)
        return load_torch_generator(converted, device="cpu")


def _set_precision(model, precision: str) -> str:
    if precision == "artifact":
        return "bf16" if model.compute_dtype == torch.bfloat16 else "fp32"
    dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    if precision not in dtypes:
        raise ValueError("precision must be artifact, fp32, bf16, or fp16")
    model.compute_dtype = dtypes[precision]
    model.model.compute_dtype = dtypes[precision]
    return precision


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_outputs(
    output: Path, raw: Tensor, images: Tensor, metadata: dict[str, Any]
) -> tuple[Path, Path, tuple[Path, ...]]:
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "samples.pt"
    raw_temporary = raw_path.with_name(f".{raw_path.name}.tmp-{os.getpid()}")
    torch.save(raw.cpu(), raw_temporary)
    os.replace(raw_temporary, raw_path)
    image_paths = []
    arrays = (
        images.detach().cpu().permute(0, 2, 3, 1).numpy().clip(0, 1) * 255
    ).round().astype(np.uint8)
    for index, array in enumerate(arrays):
        path = output / f"sample-{index:04d}-class-{metadata['class_ids'][index]:04d}.png"
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        Image.fromarray(array).save(temporary, format="PNG")
        os.replace(temporary, path)
        image_paths.append(path)
    metadata_path = output / "metadata.json"
    metadata["raw_path"] = str(raw_path)
    metadata["image_paths"] = [str(path) for path in image_paths]
    _atomic_json(metadata, metadata_path)
    return raw_path, metadata_path, tuple(image_paths)


def generate(request: InferenceRequest) -> InferenceResult:
    if not request.class_ids:
        raise ValueError("class_ids must contain at least one label")
    loaded = _load_source(request)
    model = loaded.model
    if any(label < 0 or label >= model.num_classes for label in request.class_ids):
        raise ValueError(f"class ids must be in [0, {model.num_classes})")
    device = resolve_device(request.device)
    precision = _set_precision(model, request.precision)
    if precision == "fp16" and device.type == "cpu":
        raise ValueError("fp16 inference is not supported on CPU")
    model.to(device).eval()
    batch = len(request.class_ids)
    generator = torch.Generator(device="cpu").manual_seed(int(request.seed))
    labels = torch.tensor(request.class_ids, dtype=torch.long, device=device)
    noise = request.noise
    if noise is None:
        noise = torch.randn(
            batch, model.in_channels, model.input_size, model.input_size,
            generator=generator,
        )
    noise_labels = request.noise_labels
    if noise_labels is None:
        noise_labels = torch.randint(
            max(1, model.noise_classes),
            (batch, max(1, model.noise_coords)),
            generator=generator,
        )
    with torch.inference_mode():
        raw = model(
            labels,
            cfg_scale=request.cfg_scale,
            temp=request.temperature,
            noise=noise.to(device),
            noise_labels=noise_labels.to(device),
        ).samples.float()
        if model.out_channels == 4:
            vae = request.vae or VAECodec.from_pretrained(
                cache_dir=None if request.hf_cache_dir is None else str(request.hf_cache_dir),
                local_files_only=request.local_files_only,
                device=device,
            )
            pixels = vae.decode(raw)
        else:
            pixels = raw
        images = ((pixels + 1) / 2).clamp(0, 1)
    weights = loaded.manifest.files["weights"]
    metadata = {
        "artifact": str(loaded.root),
        "artifact_sha256": weights.sha256,
        "artifact_step": loaded.manifest.step,
        "class_ids": list(request.class_ids),
        "cfg_scale": float(request.cfg_scale),
        "temperature": float(request.temperature),
        "seed": int(request.seed),
        "device": str(device),
        "precision": precision,
        "sample_shape": list(raw.shape),
        "image_shape": list(images.shape),
        "sample_finite": bool(torch.isfinite(raw).all()),
        "image_finite": bool(torch.isfinite(images).all()),
        "sample_mean": float(raw.mean()),
        "sample_std": float(raw.std(unbiased=False)),
    }
    raw_path = metadata_path = None
    image_paths: tuple[Path, ...] = ()
    if request.output_dir is not None:
        raw_path, metadata_path, image_paths = _write_outputs(
            Path(request.output_dir).resolve(), raw, images, metadata
        )
    return InferenceResult(
        raw_samples=raw.cpu(),
        images=images.cpu(),
        metadata=metadata,
        raw_path=raw_path,
        metadata_path=metadata_path,
        image_paths=image_paths,
    )


__all__ = ["InferenceRequest", "InferenceResult", "generate"]
