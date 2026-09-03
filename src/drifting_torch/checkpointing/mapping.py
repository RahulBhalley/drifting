"""Exhaustive, fail-closed mappings from Flax parameter trees to PyTorch."""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor


class ConversionError(ValueError):
    """Raised when checkpoint conversion is incomplete or shape-unsafe."""


def convert_leaf(source_name: str, value: np.ndarray, *, target_shape: tuple[int, ...]) -> np.ndarray:
    converted = np.asarray(value)
    if source_name.endswith("/kernel"):
        if converted.ndim == 2:
            converted = converted.T
        elif converted.ndim == 4:
            converted = converted.transpose(3, 2, 0, 1)
        else:
            raise ConversionError(
                f"{source_name}: unsupported kernel rank {converted.ndim}; expected dense or convolution"
            )
    if tuple(converted.shape) != tuple(target_shape):
        raise ConversionError(
            f"{source_name}: expected converted shape {tuple(target_shape)}, got {tuple(converted.shape)}"
        )
    return np.array(converted, copy=True, order="C")


def _linear_source(prefix: str, leaf: str) -> str:
    flax_leaf = "kernel" if leaf == "weight" else leaf
    return f"{prefix}/Dense_0/{flax_leaf}"


def _source_for_generator_target(target: str) -> str | None:
    direct = {
        "class_embed.weight": "Embed_0/embedding",
        "cfg_norm.weight": "RMSNorm_0/weight",
        "model.pos_embed": "LightningDiT_0/pos_embed",
        "model.cls_embed": "LightningDiT_0/cls_embed",
    }
    if target in direct:
        return direct[target]

    match = re.fullmatch(r"noise_embeds\.(\d+)\.weight", target)
    if match:
        return f"noise_embeds_{match.group(1)}/embedding"

    match = re.fullmatch(r"cfg_embedder\.linear_([12])\.(weight|bias)", target)
    if match:
        index = int(match.group(1)) - 1
        return _linear_source(f"TimestepEmbedder_0/TorchLinear_{index}", match.group(2))

    match = re.fullmatch(r"model\.(patch_embed|class_token_proj)\.(weight|bias)", target)
    if match:
        index = 0 if match.group(1) == "patch_embed" else 1
        return _linear_source(f"LightningDiT_0/TorchLinear_{index}", match.group(2))

    match = re.fullmatch(r"model\.blocks\.(\d+)\.norm([12])\.weight", target)
    if match:
        block, norm = match.groups()
        return f"LightningDiT_0/blocks_{block}/RMSNorm_{int(norm) - 1}/weight"

    match = re.fullmatch(
        r"model\.blocks\.(\d+)\.attn\.(qkv|proj)\.(weight|bias)", target
    )
    if match:
        block, layer, leaf = match.groups()
        index = 0 if layer == "qkv" else 1
        return _linear_source(
            f"LightningDiT_0/blocks_{block}/Attention_0/TorchLinear_{index}", leaf
        )

    match = re.fullmatch(
        r"model\.blocks\.(\d+)\.attn\.(q_norm|k_norm)\.(weight|bias)", target
    )
    if match:
        block, norm, leaf = match.groups()
        flax_leaf = "scale" if leaf == "weight" else leaf
        # RMSNorm calls its affine parameter weight; LayerNorm calls it scale.
        rms_candidate = f"LightningDiT_0/blocks_{block}/Attention_0/{norm}/weight"
        layer_candidate = f"LightningDiT_0/blocks_{block}/Attention_0/{norm}/{flax_leaf}"
        return f"{rms_candidate}|{layer_candidate}"

    match = re.fullmatch(
        r"model\.blocks\.(\d+)\.mlp\.(w1|w3|w2|fc1|fc2)\.(weight|bias)", target
    )
    if match:
        block, layer, leaf = match.groups()
        if layer in {"w1", "w3", "w2"}:
            index = {"w1": 0, "w3": 1, "w2": 2}[layer]
            prefix = f"LightningDiT_0/blocks_{block}/SwiGLUFFN_0/TorchLinear_{index}"
        else:
            index = {"fc1": 0, "fc2": 1}[layer]
            prefix = f"LightningDiT_0/blocks_{block}/StandardMLP_0/TorchLinear_{index}"
        return _linear_source(prefix, leaf)

    match = re.fullmatch(r"model\.blocks\.(\d+)\.ada_ln\.(weight|bias)", target)
    if match:
        block, leaf = match.groups()
        return _linear_source(f"LightningDiT_0/blocks_{block}/TorchLinear_0", leaf)

    match = re.fullmatch(r"model\.final_layer\.norm\.weight", target)
    if match:
        return "LightningDiT_0/FinalLayer_0/RMSNorm_0/weight"

    match = re.fullmatch(r"model\.final_layer\.(ada_ln|linear)\.(weight|bias)", target)
    if match:
        layer, leaf = match.groups()
        index = 0 if layer == "ada_ln" else 1
        return _linear_source(f"LightningDiT_0/FinalLayer_0/TorchLinear_{index}", leaf)
    return None


def _choose_source(candidate: str | None, source: Mapping[str, np.ndarray]) -> str | None:
    if candidate is None:
        return None
    choices = candidate.split("|")
    return next((choice for choice in choices if choice in source), choices[0])


def map_generator_state(
    source: Mapping[str, np.ndarray],
    expected_target: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Map a flattened Flax generator tree and reject every incomplete mapping."""
    mapped: dict[str, Tensor] = {}
    consumed: set[str] = set()
    missing_targets: list[str] = []
    for target_name, target_value in expected_target.items():
        source_name = _choose_source(_source_for_generator_target(target_name), source)
        if source_name is None or source_name not in source:
            missing_targets.append(target_name)
            continue
        converted = convert_leaf(
            source_name,
            source[source_name],
            target_shape=tuple(target_value.shape),
        )
        mapped[target_name] = torch.from_numpy(converted).to(dtype=target_value.dtype)
        consumed.add(source_name)

    unused_sources = sorted(set(source) - consumed)
    if unused_sources:
        raise ConversionError("unmapped source parameters: " + ", ".join(unused_sources))
    if missing_targets:
        raise ConversionError("unmapped target parameters: " + ", ".join(sorted(missing_targets)))
    return mapped
