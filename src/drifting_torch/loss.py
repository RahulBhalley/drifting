"""Native PyTorch implementation of the released drifting objective."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def _cdist(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """Pairwise distance in the same operation order as the JAX oracle."""
    xy_dot = torch.einsum("bnd,bmd->bnm", x, y)
    x_norms = torch.einsum("bnd,bnd->bn", x, x)
    y_norms = torch.einsum("bmd,bmd->bm", y, y)
    squared = x_norms[:, :, None] + y_norms[:, None, :] - 2 * xy_dot
    return squared.clamp_min(eps).sqrt()


def _validate(
    gen: Tensor,
    fixed_pos: Tensor,
    fixed_neg: Tensor | None,
    radii: tuple[float, ...],
) -> None:
    tensors = [gen, fixed_pos] + ([] if fixed_neg is None else [fixed_neg])
    if any(value.ndim != 3 for value in tensors):
        raise ValueError("gen, fixed_pos, and fixed_neg must have rank 3")
    if fixed_pos.shape[1] == 0:
        raise ValueError("at least one positive target is required")
    if any(value.shape[0] != gen.shape[0] or value.shape[2] != gen.shape[2] for value in tensors[1:]):
        raise ValueError("all feature tensors must share batch and feature dimensions")
    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError("each radius must be positive")


def drift_loss(
    gen: Tensor,
    fixed_pos: Tensor,
    fixed_neg: Tensor | None = None,
    weight_gen: Tensor | None = None,
    weight_pos: Tensor | None = None,
    weight_neg: Tensor | None = None,
    R_list: Sequence[float] = (0.02, 0.05, 0.2),
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute stopped drifting targets and live generated-feature MSE.

    Inputs use ``(batch, candidates, features)`` layout. As in the released
    implementation, arithmetic is performed in float32 even when callers pass
    lower- or higher-precision tensors.
    """
    radii = tuple(float(radius) for radius in R_list)
    _validate(gen, fixed_pos, fixed_neg, radii)
    batch, generated_count, feature_size = gen.shape
    del batch
    positive_count = fixed_pos.shape[1]
    if fixed_neg is None:
        fixed_neg = gen[:, :0, :].detach()
    negative_count = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])
    expected_weights = (
        ("weight_gen", weight_gen, generated_count),
        ("weight_pos", weight_pos, positive_count),
        ("weight_neg", weight_neg, negative_count),
    )
    for name, weight, count in expected_weights:
        if weight.shape != (gen.shape[0], count):
            raise ValueError(f"{name} has shape {tuple(weight.shape)}, expected {(gen.shape[0], count)}")

    live_gen = gen.float()
    with torch.no_grad():
        old_gen = live_gen.detach()
        positive = fixed_pos.detach().float()
        negative = fixed_neg.detach().float()
        generated_weight = weight_gen.detach().float()
        positive_weight = weight_pos.detach().float()
        negative_weight = weight_neg.detach().float()
        targets = torch.cat((old_gen, negative, positive), dim=1)
        target_weights = torch.cat(
            (generated_weight, negative_weight, positive_weight), dim=1
        )

        distance = _cdist(old_gen, targets)
        weighted_distance = distance * target_weights[:, None, :]
        scale = weighted_distance.mean() / target_weights.mean()
        scale_inputs = (scale / feature_size**0.5).clamp_min(1e-3)
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs
        distance_normalized = distance / scale.clamp_min(1e-3)

        diagonal = torch.eye(
            generated_count, device=gen.device, dtype=torch.float32
        )
        block_mask = torch.nn.functional.pad(
            diagonal, (0, negative_count + positive_count)
        ).unsqueeze(0)
        distance_normalized = distance_normalized + block_mask * 100.0

        force = torch.zeros_like(old_gen_scaled)
        info: dict[str, Tensor] = {"scale": scale}
        split_index = generated_count + negative_count
        for radius in radii:
            logits = -distance_normalized / radius
            affinity = (
                torch.softmax(logits, dim=-1) * torch.softmax(logits, dim=-2)
            ).clamp_min(1e-6).sqrt()
            affinity = affinity * target_weights[:, None, :]
            affinity_negative = affinity[:, :, :split_index]
            affinity_positive = affinity[:, :, split_index:]
            sum_positive = affinity_positive.sum(dim=-1, keepdim=True)
            coefficient_negative = -affinity_negative * sum_positive
            sum_negative = affinity_negative.sum(dim=-1, keepdim=True)
            coefficient_positive = affinity_positive * sum_negative
            coefficient = torch.cat(
                (coefficient_negative, coefficient_positive), dim=2
            )
            force_radius = torch.einsum("biy,byx->bix", coefficient, targets_scaled)
            total_coefficients = coefficient.sum(dim=-1)
            force_radius = force_radius - total_coefficients[..., None] * old_gen_scaled
            force_norm = (force_radius**2).mean()
            info[f"loss_{radius}"] = force_norm
            force_scale = force_norm.clamp_min(1e-8).sqrt()
            force = force + force_radius / force_scale
        goal_scaled = old_gen_scaled + force

    generated_scaled = live_gen / scale_inputs
    loss = ((generated_scaled - goal_scaled) ** 2).mean(dim=(-1, -2))
    return loss, {name: value.mean() for name, value in info.items()}


__all__ = ["drift_loss"]
