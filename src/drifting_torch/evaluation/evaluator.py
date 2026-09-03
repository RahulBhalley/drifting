"""Generator evaluation with validity-mask-aware distributed aggregation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import time

import numpy as np
import torch
from torch import Tensor

from drifting_common.metrics import MetricResult

from .precision_recall import compute_precision_recall
from .statistics import (
    compute_frechet_distance,
    compute_inception_score,
    compute_statistics,
)


def evaluate_generator(
    batches: Iterable,
    generate_batch: Callable[[object, int], Tensor],
    inception,
    *,
    reference_mu: np.ndarray,
    reference_sigma: np.ndarray,
    num_samples: int,
    reference_features: np.ndarray | None = None,
    compute_is: bool = True,
    compute_pr: bool = False,
    splits: int = 10,
) -> MetricResult:
    started = time.perf_counter()
    samples, masks = [], []
    collected = 0
    for batch_index, batch in enumerate(batches):
        generated = generate_batch(batch, batch_index)
        if generated.ndim != 4:
            raise ValueError("generated samples must have rank four")
        if generated.shape[1] != 3 and generated.shape[-1] == 3:
            generated = generated.permute(0, 3, 1, 2)
        remaining = num_samples - collected
        valid = min(generated.shape[0], remaining)
        samples.append((generated.clamp(0, 1) * 255).to(torch.uint8))
        mask = torch.zeros(generated.shape[0], dtype=torch.bool, device=generated.device)
        mask[:valid] = True
        masks.append(mask)
        collected += valid
        if collected >= num_samples:
            break
    if not samples:
        raise ValueError("evaluation produced no samples")
    images = torch.cat(samples).float()
    valid_mask = torch.cat(masks)
    statistics = compute_statistics(
        images,
        valid_mask,
        inception,
        num_samples=num_samples,
        keep_features=compute_pr,
        keep_logits=compute_is,
    )
    metrics = {
        "fid": compute_frechet_distance(
            reference_mu, statistics.mu, reference_sigma, statistics.sigma
        )
    }
    if compute_is and statistics.logits is not None:
        metrics["isc_mean"], metrics["isc_std"] = compute_inception_score(
            statistics.logits, splits=splits
        )
    if compute_pr:
        if reference_features is None or statistics.features is None:
            raise ValueError("reference_features are required for precision/recall")
        metrics["precision"], metrics["recall"] = compute_precision_recall(
            reference_features, statistics.features, k=3
        )
    metrics["fid_time"] = float(time.perf_counter() - started)
    return MetricResult(metrics=metrics, num_samples=statistics.num_samples, backend="torch")


__all__ = ["evaluate_generator"]
