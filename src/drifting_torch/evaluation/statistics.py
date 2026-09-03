"""Masked Inception statistics, FID, and Inception Score."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class InceptionStatistics:
    mu: np.ndarray
    sigma: np.ndarray
    features: np.ndarray | None
    logits: np.ndarray | None
    num_samples: int


def _distributed_gather(value: Tensor) -> Tensor:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value
    gathered = [torch.empty_like(value) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, value)
    return torch.cat(gathered, dim=0)


@torch.no_grad()
def compute_statistics(
    samples: Tensor,
    valid_mask: Tensor,
    model,
    *,
    batch_size: int = 200,
    num_samples: int | None = None,
    keep_features: bool = False,
    keep_logits: bool = True,
) -> InceptionStatistics:
    if samples.ndim != 4 or samples.shape[1] != 3:
        raise ValueError("samples must use NCHW RGB layout")
    if valid_mask.shape != (samples.shape[0],):
        raise ValueError("valid_mask must have one entry per sample")
    features, logits = [], []
    device = next(model.parameters()).device
    for start in range(0, len(samples), batch_size):
        pooled, _, batch_logits = model(samples[start : start + batch_size].to(device))
        features.append(pooled.detach())
        if keep_logits:
            logits.append(batch_logits.detach())
    feature_tensor = _distributed_gather(torch.cat(features))
    mask_tensor = _distributed_gather(valid_mask.to(device=device, dtype=torch.bool))
    limit = int(mask_tensor.sum()) if num_samples is None else int(num_samples)
    feature_tensor = feature_tensor[mask_tensor][:limit]
    if feature_tensor.shape[0] < 2:
        raise ValueError("at least two valid samples are required for covariance")
    if not torch.isfinite(feature_tensor).all():
        raise ValueError("Inception features contain non-finite values")
    feature_array = feature_tensor.double().cpu().numpy()
    logits_array = None
    if keep_logits:
        logits_tensor = _distributed_gather(torch.cat(logits))
        logits_array = logits_tensor[mask_tensor][:limit].float().cpu().numpy()
        if not np.isfinite(logits_array).all():
            raise ValueError("Inception logits contain non-finite values")
    return InceptionStatistics(
        mu=np.mean(feature_array, axis=0),
        sigma=np.cov(feature_array, rowvar=False),
        features=feature_tensor.float().cpu().numpy() if keep_features else None,
        logits=logits_array,
        num_samples=feature_array.shape[0],
    )


def compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps: float = 1e-6) -> float:
    mu1, mu2 = np.atleast_1d(mu1).astype(np.float64), np.atleast_1d(mu2).astype(np.float64)
    sigma1 = np.atleast_2d(sigma1).astype(np.float64)
    sigma2 = np.atleast_2d(sigma2).astype(np.float64)
    if mu1.shape != mu2.shape or sigma1.shape != sigma2.shape:
        raise ValueError("Frechet inputs have incompatible shapes")
    product = sigma1.dot(sigma2)
    eigenvalues = np.linalg.eigvals(product).astype("complex128")
    trace_covariance_mean = np.sqrt(eigenvalues).real.sum()
    if not np.isfinite(trace_covariance_mean):
        offset = np.eye(sigma1.shape[0]) * eps
        eigenvalues = np.linalg.eigvals((sigma1 + offset).dot(sigma2 + offset)).astype("complex128")
        trace_covariance_mean = np.sqrt(eigenvalues).real.sum()
    difference = mu1 - mu2
    return float(
        difference.dot(difference)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2 * trace_covariance_mean
    )


def compute_inception_score(logits, splits: int = 10) -> tuple[float, float]:
    logits = np.asarray(logits)
    if splits <= 0 or logits.shape[0] < splits:
        raise ValueError("Inception Score requires at least one sample per split")
    logits = logits[np.random.RandomState(2020).permutation(logits.shape[0])]
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities = (probabilities / probabilities.sum(axis=-1, keepdims=True)).astype(np.float64)
    split_size = probabilities.shape[0] // splits
    probabilities = probabilities[: split_size * splits]
    scores = []
    for index in range(splits):
        part = probabilities[index * split_size : (index + 1) * split_size]
        marginal = np.mean(part, axis=0, keepdims=True)
        divergence = part * (np.log(part + 1e-10) - np.log(marginal + 1e-10))
        scores.append(np.exp(np.mean(np.sum(divergence, axis=1))))
    return float(np.mean(scores)), float(np.std(scores))


__all__ = [
    "InceptionStatistics",
    "compute_frechet_distance",
    "compute_inception_score",
    "compute_statistics",
]
