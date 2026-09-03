"""Released improved precision/recall manifold calculation in NumPy."""

from __future__ import annotations

import numpy as np


def pairwise_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    norm_first = np.sum(np.square(first), axis=1)[:, None]
    norm_second = np.sum(np.square(second), axis=1)[None, :]
    return np.maximum(norm_first - 2 * first.dot(second.T) + norm_second, 0.0)


def manifold_radii(
    features: np.ndarray,
    nhood_sizes: tuple[int, ...],
    *,
    row_batch_size: int = 10_000,
    col_batch_size: int = 10_000,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if len(features) <= max(nhood_sizes):
        raise ValueError("not enough samples for requested neighborhood")
    radii = np.zeros((len(features), len(nhood_sizes)), dtype=np.float64)
    for row_start in range(0, len(features), row_batch_size):
        row_end = min(row_start + row_batch_size, len(features))
        distances = np.zeros((row_end - row_start, len(features)), dtype=np.float64)
        for col_start in range(0, len(features), col_batch_size):
            col_end = min(col_start + col_batch_size, len(features))
            distances[:, col_start:col_end] = pairwise_distances(
                features[row_start:row_end], features[col_start:col_end]
            )
        partitioned = np.partition(
            distances, np.arange(max(nhood_sizes) + 1), axis=1
        )
        radii[row_start:row_end] = partitioned[:, nhood_sizes]
    return radii


def _evaluate_pr(
    real: np.ndarray,
    real_radii: np.ndarray,
    fake: np.ndarray,
    fake_radii: np.ndarray,
    *,
    row_batch_size: int = 10_000,
    col_batch_size: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    real_status = np.zeros((len(real), fake_radii.shape[1]), dtype=bool)
    fake_status = np.zeros((len(fake), real_radii.shape[1]), dtype=bool)
    for row_start in range(0, len(real), row_batch_size):
        row_end = min(row_start + row_batch_size, len(real))
        for col_start in range(0, len(fake), col_batch_size):
            col_end = min(col_start + col_batch_size, len(fake))
            distances = pairwise_distances(
                real[row_start:row_end], fake[col_start:col_end]
            )
            real_status[row_start:row_end] |= np.any(
                distances[..., None] <= fake_radii[col_start:col_end], axis=1
            )
            fake_status[col_start:col_end] |= np.any(
                distances[..., None] <= real_radii[row_start:row_end, None], axis=0
            )
    return fake_status.astype(np.float64).mean(0), real_status.astype(np.float64).mean(0)


def compute_precision_recall(
    features_real, features_fake, k: int | tuple[int, ...] = 3
) -> tuple[float, float]:
    real = np.asarray(features_real, dtype=np.float64)
    fake = np.asarray(features_fake, dtype=np.float64)
    neighborhoods = (k,) if isinstance(k, int) else tuple(k)
    real_radii = manifold_radii(real, neighborhoods)
    fake_radii = manifold_radii(fake, neighborhoods)
    precision, recall = _evaluate_pr(real, real_radii, fake, fake_radii)
    return float(precision[0]), float(recall[0])


__all__ = ["compute_precision_recall", "manifold_radii", "pairwise_distances"]
