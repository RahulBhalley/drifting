"""Fail-closed numerical comparison utilities for backend parity."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ParityPolicy:
    max_abs_error: float
    mean_abs_error: float
    nonfinite_count: int
    cosine_similarity: float
    psnr: float
    ssim: float
    uint8_mismatch_rate: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, float | int]) -> "ParityPolicy":
        required = {field for field in cls.__dataclass_fields__}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError("parity policy is missing metrics: " + ", ".join(missing))
        policy = cls(**{name: value[name] for name in required})
        for name, threshold in policy.__dict__.items():
            if isinstance(threshold, bool) or not math.isfinite(float(threshold)):
                raise ValueError(f"parity threshold {name!r} must be finite")
        return policy


@dataclass(frozen=True)
class TensorComparison:
    metrics: Mapping[str, float | int]
    passed: bool
    failures: tuple[str, ...]


def _global_ssim(reference: np.ndarray, candidate: np.ndarray, data_range: float) -> float:
    reference = reference.astype(np.float64, copy=False)
    candidate = candidate.astype(np.float64, copy=False)
    mean_ref, mean_candidate = reference.mean(), candidate.mean()
    var_ref, var_candidate = reference.var(), candidate.var()
    covariance = ((reference - mean_ref) * (candidate - mean_candidate)).mean()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    denominator = (mean_ref**2 + mean_candidate**2 + c1) * (
        var_ref + var_candidate + c2
    )
    if denominator == 0:
        return 1.0 if np.array_equal(reference, candidate) else 0.0
    return float(
        ((2 * mean_ref * mean_candidate + c1) * (2 * covariance + c2)) / denominator
    )


def compare_tensors(
    reference,
    candidate,
    policy: ParityPolicy | Mapping[str, float | int],
    *,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> TensorComparison:
    if not isinstance(policy, ParityPolicy):
        policy = ParityPolicy.from_mapping(policy)
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(f"tensor shape mismatch: {reference.shape} != {candidate.shape}")

    finite = np.isfinite(reference) & np.isfinite(candidate)
    nonfinite_count = int(finite.size - finite.sum())
    if finite.any():
        difference = np.abs(reference[finite] - candidate[finite])
        max_abs = float(difference.max(initial=0.0))
        mean_abs = float(difference.mean())
        ref_finite = reference[finite]
        candidate_finite = candidate[finite]
        denominator = float(np.linalg.norm(ref_finite) * np.linalg.norm(candidate_finite))
        if denominator == 0:
            cosine = 1.0 if np.array_equal(ref_finite, candidate_finite) else 0.0
        else:
            cosine = float(np.dot(ref_finite, candidate_finite) / denominator)
        mse = float(np.mean((ref_finite - candidate_finite) ** 2))
    else:
        max_abs = mean_abs = cosine = mse = float("nan")

    lower, upper = value_range
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError(f"value_range must be a finite increasing pair, got {value_range}")
    data_range = upper - lower
    psnr = float(
        20 * math.log10(data_range)
        - 10 * math.log10(max(mse, np.finfo(float).tiny))
    )
    ssim = _global_ssim(reference[finite], candidate[finite], data_range) if finite.any() else float("nan")
    safe_reference = np.nan_to_num(reference, nan=0.0, posinf=1.0, neginf=-1.0)
    safe_candidate = np.nan_to_num(candidate, nan=0.0, posinf=1.0, neginf=-1.0)
    ref_u8 = np.rint(np.clip((safe_reference - lower) / data_range, 0, 1) * 255).astype(np.uint8)
    candidate_u8 = np.rint(np.clip((safe_candidate - lower) / data_range, 0, 1) * 255).astype(np.uint8)
    uint8_mismatch = float(np.mean(ref_u8 != candidate_u8))

    metrics: dict[str, float | int] = {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "nonfinite_count": nonfinite_count,
        "cosine_similarity": cosine,
        "psnr": psnr,
        "ssim": ssim,
        "uint8_mismatch_rate": uint8_mismatch,
    }
    failures: list[str] = []
    for name, value in metrics.items():
        if not math.isfinite(float(value)):
            failures.append(f"{name} is non-finite")
    upper_bounds = {
        "max_abs_error": policy.max_abs_error,
        "mean_abs_error": policy.mean_abs_error,
        "nonfinite_count": policy.nonfinite_count,
        "uint8_mismatch_rate": policy.uint8_mismatch_rate,
    }
    lower_bounds = {
        "cosine_similarity": policy.cosine_similarity,
        "psnr": policy.psnr,
        "ssim": policy.ssim,
    }
    failures.extend(
        f"{name}={metrics[name]} exceeds {bound}"
        for name, bound in upper_bounds.items()
        if metrics[name] > bound
    )
    failures.extend(
        f"{name}={metrics[name]} is below {bound}"
        for name, bound in lower_bounds.items()
        if metrics[name] < bound
    )
    return TensorComparison(metrics=metrics, passed=not failures, failures=tuple(failures))
