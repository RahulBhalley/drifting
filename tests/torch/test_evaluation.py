import numpy as np
import pytest
import torch

from drifting_torch.evaluation.inception import ReleasedInception
from drifting_torch.evaluation.statistics import (
    compute_frechet_distance,
    compute_inception_score,
    compute_statistics,
)


class DummyInception(torch.nn.Module):
    def __init__(self, nonfinite=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.nonfinite = nonfinite

    def forward(self, images):
        mean = images.mean(dim=(2, 3))
        std = images.std(dim=(2, 3), unbiased=False)
        pooled = torch.cat((mean, std), dim=1)
        if self.nonfinite:
            pooled[0, 0] = float("nan")
        spatial = images[:, :1]
        logits = torch.cat((mean, -mean), dim=1)
        return pooled, spatial, logits


def test_padding_does_not_change_statistics():
    generator = torch.Generator().manual_seed(3)
    images = torch.rand(8, 3, 7, 7, generator=generator)
    base = compute_statistics(images, torch.ones(8, dtype=torch.bool), DummyInception())
    padded = torch.cat((images, torch.zeros_like(images[:3])))
    mask = torch.tensor([True] * 8 + [False] * 3)
    candidate = compute_statistics(padded, mask, DummyInception())
    np.testing.assert_array_equal(base.mu, candidate.mu)
    np.testing.assert_array_equal(base.sigma, candidate.sigma)
    np.testing.assert_array_equal(base.logits, candidate.logits)


def test_nonfinite_features_fail_closed():
    with pytest.raises(ValueError, match="non-finite"):
        compute_statistics(
            torch.ones(3, 3, 4, 4), torch.ones(3, dtype=torch.bool),
            DummyInception(nonfinite=True),
        )


def test_frechet_singular_covariance_and_inception_score_are_finite():
    mean = np.array([0.0, 1.0])
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert compute_frechet_distance(mean, mean, singular, singular) == pytest.approx(0, abs=1e-7)
    logits = np.arange(60, dtype=np.float32).reshape(20, 3) / 20
    score_mean, score_std = compute_inception_score(logits, splits=5)
    assert np.isfinite(score_mean) and np.isfinite(score_std)


def test_release_average_pool_excludes_padding():
    image = torch.ones(1, 1, 2, 2)
    torch.testing.assert_close(ReleasedInception._average_pool(image), image)
