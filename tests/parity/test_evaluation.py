import numpy as np
import torch

from drifting_jax.evaluation.evaluator import _compute_inception_score as jax_inception_score
from drifting_jax.evaluation.frechet import compute_frechet_distance as jax_frechet
from drifting_jax.evaluation.precision_recall import compute_precision_recall as jax_pr
from drifting_jax.evaluation.resize import forward as jax_resize
from drifting_torch.evaluation.inception import resize_for_inception
from drifting_torch.evaluation.precision_recall import compute_precision_recall
from drifting_torch.evaluation.statistics import compute_frechet_distance, compute_inception_score


def test_resize_matches_release_torchscript_helper_exactly():
    images = torch.arange(2 * 3 * 31 * 47, dtype=torch.float32).reshape(2, 3, 31, 47) % 256
    torch.testing.assert_close(resize_for_inception(images), jax_resize(images), rtol=0, atol=0)


def test_fid_is_and_precision_recall_match_jax_metrics():
    rng = np.random.default_rng(71)
    first = rng.normal(size=(12, 5))
    second = rng.normal(size=(13, 5))
    mu1, mu2 = first.mean(0), second.mean(0)
    sigma1, sigma2 = np.cov(first, rowvar=False), np.cov(second, rowvar=False)
    np.testing.assert_allclose(
        compute_frechet_distance(mu1, mu2, sigma1, sigma2),
        jax_frechet(mu1, mu2, sigma1, sigma2),
        atol=1e-10,
        rtol=1e-10,
    )

    logits = rng.normal(size=(20, 7)).astype(np.float32)
    np.testing.assert_allclose(
        compute_inception_score(logits, splits=5),
        jax_inception_score(logits, splits=5),
        atol=1e-7,
        rtol=1e-7,
    )
    expected_pr = jax_pr(first, second, k=3)
    actual_pr = compute_precision_recall(first, second, k=3)
    np.testing.assert_allclose(actual_pr, expected_pr, atol=0, rtol=0)
