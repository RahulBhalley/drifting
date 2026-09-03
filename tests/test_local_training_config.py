from __future__ import annotations

from pathlib import Path

import pytest

from train import _create_memory_banks, _should_evaluate
from utils.misc import load_config


LOCAL_CONFIGS = (
    Path("configs/local/m1_fake_smoke.yaml"),
    Path("configs/local/m1_cifar10_smoke.yaml"),
)


@pytest.mark.parametrize("path", LOCAL_CONFIGS)
def test_local_configs_are_resource_bounded_and_disable_fid(path: Path) -> None:
    assert path.is_file()
    config = load_config(str(path))

    assert config.dataset.source in {"fake", "cifar10"}
    assert config.dataset.num_classes == 10
    assert config.dataset.batch_size <= 16
    assert config.dataset.eval_batch_size <= 16
    assert config.dataset.kwargs.num_workers == 0
    assert config.train.total_steps <= 2
    assert config.train.enable_eval is False
    assert config.logging.use_wandb is False
    assert config.feature.use_mae is False
    assert config.feature.use_convnext is False
    assert config.model.input_size == config.dataset.resolution


def test_disabled_evaluation_skips_first_and_final_step() -> None:
    assert not _should_evaluate(
        step=1,
        total_steps=1,
        eval_per_step=1,
        enable_eval=False,
    )


def test_enabled_evaluation_preserves_first_step_sanity_check() -> None:
    assert _should_evaluate(
        step=1,
        total_steps=10,
        eval_per_step=5,
        enable_eval=True,
    )


def test_positive_memory_bank_uses_dataset_class_count() -> None:
    positive, negative = _create_memory_banks(
        num_classes=10,
        positive_bank_size=4,
        negative_bank_size=8,
    )

    assert positive.num_classes == 10
    assert positive.max_size == 4
    assert negative.num_classes == 1
    assert negative.max_size == 8
