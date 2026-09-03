from __future__ import annotations

import argparse

from drifting_jax.runtime import EasyDict, add_dataset_override_args, apply_dataset_overrides


def _config() -> EasyDict:
    return EasyDict(
        dataset=EasyDict(
            source="fake",
            resolution=16,
            num_classes=10,
            kwargs=EasyDict(fake_size=32),
        )
    )


def test_dataset_cli_overrides_replace_local_source_and_paths() -> None:
    parser = argparse.ArgumentParser()
    add_dataset_override_args(parser)
    args = parser.parse_args(
        [
            "--dataset-source", "imagenet",
            "--data-root", "/datasets/imagenet",
            "--cache-root", "/datasets/imagenet-latents",
            "--dataset-resolution", "256",
            "--num-classes", "1000",
            "--no-download-dataset",
        ]
    )
    config = _config()

    apply_dataset_overrides(config, args)

    assert config.dataset.source == "imagenet"
    assert config.dataset.resolution == 256
    assert config.dataset.num_classes == 1000
    assert config.dataset.kwargs.data_root == "/datasets/imagenet"
    assert config.dataset.kwargs.cache_root == "/datasets/imagenet-latents"
    assert config.dataset.kwargs.download is False


def test_absent_dataset_cli_overrides_preserve_config() -> None:
    parser = argparse.ArgumentParser()
    add_dataset_override_args(parser)
    config = _config()

    apply_dataset_overrides(config, parser.parse_args([]))

    assert config.dataset.source == "fake"
    assert config.dataset.resolution == 16
    assert config.dataset.num_classes == 10
    assert config.dataset.kwargs == {"fake_size": 32}
