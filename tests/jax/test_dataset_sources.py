from __future__ import annotations

import jax.numpy as jnp
import pytest

import drifting_jax.data.datasets as dataset_module
from drifting_jax.data.datasets import create_dataset_split


def test_fake_split_emits_requested_bhwc_images() -> None:
    loader, preprocess, postprocess = create_dataset_split(
        source="fake",
        resolution=16,
        batch_size=4,
        split="train",
        num_classes=10,
        fake_size=8,
        num_workers=0,
    )

    batch = preprocess(next(iter(loader)))

    assert batch["images"].shape == (4, 16, 16, 3)
    assert batch["labels"].shape == (4,)
    assert batch["labels"].dtype == jnp.int32
    assert float(batch["images"].min()) >= -1.0
    assert float(batch["images"].max()) <= 1.0
    assert postprocess(batch["images"]).shape == (4, 3, 16, 16)


def test_fake_split_uses_independent_configurable_sizes() -> None:
    train, _, _ = create_dataset_split(
        source="fake",
        resolution=8,
        batch_size=2,
        split="train",
        num_classes=3,
        fake_size=99,
        fake_train_size=6,
        fake_val_size=4,
        num_workers=0,
    )
    val, _, _ = create_dataset_split(
        source="fake",
        resolution=8,
        batch_size=2,
        split="val",
        num_classes=3,
        fake_size=99,
        fake_train_size=6,
        fake_val_size=4,
        num_workers=0,
    )

    assert len(train.dataset) == 6
    assert len(val.dataset) == 4


@pytest.mark.parametrize("source", ["fake", "cifar10"])
@pytest.mark.parametrize("flag", ["use_latent", "use_cache"])
def test_alternate_sources_reject_latent_modes(source: str, flag: str) -> None:
    with pytest.raises(ValueError, match="pixel-only"):
        create_dataset_split(
            source=source,
            resolution=16,
            batch_size=2,
            split="train",
            num_classes=10,
            num_workers=0,
            **{flag: True},
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source": "mnist"}, "Supported sources"),
        ({"source": "fake", "resolution": 0}, "resolution"),
        ({"source": "fake", "batch_size": 0}, "batch_size"),
        ({"source": "fake", "num_classes": 0}, "num_classes"),
        ({"source": "fake", "fake_size": 0}, "fake_size"),
        ({"source": "fake", "split": "test"}, "split"),
        ({"source": "cifar10", "num_classes": 11}, "10 classes"),
    ],
)
def test_invalid_dataset_options_raise_value_error(kwargs: dict, message: str) -> None:
    options = {
        "source": "fake",
        "resolution": 8,
        "batch_size": 2,
        "split": "train",
        "num_classes": 10,
        "fake_size": 4,
        "num_workers": 0,
    }
    options.update(kwargs)

    with pytest.raises(ValueError, match=message):
        create_dataset_split(**options)


def test_absent_source_dispatches_to_existing_imagenet_loader(monkeypatch) -> None:
    sentinel = (object(), object(), object())
    received = {}

    def fake_imagenet_split(**kwargs):
        received.update(kwargs)
        return sentinel

    monkeypatch.setattr(dataset_module, "create_imagenet_split", fake_imagenet_split)

    result = create_dataset_split(
        resolution=32,
        batch_size=4,
        split="val",
        use_aug=True,
        num_workers=0,
    )

    assert result is sentinel
    assert received == {
        "resolution": 32,
        "batch_size": 4,
        "split": "val",
        "use_aug": True,
        "use_latent": False,
        "use_cache": False,
        "num_workers": 0,
        "prefetch_factor": 2,
        "pin_memory": False,
        "local": None,
        "data_root": None,
        "cache_root": None,
    }


def test_imagenet_data_root_can_override_environment_path(monkeypatch, tmp_path) -> None:
    received = {}
    sentinel = object()

    def fake_image_folder(*, root, transform):
        received["root"] = root
        received["transform"] = transform
        return sentinel

    monkeypatch.setattr(dataset_module, "ImageFolder", fake_image_folder)

    result = dataset_module._build_imagenet_dataset(
        resolution=16,
        use_aug=False,
        use_cache=False,
        split="train",
        data_root=str(tmp_path),
    )

    assert result is sentinel
    assert received["root"] == str(tmp_path / "train")
