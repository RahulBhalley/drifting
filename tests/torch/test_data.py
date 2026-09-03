from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from drifting_torch.data.datasets import create_dataset_split
from drifting_torch.data.transforms import adm_center_crop, pixel_transform


def mapping(source: str, root: Path) -> dict:
    return {
        "source": source,
        "resolution": 16,
        "num_classes": 10 if source == "cifar10" else 2,
        "batch_size": 2,
        "eval_batch_size": 2,
        "kwargs": {
            "data_root": str(root),
            "download": False,
            "fake_size": 4,
            "num_workers": 0,
        },
    }


def write_imagefolder(root: Path) -> None:
    for split in ("train", "val"):
        for class_name, offset in (("a", 0), ("b", 40)):
            folder = root / split / class_name
            folder.mkdir(parents=True)
            values = np.arange(20 * 30 * 3, dtype=np.uint16).reshape(20, 30, 3)
            Image.fromarray(((values + offset) % 256).astype(np.uint8)).save(folder / "0.png")


def test_fake_dataset_pipeline_returns_nchw_and_resumable_sampler(tmp_path: Path):
    pipeline = create_dataset_split(mapping("fake", tmp_path), {"device": "cpu"}, "train")
    batch = pipeline.preprocess(next(iter(pipeline.loader)))
    assert batch.images.shape == (2, 3, 16, 16)
    assert batch.labels.dtype == torch.int64
    assert batch.images.min() >= -1 and batch.images.max() <= 1
    state = pipeline.sampler.state_dict()
    assert state["cursor"] == 2
    pipeline.sampler.load_state_dict(state)


def test_imagenet_imagefolder_pipeline_uses_path_override(tmp_path: Path):
    write_imagefolder(tmp_path)
    config = mapping("imagenet", tmp_path)
    pipeline = create_dataset_split(config, {"device": "cpu"}, "val")
    batch = pipeline.preprocess(next(iter(pipeline.loader)))
    assert batch.images.shape == (2, 3, 16, 16)
    assert sorted(batch.labels.tolist()) == [0, 1]


def test_cifar_dispatch_can_be_replaced_without_changing_arguments(monkeypatch, tmp_path: Path):
    captured = {}

    class StubCifar(torch.utils.data.Dataset):
        def __init__(self, root, train, transform, download):
            captured.update(root=root, train=train, download=download)
            self.transform = transform

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return self.transform(Image.new("RGB", (32, 32), color=index)), index % 10

    monkeypatch.setattr("drifting_torch.data.datasets.CIFAR10", StubCifar)
    pipeline = create_dataset_split(mapping("cifar10", tmp_path), {"device": "cpu"}, "train")
    batch = pipeline.preprocess(next(iter(pipeline.loader)))
    assert batch.images.shape == (2, 3, 16, 16)
    assert captured == {"root": str(tmp_path), "train": True, "download": False}


def test_invalid_source_and_latent_flags_are_rejected(tmp_path: Path):
    config = mapping("fake", tmp_path)
    config["source"] = "unknown"
    with pytest.raises(ValueError, match="Unsupported"):
        create_dataset_split(config, {"device": "cpu"}, "train")
    config = mapping("fake", tmp_path)
    config["use_cache"] = True
    with pytest.raises(ValueError, match="cache_root"):
        create_dataset_split(config, {"device": "cpu"}, "train")


def test_adm_crop_and_pixel_transform_are_deterministic():
    image = Image.fromarray(np.arange(33 * 51 * 3, dtype=np.uint8).reshape(33, 51, 3))
    first = adm_center_crop(image, 16)
    second = adm_center_crop(image, 16)
    assert np.array_equal(np.asarray(first), np.asarray(second))
    tensor = pixel_transform(16, use_aug=False, split="val")(image)
    assert tensor.shape == (3, 16, 16)

