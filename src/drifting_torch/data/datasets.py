"""Native NCHW PyTorch input pipelines for all released dataset modes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sized
from dataclasses import dataclass
from functools import partial
import os
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.datasets import CIFAR10, FakeData, ImageFolder

from .latent import LatentCacheDataset
from .transforms import pixel_transform
from .vae import VAECodec


@dataclass(frozen=True)
class DataBatch:
    images: Tensor
    labels: Tensor


@dataclass
class DataPipeline:
    loader: DataLoader
    sampler: "StatefulSampler"
    preprocess: Any
    postprocess: Any


class StatefulSampler(Sampler[int]):
    """Deterministic sharded sampler with a serializable epoch/cursor."""

    def __init__(
        self,
        dataset: Sized,
        *,
        shuffle: bool,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.cursor = 0

    def _indices(self) -> list[int]:
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            values = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            values = list(range(len(self.dataset)))
        return values[self.rank :: self.world_size]

    def __iter__(self) -> Iterator[int]:
        indices = self._indices()
        if self.cursor >= len(indices):
            self.epoch += 1
            self.cursor = 0
            indices = self._indices()
        while self.cursor < len(indices):
            index = indices[self.cursor]
            self.cursor += 1
            yield index

    def __len__(self) -> int:
        return len(range(self.rank, len(self.dataset), self.world_size))

    def state_dict(self) -> dict[str, int | bool]:
        return {
            "schema_version": 1,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = (self.shuffle, self.seed, self.rank, self.world_size)
        actual = (
            bool(state["shuffle"]),
            int(state["seed"]),
            int(state["rank"]),
            int(state["world_size"]),
        )
        if state.get("schema_version") != 1 or actual != expected:
            raise ValueError("sampler state is incompatible with this data pipeline")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        if not 0 <= self.cursor <= len(self._indices()):
            raise ValueError("sampler cursor is out of range")


def _mapping(config: Any) -> dict[str, Any]:
    if hasattr(config, "dataset"):
        config = config.dataset
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return dict(config)


def _runtime_mapping(runtime: Any) -> dict[str, Any]:
    if runtime is None:
        return {}
    if hasattr(runtime, "runtime"):
        runtime = runtime.runtime
    if hasattr(runtime, "to_dict"):
        return runtime.to_dict()
    return dict(runtime)


def _worker_init(worker_id: int, *, seed: int, rank: int) -> None:
    worker_seed = seed + worker_id + rank * 1000
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))


def create_dataset_split(config: Any, runtime: Any, split: str) -> DataPipeline:
    """Build an explicit fake, CIFAR-10, ImageNet, or cached-latent pipeline."""
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    values = _mapping(config)
    runtime_values = _runtime_mapping(runtime)
    kwargs = dict(values.get("kwargs", {}))
    source = str(values.get("source", "imagenet")).lower()
    if source not in {"fake", "cifar10", "imagenet"}:
        raise ValueError(f"Unsupported dataset source {source!r}")
    resolution = int(values["resolution"])
    num_classes = int(values["num_classes"])
    batch_size = int(values["batch_size"] if split == "train" else values["eval_batch_size"])
    use_aug = bool(values.get("use_aug", False))
    use_cache = bool(values.get("use_cache", False))
    use_latent = bool(values.get("use_latent", False))
    data_root = kwargs.get("data_root", "data")
    cache_root = kwargs.get("cache_root")
    seed = int(kwargs.get("seed", 0))
    rank = int(runtime_values.get("rank", os.environ.get("RANK", 0)))
    world_size = int(runtime_values.get("world_size", os.environ.get("WORLD_SIZE", 1)))

    codec = kwargs.get("codec")
    if use_cache:
        if cache_root is None:
            raise ValueError("cache_root is required when use_cache=true")
        dataset: Dataset = LatentCacheDataset(os.path.join(cache_root, split), seed=seed)
    else:
        transform = pixel_transform(
            resolution, use_aug=use_aug, split=split, imagenet_style=source == "imagenet"
        )
        if source == "fake":
            size = int(
                kwargs.get(
                    f"fake_{split}_size", kwargs.get("fake_size", 1024)
                )
            )
            dataset = FakeData(
                size=size,
                image_size=(3, resolution, resolution),
                num_classes=num_classes,
                transform=transform,
                random_offset=0 if split == "train" else 1_000_000,
            )
        elif source == "cifar10":
            if num_classes != 10:
                raise ValueError("CIFAR-10 requires exactly 10 classes")
            dataset = CIFAR10(
                root=data_root,
                train=split == "train",
                transform=transform,
                download=bool(kwargs.get("download", False)),
            )
        else:
            dataset = ImageFolder(os.path.join(data_root, split), transform=transform)
    if use_latent and use_cache:
        raise ValueError("use_latent and use_cache are mutually exclusive")
    if (use_latent or use_cache) and codec is None:
        codec = VAECodec.from_pretrained(
            cache_dir=kwargs.get("hf_cache_dir"),
            local_files_only=bool(kwargs.get("local_files_only", False)),
        )

    sampler = StatefulSampler(
        dataset,
        shuffle=split == "train",
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
    num_workers = int(kwargs.get("num_workers", runtime_values.get("num_workers", 0)))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=split == "train",
        num_workers=num_workers,
        prefetch_factor=int(kwargs.get("prefetch_factor", 2)) if num_workers else None,
        persistent_workers=num_workers > 0,
        pin_memory=bool(kwargs.get("pin_memory", runtime_values.get("pin_memory", False))),
        worker_init_fn=partial(_worker_init, seed=seed, rank=rank),
    )
    device_name = runtime_values.get("device", "cpu")
    device = torch.device("cpu" if device_name == "auto" else device_name)

    def preprocess(raw_batch, *, noise: Tensor | None = None) -> DataBatch:
        images, labels = raw_batch
        images = torch.as_tensor(images, device=device)
        labels = torch.as_tensor(labels, device=device, dtype=torch.int64)
        if use_latent:
            images = codec.encode(images, noise=noise)
        return DataBatch(images=images, labels=labels)

    def postprocess(images: Tensor) -> Tensor:
        if use_latent or use_cache:
            if codec is None:
                raise RuntimeError("a VAE codec is required to decode cached latents")
            images = codec.decode(images)
        return ((images + 1) / 2).clamp(0, 1)

    return DataPipeline(loader=loader, sampler=sampler, preprocess=preprocess, postprocess=postprocess)


__all__ = ["DataBatch", "DataPipeline", "StatefulSampler", "create_dataset_split"]
