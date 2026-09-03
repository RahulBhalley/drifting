"""Dataset pipelines for the Drifting release.

``create_imagenet_split`` retains the original ImageNet implementation.
``create_dataset_split`` dispatches to ImageNet or local pixel-only datasets.

Flag meanings:
- `use_aug`: enable stronger pixel-space augmentation in train split.
- `use_latent`: encode RGB images to VAE latent online.
- `use_cache`: read precomputed latent `.pt` files from `IMAGENET_CACHE_PATH`.
"""

from __future__ import annotations

import os
import random
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10, FakeData, ImageFolder

from dataset.latent import LatentDataset
from dataset.vae import vae_enc_decode
from utils.env import IMAGENET_PATH, IMAGENET_CACHE_PATH
from utils.logging import log_for_0

def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """Center-crop image with ADM preprocessing style."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def _build_transforms(resolution: int, use_aug: bool, split: str):
    """Build torchvision transforms for ImageNet.

    Args:
        resolution: output spatial size.
        use_aug: whether to use strong train-time augmentation.
        split: `train` or `val`.

    Notes:
        - when `use_aug=True`, train split uses random resized crop.
        - otherwise uses center-crop style preprocessing.
    """
    if use_aug and split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(resolution, scale=(0.2, 1.0), interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: center_crop_arr(img, resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def _build_imagenet_dataset(
    *,
    resolution: int,
    use_aug: bool,
    use_cache: bool,
    split: str,
    data_root: str | None = None,
    cache_root: str | None = None,
):
    """Create dataset object for one ImageNet split."""
    if use_cache:
        return LatentDataset(root=os.path.join(cache_root or IMAGENET_CACHE_PATH, split))

    transform = _build_transforms(resolution, use_aug=use_aug, split=split)
    return ImageFolder(root=os.path.join(data_root or IMAGENET_PATH, split), transform=transform)


def worker_init_fn(worker_id: int, rank: int) -> None:
    """Initialize deterministic RNG for each data-loader worker."""
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def create_imagenet_split(
    *,
    resolution: int,
    batch_size: int,
    split: str,
    use_aug: bool = False,
    use_latent: bool = False,
    use_cache: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    local: bool | None = None,
    data_root: str | None = None,
    cache_root: str | None = None,
):
    """Create ImageNet split loader and preprocess/postprocess functions.

    Args:
        resolution: image resolution.
        use_aug: whether enable random resized crop augmentation for train split.
        use_latent: encode RGB image to latent online.
        use_cache: read precomputed latent cache from disk.
        batch_size: per-process batch size.
        split: `train` or `val`.
        num_workers: dataloader workers.
        prefetch_factor: dataloader prefetch factor.
        pin_memory: dataloader pin_memory.
        local: legacy config field kept for release compatibility; ignored.
        data_root: optional ImageNet root override; defaults to ``IMAGENET_PATH``.
        cache_root: optional latent-cache root override; defaults to
            ``IMAGENET_CACHE_PATH``.

    Returns:
        `(loader, preprocess_fn, postprocess_fn)`.
        - `preprocess_fn` converts one batch to dict with keys `images`, `labels`.
        - `postprocess_fn` converts model-space tensor to visualization tensor in `[0,1]`.

    Shape conventions:
        - Raw image batches from DataLoader are `BCHW` (torch/numpy).
        - Model runtime image/latent tensors are `BHWC` (JAX path).
        - Pixel postprocess returns `BCHW`.
    """
    del local
    ds = _build_imagenet_dataset(
        resolution=resolution,
        use_aug=use_aug,
        use_cache=use_cache,
        split=split,
        data_root=data_root,
        cache_root=cache_root,
    )
    log_for_0(ds)

    rank = jax.process_index()
    sampler = DistributedSampler(ds, num_replicas=jax.process_count(), rank=rank, shuffle=True)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=(split == "train"),
        worker_init_fn=partial(worker_init_fn, rank=rank),
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )

    if use_latent or use_cache:
        encode_fn, decode_fn = vae_enc_decode()
        if use_cache:
            def preprocess_fn(batch, rng=jax.random.PRNGKey(0)):
                del rng
                cached_latent, label = batch
                # cached_latent: BHWC latent, label: B
                return {"images": cached_latent, "labels": label}
        else:
            def preprocess_fn(batch, rng=jax.random.PRNGKey(0)):
                image, label = batch
                # image: BCHW -> VAE encode -> BHWC latent
                return {"images": encode_fn(image, rng), "labels": label}

        def postprocess_fn(images):
            # images: BHWC latent -> decode -> BCHW pixel in [0,1]
            return jnp.clip((decode_fn(images) + 1) / 2, 0, 1)

        return loader, preprocess_fn, postprocess_fn

    def preprocess_fn(batch, rng=jax.random.PRNGKey(0)):
        del rng
        image, label = batch
        # image: BCHW -> BHWC
        return {"images": jnp.array(image).transpose(0, 2, 3, 1), "labels": jnp.array(label, dtype=jnp.int32)}

    def postprocess_fn(images):
        # images: BHWC pixel in [-1,1] -> BCHW pixel in [0,1]
        return jnp.clip((images + 1) / 2, 0, 1).transpose(0, 3, 1, 2)

    return loader, preprocess_fn, postprocess_fn


def _build_pixel_transform(*, resolution: int, use_aug: bool, split: str):
    operations = [
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
    ]
    if use_aug and split == "train":
        operations.append(transforms.RandomHorizontalFlip())
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(operations)


def _create_pixel_loader(
    dataset,
    *,
    batch_size: int,
    split: str,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
):
    rank = jax.process_index()
    sampler = DistributedSampler(
        dataset,
        num_replicas=jax.process_count(),
        rank=rank,
        shuffle=(split == "train"),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=(split == "train"),
        worker_init_fn=partial(worker_init_fn, rank=rank),
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    def preprocess_fn(batch, rng=jax.random.PRNGKey(0)):
        del rng
        image, label = batch
        return {
            "images": jnp.asarray(image).transpose(0, 2, 3, 1),
            "labels": jnp.asarray(label, dtype=jnp.int32),
        }

    def postprocess_fn(images):
        return jnp.clip((images + 1) / 2, 0, 1).transpose(0, 3, 1, 2)

    return loader, preprocess_fn, postprocess_fn


def create_dataset_split(
    *,
    resolution: int,
    batch_size: int,
    split: str,
    source: str = "imagenet",
    num_classes: int = 1000,
    data_root: str | None = None,
    cache_root: str | None = None,
    download: bool = False,
    fake_size: int = 1024,
    fake_train_size: int | None = None,
    fake_val_size: int | None = None,
    use_aug: bool = False,
    use_latent: bool = False,
    use_cache: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    local: bool | None = None,
):
    """Create an ImageNet, deterministic fake, or resized CIFAR-10 split."""
    source = str(source).strip().lower()
    supported = ("imagenet", "fake", "cifar10")
    if source not in supported:
        raise ValueError(f"Unsupported dataset source {source!r}. Supported sources: {', '.join(supported)}")
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    for name, value in (
        ("resolution", resolution),
        ("batch_size", batch_size),
        ("num_classes", num_classes),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")

    if source == "imagenet":
        return create_imagenet_split(
            resolution=resolution,
            batch_size=batch_size,
            split=split,
            use_aug=use_aug,
            use_latent=use_latent,
            use_cache=use_cache,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            local=local,
            data_root=data_root,
            cache_root=cache_root,
        )

    if use_latent or use_cache:
        raise ValueError(f"Dataset source {source!r} is pixel-only; latent encoding and caches are unsupported")
    if source == "cifar10" and int(num_classes) != 10:
        raise ValueError("CIFAR-10 requires exactly 10 classes")

    transform = _build_pixel_transform(resolution=resolution, use_aug=use_aug, split=split)
    if source == "fake":
        selected_size = fake_train_size if split == "train" else fake_val_size
        selected_size = fake_size if selected_size is None else selected_size
        if int(selected_size) <= 0:
            raise ValueError("fake_size for the selected split must be positive")
        dataset = FakeData(
            size=int(selected_size),
            image_size=(3, resolution, resolution),
            num_classes=int(num_classes),
            transform=transform,
            random_offset=0 if split == "train" else 1_000_000,
        )
    else:
        dataset = CIFAR10(
            root=data_root or "data",
            train=(split == "train"),
            transform=transform,
            download=download,
        )

    log_for_0(dataset)
    return _create_pixel_loader(
        dataset,
        batch_size=batch_size,
        split=split,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )


def get_postprocess_fn(*, use_aug: bool = False, use_latent: bool = False, use_cache: bool = False, has_clip: bool = True):
    """Return postprocess function for generated samples by dataset mode flags."""
    if use_latent or use_cache:
        _, decode_fn = vae_enc_decode()

        def postprocess(images):
            out = (decode_fn(images) + 1) / 2
            return jnp.clip(out, 0, 1) if has_clip else out

        return postprocess

    if use_aug or (not use_latent and not use_cache):

        def postprocess(images):
            out = (images + 1) / 2
            out = jnp.clip(out, 0, 1) if has_clip else out
            return out.transpose(0, 3, 1, 2)

        return postprocess

    raise ValueError("Unsupported dataset flags.")


def infinite_sampler(it, start_step: int = 0):
    """Yield `(image, label)` batches forever, resuming at `start_step`."""
    step_per_epoch = len(it)
    epoch_idx = start_step // step_per_epoch
    it.sampler.set_epoch(epoch_idx)
    skip_batches = start_step % step_per_epoch
    while True:
        for i, batch in enumerate(it):
            if skip_batches > 0 and i < skip_batches:
                continue
            image, label = batch
            yield (image.numpy(), label.numpy())
        skip_batches = 0
        epoch_idx += 1
        it.sampler.set_epoch(epoch_idx)


def epoch0_sampler(it):
    """Yield one deterministic epoch (`sampler.set_epoch(0)`)."""
    it.sampler.set_epoch(0)
    for batch in it:
        image, label = batch
        yield (image.numpy(), label.numpy())
