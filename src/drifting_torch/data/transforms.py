"""Image transforms shared by PyTorch training and cache creation."""

from __future__ import annotations

import numpy as np
from PIL import Image
from torchvision import transforms


def adm_center_crop(image: Image.Image, image_size: int) -> Image.Image:
    """Center crop with the exact resize sequence used by ADM and the JAX path."""
    while min(*image.size) >= 2 * image_size:
        image = image.resize(tuple(value // 2 for value in image.size), resample=Image.BOX)
    scale = image_size / min(*image.size)
    image = image.resize(
        tuple(round(value * scale) for value in image.size), resample=Image.BICUBIC
    )
    array = np.asarray(image)
    crop_y = (array.shape[0] - image_size) // 2
    crop_x = (array.shape[1] - image_size) // 2
    return Image.fromarray(
        array[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def pixel_transform(
    resolution: int,
    *,
    use_aug: bool,
    split: str,
    imagenet_style: bool = False,
):
    if imagenet_style and use_aug and split == "train":
        operations = [
            transforms.RandomResizedCrop(
                resolution,
                scale=(0.2, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
        ]
    elif imagenet_style:
        # This includes the released path's flip policy for both train and val.
        operations = [
            transforms.Lambda(lambda image: adm_center_crop(image, resolution)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        operations = [
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        ]
        if use_aug and split == "train":
            operations.append(transforms.RandomHorizontalFlip())
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return transforms.Compose(operations)


__all__ = ["adm_center_crop", "pixel_transform"]
