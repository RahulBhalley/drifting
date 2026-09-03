"""Hash-validated latent-cache format and ImageFolder-compatible reader."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from drifting_common.artifacts import sha256_file

from .transforms import adm_center_crop


@dataclass(frozen=True)
class CacheEntry:
    path: str
    source_path: str
    label: int
    shape: tuple[int, ...]
    dtype: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class CacheManifest:
    schema_version: int
    source_root: str
    resolution: int
    vae_model_id: str
    vae_revision: str | None
    scaling_factor: float
    split_counts: dict[str, int]
    entries: tuple[CacheEntry, ...]

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)

    @classmethod
    def from_json(cls, path: str | Path) -> "CacheManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise ValueError("unsupported latent-cache schema version")
        return cls(
            schema_version=1,
            source_root=raw["source_root"],
            resolution=int(raw["resolution"]),
            vae_model_id=raw["vae_model_id"],
            vae_revision=raw.get("vae_revision"),
            scaling_factor=float(raw["scaling_factor"]),
            split_counts={name: int(value) for name, value in raw["split_counts"].items()},
            entries=tuple(
                CacheEntry(
                    path=item["path"],
                    source_path=item["source_path"],
                    label=int(item["label"]),
                    shape=tuple(item["shape"]),
                    dtype=item["dtype"],
                    sha256=item["sha256"],
                    size_bytes=int(item["size_bytes"]),
                )
                for item in raw["entries"]
            ),
        )


class LatentCacheDataset(Dataset):
    """Read versioned cache entries and legacy ``moments`` cache files."""

    def __init__(self, root: str | Path, *, seed: int = 0):
        self.root = Path(root)
        classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        class_to_index = {name: index for index, name in enumerate(classes)}
        self.samples = [
            (path, class_to_index[path.parent.name])
            for path in sorted(self.root.glob("*/*.pt"))
        ]
        if not self.samples:
            raise FileNotFoundError(f"no latent .pt entries under {self.root}")
        self.generator = torch.Generator().manual_seed(int(seed))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path, label = self.samples[index]
        data = torch.load(path, map_location="cpu", weights_only=False)
        first = data.get("latent", data.get("moments"))
        flipped = data.get("latent_flip", data.get("moments_flip", first))
        if first is None:
            raise ValueError(f"latent cache entry has no supported tensor: {path}")
        use_flip = bool(torch.rand((), generator=self.generator) < 0.5)
        value = flipped if use_flip else first
        return torch.as_tensor(value, dtype=torch.float32), label


def _cache_file_valid(path: Path, entry: CacheEntry | None) -> bool:
    return bool(
        entry
        and path.is_file()
        and path.stat().st_size == entry.size_bytes
        and sha256_file(path) == entry.sha256
    )


def _atomic_torch_save(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def build_latent_cache(
    data_root: str | Path,
    cache_root: str | Path,
    codec,
    *,
    resolution: int = 256,
    batch_size: int = 32,
    num_workers: int = 0,
) -> CacheManifest:
    """Build or resume a two-split cache with same-directory atomic writes."""
    del num_workers  # Writes are intentionally synchronous and crash-consistent.
    source_root = Path(data_root).resolve()
    target_root = Path(cache_root).resolve()
    old_entries: dict[str, CacheEntry] = {}
    manifest_path = target_root / "manifest.json"
    if manifest_path.is_file():
        try:
            old = CacheManifest.from_json(manifest_path)
            if (
                old.source_root == str(source_root)
                and old.resolution == resolution
                and old.vae_model_id == codec.model_id
                and old.vae_revision == codec.revision
            ):
                old_entries = {entry.path: entry for entry in old.entries}
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            old_entries = {}

    entries: list[CacheEntry] = []
    split_counts: dict[str, int] = {}
    for split in ("train", "val"):
        dataset = ImageFolder(source_root / split)
        split_counts[split] = len(dataset.samples)
        pending: list[tuple[Path, int, str, Path]] = []
        for source_name, label in dataset.samples:
            source_path = Path(source_name)
            relative_source = source_path.relative_to(source_root).as_posix()
            relative_output = Path(split, source_path.parent.name, source_path.stem + ".pt")
            output = target_root / relative_output
            previous = old_entries.get(relative_output.as_posix())
            if _cache_file_valid(output, previous):
                entries.append(previous)
            else:
                pending.append((source_path, label, relative_source, relative_output))

        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            tensors = []
            for source_path, _, _, _ in chunk:
                image = dataset.loader(str(source_path)).convert("RGB")
                image = adm_center_crop(image, resolution)
                array = torch.from_numpy(np.array(image, copy=True))
                tensors.append(array.permute(2, 0, 1).float().div(127.5).sub(1))
            images = torch.stack(tensors)
            if hasattr(codec, "model"):
                parameter = next(codec.model.parameters(), None)
                if parameter is not None:
                    images = images.to(parameter.device)
            with torch.inference_mode():
                latent = codec.encode(images).detach().cpu()
                latent_flip = codec.encode(images.flip(-1)).detach().cpu()
            for item, value, value_flip in zip(chunk, latent, latent_flip):
                _, label, relative_source, relative_output = item
                output = target_root / relative_output
                _atomic_torch_save(
                    {"moments": value, "moments_flip": value_flip}, output
                )
                entries.append(
                    CacheEntry(
                        path=relative_output.as_posix(),
                        source_path=relative_source,
                        label=int(label),
                        shape=tuple(value.shape),
                        dtype=str(value.dtype).removeprefix("torch."),
                        sha256=sha256_file(output),
                        size_bytes=output.stat().st_size,
                    )
                )
    manifest = CacheManifest(
        schema_version=1,
        source_root=str(source_root),
        resolution=int(resolution),
        vae_model_id=codec.model_id,
        vae_revision=codec.revision,
        scaling_factor=float(codec.scaling_factor),
        split_counts=split_counts,
        entries=tuple(sorted(entries, key=lambda item: item.path)),
    )
    manifest.write(manifest_path)
    return manifest


__all__ = ["CacheEntry", "CacheManifest", "LatentCacheDataset", "build_latent_cache"]
