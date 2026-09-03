import json
from pathlib import Path

from PIL import Image
import torch

from drifting_torch.data.latent import LatentCacheDataset, build_latent_cache


class StubCodec:
    model_id = "stub/vae"
    revision = "test-revision"
    scaling_factor = 0.18215

    def encode(self, images: torch.Tensor, noise=None) -> torch.Tensor:
        pooled = torch.nn.functional.avg_pool2d(images, kernel_size=2)
        return torch.cat((pooled, pooled[:, :1]), dim=1)


def write_source(root: Path) -> None:
    for split in ("train", "val"):
        folder = root / split / "class0"
        folder.mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (20, 24), color=(index * 30, 10, 20)).save(folder / f"{index}.png")


def test_cache_build_is_atomic_manifested_and_resumable(tmp_path: Path):
    source = tmp_path / "source"
    target = tmp_path / "cache"
    write_source(source)
    first = build_latent_cache(
        data_root=source,
        cache_root=target,
        codec=StubCodec(),
        resolution=16,
        batch_size=2,
        num_workers=0,
    )
    assert first.split_counts == {"train": 2, "val": 2}
    assert len(first.entries) == 4
    assert all((target / entry.path).is_file() for entry in first.entries)
    assert not list(target.rglob("*.tmp-*"))
    before = {entry.path: (target / entry.path).stat().st_mtime_ns for entry in first.entries}
    second = build_latent_cache(
        data_root=source,
        cache_root=target,
        codec=StubCodec(),
        resolution=16,
        batch_size=2,
        num_workers=0,
    )
    after = {entry.path: (target / entry.path).stat().st_mtime_ns for entry in second.entries}
    assert before == after
    raw = json.loads((target / "manifest.json").read_text())
    assert raw["vae_revision"] == "test-revision"


def test_latent_dataset_loads_new_and_legacy_entries(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    target = tmp_path / "cache"
    write_source(source)
    build_latent_cache(source, target, StubCodec(), resolution=16, batch_size=2, num_workers=0)
    dataset = LatentCacheDataset(target / "train", seed=0)
    latent, label = dataset[0]
    assert latent.shape == (4, 8, 8)
    assert label == 0
    assert latent.dtype == torch.float32

