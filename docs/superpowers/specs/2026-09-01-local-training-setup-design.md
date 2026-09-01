# Local Training Setup Design

## Goal

Provide two reproducible local training workflows while preserving every
upstream Colab notebook and TPU/ImageNet configuration unchanged:

1. the authors' PyTorch toy drifting notebook for Swiss-roll and checkerboard
   training; and
2. a small JAX generator-training path for an Apple M1 Mac with 16 GB memory,
   using either deterministic fake images or CIFAR-10 resized to a configurable
   resolution until ImageNet is available.

The local JAX path is a functional algorithm and integration smoke test. It is
not an ImageNet benchmark reproduction and must never overwrite or masquerade
as an upstream SOTA configuration.

## Constraints

- Preserve all files under `configs/gen/` and the upstream
  `notebooks/inference_demo.ipynb` byte-for-byte.
- Save the linked authors' toy notebook as an original source artifact without
  editing its cells.
- Put local-only configs under `configs/local/` with names that state their
  hardware and dataset purpose.
- Keep toy PyTorch dependencies isolated from the existing JAX inference
  `.venv` to avoid changing the verified inference environment.
- Default dataset behavior remains ImageNet when `dataset.source` is absent.
- Dataset selection is configuration-driven: `imagenet`, `fake`, or `cifar10`.
- CIFAR-10 resizing is controlled by the existing `dataset.resolution` value,
  so ImageNet can later replace it through configuration rather than code edits.
- Fake and CIFAR-10 runs disable ImageNet FID evaluation because the repository
  only ships an ImageNet-256 reference-statistics path.
- Verification claims must distinguish config parsing, loader/model smoke tests,
  one optimization step, full toy training, and benchmark reproduction.

## Architecture

### Toy PyTorch workflow

- `notebooks/drifting_model_demo_original.ipynb` is downloaded from the authors'
  website and retained unchanged.
- `.venv-toy` contains the notebook's PyTorch, torchvision, matplotlib, tqdm,
  einops, and notebook-execution dependencies.
- `requirements-toy.txt` records reproducible local dependencies.
- A local execution helper copies the original notebook in memory, skips only
  its Colab-style `pip install` cell, executes the remaining cells with the
  installed kernel, and writes the executed result under `outputs/toy/`.
- Verification runs the authors' Swiss-roll and checkerboard training and checks
  that one-pass generated samples have shape `(N, 2)` and finite values.

### Configurable JAX dataset workflow

`dataset/dataset.py` gains a general `create_dataset_split` entry point. The
existing `create_imagenet_split` remains available and retains its current
behavior. The general entry point dispatches by `source`:

- `imagenet`: existing ImageFolder or latent-cache path;
- `fake`: torchvision `FakeData`, with configurable split sizes, deterministic
  offsets, class count, resolution, and normalized RGB tensors;
- `cifar10`: torchvision CIFAR-10 train/test splits, optional download, resized
  to `dataset.resolution`, normalized to `[-1, 1]`.

All pixel sources expose the same loader contract:

```text
PyTorch BCHW batch -> preprocess_fn -> JAX BHWC images + int32 labels
```

Latent encoding/cache flags are rejected for `fake` and `cifar10` with a clear
error because those modes are intended for pixel-space local smoke training.

`utils/model_builder.py` reads `dataset.source`, defaults it to `imagenet`, and
passes source-specific options to the loader. It reports a dataset name that
matches the source instead of labeling every run as ImageNet.

### Local JAX configurations

Two separate configurations are added:

- `configs/local/m1_fake_smoke.yaml`: deterministic, no-download, minimal run
  used in automated tests and one-step verification;
- `configs/local/m1_cifar10_smoke.yaml`: CIFAR-10 download/cache with configurable
  upscaling through `dataset.resolution`.

Both use one JAX CPU device, a deliberately small pixel-space generator, tiny
memory banks and batch sizes, no external MAE/ConvNeXt feature model, no W&B,
and no FID evaluation. The unmodified `configs/gen/*.yaml` remain the source of
truth for ImageNet/TPU experiments.

### Training-loop compatibility

The generator training loop receives `num_classes` from the selected dataset
instead of hard-coding 1000 in its positive memory bank. An `enable_eval` flag,
defaulting to `true`, guards the ImageNet FID block. Upstream configurations do
not set either new option and therefore retain their existing 1000-class,
evaluation-enabled behavior.

## Interfaces

```python
def create_dataset_split(
    *,
    source: str = "imagenet",
    resolution: int,
    batch_size: int,
    split: str,
    num_classes: int = 1000,
    data_root: str = "data",
    download: bool = False,
    fake_size: int = 1024,
    **loader_options,
) -> tuple[DataLoader, Callable, Callable]:
    ...
```

`create_imagenet_split` remains source-compatible. Local configuration fields
are consumed explicitly rather than forwarded blindly to `DataLoader`.

## Error Handling

- Unknown dataset sources raise `ValueError` listing supported values.
- `fake` or `cifar10` combined with `use_latent`/`use_cache` raises `ValueError`
  explaining that local alternate datasets are pixel-only.
- CIFAR-10 with `download: false` and no cached dataset preserves torchvision's
  actionable missing-data error.
- Local non-ImageNet configs set `enable_eval: false`; attempting ImageNet FID
  for another dataset remains an explicit unsupported-operation error.
- Config validation checks that batch sizes, fake split sizes, class counts, and
  resolutions are positive.

## Testing and Verification

1. Unit tests cover fake loader shapes/ranges/labels, source validation,
   pixel-only constraints, and unchanged ImageNet dispatch defaults.
2. CIFAR-10 loader verification downloads the official torchvision dataset,
   loads both splits, and confirms configurable resized tensor shapes.
3. Config tests parse every upstream and local YAML file, assert upstream files
   are unchanged, and validate local resource limits.
4. A one-step fake-data generator run verifies dataset loading, model creation,
   mesh initialization, drift loss, optimizer update, checkpoint/output writing,
   and finite metrics on the M1 CPU.
5. The original toy notebook executes in `.venv-toy`; both datasets train and
   produce finite one-pass samples. Executed notebooks and image outputs are
   retained under `outputs/toy/`.
6. Existing pretrained `pixel_B_sota` inference is rerun to ensure the new
   training setup does not regress the established inference path.

## Acceptance Criteria

- Original upstream configs and inference notebook remain unchanged.
- The original toy notebook is present locally and hash-recorded.
- Toy Swiss-roll and checkerboard training plus one-pass sampling complete.
- Fake and CIFAR-10 loaders are selectable by config and emit the requested
  resolution.
- The M1 fake-data config completes at least one real optimization step with
  finite loss and leaves a checkpoint or metrics artifact.
- CIFAR-10 is downloaded and loader-verified; a short CIFAR-10 training run is
  attempted within the 16 GB memory boundary and reported precisely.
- No ImageNet/FID or TPU benchmark claim is made from fake/CIFAR-10 results.
- Temporary dependency/download caches are removed; intentional environments,
  datasets, checkpoints, notebooks, and verification outputs are retained.
