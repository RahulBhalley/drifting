# Local Training on Apple Silicon

This guide provides two separate local workflows:

1. the authors' complete PyTorch toy notebook, covering Swiss roll and
   checkerboard training; and
2. the released JAX generator-training pipeline with a deliberately tiny model
   and either deterministic fake images or resized CIFAR-10.

The local JAX runs validate dataset loading, model construction, the drifting
loss, an optimizer update, and checkpoint writing. They are not ImageNet
benchmark reproductions. The original `configs/gen/*.yaml` files remain the
authoritative ImageNet/TPU configurations and are unchanged.

## Requirements

- Apple-silicon Mac
- Python 3.11 (Python 3.10 also satisfies the repository's upstream guidance)
- Enough disk space for the environments, outputs, and optionally CIFAR-10

Commands below run from the repository root.

## Authors' Toy Notebook

The unmodified source notebook is stored at
`notebooks/drifting_model_demo_original.ipynb`. Its expected SHA-256 digest is
stored beside it and can be checked with:

```bash
cd notebooks
shasum -a 256 -c drifting_model_demo_original.sha256
cd ..
```

Create the isolated environment and execute a derived copy:

```bash
python3.11 -m venv .venv-toy
.venv-toy/bin/python -m pip install -r requirements-toy.txt

JUPYTER_CONFIG_DIR="$PWD/work/jupyter-config" \
JUPYTER_DATA_DIR="$PWD/work/jupyter-data" \
JUPYTER_RUNTIME_DIR="$PWD/work/jupyter-runtime" \
IPYTHONDIR="$PWD/work/ipython" \
MPLCONFIGDIR="$PWD/work/matplotlib" \
.venv-toy/bin/python scripts/run_toy_notebook.py
```

The runner leaves the original notebook byte-for-byte unchanged, skips only its
Colab `pip install` cell, and appends a local assertion cell to the derived
notebook. That assertion performs one forward pass through each trained model
and checks for `(64, 2)` finite samples. The result is written to:

```text
outputs/toy/drifting_model_demo_executed.ipynb
```

Jupyter kernels use loopback sockets. In a restricted sandbox, run this command
with permission to bind local sockets.

## Local JAX Training Environment

The upstream `requirements.txt` intentionally remains TPU/Linux-oriented. Use
the macOS variant for local training:

```bash
python3.11 -m venv .venv-training
.venv-training/bin/python -m pip install -r requirements-macos-training.txt
```

Force the single-process CPU path for these small compatibility runs:

```bash
export JAX_PLATFORMS=cpu
```

This avoids multi-host initialization. It does not modify the upstream
`JAX_PLATFORMS=tpu,cpu` workflow.

## Deterministic Fake-Data Smoke Run

This run requires no dataset download:

```bash
JAX_PLATFORMS=cpu .venv-training/bin/python main.py \
  --gen \
  --config configs/local/m1_fake_smoke.yaml \
  --workdir outputs/training-smoke/fake
```

## Resized CIFAR-10 Smoke Run

The checked-in config downloads CIFAR-10 when it is not already present and
resizes it to `dataset.resolution`:

```bash
JAX_PLATFORMS=cpu .venv-training/bin/python main.py \
  --gen \
  --config configs/local/m1_cifar10_smoke.yaml \
  --workdir outputs/training-smoke/cifar10
```

To reuse an existing cache or change the resolution without editing YAML:

```bash
JAX_PLATFORMS=cpu .venv-training/bin/python main.py \
  --gen \
  --config configs/local/m1_cifar10_smoke.yaml \
  --data-root /path/to/cifar-cache \
  --dataset-resolution 32 \
  --no-download-dataset \
  --workdir outputs/training-smoke/cifar10-custom
```

## Replacing the Local Dataset Later

Dataset selection and paths can be replaced through CLI arguments:

```text
--dataset-source {imagenet,fake,cifar10}
--data-root PATH
--cache-root PATH
--dataset-resolution N
--num-classes N
--download-dataset / --no-download-dataset
```

For an extracted ImageNet pixel dataset, use an original generator config and
override only machine-specific paths when useful:

```bash
JAX_PLATFORMS=tpu,cpu python main.py \
  --gen \
  --config configs/gen/pixel_sota_B.yaml \
  --dataset-source imagenet \
  --data-root /path/to/imagenet \
  --dataset-resolution 256 \
  --num-classes 1000 \
  --workdir runs/gen_pixel_sota_B
```

For latent training, add `--cache-root /path/to/latent_cache`. ImageNet FID
still requires the reference statistics configured in `utils/env.py`.

The local fake/CIFAR configs set `train.enable_eval: false` because the release
only provides an ImageNet-256 FID reference path. Do not interpret their losses
as FID, image quality, or benchmark parity.

## Outputs

Each JAX work directory contains:

```text
<workdir>/
├── checkpoints/
├── params_ema/
│   ├── ema_params.msgpack
│   └── metadata.json
└── log/
    └── metrics.jsonl
```

The one-step configurations are intentionally small. Increase model, bank, and
batch sizes only after confirming the available memory and intended scientific
comparison.
