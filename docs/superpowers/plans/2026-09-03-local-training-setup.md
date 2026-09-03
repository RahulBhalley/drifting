# Local Training Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible local toy-model training and a configurable, resource-bounded JAX generator-training path using fake data or resized CIFAR-10 while preserving the upstream ImageNet workflow.

**Architecture:** Keep the authors' notebook immutable and execute a derived in-memory copy through a dedicated PyTorch environment. Generalize the existing dataset factory behind a source dispatcher, thread dataset metadata into model/training construction, and use separate local YAML configurations with ImageNet-only evaluation disabled.

**Tech Stack:** Python 3.10+, PyTorch, torchvision, Jupyter/nbclient, JAX, Flax, Optax, PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-09-01-local-training-setup-design.md`

## Global Constraints

- Preserve every existing file under `configs/gen/` byte-for-byte.
- Preserve `notebooks/inference_demo.ipynb` byte-for-byte.
- Save `notebooks/drifting_model_demo_original.ipynb` exactly as published by the authors.
- Keep PyTorch notebook dependencies in `.venv-toy`, separate from `.venv`.
- Keep `imagenet` as the default when `dataset.source` is absent.
- Permit `dataset.source` values `imagenet`, `fake`, and `cifar10` only.
- Treat fake/CIFAR-10 execution as a pipeline smoke test, never an ImageNet benchmark reproduction.
- Disable ImageNet FID evaluation in local alternate-dataset configurations.

---

### Task 1: Preserve and Execute the Authors' Toy Notebook

**Files:**
- Create: `notebooks/drifting_model_demo_original.ipynb`
- Create: `requirements-toy.txt`
- Create: `scripts/run_toy_notebook.py`
- Create: `tests/test_run_toy_notebook.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: the published raw notebook URL and a Python interpreter containing the dependencies in `requirements-toy.txt`.
- Produces: `prepare_notebook(source: Path) -> nbformat.NotebookNode`, `execute_notebook(source: Path, output: Path, kernel_name: str, timeout: int) -> Path`, and an executed notebook under `outputs/toy/`.

- [x] **Step 1: Download and hash the immutable notebook**

Run:

```bash
curl -L https://raw.githubusercontent.com/lambertae/lambertae.github.io/main/projects/drifting/notebooks/drifting_model_demo.ipynb -o notebooks/drifting_model_demo_original.ipynb
shasum -a 256 notebooks/drifting_model_demo_original.ipynb
```

Record the digest in `notebooks/drifting_model_demo_original.sha256`. Do not modify the downloaded JSON.

- [x] **Step 2: Write a failing runner test**

Create a minimal temporary notebook containing a `%pip install` cell and a normal code cell. Assert that `prepare_notebook` leaves the source unchanged, replaces only the install cell with a markdown explanation, and preserves the normal cell.

```python
def test_prepare_notebook_skips_only_colab_install_cell(tmp_path):
    source = tmp_path / "source.ipynb"
    output = tmp_path / "executed.ipynb"
    original = make_notebook("%pip install torch", "answer = 42")
    nbformat.write(original, source)

    prepared = prepare_notebook(source)

    assert nbformat.read(source, as_version=4).cells[0].source == "%pip install torch"
    assert prepared.cells[0].cell_type == "markdown"
    assert prepared.cells[1].source == "answer = 42"
```

- [x] **Step 3: Run the focused test and confirm RED**

Run: `.venv/bin/python -m pytest tests/test_run_toy_notebook.py -v`

Expected: collection fails because `scripts.run_toy_notebook` does not exist.

- [x] **Step 4: Implement the notebook preparation and execution helper**

The CLI accepts `--source`, `--output`, `--kernel-name`, and `--timeout`. It reads the notebook, deep-copies it, converts cells whose stripped source begins with `%pip install`, `!pip install`, or `pip install` to markdown, executes with `nbclient.NotebookClient`, and writes only the derived output.

```python
def prepare_notebook(source: Path) -> nbformat.NotebookNode:
    notebook = nbformat.read(source, as_version=4)
    prepared = copy.deepcopy(notebook)
    for index, cell in enumerate(prepared.cells):
        first_line = cell.source.lstrip().splitlines()[0] if cell.source.strip() else ""
        if first_line.startswith(("%pip install", "!pip install", "pip install")):
            prepared.cells[index] = nbformat.v4.new_markdown_cell(
                "Local dependency installation skipped; use requirements-toy.txt."
            )
    return prepared
```

- [x] **Step 5: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_run_toy_notebook.py -v`

Expected: PASS.

- [x] **Step 6: Add isolated dependency and ignore declarations**

`requirements-toy.txt` pins compatible versions of `torch`, `torchvision`, `einops`, `matplotlib`, `tqdm`, `jupyter`, `nbclient`, `nbformat`, and `ipykernel`. Add `.venv-toy/`, `outputs/`, `data/`, and `work/` to `.gitignore` without removing existing entries.

- [x] **Step 7: Create `.venv-toy` and execute the full notebook**

Run:

```bash
python3 -m venv .venv-toy
.venv-toy/bin/python -m pip install -r requirements-toy.txt
.venv-toy/bin/python scripts/run_toy_notebook.py --kernel-name python3 --output outputs/toy/drifting_model_demo_executed.ipynb
```

Inspect cell outputs and assert that both Swiss-roll and checkerboard training complete with finite `(N, 2)` samples.

- [x] **Step 8: Commit the toy workflow**

```bash
git add .gitignore notebooks/drifting_model_demo_original.ipynb notebooks/drifting_model_demo_original.sha256 requirements-toy.txt scripts/run_toy_notebook.py tests/test_run_toy_notebook.py
git commit -m "feat: add reproducible toy notebook training"
```

### Task 2: Add Configurable Fake and CIFAR-10 Dataset Sources

**Files:**
- Modify: `dataset/dataset.py`
- Create: `tests/test_dataset_sources.py`

**Interfaces:**
- Consumes: `source`, `resolution`, `batch_size`, `split`, `num_classes`, `data_root`, `download`, `fake_size`, plus existing loader options.
- Produces: `create_dataset_split(...) -> tuple[DataLoader, Callable, Callable]`; retains `create_imagenet_split(...)` unchanged as the ImageNet implementation.

- [x] **Step 1: Write failing fake-data behavior tests**

Test real loader output, preprocessing, postprocessing, validation, and pixel-only constraints:

```python
def test_fake_split_emits_requested_bhwc_images():
    loader, preprocess, postprocess = create_dataset_split(
        source="fake", resolution=16, batch_size=4, split="train",
        num_classes=10, fake_size=8, num_workers=0,
    )
    batch = preprocess(next(iter(loader)))
    assert batch["images"].shape == (4, 16, 16, 3)
    assert batch["labels"].dtype == jnp.int32
    assert float(batch["images"].min()) >= -1.0
    assert float(batch["images"].max()) <= 1.0
    assert postprocess(batch["images"]).shape == (4, 3, 16, 16)
```

Also assert unknown sources, non-positive dimensions, and fake/CIFAR with `use_latent=True` or `use_cache=True` raise `ValueError`.

- [x] **Step 2: Run focused tests and confirm RED**

Run: `.venv/bin/python -m pytest tests/test_dataset_sources.py -v`

Expected: import fails because `create_dataset_split` does not exist.

- [x] **Step 3: Implement source validation and datasets**

Add `FakeData` and `CIFAR10` imports, a shared normalized pixel transform, and dispatch without altering `create_imagenet_split` behavior.

```python
def create_dataset_split(*, source="imagenet", resolution, batch_size, split,
                         num_classes=1000, data_root="data", download=False,
                         fake_size=1024, **loader_options):
    source = source.strip().lower()
    if source == "imagenet":
        return create_imagenet_split(
            resolution=resolution, batch_size=batch_size, split=split,
            **loader_options,
        )
    _validate_pixel_source(...)
    dataset = _build_local_pixel_dataset(...)
    return _create_pixel_loader(dataset, batch_size=batch_size, split=split, ...)
```

Use distinct deterministic `FakeData` offsets for train and validation. Map CIFAR-10 `val` to torchvision's `train=False` split and resize to `resolution` before tensor conversion and normalization.

- [x] **Step 4: Verify GREEN and regression coverage**

Run: `.venv/bin/python -m pytest tests/test_dataset_sources.py -v`

Expected: all tests PASS, including an ImageNet-default dispatch test that substitutes only `_build_imagenet_dataset` to avoid requiring ImageNet.

- [x] **Step 5: Commit dataset support**

```bash
git add dataset/dataset.py tests/test_dataset_sources.py
git commit -m "feat: add configurable local datasets"
```

### Task 3: Thread Dataset Metadata Through Generator Training

**Files:**
- Modify: `utils/model_builder.py`
- Modify: `train.py`
- Create: `tests/test_local_training_config.py`
- Create: `configs/local/m1_fake_smoke.yaml`
- Create: `configs/local/m1_cifar10_smoke.yaml`
- Create: `requirements-macos-training.txt`

**Interfaces:**
- Consumes: `dataset.source`, `dataset.num_classes`, local dataset kwargs, and `train.enable_eval`.
- Produces: source-correct `dataset_name`; `train_gen(..., num_classes: int = 1000, enable_eval: bool = True)`.

- [x] **Step 1: Write failing configuration and training-contract tests**

Parse all YAML files and assert the two local configs are bounded and self-consistent. Test the public training-loop configuration contract via signature inspection and a helper that decides when evaluation runs.

```python
def test_local_configs_disable_imagenet_evaluation():
    for path in Path("configs/local").glob("*.yaml"):
        config = load_config(str(path))
        assert config.dataset.source in {"fake", "cifar10"}
        assert config.train.enable_eval is False
        assert config.dataset.num_classes == 10
        assert config.dataset.batch_size <= 16
        assert config.train.total_steps <= 2
```

Add a behavior test asserting `_should_evaluate(step=1, total_steps=1, eval_per_step=1, enable_eval=False)` is false and the same call with `enable_eval=True` is true.

- [x] **Step 2: Run focused tests and confirm RED**

Run: `.venv/bin/python -m pytest tests/test_local_training_config.py -v`

Expected: failure because local configs and `_should_evaluate` are absent.

- [x] **Step 3: Implement builder dispatch and metadata**

Replace imports/calls of `create_imagenet_split` in `utils/model_builder.py` with `create_dataset_split`, read `dataset.source` with the ImageNet default, and forward only explicit source options.

```python
source = str(config.dataset.get("source", "imagenet")).strip().lower()
dataset_options = dict(config.dataset.kwargs)
train_loader, preprocess_fn, postprocess_fn = create_dataset_split(
    source=source,
    resolution=resolution,
    num_classes=int(config.dataset.num_classes),
    batch_size=batch_size_per_node,
    split="train",
    **dataset_options,
)
```

Return `dataset_name=f"{source}{resolution}"` and `num_classes=int(config.dataset.num_classes)`.

- [x] **Step 4: Implement training-loop controls**

Add `num_classes=1000` and `enable_eval=True` parameters to `train_gen`, initialize `ArrayMemoryBank(num_classes=num_classes, ...)`, and guard all FID work through:

```python
def _should_evaluate(*, step, total_steps, eval_per_step, enable_eval):
    return bool(enable_eval) and (
        step % eval_per_step == 0 or step == 1 or step == total_steps
    )
```

Pass `model_dict.num_classes` from `main_gen` into `train_gen`.

- [x] **Step 5: Add local configurations**

Both configs use 16x16 RGB input, patch size 4, hidden size 64, depth 2, four heads, batch size 4, `num_workers: 0`, `use_mae: false`, `use_convnext: false`, `use_wandb: false`, `enable_eval: false`, one optimization step, and minimal positive/negative banks. The fake config uses `source: fake`; the CIFAR config uses `source: cifar10`, `data_root: data/cifar10`, and `download: true`.

Add `requirements-macos-training.txt` as the macOS-compatible counterpart to the TPU-oriented upstream requirements. It includes the same JAX/Flax/Optax stack already verified locally, macOS wheels `torch==2.4.0` and `torchvision==0.19.0` without the Linux-only `+cpu` suffix, and `pytest` for verification. Do not modify `requirements.txt`.

- [x] **Step 6: Verify GREEN and upstream preservation**

Run:

```bash
.venv/bin/python -m pytest tests/test_dataset_sources.py tests/test_local_training_config.py -v
git diff accd0cf --exit-code -- configs/gen notebooks/inference_demo.ipynb
```

Expected: all tests PASS and the preservation diff exits zero.

- [x] **Step 7: Commit training integration**

```bash
git add utils/model_builder.py train.py tests/test_local_training_config.py configs/local/m1_fake_smoke.yaml configs/local/m1_cifar10_smoke.yaml requirements-macos-training.txt
git commit -m "feat: add local generator training configs"
```

### Task 4: Execute Local Training and Regression Verification

**Files:**
- Modify: `README.md`
- Create: `docs/local-training.md`
- Retain intentionally: `data/cifar10/`, `outputs/toy/`, and `outputs/training-smoke/` as ignored runtime artifacts.

**Interfaces:**
- Consumes: both local configs, `.venv`, `.venv-toy`, and existing `local_inference_smoke.py` when present.
- Produces: documented commands, finite training metrics/checkpoints, verified toy outputs, and a precise reproduction-boundary report.

- [x] **Step 1: Run one-step fake-data generator training**

Run:

```bash
JAX_PLATFORMS=cpu .venv/bin/python main.py --gen --config configs/local/m1_fake_smoke.yaml --workdir outputs/training-smoke/fake
```

Require exit code zero, at least one finite `loss` value in `log/metrics.jsonl`, and a checkpoint or EMA parameters artifact.

- [x] **Step 2: Download and verify CIFAR-10 loader**

Run a loader-only script through `.venv/bin/python` using `create_dataset_split(source="cifar10", resolution=16, download=True, ...)`. Require train and validation shapes `(B, 16, 16, 3)` after preprocessing.

- [x] **Step 3: Attempt bounded CIFAR-10 training**

Run:

```bash
JAX_PLATFORMS=cpu .venv/bin/python main.py --gen --config configs/local/m1_cifar10_smoke.yaml --workdir outputs/training-smoke/cifar10
```

Stop and report precisely if the 16 GB machine cannot compile or execute the configured step; do not increase resource limits or claim a benchmark result.

- [x] **Step 4: Re-run pretrained inference regression**

Run the established one-image `pixel_B_sota`, class-95, seed-0 inference smoke command. Confirm output shape `(1, 256, 256, 3)`, finite pixels, and the expected PNG/JSON artifacts.

- [x] **Step 5: Document exact local commands and boundaries**

`docs/local-training.md` documents environment creation, toy execution, fake/CIFAR config overrides, later ImageNet replacement, artifact locations, and explicitly separates observed smoke-test execution from the authors' reported ImageNet/TPU benchmark.

- [x] **Step 6: Run comprehensive verification**

Run:

```bash
.venv/bin/python -m pytest tests -v
.venv/bin/python -m compileall dataset models utils scripts train.py main.py local_inference_smoke.py
git diff accd0cf --exit-code -- configs/gen notebooks/inference_demo.ipynb
git diff --check
```

Inspect the complete diff against the spec and confirm no temporary download/build directories remain outside ignored intentional environments, datasets, and outputs.

- [x] **Step 7: Commit documentation**

```bash
git add README.md docs/local-training.md
git commit -m "docs: document local drifting training"
```
