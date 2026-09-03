# Drifting Models PyTorch Backend Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build separately packaged JAX and PyTorch Drifting Models backends with shared scientific configuration and verified model, training, inference, checkpoint, data, evaluation, and distributed parity.

**Architecture:** A framework-free `drifting_common` package owns validated configuration and artifact contracts. `drifting_jax` packages the existing implementation as the executable oracle; `drifting_torch` is a native NCHW PyTorch implementation with explicit conversion/parity tools. Neither backend is selected implicitly.

**Tech Stack:** Python 3.10+, JAX 0.4/Flax/Optax/Orbax, PyTorch 2.4+, torchvision, Diffusers, Hugging Face Hub, safetensors, NumPy, SciPy, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-pytorch-backend-port-design.md`

## Global Constraints

- Preserve `configs/gen/*.yaml`, `configs/mae/*.yaml`, `configs/local/*.yaml`, and `notebooks/drifting_model_demo_original.ipynb` byte-for-byte.
- Keep `drifting_common` free of JAX and PyTorch imports.
- Keep JAX and PyTorch separately installable; parity installs both.
- Expose only explicit backend commands; no generic command may choose a default.
- PyTorch image tensors are NCHW internally; JAX image tensors remain NHWC.
- Backend comparison uses externally supplied identical tensors, noise labels, and MAE masks.
- Conversion rejects every missing, extra, duplicate, or shape-incompatible tensor.
- FP32 CPU reference parity is the strict gate; BF16, fused kernels, and unavailable accelerator paths have separate evidence.
- All ranks participate in collective checkpoint calls; only rank zero performs non-collective logging and image writes.
- Full ImageNet-50K metric reproduction is not claimed without ImageNet, reference statistics, and suitable accelerators.

---

### Task 1: Package Foundation and Backend-Neutral Contracts

**Files:**
- Create: `pyproject.toml`
- Create: `src/drifting_common/__init__.py`
- Create: `src/drifting_common/config/__init__.py`
- Create: `src/drifting_common/config/schema.py`
- Create: `src/drifting_common/config/loader.py`
- Create: `src/drifting_common/artifacts/__init__.py`
- Create: `src/drifting_common/artifacts/schema.py`
- Create: `src/drifting_common/artifacts/hashing.py`
- Create: `src/drifting_common/data_contracts/__init__.py`
- Create: `src/drifting_common/data_contracts/types.py`
- Create: `src/drifting_common/metrics/__init__.py`
- Create: `src/drifting_common/metrics/types.py`
- Create: `configs/runtime/jax/cpu.yaml`
- Create: `configs/runtime/torch/cpu.yaml`
- Create: `configs/runtime/torch/mps.yaml`
- Create: `configs/runtime/torch/cuda.yaml`
- Test: `tests/common/test_import_isolation.py`
- Test: `tests/common/test_config.py`
- Test: `tests/common/test_artifact_schema.py`

**Interfaces:**
- Produces: `compose_config(scientific_path: Path, runtime_path: Path | None, overrides: Sequence[str]) -> ExperimentConfig`
- Produces: `ArtifactManifest.from_json(path: Path) -> ArtifactManifest`
- Produces: `ArtifactManifest.write(path: Path) -> None`
- Produces: `sha256_file(path: Path) -> str`
- Produces: immutable `DatasetContract` and `MetricResult` dataclasses.

- [x] **Step 1: Write import-isolation and configuration tests**

```python
def test_common_imports_no_tensor_backend(monkeypatch):
    for name in list(sys.modules):
        if name == "drifting_common" or name.startswith("drifting_common."):
            sys.modules.pop(name)
    import drifting_common
    assert "jax" not in sys.modules
    assert "torch" not in sys.modules

def test_compose_config_precedence(tmp_path):
    cfg = compose_config(SCIENTIFIC, RUNTIME, ["train.total_steps=3"])
    assert cfg.train.total_steps == 3
    assert cfg.runtime.device == "cpu"
```

- [x] **Step 2: Run the common tests and verify failure**

Run: `.venv-training/bin/python -m pytest -q tests/common`

Expected: collection fails because `drifting_common` does not exist.

- [x] **Step 3: Add packaging and typed configuration composition**

```python
def compose_config(scientific_path, runtime_path=None, overrides=()):
    raw = _read_yaml(scientific_path)
    if runtime_path is not None:
        raw = deep_merge(raw, {"runtime": _read_yaml(runtime_path)})
    raw = apply_dot_overrides(raw, overrides)
    return ExperimentConfig.from_mapping(raw)
```

Define explicit `DatasetConfig`, `ModelConfig`, `OptimizerConfig`,
`TrainingConfig`, `LoggingConfig`, and `RuntimeConfig` validation. Preserve
unknown model/feature fields in typed mappings, but reject unknown top-level,
dataset, optimizer, training, logging, and runtime keys. Treat `hsdp_dim` as a
legacy JAX-only input field. The loader composes scientific YAML, runtime YAML,
then CLI overrides in that order.

- [x] **Step 4: Add versioned artifact and result records**

```python
@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: int
    kind: Literal["generator", "mae", "training_state"]
    backend: Literal["jax", "torch"]
    model_config: dict[str, JSONValue]
    step: int
    ema_decay: float | None
    files: dict[str, ArtifactFile]
    source: ArtifactSource | None = None
    conversion: ConversionRecord | None = None
```

Write JSON atomically with a sibling temporary file followed by `os.replace`.
Validate schema version, hashes, relative paths, and non-negative step.

- [x] **Step 5: Run tests and packaging checks**

Run: `uv pip install --python .venv-training/bin/python -e . --no-deps`

Run: `.venv-training/bin/python -m pytest -q tests/common`

Run: `.venv-training/bin/python -c "import drifting_common; import sys; assert 'jax' not in sys.modules and 'torch' not in sys.modules"`

Expected: all pass.

- [x] **Step 6: Commit**

```bash
git add pyproject.toml src/drifting_common configs/runtime tests/common
git commit -m "feat: add backend-neutral package contracts"
```

---

### Task 2: Package the JAX Oracle and Preserve Compatibility

**Files:**
- Create: `src/drifting_jax/__init__.py`
- Create: `src/drifting_jax/models/{generator,mae,convnext,hf}.py`
- Create: `src/drifting_jax/data/{datasets,latent,vae}.py`
- Create: `src/drifting_jax/training/{generator,mae,builder}.py`
- Create: `src/drifting_jax/evaluation/{fid,inception,precision_recall,resize,frechet,weights}.py`
- Create: `src/drifting_jax/checkpointing/{checkpoint,initialize}.py`
- Create: `src/drifting_jax/distributed.py`
- Create: `src/drifting_jax/{inference,logging,runtime,cli}.py`
- Modify: `pyproject.toml`
- Modify: `main.py`
- Modify: `train.py`
- Modify: `train_mae.py`
- Modify: `inference.py`
- Modify: `models/*.py`
- Modify: `dataset/*.py`
- Modify: `utils/*.py`
- Modify: `utils/jax_fid/*.py`
- Test: `tests/jax/test_packaged_imports.py`
- Test: `tests/jax/test_cli.py`
- Move: current tests to `tests/jax/` while retaining their assertions.

**Interfaces:**
- Consumes: `ExperimentConfig` and artifact schemas from Task 1.
- Produces: `drifting_jax.training.generator.main_gen(config, output_dir)`
- Produces: `drifting_jax.training.mae.main_mae(config, output_dir)`
- Produces: `drifting_jax.inference.generate(...)`
- Produces explicit console commands `drifting-jax-train`, `drifting-jax-infer`, and `drifting-jax-cache`.

- [x] **Step 1: Write package and CLI regression tests**

```python
def test_jax_entrypoints_are_explicit():
    scripts = {
        entry.name
        for entry in metadata.entry_points(group="console_scripts")
        if entry.dist and entry.dist.name == "drifting-models"
    }
    assert {"drifting-jax-train", "drifting-jax-infer", "drifting-jax-cache"} <= scripts
    assert "drifting-train" not in scripts
    assert callable(jax_cli.train_main)
    assert callable(jax_cli.infer_main)

def test_legacy_generator_is_packaged_generator():
    from models.generator import DitGen as legacy
    from drifting_jax.models.generator import DitGen
    assert legacy is DitGen
```

- [x] **Step 2: Run tests and verify failure**

Run: `JAX_PLATFORMS=cpu .venv-training/bin/python -m pytest -q tests/jax/test_packaged_imports.py tests/jax/test_cli.py`

Expected: fails because `drifting_jax` is absent.

- [x] **Step 3: Move implementation into focused package modules**

Use `git mv` for the authoritative implementations, rewrite internal imports to
absolute `drifting_jax.*` or `drifting_common.*` imports, and leave thin root
compatibility modules containing only imports and `main()` delegation:

```python
from drifting_jax.models.generator import *  # noqa: F403
```

Root compatibility modules are not registered as generic console commands.

- [x] **Step 4: Add external-noise and external-mask oracle hooks**

```python
def __call__(self, c, cfg_scale=1.0, temp=1.0, deterministic=True,
             train=False, noise=None, noise_labels=None):
    if noise is None or noise_labels is None:
        noise, noise_labels = self.sample_noise(c, temp=temp)
    return self.forward_with_noise(c, cfg_scale, noise, noise_labels, deterministic)
```

Add an optional explicit MAE mask without changing behavior when omitted. Test
that native RNG and explicit-input paths return identical values when fed the
captured native samples.

- [x] **Step 5: Run the complete split-environment JAX baseline**

Run: `JAX_PLATFORMS=cpu .venv-training/bin/python -m pytest -q tests/common tests/jax --ignore=tests/jax/test_run_toy_notebook.py --ignore=tests/jax/test_packaged_imports.py`

Run: `JAX_PLATFORMS=cpu .venv-training/bin/python -m pytest -q tests/jax/test_packaged_imports.py`

Run: `.venv-toy/bin/python -m pytest -q tests/jax/test_run_toy_notebook.py`

Expected: 27 existing assertions plus new package/CLI/oracle-hook assertions pass.

- [x] **Step 6: Run the existing local JAX inference smoke**

Run: `JAX_PLATFORMS=cpu PYTHONPATH=src .venv-training/bin/python local_inference_smoke.py`

Expected: finite `(1, 256, 256, 3)` pixel output and JSON report. The script
reuses the repository-local artifact cache when present and writes its report
under the ignored `outputs/inference-smoke/` directory.

- [x] **Step 7: Commit**

```bash
git add src/drifting_jax main.py train.py train_mae.py inference.py models dataset utils tests/jax pyproject.toml
git commit -m "refactor: package jax reference backend"
```

---

### Task 3: PyTorch Generator and Mathematical Primitives

**Files:**
- Create: `src/drifting_torch/__init__.py`
- Create: `src/drifting_torch/models/__init__.py`
- Create: `src/drifting_torch/models/primitives.py`
- Create: `src/drifting_torch/models/generator.py`
- Create: `src/drifting_torch/runtime.py`
- Test: `tests/torch/test_generator_primitives.py`
- Test: `tests/torch/test_generator.py`
- Test: `tests/parity/test_generator_primitives.py`

**Interfaces:**
- Produces: `RMSNorm`, `apply_rope`, `modulate`, `TimestepEmbedder`.
- Produces: `DitGen.forward(labels, cfg_scale=1.0, temp=1.0, noise=None, noise_labels=None) -> GenerationOutput`.
- Produces: `build_generator(model_config: Mapping[str, Any]) -> DitGen`.
- `GenerationOutput.samples` is NCHW and `GenerationOutput.noise` records NCHW Gaussian noise plus discrete labels.

- [x] **Step 1: Write primitive and shape tests**

```python
def test_modulate_broadcasts_over_tokens():
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    shift = torch.ones(2, 4)
    scale = torch.full((2, 4), 0.5)
    torch.testing.assert_close(modulate(x, shift, scale), x * 1.5 + 1)

def test_generator_explicit_noise_is_repeatable(tiny_generator):
    out1 = tiny_generator(LABELS, noise=NOISE, noise_labels=NOISE_LABELS)
    out2 = tiny_generator(LABELS, noise=NOISE, noise_labels=NOISE_LABELS)
    torch.testing.assert_close(out1.samples, out2.samples, rtol=0, atol=0)
```

- [x] **Step 2: Verify tests fail**

Run: `.venv-training/bin/python -m pytest -q tests/torch/test_generator_primitives.py tests/torch/test_generator.py`

Expected: import failure for `drifting_torch.models`.

- [x] **Step 3: Implement transparent PyTorch primitives and DiT blocks**

```python
def apply_rope(q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    half = q.shape[-1] // 2
    freq = 1.0 / (10000 ** (torch.arange(half, device=q.device) / half))
    phase = torch.outer(torch.arange(q.shape[1], device=q.device), freq)
    phase = torch.cat((phase, phase), dim=-1)[None, :, None, :]
    return q * phase.cos() + rotate_half(q) * phase.sin(), \
           k * phase.cos() + rotate_half(k) * phase.sin()
```

Implement the complete LightningDiT path with explicit QK scaling, matmul,
softmax, value aggregation, AdaLN gates,
SwiGLU sizing, class tokens, patch transforms, zero initialization, CFG
embedding, noise embeddings, activation checkpointing, and FP32 attention.

- [x] **Step 4: Add deterministic tiny-model JAX/PyTorch primitive parity**

Use fixed NumPy arrays converted into each backend. Compare sinusoidal
embeddings, RMSNorm, RoPE, modulation, patchify, and unpatchify after explicit
NHWC/NCHW conversion. Start at `rtol=1e-5, atol=1e-6`; record actual maxima.

- [x] **Step 5: Run targeted tests**

Run: `JAX_PLATFORMS=cpu .venv-training/bin/python -m pytest -q tests/torch/test_generator_primitives.py tests/torch/test_generator.py tests/parity/test_generator_primitives.py`

Expected: pass with finite gradients and recorded FP32 errors below thresholds.

- [x] **Step 6: Commit**

```bash
git add src/drifting_torch/models src/drifting_torch/runtime.py tests/torch tests/parity/test_generator_primitives.py pyproject.toml
git commit -m "feat: port drifting generator to pytorch"
```

---

### Task 4: JAX Checkpoint Conversion and Generator Parity

**Files:**
- Create: `src/drifting_torch/checkpointing/__init__.py`
- Create: `src/drifting_torch/checkpointing/mapping.py`
- Create: `src/drifting_torch/checkpointing/converter.py`
- Create: `src/drifting_torch/checkpointing/artifact.py`
- Create: `src/drifting_torch/parity.py`
- Create: `tools/convert_checkpoint.py`
- Create: `tools/compare_backends.py`
- Test: `tests/torch/test_checkpoint_mapping.py`
- Test: `tests/parity/test_generator_model.py`
- Test: `tests/parity/test_official_generator.py`

**Interfaces:**
- Produces: `convert_jax_generator(source: Path, destination: Path) -> ConversionReport`.
- Produces: `load_torch_generator(source: str | Path, device: torch.device) -> LoadedGenerator`.
- Produces: `compare_tensors(reference, candidate, policy) -> TensorComparison`.

- [x] **Step 1: Write exhaustive mapping tests**

```python
def test_dense_kernel_transposes():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    got = convert_leaf("Dense_0/kernel", src, target_shape=(4, 3))
    np.testing.assert_array_equal(got, src.T)

def test_converter_rejects_unmapped_source():
    with pytest.raises(ConversionError, match="unmapped source"):
        map_state({"unknown/kernel": np.ones((2, 2))}, EXPECTED_TARGET)
```

- [x] **Step 2: Verify tests fail**

Run: `.venv-training/bin/python -m pytest -q tests/torch/test_checkpoint_mapping.py`

Expected: converter imports fail.

- [x] **Step 3: Implement named mapping and safe artifact export**

```python
def convert_leaf(rule: TensorRule, value: np.ndarray) -> np.ndarray:
    if rule.layout == "dense_io_to_oi":
        value = value.T
    elif rule.layout == "conv_hwio_to_oihw":
        value = value.transpose(3, 2, 0, 1)
    if tuple(value.shape) != rule.target_shape:
        raise ConversionError(rule.describe_shape_error(value.shape))
    return np.ascontiguousarray(value)
```

Export EMA weights as safetensors, write the Task 1 manifest and SHA-256 values,
and refuse output publication until the model accepts the entire mapped state
with no missing or unexpected keys.

- [x] **Step 4: Add tiny full-generator parity**

Initialize a tiny JAX generator, convert its full tree, supply fixed noise and
noise labels, and compare conditioning, every block output, pre-unpatchified
output, and final NCHW result. Use hooks/debug returns only in test mode.

- [x] **Step 5: Convert and compare official pixel and latent artifacts**

Run:

```bash
JAX_PLATFORMS=cpu .venv-training/bin/python tools/convert_checkpoint.py \
  --kind generator --source hf://pixel_B_sota --output work/converted/pixel_B_sota
JAX_PLATFORMS=cpu .venv-training/bin/python tools/compare_backends.py \
  --artifact work/converted/pixel_B_sota --labels 95 --seed 0 --report work/parity/pixel_B_sota.json
```

Repeat for `hf://latent_B_sota`. Capture shared noise as NumPy and compare raw
model outputs after layout conversion. Compare pixel postprocessing here;
latent VAE decoding parity is completed in Task 8 after both codecs exist.

- [x] **Step 6: Set empirical tolerance policy**

Write measured FP32 CPU thresholds into a versioned JSON policy under
`tests/parity/policies/`. The policy includes max/mean absolute error,
non-finite counts, cosine similarity, PSNR, SSIM, and uint8 mismatch bounds.
Tests must fail closed when a metric is absent or non-finite.

- [x] **Step 7: Run converter and parity tests**

Run: `JAX_PLATFORMS=cpu .venv-training/bin/python -m pytest -q tests/torch/test_checkpoint_mapping.py tests/parity/test_generator_model.py tests/parity/test_official_generator.py`

Expected: all mappings consumed exactly once and official comparisons satisfy
the measured policy.

- [x] **Step 8: Commit**

```bash
git add src/drifting_torch/checkpointing src/drifting_torch/parity.py tools tests/torch/test_checkpoint_mapping.py tests/parity
git commit -m "feat: convert and validate jax generator checkpoints"
```

---

### Task 5: PyTorch MAE-ResNet and Activation Parity

**Files:**
- Create: `src/drifting_torch/models/mae.py`
- Test: `tests/torch/test_mae.py`
- Test: `tests/parity/test_mae.py`
- Modify: `src/drifting_torch/checkpointing/mapping.py`
- Modify: `tools/convert_checkpoint.py`

**Interfaces:**
- Produces: `MAEResNet.forward(images, labels, mask=None, ...) -> MAEOutput`.
- Produces: `MAEResNet.get_activations(images, ...) -> dict[str, Tensor]` with values shaped `(B,T,C)`.
- Extends: `convert_jax_mae(source, destination) -> ConversionReport`.

- [x] **Step 1: Write MAE shape, loss, and activation-key tests**

```python
def test_mae_activation_contract(tiny_mae):
    out = tiny_mae.get_activations(torch.randn(2, 3, 32, 32), every_k_block=2)
    assert {"norm_x", "conv1", "layer1", "layer4"} <= out.keys()
    assert all(value.ndim == 3 for value in out.values())
```

- [x] **Step 2: Verify tests fail**

Run: `.venv-training/bin/python -m pytest -q tests/torch/test_mae.py tests/parity/test_mae.py`

Expected: MAE module missing.

- [x] **Step 3: Implement NCHW ResNet encoder and U-Net decoder**

```python
def patch_input(x: Tensor, patch: int) -> Tensor:
    return rearrange(x, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=patch, p2=patch)
```

Match convolution padding/stride/bias, dynamic GroupNorm groups, dropout,
projection skips, bilinear resize convention, decoder concatenation, logits,
mask-weighted reconstruction loss, and optional classification loss.

- [x] **Step 4: Implement complete activation extraction**

Preserve all released keys and `(B,T,C)` values for raw, per-block, global
mean/std, and patch mean/std outputs. Use population standard deviation in FP32
with the JAX epsilon placement.

- [x] **Step 5: Extend conversion and compare an official MAE artifact**

Convert `hf://mae_latent_256` first, supply identical images and explicit masks,
and compare encoder stages, decoder output, logits, losses, and activation
dictionaries. Then run a shape/mapping conversion gate for the 640-channel
pixel and latent artifacts.

- [x] **Step 6: Run tests**

Run: `JAX_PLATFORMS=cpu .venv-training/bin/python -m pytest -q tests/torch/test_mae.py tests/parity/test_mae.py`

Expected: all MAE outputs and official converted activations satisfy policy.

- [x] **Step 7: Commit**

```bash
git add src/drifting_torch/models/mae.py src/drifting_torch/checkpointing tools/convert_checkpoint.py tests/torch/test_mae.py tests/parity/test_mae.py
git commit -m "feat: port mae feature encoder to pytorch"
```

---

### Task 6: ConvNeXtV2 and Feature Composition

**Files:**
- Create: `src/drifting_torch/models/convnext.py`
- Create: `src/drifting_torch/models/features.py`
- Test: `tests/torch/test_features.py`
- Test: `tests/parity/test_convnext.py`

**Interfaces:**
- Produces: `build_activation_function(config, postprocess_fn) -> FrozenFeatureExtractor`.
- `FrozenFeatureExtractor.forward(images) -> dict[str, Tensor]` returns detached reference activations only when explicitly requested; generated-image activations remain differentiable.

- [x] **Step 1: Write freezing and gradient-flow tests**

```python
def test_feature_weights_frozen_but_input_gradient_flows(extractor):
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    sum(v.sum() for v in extractor(x).values()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is None for p in extractor.parameters())
```

- [x] **Step 2: Verify tests fail**

Run: `.venv-training/bin/python -m pytest -q tests/torch/test_features.py tests/parity/test_convnext.py`

Expected: feature modules missing.

- [x] **Step 3: Implement native ConvNeXtV2 activation adapter**

Load the same Hugging Face model revision as the JAX conversion path, freeze its
parameters, preserve normalization/preprocessing and activation names, and
return `(B,T,C)` values.

- [x] **Step 4: Implement MAE plus ConvNeXt composition**

```python
def forward(self, images: Tensor) -> dict[str, Tensor]:
    outputs = {}
    if self.mae is not None:
        outputs.update(self.mae.get_activations(images, **self.mae_options))
    if self.convnext is not None:
        outputs.update({f"convnext/{k}": v for k, v in self.convnext(images).items()})
    return outputs
```

- [x] **Step 5: Compare fixed-image ConvNeXt activations**

Run JAX-converted and native PyTorch paths on the same postprocessed images.
Compare every requested stage after layout/token conversion.

- [x] **Step 6: Run tests and commit**

Run: `JAX_PLATFORMS=cpu .venv-training/bin/python -m pytest -q tests/torch/test_features.py tests/parity/test_convnext.py`

```bash
git add src/drifting_torch/models tests/torch/test_features.py tests/parity/test_convnext.py
git commit -m "feat: port drifting feature extraction"
```

---

### Task 7: Drifting Loss and Serializable Memory Banks

**Files:**
- Create: `src/drifting_torch/loss.py`
- Create: `src/drifting_torch/memory_bank.py`
- Test: `tests/torch/test_loss.py`
- Test: `tests/torch/test_memory_bank.py`
- Test: `tests/parity/test_loss.py`

**Interfaces:**
- Produces: `drift_loss(gen, fixed_pos, fixed_neg, weight_gen, weight_pos, weight_neg, R_list) -> tuple[Tensor, dict[str, Tensor]]`.
- Produces: `ClassMemoryBank.add`, `.sample`, `.state_dict`, and `.load_state_dict`.

- [x] **Step 1: Write forward, gradient, and ring-buffer tests**

```python
def test_drift_loss_target_is_stopped():
    loss, _ = drift_loss(GEN, POS.requires_grad_(), NEG.requires_grad_(), WG, WP, WN, (0.2, 0.05))
    loss.mean().backward()
    assert GEN.grad is not None
    assert POS.grad is None and NEG.grad is None

def test_memory_bank_round_trip(bank):
    restored = ClassMemoryBank.from_state_dict(bank.state_dict())
    assert restored.state_dict_equal(bank)
```

- [x] **Step 2: Verify tests fail**

Run: `.venv-training/bin/python -m pytest -q tests/torch/test_loss.py tests/torch/test_memory_bank.py tests/parity/test_loss.py`

Expected: loss and memory-bank modules missing.

- [x] **Step 3: Port the loss in the same operation order**

```python
dist = (gen[:, :, None] - targets[:, None]).square().sum(dim=-1).clamp_min(eps).sqrt()
affinity = (torch.softmax(-dist_normed / radius, -1) *
            torch.softmax(-dist_normed / radius, -2)).clamp_min(1e-6).sqrt()
```

Construct the target inside `torch.no_grad()`, then compute MSE from the live
generated features to the stopped target. Preserve every reduction axis and
`loss_<radius>` metric.

- [x] **Step 4: Implement deterministic memory banks**

Use CPU tensors and a private `torch.Generator`. Validate labels, shape, dtype,
and capacity; sample only initialized slots; include generator state in
serialization.

- [x] **Step 5: Compare JAX/PyTorch loss values and gradients**

Use float64 diagnostic fixtures and float32 production fixtures with unequal
positive/negative counts, multiple radii, small scale, and non-uniform weights.

- [x] **Step 6: Run tests and commit**

Run: `JAX_PLATFORMS=cpu .venv-training/bin/python -m pytest -q tests/torch/test_loss.py tests/torch/test_memory_bank.py tests/parity/test_loss.py`

```bash
git add src/drifting_torch/loss.py src/drifting_torch/memory_bank.py tests/torch tests/parity/test_loss.py
git commit -m "feat: port drifting loss and memory banks"
```

---

### Task 8: PyTorch Data, VAE, and Latent Cache

**Files:**
- Create: `src/drifting_torch/data/__init__.py`
- Create: `src/drifting_torch/data/datasets.py`
- Create: `src/drifting_torch/data/transforms.py`
- Create: `src/drifting_torch/data/latent.py`
- Create: `src/drifting_torch/data/vae.py`
- Create: `src/drifting_torch/data/cli.py`
- Test: `tests/torch/test_data.py`
- Test: `tests/torch/test_latent_cache.py`
- Test: `tests/parity/test_preprocessing.py`

**Interfaces:**
- Produces: `create_dataset_split(config, runtime, split) -> DataPipeline`.
- Produces: `VAECodec.encode(images, noise=None) -> Tensor` and `.decode(latents) -> Tensor`.
- Produces: `build_latent_cache(config) -> CacheManifest`.

- [x] **Step 1: Write dataset dispatch and preprocessing tests**

```python
@pytest.mark.parametrize("source", ["fake", "cifar10", "imagenet"])
def test_dataset_pipeline_returns_nchw(source, fixture_root):
    pipeline = create_dataset_split(CONFIGS[source], CPU_RUNTIME, "train")
    batch = pipeline.preprocess(next(iter(pipeline.loader)))
    assert batch.images.ndim == 4 and batch.images.shape[1] in (3, 4)
    assert batch.labels.dtype == torch.int64
```

- [x] **Step 2: Verify tests fail**

Run: `.venv-training/bin/python -m pytest -q tests/torch/test_data.py tests/torch/test_latent_cache.py tests/parity/test_preprocessing.py`

Expected: data modules missing.

- [x] **Step 3: Port loaders and deterministic transforms**

Preserve ADM center crop, resize interpolation, random-resized-crop parameters,
flip policy, normalization, split shuffle/drop-last behavior, path overrides,
worker seeding, and legacy latent-cache loading. Use a stateful sampler with
serializable epoch/cursor state.

- [x] **Step 4: Implement native PyTorch VAE codec**

Use Diffusers `AutoencoderKL` with the same model revision, posterior sampling,
latent scaling, flips, pixel range, and decode clamp as the JAX path. Allow
explicit posterior noise for parity.

- [x] **Step 5: Implement atomic cache building and manifesting**

Write each `.pt` entry to a same-directory temporary file and publish with
`os.replace`. Record split counts, relative paths, source metadata, VAE
revision, dtype, shape, and hashes in the cache manifest. Resume skips only
entries whose manifest and hash validate.

- [x] **Step 6: Compare fixed preprocessing and VAE fixtures**

Compare crop pixels exactly, normalized tensors after layout conversion, latent
moments, explicit-noise latent samples, and decoded images.

- [x] **Step 7: Run tests and commit**

Run: `JAX_PLATFORMS=cpu .venv-training/bin/python -m pytest -q tests/torch/test_data.py tests/torch/test_latent_cache.py tests/parity/test_preprocessing.py`

```bash
git add src/drifting_torch/data tests/torch/test_data.py tests/torch/test_latent_cache.py tests/parity/test_preprocessing.py pyproject.toml
git commit -m "feat: port datasets and latent cache to pytorch"
```

---

### Task 9: PyTorch Generator Training and Exact Resume

**Files:**
- Create: `src/drifting_torch/training/__init__.py`
- Create: `src/drifting_torch/training/state.py`
- Create: `src/drifting_torch/training/schedules.py`
- Create: `src/drifting_torch/training/generator.py`
- Create: `src/drifting_torch/training/engine.py`
- Create: `src/drifting_torch/logging.py`
- Create: `src/drifting_torch/checkpointing/training.py`
- Test: `tests/torch/test_schedules.py`
- Test: `tests/torch/test_generator_step.py`
- Test: `tests/torch/test_generator_resume.py`
- Test: `tests/parity/test_generator_step.py`

**Interfaces:**
- Produces: `generator_train_step(state, batch, banks, feature_extractor, options) -> StepResult`.
- Produces: `train_generator(config, runtime, workdir) -> TrainingSummary`.
- Produces: `save_training_state` and `load_training_state` with completed-step semantics.

- [ ] **Step 1: Write schedule, step, and resume tests**

```python
def test_resume_matches_uninterrupted(run_three_steps, run_one_then_resume):
    direct = run_three_steps()
    resumed = run_one_then_resume(final_step=3)
    assert_state_dict_close(direct.model, resumed.model, rtol=0, atol=0)
    assert_state_dict_close(direct.ema, resumed.ema, rtol=0, atol=0)
    assert direct.banks.state_dict_equal(resumed.banks)
```

- [ ] **Step 2: Verify tests fail**

Run: `.venv-training/bin/python -m pytest -q tests/torch/test_schedules.py tests/torch/test_generator_step.py tests/torch/test_generator_resume.py`

Expected: training modules missing.

- [ ] **Step 3: Implement matching schedules and AdamW setup**

```python
def learning_rate(step, *, base_lr, warmup_steps, total_steps, schedule):
    if step < warmup_steps:
        return 1e-6 + (base_lr - 1e-6) * step / max(warmup_steps, 1)
    if schedule == "const":
        return base_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * (1e-6 + (1 - 1e-6) * 0.5 * (1 + math.cos(math.pi * progress)))
```

Configure AdamW beta, epsilon, weight decay, and LR assignment to match Optax.

- [ ] **Step 4: Implement generator step and loop**

Preserve CFG power-law sampling, no-CFG fraction, class repetition, stopped
reference features, differentiable generated features, per-feature drifting
loss aggregation, global-norm clipping, optimizer step, EMA order, memory-bank
push/fill policy, timing metrics, checkpoints, sanity evaluation, CFG sweep,
and best-FID reporting.

- [ ] **Step 5: Implement exact completed-step checkpoints**

Store model, EMA, optimizer, scheduler, scaler, completed step, seed streams,
CPU/CUDA/MPS RNG where available, sampler state, both memory banks, and config
trajectory hash. Validate compatibility before restoration.

- [ ] **Step 6: Compare one JAX/PyTorch optimizer transition**

Use converted tiny generator/feature weights, identical fixed images, labels,
banks, CFG values, noise, and optimizer state. Compare loss terms, gradient
norm, selected parameter gradients, updated weights, and EMA.

- [ ] **Step 7: Run fake and CIFAR smoke training**

Run:

```bash
drifting-torch-train --config configs/local/m1_fake_smoke.yaml \
  --runtime configs/runtime/torch/cpu.yaml --workdir work/smoke/fake
drifting-torch-train --config configs/local/m1_cifar10_smoke.yaml \
  --runtime configs/runtime/torch/mps.yaml --workdir work/smoke/cifar10
```

Expected: at least one optimizer step, finite metrics, native checkpoint, EMA
artifact, and successful resume/inference load.

- [ ] **Step 8: Run tests and commit**

Run: `JAX_PLATFORMS=cpu .venv-training/bin/python -m pytest -q tests/torch/test_schedules.py tests/torch/test_generator_step.py tests/torch/test_generator_resume.py tests/parity/test_generator_step.py`

```bash
git add src/drifting_torch/training src/drifting_torch/checkpointing/training.py src/drifting_torch/logging.py tests/torch tests/parity/test_generator_step.py pyproject.toml
git commit -m "feat: add pytorch generator training and resume"
```

---

### Task 10: PyTorch MAE Training and State Parity

**Files:**
- Create: `src/drifting_torch/training/mae.py`
- Test: `tests/torch/test_mae_step.py`
- Test: `tests/torch/test_mae_resume.py`
- Test: `tests/parity/test_mae_step.py`

**Interfaces:**
- Produces: `mae_train_step(state, batch, mask, options) -> StepResult`.
- Produces: `evaluate_mae(state, loader, options) -> dict[str, float]`.
- Produces: `train_mae(config, runtime, workdir) -> TrainingSummary`.

- [ ] **Step 1: Write MAE transition and finetune-ramp tests**

```python
def test_classifier_weight_ramp():
    assert classifier_weight(97, total=100, finetune_steps=4, warmup=2, target=0.1) == 0.05
    assert classifier_weight(99, total=100, finetune_steps=4, warmup=2, target=0.1) == 0.1
```

- [ ] **Step 2: Verify tests fail**

Run: `.venv-training/bin/python -m pytest -q tests/torch/test_mae_step.py tests/torch/test_mae_resume.py tests/parity/test_mae_step.py`

Expected: MAE trainer missing.

- [ ] **Step 3: Implement MAE train/evaluation loops**

Preserve masking ranges, reconstruction/classification metrics, no-mask
evaluation, EMA/non-EMA evaluation, final classifier finetune ramp, profiling,
logging, checkpoint cadence, and EMA artifact export.

- [ ] **Step 4: Compare one optimizer transition and resume**

Use converted tiny weights, an explicit mask, identical labels/images and AdamW
state. Compare loss components, gradients, updated weights, EMA, and three-step
uninterrupted versus resumed state.

- [ ] **Step 5: Run tests and commit**

Run: `JAX_PLATFORMS=cpu .venv-training/bin/python -m pytest -q tests/torch/test_mae_step.py tests/torch/test_mae_resume.py tests/parity/test_mae_step.py`

```bash
git add src/drifting_torch/training/mae.py tests/torch/test_mae_step.py tests/torch/test_mae_resume.py tests/parity/test_mae_step.py
git commit -m "feat: add pytorch mae training"
```

---

### Task 11: PyTorch Inference and Native Artifacts

**Files:**
- Create: `src/drifting_torch/inference.py`
- Create: `src/drifting_torch/cli.py`
- Test: `tests/torch/test_inference.py`
- Test: `tests/torch/test_artifact_loading.py`
- Create: `notebooks/torch/inference_demo.ipynb`

**Interfaces:**
- Produces: `generate(request: InferenceRequest) -> InferenceResult`.
- Produces explicit `drifting-torch-infer` CLI.
- Loads `hf://`, native directory, and explicitly converted JAX artifact sources.

- [ ] **Step 1: Write inference request and artifact validation tests**

```python
def test_inference_metadata_records_reproducibility_fields(result):
    assert result.metadata.keys() >= {
        "artifact_sha256", "class_ids", "cfg_scale", "temperature",
        "seed", "device", "precision", "sample_shape", "sample_finite"
    }
```

- [ ] **Step 2: Verify tests fail**

Run: `.venv-training/bin/python -m pytest -q tests/torch/test_inference.py tests/torch/test_artifact_loading.py`

Expected: inference module missing.

- [ ] **Step 3: Implement inference loading and execution**

Validate artifact hashes/configuration before construction, resolve device and
precision explicitly, use inference mode/autocast safely, accept labels/CFG/
temperature/seed or external noise, and write images/raw tensors/JSON metadata
atomically.

- [ ] **Step 4: Build and execute the PyTorch notebook**

The notebook installs the torch extra, downloads or converts an official
artifact, generates the same label set as the JAX notebook, shows outputs, and
records finite shape/statistics. Execute a derived copy; retain the source
notebook unexecuted and deterministic.

- [ ] **Step 5: Compare official inference outputs**

Run explicit JAX and PyTorch commands with shared captured noise for class 95
and a multi-label batch. Validate raw, postprocessed float, and uint8 policy.

- [ ] **Step 6: Run tests and commit**

Run: `.venv-training/bin/python -m pytest -q tests/torch/test_inference.py tests/torch/test_artifact_loading.py`

```bash
git add src/drifting_torch/inference.py src/drifting_torch/cli.py notebooks/torch tests/torch pyproject.toml
git commit -m "feat: add pytorch inference workflow"
```

---

### Task 12: PyTorch FID, Inception Score, Precision, and Recall

**Files:**
- Create: `src/drifting_torch/evaluation/__init__.py`
- Create: `src/drifting_torch/evaluation/inception.py`
- Create: `src/drifting_torch/evaluation/statistics.py`
- Create: `src/drifting_torch/evaluation/precision_recall.py`
- Create: `src/drifting_torch/evaluation/evaluator.py`
- Test: `tests/torch/test_evaluation.py`
- Test: `tests/parity/test_evaluation.py`

**Interfaces:**
- Produces: `compute_statistics(samples, valid_mask, options) -> InceptionStatistics`.
- Produces: `evaluate_generator(...) -> MetricResult`.

- [ ] **Step 1: Write finite, masking, and duplicate-removal tests**

```python
def test_padding_does_not_change_statistics(evaluator, images):
    base = evaluator.statistics(images, torch.ones(len(images), dtype=torch.bool))
    padded = torch.cat((images, torch.zeros_like(images[:3])))
    mask = torch.tensor([True] * len(images) + [False] * 3)
    assert_statistics_close(base, evaluator.statistics(padded, mask))
```

- [ ] **Step 2: Verify tests fail**

Run: `.venv-training/bin/python -m pytest -q tests/torch/test_evaluation.py tests/parity/test_evaluation.py`

Expected: evaluation modules missing.

- [ ] **Step 3: Port the exact released Inception/evaluation semantics**

Match resize, input range, convolution/pool padding, feature/logit selection,
float64 mean/covariance, Frechet calculation, fixed IS permutation/splits, and
the released manifold precision/recall algorithm. Convert the same Inception
weights by named rules rather than substituting an unverified library metric.

- [ ] **Step 4: Add duplicate-safe distributed aggregation**

Gather validity masks with logits/features, apply masks before truncating to
`num_samples`, and calculate/log metrics only once. Preview images are rank-zero
side effects.

- [ ] **Step 5: Compare JAX/PyTorch features and metrics**

Use deterministic image fixtures to compare resize output, selected internal
Inception activations, final logits/features, means, covariances, FID, IS, and
precision/recall. Use a singular-covariance fixture to cover epsilon fallback.

- [ ] **Step 6: Run tests and commit**

Run: `JAX_PLATFORMS=cpu .venv-training/bin/python -m pytest -q tests/torch/test_evaluation.py tests/parity/test_evaluation.py`

```bash
git add src/drifting_torch/evaluation tests/torch/test_evaluation.py tests/parity/test_evaluation.py
git commit -m "feat: port release evaluation metrics to pytorch"
```

---

### Task 13: DDP, FSDP, HSDP, and Collective Checkpointing

**Files:**
- Create: `src/drifting_torch/distributed/__init__.py`
- Create: `src/drifting_torch/distributed/runtime.py`
- Create: `src/drifting_torch/distributed/strategies.py`
- Create: `src/drifting_torch/distributed/collectives.py`
- Modify: `src/drifting_torch/checkpointing/training.py`
- Modify: `src/drifting_torch/training/engine.py`
- Test: `tests/distributed/test_runtime.py`
- Test: `tests/distributed/test_ddp_training.py`
- Test: `tests/distributed/test_collective_checkpoint.py`
- Test: `tests/distributed/test_topology_validation.py`

**Interfaces:**
- Produces: `DistributedContext.initialize(runtime) -> DistributedContext`.
- Produces: `wrap_model(model, context) -> nn.Module`.
- Produces: `gather_valid(tensor, valid_mask, context) -> tuple[Tensor, Tensor]`.

- [ ] **Step 1: Write topology and two-process tests**

```python
def worker(rank, world_size, rendezvous, result_dir):
    context = DistributedContext.initialize(cpu_ddp(rank, world_size, rendezvous))
    summary = run_tiny_generator_step(context, result_dir)
    context.barrier()
    if rank == 0:
        assert summary.completed_step == 1
```

- [ ] **Step 2: Verify tests fail**

Run: `.venv-training/bin/python -m pytest -q tests/distributed/test_runtime.py tests/distributed/test_topology_validation.py`

Expected: distributed package missing.

- [ ] **Step 3: Implement explicit runtime lifecycle**

Resolve rank/local-rank/world-size from validated config/environment, initialize
only for distributed strategies, choose Gloo for CPU and NCCL for CUDA, set the
CUDA device before collectives, provide barriers/reductions, and always destroy
owned process groups.

- [ ] **Step 4: Implement DDP, FSDP, and HSDP wrappers**

DDP replicates the model and preserves process-local banks. FSDP shards along a
one-dimensional mesh. HSDP validates `replicate_size * shard_size == world_size`
and creates a two-dimensional device mesh. Activation checkpointing composes
before distributed wrapping.

- [ ] **Step 5: Make checkpoint operations collective**

Every rank enters distributed save/load. Publish only after all shards and the
manifest validate. Bind resumable state to strategy, world size, precision,
optimizer, and trajectory hash; fail with a compatibility report otherwise.

- [ ] **Step 6: Run multiprocess CPU verification**

Run: `.venv-training/bin/python -m pytest -q tests/distributed/test_ddp_training.py tests/distributed/test_collective_checkpoint.py`

Expected: two Gloo processes train, checkpoint, restore, and finish without
duplicate logs or partial checkpoint publication.

- [ ] **Step 7: Run topology tests and commit**

Run: `.venv-training/bin/python -m pytest -q tests/distributed/test_runtime.py tests/distributed/test_topology_validation.py`

```bash
git add src/drifting_torch/distributed src/drifting_torch/checkpointing/training.py src/drifting_torch/training/engine.py tests/distributed configs/runtime/torch
git commit -m "feat: add pytorch distributed execution"
```

---

### Task 14: Documentation, Packaging, Notebooks, and Completion Audit

**Files:**
- Modify: `README.md`
- Create: `docs/pytorch.md`
- Create: `docs/parity.md`
- Create: `docs/distributed.md`
- Modify: `docs/local-training.md`
- Create: `tests/test_packaging.py`
- Create: `tests/test_preserved_sources.py`
- Modify: `.gitignore`

**Interfaces:**
- Documents every explicit CLI, installation extra, config composition rule,
  conversion workflow, verification command, and hardware boundary.
- Produces a clean source distribution and wheel containing all three packages,
  runtime configs, and required metadata.

- [ ] **Step 1: Write preservation and packaging tests**

```python
def test_original_configs_and_notebook_match_recorded_hashes():
    for path, expected in load_preservation_manifest().items():
        assert sha256_file(ROOT / path) == expected

def test_wheel_imports_from_outside_checkout(built_wheel, clean_venv):
    clean_venv.install(built_wheel, extra="torch")
    clean_venv.run("python", "-c", "import drifting_common, drifting_torch")
```

- [ ] **Step 2: Verify new tests fail**

Run: `.venv-training/bin/python -m pytest -q tests/test_packaging.py tests/test_preserved_sources.py`

Expected: preservation manifest and final package inspection are absent.

- [ ] **Step 3: Write user and developer documentation**

Document separate installations/commands, source layouts, scientific/runtime
config composition, JAX conversion, pixel/latent inference, generator/MAE
training, fake/CIFAR replacement, ImageNet paths, latent cache, metrics,
distributed launch, exact resume boundary, notebook use, and parity reports.

- [ ] **Step 4: Execute both notebooks from derived copies**

Run the unchanged toy notebook and the PyTorch inference notebook in isolated
Jupyter directories. Assert finite outputs and expected shapes. Keep executed
copies under `work/verification/notebooks`, excluded from source control.

- [ ] **Step 5: Run comprehensive local verification**

Run:

```bash
JAX_PLATFORMS=cpu .venv-training/bin/python -m pytest -q \
  tests/common tests/jax tests/torch tests/parity tests/distributed \
  --ignore=tests/jax/test_run_toy_notebook.py \
  --ignore=tests/jax/test_packaged_imports.py
.venv-training/bin/python -m pytest -q tests/jax/test_packaged_imports.py
.venv-toy/bin/python -m pytest -q tests/jax/test_run_toy_notebook.py
.venv-training/bin/python -m build
.venv-training/bin/python -m pip check
git diff --check
```

Run the fake and CIFAR-10 generator smoke commands, a tiny MAE smoke run, native
checkpoint resume, official pixel/latent inference comparison, and two-process
CPU checkpoint test. Record exact commands and outputs in `docs/parity.md`.

- [ ] **Step 6: Inspect built artifacts and import isolation**

Verify the original toy notebook remains byte-for-byte identical. Verify
wheel/sdist contents exclude `.venv*`, `work/`, outputs, caches,
checkpoints, downloaded weights, executed notebooks, and bytecode. Install JAX
and PyTorch extras in separate clean environments and verify neither imports the
other backend package during its CLI startup.

- [ ] **Step 7: Perform requirement-by-requirement completion audit**

For every acceptance criterion in the design spec, link an authoritative test,
command output, artifact hash/report, or explicitly unverified hardware/scale
boundary. Any missing evidence keeps the port incomplete.

- [ ] **Step 8: Clean temporary artifacts and commit**

Remove only task-created caches, downloads, build directories, test outputs,
and derived notebooks after retaining the requested reports. Verify exact paths
are gone and the worktree contains only intentional files.

```bash
git add README.md docs .gitignore tests pyproject.toml notebooks/torch
git commit -m "docs: complete pytorch port verification"
```
