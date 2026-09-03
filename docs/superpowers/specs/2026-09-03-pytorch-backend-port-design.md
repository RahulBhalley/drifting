# Drifting Models Dual-Backend Architecture

**Status:** Approved direction; implementation specification  
**Date:** 2026-09-03  
**Base branch:** `codex/local-training`  
**Implementation branch:** `codex/pytorch-port`

## 1. Objective

Port the complete released Drifting Models implementation from JAX/Flax to a
native, production-quality PyTorch backend without making either backend the
implicit default. Training, inference, datasets, latent-cache construction,
feature extraction, checkpointing, evaluation, logging, and distributed
execution must work through explicit backend entry points.

The existing JAX behavior remains executable as the scientific reference. The
PyTorch backend must be verified against it with shared inputs and converted
weights. Visual similarity alone is not parity evidence.

## 2. Requirements

### 2.1 Functional scope

The PyTorch backend must provide equivalents for:

- the LightningDiT/DitGen pixel and latent generators;
- the MAE-ResNet encoder, decoder, training loss, and activation extraction;
- the optional ConvNeXtV2 activation path;
- the multi-temperature drifting loss;
- class-wise positive and negative memory banks;
- ImageNet, latent-cache, deterministic fake-data, and resized CIFAR-10 data;
- Stable Diffusion VAE encoding and decoding for latent workflows;
- generator and MAE optimization, EMA, schedules, clipping, logging, evaluation,
  and exact completed-step resume;
- local and Hugging Face artifact loading;
- FID, Inception Score, precision, and recall evaluation;
- CPU, Apple MPS, CUDA, DDP, FSDP, and two-dimensional HSDP configuration;
- the local toy workflow and both JAX and PyTorch inference notebooks.

### 2.2 Preservation requirements

- Preserve the authors' original scientific YAML files under `configs/gen/`
  and `configs/mae/` byte-for-byte.
- Preserve the existing local fake/CIFAR configurations under `configs/local/`.
- Preserve the original toy notebook and its recorded checksum.
- Keep JAX and PyTorch separately installable.
- Do not silently select a backend.
- Do not require JAX in a PyTorch-only installation or PyTorch in a JAX-only
  installation except where the backend itself already consumes a PyTorch data
  or model dependency.

### 2.3 Verification requirements

- Compare backend-neutral math, layers, models, losses, and state transitions.
- Compare with identical externally supplied noise/masks; equal integer seeds
  are insufficient because JAX and PyTorch use different random algorithms.
- Exercise one official pixel generator and one official latent generator.
- Exercise one official MAE artifact used by generator training.
- Verify one generator optimizer step and one MAE optimizer step.
- Verify checkpoint conversion, native save/load, EMA export, and resume.
- Run fake-data and CIFAR-10 training smoke tests locally.
- Run at least a two-process CPU distributed test.
- Record unsupported hardware verification honestly; configuration support is
  not evidence of successful execution on unavailable hardware.

## 3. Non-goals and Verification Boundaries

- Exact bitwise equality is not required across framework kernels. Numeric
  tolerances must be empirical, documented, and tight enough to catch layout,
  precision, normalization, conditioning, or checkpoint mapping errors.
- A 50,000-sample ImageNet FID reproduction is not a local acceptance gate
  without ImageNet, reference statistics, and suitable accelerator resources.
  The evaluator itself must still be ported and cross-checked on fixed samples.
- PyTorch is not declared scientifically equivalent based only on a tiny model
  or random initialization. Official converted artifacts are required.
- Fast fused kernels may not replace the transparent parity path until their
  drift has been measured against it.

## 4. Package Layout

Use a `src/` layout so tests and CLIs exercise installed packages instead of
accidentally importing files from the repository root.

```text
drifting/
├── pyproject.toml
├── src/
│   ├── drifting_common/
│   │   ├── config/
│   │   ├── artifacts/
│   │   ├── data_contracts/
│   │   └── metrics/
│   ├── drifting_jax/
│   │   ├── models/
│   │   ├── data/
│   │   ├── training/
│   │   ├── evaluation/
│   │   ├── checkpointing/
│   │   ├── distributed/
│   │   └── inference.py
│   └── drifting_torch/
│       ├── models/
│       ├── data/
│       ├── training/
│       ├── evaluation/
│       ├── checkpointing/
│       ├── distributed/
│       └── inference.py
├── configs/
│   ├── gen/
│   ├── mae/
│   ├── local/
│   └── runtime/
│       ├── jax/
│       └── torch/
├── tools/
│   ├── convert_checkpoint.py
│   └── compare_backends.py
├── notebooks/
│   ├── jax/
│   └── torch/
└── tests/
    ├── common/
    ├── jax/
    ├── torch/
    ├── parity/
    └── distributed/
```

The current flat modules become compatibility shims during migration. They are
removed only after every original entry point has an explicit replacement and
the JAX regression suite passes from its packaged location.

## 5. Dependency and Command Model

`pyproject.toml` defines a minimal backend-neutral core and optional extras:

- `.[jax]`: JAX/Flax/Optax/Orbax and JAX evaluation dependencies;
- `.[torch]`: PyTorch/torchvision and PyTorch evaluation dependencies;
- `.[parity]`: both backends plus comparison dependencies;
- `.[dev]`: test, lint, type-check, and notebook tooling.

No generic command chooses a backend. Public commands are explicit:

```text
drifting-jax-train
drifting-jax-infer
drifting-jax-cache
drifting-torch-train
drifting-torch-infer
drifting-torch-cache
drifting-convert
drifting-compare
```

Importing `drifting_common` must not import either tensor framework. Importing
one backend must not initialize or import the other backend.

## 6. Common/Backend Boundary

`drifting_common` owns only serializable or framework-neutral contracts:

- validated scientific and runtime configuration schemas;
- configuration composition and legacy-key normalization;
- artifact metadata schemas and format versions;
- dataset names, split metadata, class-count and path validation;
- metric names and serializable results;
- logger event schemas and filesystem layout;
- parity reports and tolerance policy data.

It does not own tensors, random generators, models, optimizers, dataloaders,
collectives, device selection, or framework checkpoint state.

Each backend owns:

- tensor layout and device placement;
- random-state implementation;
- preprocessing and dataloaders;
- models and parameter initialization;
- autograd, optimizers, schedules, mixed precision, and compilation;
- distributed wrapping and collective operations;
- native training checkpoints;
- backend-specific inference and evaluation execution.

## 7. Configuration Composition

Scientific configuration remains shared:

```bash
drifting-torch-train \
  --config configs/gen/pixel_sota_B.yaml \
  --runtime configs/runtime/torch/mps.yaml \
  --workdir runs/torch-pixel-b
```

Composition order is deterministic:

1. shared scientific YAML;
2. optional backend runtime YAML;
3. explicit CLI overrides.

Unknown keys and invalid combinations fail before model or distributed
initialization. Original `hsdp_dim` is accepted as a legacy JAX field. It is not
silently interpreted as a PyTorch topology; PyTorch distributed topology comes
from its runtime schema.

Runtime profiles include:

- JAX CPU/single-host and original TPU/HSDP;
- PyTorch CPU, MPS, CUDA, DDP, FSDP, and HSDP;
- small local fake/CIFAR profiles without changing ImageNet configs.

## 8. Tensor and Public API Contracts

PyTorch uses `NCHW` internally for image-like tensors. JAX retains `NHWC`.
Backend public results record their layout, and parity adapters perform an
explicit transpose before comparison. Silent layout guessing is forbidden.

Generator inference accepts:

- class labels;
- CFG scale (scalar or per-example);
- temperature;
- optional externally supplied Gaussian input noise;
- optional externally supplied discrete noise labels;
- deterministic/evaluation mode.

When external noise is absent, each backend uses its native seeded generator.
When present, no backend RNG may alter the supplied values. This is the basis
for end-to-end parity.

MAE training accepts an optional externally supplied patch mask for parity.
Production training may generate masks natively.

Feature activation dictionaries retain the JAX key names and semantic shapes
`(batch, tokens, channels)` so the drifting loss sees the same contract in both
backends.

## 9. Model Port

### 9.1 Generator

Port and test:

- sinusoidal position embeddings;
- PyTorch-compatible dense initialization;
- RMSNorm;
- AdaLN modulation;
- rotary embeddings;
- SwiGLU and standard MLP paths;
- Q/K normalization and attention precision behavior;
- class tokens;
- CFG timestep embedding and normalization;
- discrete noise-label embeddings;
- patchify/unpatchify behavior;
- activation checkpointing;
- pixel and latent output shapes.

The transparent parity attention path uses explicit scale, matrix multiply,
softmax, and value aggregation matching JAX operation order. An SDPA/fused path
may be enabled separately after measured comparison.

### 9.2 MAE-ResNet

Port and test:

- input patching;
- mask generation/application;
- ResNet basic blocks and projection skips;
- GroupNorm behavior and epsilon;
- bilinear decoder resize settings;
- reconstruction and classification heads;
- reconstruction, classification, and combined losses;
- all activation dictionary variants, including per-block, mean, standard
  deviation, and spatial aggregation keys.

### 9.3 ConvNeXtV2

Use the native PyTorch pretrained ConvNeXtV2 as the source model while retaining
the exact activation names and preprocessing expected by generator training.
Cross-check those activations with the current PyTorch-to-JAX conversion path.

## 10. Drifting Loss and Memory Banks

The PyTorch drifting loss must preserve:

- squared pairwise distances and epsilon behavior;
- scale calculation and clipping;
- diagonal self-mask placement;
- temperature iteration order;
- row/column affinity symmetrization;
- positive/negative weights;
- stopped-gradient target construction;
- force normalization;
- per-temperature metric keys;
- batch/token/sample reduction axes.

Parity covers forward values and gradients with fixed tensors, including edge
cases with small scales and unequal positive/negative counts.

Memory banks remain class-wise ring buffers. Their serializable state includes
bank contents, pointers, counts, dtype, shape, and RNG state. Resume must not
reinitialize or silently discard the banks.

## 11. Data and Latent Cache

Both backends support the same named sources and validation behavior:

- `imagenet`;
- `fake`;
- `cifar10`;
- latent `.pt` cache.

Shared contracts define split/path/class metadata. Backend modules own tensor
conversion and device transfer. Augmentation semantics, crop algorithm, value
range, normalization, label dtype, shuffle policy, and distributed sampling are
covered by fixture-based tests.

The PyTorch latent-cache command uses the native Diffusers `AutoencoderKL`,
preserves the released scaling and posterior-sampling semantics, and writes a
versioned manifest beside cache files. Existing cache files remain readable.

## 12. Training Semantics

Both generator and MAE trainers provide:

- deterministic seed derivation by global step and rank;
- gradient accumulation when configured;
- FP32, BF16, and supported mixed-precision modes;
- global-norm clipping;
- matching warmup plus constant/cosine schedules;
- AdamW hyperparameters matching Optax;
- EMA updated after the optimizer step;
- rank-gated progress and logging;
- periodic evaluation and artifact export;
- the released per-checkpoint CFG sweep and best-FID selection behavior;
- first-step profiling with backend-specific, rank-gated timing and memory data;
- clean interruption at completed optimizer-step boundaries;
- exact completed-step resume.

Generator training preserves per-rank/local memory-bank semantics from the JAX
release. Distributed tests assert the intended local versus synchronized state
rather than assuming all ranks share one bank.

## 13. Distributed Execution

PyTorch runtime strategies are explicit:

- `single`: CPU, MPS, or one CUDA device;
- `ddp`: replicated model with process-local optimizer and memory banks;
- `fsdp`: one-dimensional parameter sharding;
- `hsdp`: a two-dimensional device mesh combining replication and sharding.

All ranks participate in collective checkpoint operations. Rank zero alone may
write non-collective logs and preview images. Dataset samplers receive epoch or
step state so resume does not silently change ordering.

At minimum, CI/local verification runs two CPU processes with Gloo. CUDA, FSDP,
and HSDP hardware claims require execution on compatible accelerators; absent
hardware is reported as unverified, not passed.

## 14. Checkpoint and Artifact Formats

### 14.1 Backend-neutral metadata

Every exported artifact contains a versioned JSON manifest with:

- schema version and artifact kind;
- source and target backend;
- model configuration;
- step and EMA decay;
- tensor dtype/layout policy;
- source checkpoint identity/hash;
- parameter mapping version;
- conversion report and validation status.

### 14.2 PyTorch artifacts

EMA inference weights use `safetensors`. Resumable training state includes model,
EMA, optimizer, scheduler, scaler, step, sampler, memory-bank, and RNG state.
Distributed checkpoints use PyTorch distributed checkpoint APIs and an atomic
publish protocol appropriate to the filesystem.

Loading validates model kind, config compatibility, tensor names/shapes, format
version, and distributed topology constraints before mutating live state.

### 14.3 JAX conversion

The converter handles Flax msgpack, released Hugging Face artifacts, and the
supported local Orbax/Flax export. Mapping rules are explicit and audited:

- Dense kernels: `(in, out)` to `(out, in)`;
- convolutions: `HWIO` to `OIHW`;
- embeddings and scalar/vector normalization weights: no transpose;
- positional/class embeddings: preserve semantic axes;
- named sequential/block modules: deterministic name table, never shape-only
  guessing.

Every conversion rejects missing, extra, duplicate, or incompatible tensors and
emits a machine-readable report.

## 15. Inference and Evaluation

Inference supports Hugging Face references, local JAX artifacts through explicit
conversion, and native PyTorch artifacts. It supports labels, CFG scale,
temperature, seed, batch size, device, precision, output images, raw tensors,
and JSON metadata.

The PyTorch evaluator implements the released preprocessing and reports FID,
Inception Score, precision, and recall with the same reference-stat contracts.
Distributed aggregation must be duplicate-safe for padded final batches.

Training-time evaluation retains the release behavior: a bounded first-step
sanity evaluation, subsequent evaluation across `cfg_list`, and logging of the
best FID/CFG pair. Alternate local datasets keep evaluation disabled unless a
compatible reference-stat set is supplied explicitly.

Evaluation parity checks Inception logits/features, accumulated moments, and
final metrics on fixed image fixtures. Paper-number reproduction is reported
separately from evaluator correctness.

## 16. Parity Methodology

### 16.1 Levels

1. **Primitive parity:** embeddings, normalization, RoPE, patch transforms,
   distance kernels, schedules, and preprocessing.
2. **Module parity:** attention, DiT blocks, MAE blocks, decoder blocks, and
   activation extraction.
3. **Model parity:** full generator and MAE forward passes after conversion.
4. **State-transition parity:** loss, gradients, clipping, optimizer update,
   EMA, memory-bank mutation, and resume.
5. **Artifact parity:** official pixel/latent/MAE conversions and inference.
6. **Evaluation parity:** features, statistics, and metrics on fixed samples.

### 16.2 Metrics

Reports include maximum absolute error, mean absolute error, relative error,
cosine similarity where meaningful, non-finite counts, and image PSNR/SSIM.
Postprocessed uint8 comparisons include mismatch count and maximum channel error.

Initial tolerances are conservative hypotheses. They become acceptance values
only after measuring a correct FP32 CPU reference. BF16 and fused-kernel
tolerances are tracked separately and may not weaken the FP32 gate.

### 16.3 Reference fixtures

Small deterministic fixtures are checked into tests where licensing and size
permit. Large official weights remain downloaded artifacts identified by repo,
revision, path, and hash. Generated parity outputs and reports live under the
selected work directory, not in source control.

## 17. Test Strategy

### Common

- schema validation and composition;
- legacy configuration compatibility;
- artifact manifest round trips;
- no-backend-import isolation.

### JAX regression

- relocated imports and explicit CLIs;
- current 27 local tests;
- existing pixel inference regression;
- fake/CIFAR one-step workflows.

### PyTorch

- model shapes and dtype/device behavior;
- loss and memory-bank edge cases;
- data sources and transforms;
- MAE and generator optimizer steps;
- checkpoint/export/resume;
- CPU and MPS smoke tests where supported;
- inference and evaluation fixtures.

### Parity

- primitive, module, full-model, training-step, artifact, and metric comparisons;
- official pixel and latent generator artifacts;
- official MAE artifact;
- explicit failure tests for malformed mappings and exceeded tolerances.

### Distributed

- two-process CPU DDP training;
- collective checkpoint save/restore;
- rank-gated side effects;
- duplicate-safe metric aggregation;
- strategy/topology validation for FSDP and HSDP.

## 18. Migration Sequence

1. Establish packaging, common schemas, explicit JAX package, and regression
   compatibility without changing scientific behavior.
2. Implement parity primitives and checkpoint mapping infrastructure.
3. Port generator and verify tiny plus official pixel inference.
4. Port MAE/ConvNeXt and verify activation parity.
5. Port drifting loss, memory banks, generator training, and step parity.
6. Port MAE training and step parity.
7. Port data, VAE/latent cache, inference, and native artifacts end to end.
8. Port evaluation and metric parity.
9. Add DDP/FSDP/HSDP runtime and distributed checkpointing.
10. Add PyTorch notebook, documentation, packaging checks, comprehensive tests,
    and a requirement-by-requirement completion audit.

Each stage must leave both backends runnable. Compatibility shims are deleted
only after their replacements and tests exist.

## 19. Acceptance Criteria

The port is complete only when all of the following are evidenced:

- JAX and PyTorch install independently and expose only explicit commands.
- All released model, data, training, inference, checkpoint, evaluation, and
  distributed responsibilities have a PyTorch implementation.
- Original and local configurations/notebooks remain preserved as specified.
- The complete JAX regression suite passes from the packaged implementation.
- The complete PyTorch suite passes on available CPU/MPS hardware.
- Official pixel, latent, and MAE artifacts convert without unmapped tensors.
- Official fixed-input inference satisfies documented FP32 parity thresholds.
- Generator and MAE fixed-input training-step parity passes.
- Fake and CIFAR training write resumable checkpoints and inference artifacts.
- Native PyTorch resume reproduces the uninterrupted completed-step trajectory.
- Two-process CPU distributed training and collective resume pass.
- Evaluation features/statistics/metrics pass fixed-sample parity.
- Documentation distinguishes verified local behavior from unverified CUDA/TPU,
  large-scale ImageNet, FSDP, and HSDP execution.
- The worktree is clean except for intentional committed deliverables, and no
  Codex-created temporary artifacts remain.
