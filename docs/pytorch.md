# PyTorch Backend

The PyTorch backend is native NCHW code. It does not call JAX during training
or inference. JAX is needed only to convert official artifacts and run parity
checks.

## Install and choose a runtime

```bash
pip install -e '.[torch]'

# Mac/CPU smoke run
drifting-torch-train \
  --config configs/local/m1_fake_smoke.yaml \
  --runtime configs/runtime/torch/cpu.yaml \
  --workdir runs/torch-fake
```

Use `configs/runtime/torch/mps.yaml` on Apple GPU, `cuda.yaml` on one NVIDIA
GPU, or the distributed runtime files documented in `docs/distributed.md`.
Runtime YAML is composed after scientific YAML; repeated `--set key=value`
overrides are applied last. The checked-in ImageNet, MAE, and local scientific
configs are preserved unchanged.

## Data and training

The dataset source is `fake`, `cifar10`, or `imagenet`. Machine-specific values
belong under `dataset.kwargs`, and can be replaced without changing model
science:

```bash
drifting-torch-train \
  --config configs/gen/pixel_sota_B.yaml \
  --runtime configs/runtime/torch/cuda.yaml \
  --workdir runs/pixel-B \
  --set dataset.source=imagenet \
  --set dataset.kwargs.data_root=/datasets/imagenet
```

ImageNet expects `train/<class>/...` and `val/<class>/...`. CIFAR-10 may set
`dataset.kwargs.download=true`; fake data is deterministic and intended only
for smoke tests. Latent configurations use either an on-the-fly VAE
(`use_latent`) or an atomic latent cache (`use_cache` plus
`dataset.kwargs.cache_root`). Build a cache with `drifting-torch-cache --help`.

Generator and MAE training write exact-resume checkpoints, rank-zero JSONL
metrics, and native safetensors EMA artifacts. Resume is automatic when the
work directory contains a compatible completed-step checkpoint. A changed
trajectory config, strategy, world size, or precision is rejected.

```bash
drifting-torch-train-mae \
  --config configs/local/m1_fake_mae_smoke.yaml \
  --runtime configs/runtime/torch/cpu.yaml \
  --workdir runs/mae-fake
```

## Official artifact conversion and inference

Conversion is strict: every named tensor must be consumed exactly once and
have the expected shape. The source JAX artifact is retained in the output
manifest.

```bash
PYTHONPATH=src python tools/convert_checkpoint.py --help

drifting-torch-infer \
  --source work/converted/pixel_B_sota \
  --class-ids 95 22 88 108 \
  --cfg-scale 1.0 \
  --seed 123 \
  --device cpu \
  --precision fp32 \
  --output-dir outputs/torch-inference
```

The inference result includes PNGs, `samples.pt`, and deterministic metadata.
`hf://NAME` resolves converted artifacts under
`DRIFTING_TORCH_ARTIFACT_ROOT` (or `work/converted`), never by silently
executing the JAX model.

## FID, IS, precision, and recall

Training evaluation uses the exact released Inception weights and semantics,
including resize behavior, padding-aware average pooling, 1008-way logits,
float64 statistics, fixed IS permutation, and the released manifold PR
algorithm. Configure paths under `dataset.kwargs` or with environment variables:

| Config key | Environment fallback | Contents |
| --- | --- | --- |
| `inception_weights` | `DRIFTING_INCEPTION_WEIGHTS` | released FID pickle |
| `fid_stats_path` | `DRIFTING_FID_STATS` | `mu/sigma` or `ref_mu/ref_sigma` NPZ |
| `pr_reference_path` | `DRIFTING_PR_REFERENCE` | NPZ containing `features` |

Evaluation occurs at step 1 (500-sample sanity check), every
`train.eval_per_step`, and the final step. Non-sanity rounds sweep
`train.cfg_list` and log the best FID/CFG. Set `train.enable_eval=false` for
local fake/CIFAR smoke runs. A 50k ImageNet benchmark still requires the real
dataset/reference statistics and suitable accelerators.
