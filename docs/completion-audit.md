# PyTorch Port Completion Audit

This audit maps the approved acceptance criteria to reproducible evidence. It
separates verified behavior from unavailable hardware/data boundaries.

| Acceptance criterion | Evidence | Status |
| --- | --- | --- |
| Explicit, isolated backends | `tests/common/test_import_isolation.py`, `tests/test_packaging.py`, installed-wheel CLI import checks | Verified |
| Generator, MAE, ConvNeXt, loss, banks, data, VAE/cache | `tests/torch/` and named conversion reports | Verified on CPU |
| Original scientific inputs preserved | `tests/preservation_manifest.json`, `tests/test_preserved_sources.py` | Byte-for-byte verified |
| Packaged JAX regression | `tests/jax/`; separate packaged-import run | Verified |
| Official pixel/latent/MAE conversion | `tests/parity/test_official_{generator,mae}.py`; 241/241 generator tensors and 155/155 MAE tensors | Verified |
| Official fixed-input inference | FP32 policies and `tools/compare_backends.py`; multi-label pixel report | Verified |
| Loss, gradient, optimizer, EMA transitions | `tests/parity/test_{loss,generator_step,mae_step}.py` | Verified |
| Fake and replaceable CIFAR workflow | native one-step fake run; CIFAR scientific config run with `dataset.source=fake`; dataset dispatch/resize tests | Verified without CIFAR download |
| Native save/load and exact resume | `tests/torch/test_{generator,mae}_resume.py`; direct vs resumed three-step path | Bit-exact |
| Evaluation | fixed FID/IS/PR tests plus official released-Inception pooled/spatial/logit parity | Verified |
| DDP and collective checkpoint/resume | `tests/distributed/`; two Gloo workers, native generator step, PyTorch DCP model/optimizer state, rank-local sidecars | Verified on CPU |
| FSDP/HSDP | explicit wrappers, mixed precision, 1D/2D device-mesh topology tests | Implemented; CUDA run unavailable |
| Local notebooks | derived toy and PyTorch notebooks, zero cell errors, finite shape assertions | Verified |
| Wheel and source distribution | `uv build`; archive inspection; installed-wheel imports outside checkout | Verified |

## Comprehensive verification snapshot

- Main offline matrix: `143 passed, 10 skipped` after the final precision and
  distributed-checkpoint additions.
- Official-artifact matrix: all strict FP32 generator/MAE/Inception checks pass.
  BF16 CPU has a separate empirical policy. ConvNeXt CPU policy covers measured
  one-thread and default-thread reduction envelopes at the pinned model revision.
- JAX packaged imports: `2 passed`.
- Toy notebook runner tests: `3 passed`.
- Two-process DDP/DCP suite requires OS process/shared-memory access and passes
  outside the restricted sandbox.

## Deliberately unclaimed

- No 50,000-sample ImageNet FID/IS/precision/recall reproduction: ImageNet and
  its reference statistics were not available locally.
- No actual CIFAR-10 archive download: the official host was too slow; the user-
  approved replaceable FakeData path and CIFAR dispatch/resize behavior were
  tested instead.
- No MPS execution: MPS was not exposed on this host.
- No CUDA, NCCL, FSDP, or HSDP hardware run: no compatible GPU was available.
- No TPU-scale JAX retraining was attempted locally.
