# JAX and PyTorch Parity

Parity uses the packaged JAX implementation as the oracle and externally
supplies the same labels, noise, noise labels, images, and masks. The strict
reference gate is FP32 CPU; BF16 and accelerator paths are reported separately.

## Verified reference paths

| Component | Evidence | Result |
| --- | --- | --- |
| Pixel Drift-B generator | official artifact, four labels, shared noise | max raw error `4.706e-5`; image error `2.354e-5` |
| Latent Drift-B generator | official artifact | max error `5.17e-5` |
| MAE pixel/latent | official MAE artifact, named activations | worst max error `7.987e-5` |
| ConvNeXtV2 base | pinned revision, all 13 activations, CPU thread envelopes | worst observed max error `1.050e-3` |
| Released Inception | official MD5-verified weights | pooled `5.01e-6`, spatial `1.04e-5`, logits `6.20e-6` |
| Loss/optimizer | deterministic tiny fixtures | values, gradients, and one transition pass |
| Resume | three-step direct vs resumed | bit-exact model, EMA, RNG, sampler, banks |

Policies live in `tests/parity/policies/`. Official-artifact tests are gated by
explicit environment paths so the normal suite remains offline. Example:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
DRIFTING_INCEPTION_WEIGHTS=work/inception_v3_weights_fid.pickle \
python -m pytest -q tests/parity/test_official_inception.py
```

Reports produced during this implementation are under ignored `work/parity/`;
they are reproducible with `tools/compare_backends.py`,
`compare_mae_backends.py`, `compare_convnext_backends.py`, and
`compare_inception_backends.py`.

## Boundaries

Local verification establishes implementation parity and finite inference; it
does not reproduce the paper's 50k ImageNet metrics. That claim requires the
ImageNet validation labels/images, official FID/PR references, the requested
sample count, and appropriate accelerator capacity. MPS was unavailable on the
verification host, and CUDA FSDP/HSDP could not be executed there.
