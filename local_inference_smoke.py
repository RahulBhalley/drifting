"""Generate one ImageNet sample with the official pixel_B_sota checkpoint."""

import json
import os
import sys
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
HF_CACHE = ROOT / "artifacts" / "hf-cache"
OUTPUT_DIR = ROOT / "outputs" / "inference-smoke"
HF_CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_ROOT", str(HF_CACHE))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.hsdp_util import set_global_mesh
from utils.init_util import load_generator_model_and_params
from utils.misc import prepare_rng


def generate(batch, params, rng, apply_fn, cfg_scale):
    _, labels = batch
    samples = apply_fn(
        {"params": params},
        train=False,
        rngs=prepare_rng(rng, ["noise"]),
        c=labels,
        cfg_scale=cfg_scale,
    )["samples"]
    return jnp.clip((samples + 1) / 2, 0, 1)


model_id = "pixel_B_sota"
class_id = 95
cfg_scale = 1.0
seed = 0
artifact_base = HF_CACHE / "models" / "gen" / "jax" / model_id
artifact_candidates = (
    artifact_base / "models" / "gen" / "jax" / model_id,
    artifact_base,
)
local_artifact = next(
    (
        path
        for path in artifact_candidates
        if (path / "metadata.json").is_file()
        and (path / "ema_params.msgpack").is_file()
    ),
    None,
)
init_from = str(local_artifact) if local_artifact is not None else f"hf://{model_id}"

# Generator.apply() calls enforce_ddp(), so even a one-device CPU run needs the
# global mesh that inference.py initializes. The current Colab notebook omits it.
set_global_mesh(min(8, max(1, jax.local_device_count() * jax.process_count())))

load_started = time.perf_counter()
model, params, metadata = load_generator_model_and_params(
    init_from, hf_cache_dir=str(HF_CACHE)
)
load_seconds = time.perf_counter() - load_started

model_config = metadata.get("model_config", {})
if model_config.get("in_channels") != 3:
    raise RuntimeError(f"Expected pixel-space model, got: {model_config}")

generate_jit = jax.jit(partial(generate, apply_fn=model.apply))
batch = (
    np.zeros((1, 1), dtype=np.int32),
    np.asarray([class_id], dtype=np.int32),
)

first_started = time.perf_counter()
samples = generate_jit(
    batch,
    params=params,
    cfg_scale=cfg_scale,
    rng=jax.random.PRNGKey(seed),
)
samples.block_until_ready()
first_seconds = time.perf_counter() - first_started

warm_started = time.perf_counter()
warm_samples = generate_jit(
    batch,
    params=params,
    cfg_scale=cfg_scale,
    rng=jax.random.PRNGKey(seed + 1),
)
warm_samples.block_until_ready()
warm_seconds = time.perf_counter() - warm_started

array = np.asarray(jax.device_get(samples), dtype=np.float32)
warm_array = np.asarray(jax.device_get(warm_samples), dtype=np.float32)
image_array = np.rint(array[0] * 255).clip(0, 255).astype(np.uint8)
image_path = OUTPUT_DIR / f"{model_id}_class{class_id:03d}_seed{seed}.png"
Image.fromarray(image_array).save(image_path)

metrics = {
    "upstream_model": f"hf://{model_id}",
    "class_id": class_id,
    "cfg_scale": cfg_scale,
    "seed": seed,
    "jax_version": jax.__version__,
    "jax_devices": [str(device) for device in jax.devices()],
    "model_input_size": model_config.get("input_size"),
    "model_in_channels": model_config.get("in_channels"),
    "sample_shape": list(array.shape),
    "sample_dtype": str(array.dtype),
    "sample_finite": bool(np.isfinite(array).all()),
    "warm_sample_finite": bool(np.isfinite(warm_array).all()),
    "sample_min": float(array.min()),
    "sample_max": float(array.max()),
    "sample_mean": float(array.mean()),
    "sample_std": float(array.std()),
    "checkpoint_load_seconds": load_seconds,
    "first_compile_and_inference_seconds": first_seconds,
    "warm_inference_seconds": warm_seconds,
    "image_path": str(image_path),
}
metrics_path = OUTPUT_DIR / f"{model_id}_class{class_id:03d}_seed{seed}.json"
metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
print(json.dumps(metrics, indent=2))
