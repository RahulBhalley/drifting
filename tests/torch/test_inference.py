import json
from pathlib import Path

import pytest
import torch

from drifting_torch.checkpointing import save_torch_generator_artifact
from drifting_torch.inference import InferenceRequest, generate
from drifting_torch.models.generator import build_generator


CONFIG = {
    "cond_dim": 16,
    "num_classes": 4,
    "noise_classes": 3,
    "noise_coords": 1,
    "input_size": 4,
    "in_channels": 3,
    "n_cls_tokens": 0,
    "patch_size": 2,
    "hidden_size": 16,
    "depth": 1,
    "num_heads": 2,
    "mlp_ratio": 2.0,
    "out_channels": 3,
    "use_bf16": False,
}


def artifact(tmp_path: Path) -> Path:
    torch.manual_seed(55)
    model = build_generator(CONFIG)
    return save_torch_generator_artifact(
        tmp_path / "artifact", state_dict=model.state_dict(),
        model_config=CONFIG, step=7, ema_decay=0.99,
    )


def test_inference_is_reproducible_and_writes_atomic_outputs(tmp_path: Path):
    source = artifact(tmp_path)
    request = InferenceRequest(
        source=source,
        class_ids=(0, 2),
        cfg_scale=1.5,
        temperature=0.8,
        seed=9,
        device="cpu",
        precision="fp32",
        output_dir=tmp_path / "outputs",
    )
    first = generate(request)
    second = generate(request)
    assert torch.equal(first.raw_samples, second.raw_samples)
    assert first.metadata.keys() >= {
        "artifact_sha256", "class_ids", "cfg_scale", "temperature",
        "seed", "device", "precision", "sample_shape", "sample_finite",
    }
    assert first.metadata["sample_shape"] == [2, 3, 4, 4]
    assert first.metadata["sample_finite"]
    assert first.raw_path.is_file() and first.metadata_path.is_file()
    assert len(first.image_paths) == 2 and all(path.is_file() for path in first.image_paths)
    assert not list((tmp_path / "outputs").glob(".*.tmp-*"))
    assert json.loads(first.metadata_path.read_text())["artifact_step"] == 7


def test_inference_accepts_external_noise_and_validates_request(tmp_path: Path):
    source = artifact(tmp_path)
    result = generate(
        InferenceRequest(
            source=source, class_ids=(1,), device="cpu", precision="fp32",
            noise=torch.zeros(1, 3, 4, 4),
            noise_labels=torch.zeros(1, 1, dtype=torch.long),
        )
    )
    assert result.raw_samples.shape == (1, 3, 4, 4)
    with pytest.raises(ValueError, match="class ids"):
        generate(InferenceRequest(source=source, class_ids=(5,), device="cpu"))
    with pytest.raises(ValueError, match="fp16"):
        generate(InferenceRequest(source=source, class_ids=(1,), device="cpu", precision="fp16"))

