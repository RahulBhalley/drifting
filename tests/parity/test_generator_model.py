from flax import serialization, traverse_util
from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
import json
import numpy as np
import torch

from drifting_jax.distributed import set_global_mesh
from drifting_jax.models.generator import (
    DitGen as JaxDitGen,
    FinalLayer as JaxFinalLayer,
    LightningDiTBlock as JaxLightningDiTBlock,
)
from drifting_torch.checkpointing.mapping import map_generator_state
from drifting_torch.checkpointing import convert_jax_generator, load_torch_generator
from drifting_torch.models.generator import DitGen as TorchDitGen
from drifting_torch.parity import ParityPolicy, compare_tensors


CONFIG = {
    "cond_dim": 16,
    "num_classes": 11,
    "noise_classes": 5,
    "noise_coords": 2,
    "input_size": 8,
    "in_channels": 3,
    "patch_size": 2,
    "hidden_size": 32,
    "depth": 2,
    "num_heads": 4,
    "mlp_ratio": 2.0,
    "out_channels": 3,
    "use_qknorm": True,
    "use_swiglu": True,
    "use_rope": True,
    "use_rmsnorm": True,
    "use_bf16": False,
    "attn_fp32": True,
    "use_remat": False,
    "n_cls_tokens": 2,
}

TINY_POLICY = ParityPolicy(
    max_abs_error=2e-5,
    mean_abs_error=2e-6,
    nonfinite_count=0,
    cosine_similarity=0.999999,
    psnr=95.0,
    ssim=0.999999,
    uint8_mismatch_rate=0.0,
)


def initialized_nontrivial_jax_state():
    set_global_mesh(1)
    model = JaxDitGen(**CONFIG)
    labels = jnp.asarray([1, 3], dtype=jnp.int32)
    noise = jnp.linspace(-1, 1, 2 * 8 * 8 * 3).reshape(2, 8, 8, 3)
    noise_labels = jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(7)},
        labels,
        noise=noise,
        noise_labels=noise_labels,
    )
    params = unfreeze(variables["params"])
    flat = traverse_util.flatten_dict(params, sep="/")
    rng = np.random.default_rng(23)
    for name, value in list(flat.items()):
        is_adaln = "/blocks_" in name and "/TorchLinear_0/Dense_0/" in name
        is_final = "/FinalLayer_0/TorchLinear_" in name
        if is_adaln or is_final:
            flat[name] = jnp.asarray(rng.normal(0, 0.02, value.shape), dtype=value.dtype)
    return model, freeze(traverse_util.unflatten_dict(flat, sep="/"))


def test_tiny_generator_conditioning_and_full_output_parity():
    jax_model, jax_params = initialized_nontrivial_jax_state()
    torch_model = TorchDitGen(**CONFIG)
    flat = traverse_util.flatten_dict(unfreeze(jax_params), sep="/")
    torch_model.load_state_dict(map_generator_state(flat, torch_model.state_dict()), strict=True)

    labels_np = np.array([1, 3], dtype=np.int32)
    noise_nhwc = np.linspace(-1, 1, 2 * 8 * 8 * 3, dtype=np.float32).reshape(2, 8, 8, 3)
    noise_labels_np = np.array([[0, 1], [2, 3]], dtype=np.int32)
    jax_cond = jax_model.apply(
        {"params": jax_params},
        jnp.asarray(labels_np),
        1.25,
        jnp.asarray(noise_labels_np),
        method=jax_model.c_cfg_noise_to_cond,
    )
    torch_cond = torch_model.conditioning(
        torch.from_numpy(labels_np).long(),
        1.25,
        torch.from_numpy(noise_labels_np).long(),
    )
    cond_comparison = compare_tensors(jax_cond, torch_cond.detach().numpy(), TINY_POLICY)
    assert cond_comparison.passed, cond_comparison

    jax_result, jax_captures = jax_model.apply(
        {"params": jax_params},
        jnp.asarray(labels_np),
        cfg_scale=1.25,
        noise=jnp.asarray(noise_nhwc),
        noise_labels=jnp.asarray(noise_labels_np),
        capture_intermediates=lambda module, _: isinstance(
            module, (JaxLightningDiTBlock, JaxFinalLayer)
        ),
        mutable=["intermediates"],
    )
    jax_output = jax_result["samples"]
    captured = jax_captures["intermediates"]["LightningDiT_0"]

    torch_captures = {}
    handles = [
        block.register_forward_hook(
            lambda _module, _inputs, output, index=index: torch_captures.__setitem__(
                f"blocks_{index}", output.detach()
            )
        )
        for index, block in enumerate(torch_model.model.blocks)
    ]
    handles.append(
        torch_model.model.final_layer.register_forward_hook(
            lambda _module, _inputs, output: torch_captures.__setitem__(
                "FinalLayer_0", output.detach()
            )
        )
    )
    torch_output = torch_model(
        torch.from_numpy(labels_np).long(),
        cfg_scale=1.25,
        noise=torch.from_numpy(noise_nhwc).permute(0, 3, 1, 2),
        noise_labels=torch.from_numpy(noise_labels_np).long(),
    ).samples
    for handle in handles:
        handle.remove()

    for name in ("blocks_0", "blocks_1", "FinalLayer_0"):
        jax_stage = captured[name]["__call__"][0]
        stage_comparison = compare_tensors(
            jax_stage, torch_captures[name].numpy(), TINY_POLICY
        )
        assert stage_comparison.passed, (name, stage_comparison)
    comparison = compare_tensors(
        np.asarray(jax_output).transpose(0, 3, 1, 2),
        torch_output.detach().numpy(),
        TINY_POLICY,
    )
    assert comparison.passed, comparison


def test_comparator_fails_closed_for_missing_policy_metric_and_nonfinite_value():
    incomplete = {
        "max_abs_error": 1.0,
        "mean_abs_error": 1.0,
        "nonfinite_count": 0,
        "cosine_similarity": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }
    try:
        compare_tensors(np.zeros(2), np.zeros(2), incomplete)
    except ValueError as error:
        assert "missing metrics" in str(error)
    else:
        raise AssertionError("incomplete parity policy was accepted")

    comparison = compare_tensors(
        np.array([0.0, np.nan]), np.array([0.0, 1.0]), TINY_POLICY
    )
    assert not comparison.passed
    assert comparison.metrics["nonfinite_count"] == 1


def test_tiny_artifact_conversion_is_atomic_hash_validated_and_reloadable(tmp_path):
    _, params = initialized_nontrivial_jax_state()
    source = tmp_path / "jax"
    source.mkdir()
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "model_config": CONFIG,
                "source": {"step_loaded": 17, "ema_selected": "0.999"},
            }
        ),
        encoding="utf-8",
    )
    (source / "ema_params.msgpack").write_bytes(
        serialization.msgpack_serialize(unfreeze(params))
    )
    destination = tmp_path / "torch"

    report = convert_jax_generator(source, destination)
    loaded = load_torch_generator(destination)

    assert report.tensor_count == len(loaded.model.state_dict())
    assert loaded.manifest.step == 17
    assert loaded.manifest.ema_decay == 0.999
    loaded.manifest.verify_files(destination)
