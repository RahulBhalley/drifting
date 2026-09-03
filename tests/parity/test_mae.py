from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import torch

from drifting_jax.models.mae import MAEResNetJAX
from drifting_torch.checkpointing.mapping import map_mae_state
from drifting_torch.models.mae import MAEResNet
from drifting_torch.parity import ParityPolicy, compare_tensors


CONFIG = {
    "num_classes": 7,
    "in_channels": 3,
    "base_channels": 32,
    "patch_size": 4,
    "dropout_prob": 0.0,
    "layers": (1, 1, 1, 1),
    "use_bf16": False,
    "input_patch_size": 1,
}

POLICY = ParityPolicy(
    max_abs_error=0.0002,
    mean_abs_error=0.00002,
    nonfinite_count=0,
    cosine_similarity=0.999999,
    psnr=80.0,
    ssim=0.99999,
    uint8_mismatch_rate=0.01,
)

NEAR_ZERO_POLICY = ParityPolicy(
    max_abs_error=0.000002,
    mean_abs_error=0.0000005,
    nonfinite_count=0,
    cosine_similarity=-1.0,
    psnr=100.0,
    ssim=0.999999,
    uint8_mismatch_rate=1.0,
)


def build_models():
    jax_model = MAEResNetJAX(**CONFIG)
    images = jnp.asarray(
        np.random.default_rng(31).normal(size=(2, 16, 16, 3)), dtype=jnp.float32
    )
    labels = jnp.asarray([1, 4], dtype=jnp.int32)
    mask = jnp.asarray(np.indices((16, 16)).sum(0) % 3 == 0, dtype=jnp.float32)
    mask = jnp.broadcast_to(mask[None, ..., None], (2, 16, 16, 1))
    variables = jax_model.init(
        {"params": jax.random.PRNGKey(9), "dropout": jax.random.PRNGKey(10)},
        images,
        labels,
        mask=mask,
        train=False,
    )
    torch_model = MAEResNet(**CONFIG)
    flat = traverse_util.flatten_dict(variables["params"], sep="/")
    torch_model.load_state_dict(map_mae_state(flat, torch_model.state_dict()), strict=True)
    return jax_model, variables["params"], torch_model, images, labels, mask


def assert_parity(reference, candidate, name: str):
    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    near_zero = max(
        float(np.max(np.abs(reference_array))),
        float(np.max(np.abs(candidate_array))),
    ) < 0.00001
    comparison = compare_tensors(
        reference_array,
        candidate_array,
        NEAR_ZERO_POLICY if near_zero else POLICY,
    )
    assert comparison.passed, (name, comparison)


def test_mae_forward_decoder_logits_losses_and_activation_parity():
    jax_model, jax_params, torch_model, images, labels, mask = build_models()
    (jax_loss, jax_metrics), captures = jax_model.apply(
        {"params": jax_params},
        images,
        labels,
        mask=mask,
        lambda_cls=0.25,
        train=False,
        capture_intermediates=lambda module, _: module.name in {"decoder", "fc"},
        mutable=["intermediates"],
    )
    torch_output = torch_model(
        torch.from_numpy(np.array(images, copy=True)).permute(0, 3, 1, 2),
        torch.from_numpy(np.array(labels, copy=True)).long(),
        mask=torch.from_numpy(np.array(mask, copy=True)).permute(0, 3, 1, 2),
        lambda_cls=0.25,
        train=False,
    )
    jax_reconstruction = captures["intermediates"]["decoder"]["__call__"][0]
    jax_logits = captures["intermediates"]["fc"]["__call__"][0]
    assert_parity(jax_loss, torch_output.loss.detach().numpy(), "loss")
    assert_parity(jax_logits, torch_output.logits.detach().numpy(), "logits")
    assert_parity(
        np.asarray(jax_reconstruction).transpose(0, 3, 1, 2),
        torch_output.reconstruction.detach().numpy(),
        "reconstruction",
    )
    for name in ("cls_loss", "recon_loss", "accuracy", "mask_ratio"):
        assert_parity(jax_metrics[name], getattr(torch_output, name).detach().numpy(), name)

    jax_activations = jax_model.apply(
        {"params": jax_params}, images, every_k_block=1, method=jax_model.get_activations
    )
    torch_activations = torch_model.get_activations(
        torch.from_numpy(np.array(images, copy=True)).permute(0, 3, 1, 2), every_k_block=1
    )
    assert set(jax_activations) == set(torch_activations)
    for name in sorted(jax_activations):
        assert_parity(
            jax_activations[name], torch_activations[name].detach().numpy(), name
        )
