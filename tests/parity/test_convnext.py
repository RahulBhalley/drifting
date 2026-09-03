import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import ConvNextV2Config, ConvNextV2ForImageClassification

from drifting_jax.distributed import set_global_mesh
from drifting_jax.models.convnext import ConvNextV2, convert_weights_to_jax
from drifting_torch.models.convnext import ConvNeXtV2FeatureExtractor
from drifting_torch.parity import ParityPolicy, compare_tensors


POLICY = ParityPolicy(
    max_abs_error=0.0002,
    mean_abs_error=0.00002,
    nonfinite_count=0,
    cosine_similarity=0.999999,
    psnr=80.0,
    ssim=0.99999,
    uint8_mismatch_rate=0.01,
)

GLOBAL_POOL_POLICY = ParityPolicy(
    max_abs_error=0.003,
    mean_abs_error=0.0005,
    nonfinite_count=0,
    cosine_similarity=0.999999,
    psnr=68.0,
    ssim=0.99999,
    uint8_mismatch_rate=0.03,
)


def test_native_convnext_activations_match_jax_conversion():
    config = ConvNextV2Config(
        num_channels=3,
        patch_size=4,
        hidden_sizes=[8, 16, 32, 64],
        depths=[1, 1, 1, 1],
        num_labels=5,
        drop_path_rate=0.0,
    )
    torch.manual_seed(43)
    native = ConvNextV2ForImageClassification(config).eval()
    torch_extractor = ConvNeXtV2FeatureExtractor(native)
    jax_model = ConvNextV2(
        in_chans=3,
        num_classes=5,
        depths=(1, 1, 1, 1),
        dims=(8, 16, 32, 64),
    )
    jax_variables = jax_model.init(
        jax.random.PRNGKey(0), jnp.ones((1, 224, 224, 3), dtype=jnp.float32)
    )
    jax_variables = convert_weights_to_jax(jax_variables, native.state_dict(), hf=True)
    set_global_mesh(1)

    images = np.random.default_rng(44).normal(size=(2, 3, 32, 32)).astype(np.float32)
    torch_outputs = torch_extractor(torch.from_numpy(images))
    jax_outputs = jax_model.apply(
        jax_variables,
        jnp.asarray(images.transpose(0, 2, 3, 1)),
        method=jax_model.get_activations,
    )

    assert set(torch_outputs) == set(jax_outputs)
    for name in sorted(jax_outputs):
        comparison = compare_tensors(
            jax_outputs[name],
            torch_outputs[name].detach().numpy(),
            GLOBAL_POOL_POLICY if name == "global_mean" else POLICY,
        )
        assert comparison.passed, (name, comparison)
