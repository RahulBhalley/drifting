import numpy as np
import pytest
import torch

from drifting_torch.checkpointing.mapping import (
    ConversionError,
    convert_leaf,
    map_generator_state,
)


def test_dense_kernel_transposes():
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    actual = convert_leaf("Dense_0/kernel", source, target_shape=(4, 3))
    np.testing.assert_array_equal(actual, source.T)
    assert actual.flags.c_contiguous


def test_conv_kernel_transposes_hwio_to_oihw():
    source = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    actual = convert_leaf("Conv_0/kernel", source, target_shape=(5, 4, 2, 3))
    np.testing.assert_array_equal(actual, source.transpose(3, 2, 0, 1))


def test_leaf_shape_mismatch_is_descriptive():
    with pytest.raises(ConversionError, match="Dense_0/kernel.*expected.*got"):
        convert_leaf("Dense_0/kernel", np.ones((2, 3)), target_shape=(7, 2))


def test_generator_mapping_consumes_every_source_and_target():
    source = {
        "Embed_0/embedding": np.arange(12, dtype=np.float32).reshape(3, 4),
        "RMSNorm_0/weight": np.ones(4, dtype=np.float32),
        "TimestepEmbedder_0/TorchLinear_0/Dense_0/kernel": np.ones((2, 4), dtype=np.float32),
        "TimestepEmbedder_0/TorchLinear_0/Dense_0/bias": np.zeros(4, dtype=np.float32),
    }
    target = {
        "class_embed.weight": torch.empty(3, 4),
        "cfg_norm.weight": torch.empty(4),
        "cfg_embedder.linear_1.weight": torch.empty(4, 2),
        "cfg_embedder.linear_1.bias": torch.empty(4),
    }

    mapped = map_generator_state(source, target)

    assert set(mapped) == set(target)
    torch.testing.assert_close(mapped["class_embed.weight"], torch.from_numpy(source["Embed_0/embedding"]))
    torch.testing.assert_close(
        mapped["cfg_embedder.linear_1.weight"],
        torch.from_numpy(source["TimestepEmbedder_0/TorchLinear_0/Dense_0/kernel"].T.copy()),
    )


def test_converter_rejects_unmapped_source():
    with pytest.raises(ConversionError, match="unmapped source"):
        map_generator_state(
            {"unknown/kernel": np.ones((2, 2), dtype=np.float32)},
            {"class_embed.weight": torch.empty(2, 2)},
        )


def test_converter_rejects_unmapped_target():
    with pytest.raises(ConversionError, match="unmapped target"):
        map_generator_state(
            {"Embed_0/embedding": np.ones((2, 2), dtype=np.float32)},
            {
                "class_embed.weight": torch.empty(2, 2),
                "surprise.weight": torch.empty(2, 2),
            },
        )
