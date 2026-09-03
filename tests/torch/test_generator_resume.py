from pathlib import Path

import pytest
import torch

from drifting_torch.checkpointing.training import load_training_state, save_training_state
from drifting_torch.data.datasets import DataBatch, StatefulSampler
from drifting_torch.models.features import FrozenFeatureExtractor
from drifting_torch.training.generator import GeneratorStepOptions, generator_train_step

from .test_generator_step import banks, tiny_state


def execute(state, memory_banks, count):
    options = GeneratorStepOptions(
        gen_per_label=2, pos_per_sample=2, neg_per_sample=2, radii=(0.2,),
        base_lr=1e-3, warmup_steps=0, total_steps=3,
    )
    batch = DataBatch(torch.zeros(2, 1, 4, 4), torch.tensor([0, 1]))
    for _ in range(count):
        generator_train_step(state, batch, memory_banks, FrozenFeatureExtractor(), options)


def test_resume_matches_uninterrupted_bit_exact(tmp_path: Path):
    direct_state, direct_banks = tiny_state(), banks()
    execute(direct_state, direct_banks, 3)

    resumed_state, resumed_banks = tiny_state(), banks()
    execute(resumed_state, resumed_banks, 1)
    sampler = StatefulSampler(list(range(8)), shuffle=True, seed=11)
    list(iter(sampler))[:2]
    checkpoint = tmp_path / "step-00000001.pt"
    save_training_state(
        checkpoint, resumed_state, sampler=sampler, banks=resumed_banks,
        config={"model": "tiny", "trajectory": 1},
    )
    loaded_state, loaded_banks = tiny_state(), banks()
    loaded_sampler = StatefulSampler(list(range(8)), shuffle=True, seed=11)
    load_training_state(
        checkpoint, loaded_state, sampler=loaded_sampler, banks=loaded_banks,
        config={"model": "tiny", "trajectory": 1},
    )
    execute(loaded_state, loaded_banks, 2)

    for name, value in direct_state.model.state_dict().items():
        assert torch.equal(value, loaded_state.model.state_dict()[name]), name
        assert torch.equal(direct_state.ema[name], loaded_state.ema[name]), name
    assert direct_banks[0].state_dict_equal(loaded_banks[0])
    assert direct_banks[1].state_dict_equal(loaded_banks[1])
    assert loaded_sampler.state_dict() == sampler.state_dict()


def test_resume_rejects_changed_trajectory(tmp_path: Path):
    state, memory_banks = tiny_state(), banks()
    sampler = StatefulSampler(list(range(2)), shuffle=False)
    checkpoint = tmp_path / "state.pt"
    save_training_state(checkpoint, state, sampler=sampler, banks=memory_banks, config={"a": 1})
    with pytest.raises(ValueError, match="trajectory"):
        load_training_state(checkpoint, state, sampler=sampler, banks=memory_banks, config={"a": 2})
