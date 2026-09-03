from pathlib import Path

import torch

from drifting_torch.checkpointing.training import load_training_state, save_training_state
from drifting_torch.data.datasets import StatefulSampler
from drifting_torch.training.mae import MAEStepOptions, mae_train_step

from .test_mae_step import fixed_batch, fixed_mask, tiny_mae_state


def execute(state, count):
    options = MAEStepOptions(lambda_cls=0.2, base_lr=1e-3, total_steps=3)
    for _ in range(count):
        mae_train_step(state, fixed_batch(), fixed_mask(), options)


def test_mae_three_step_resume_is_bit_exact(tmp_path: Path):
    direct = tiny_mae_state()
    execute(direct, 3)
    partial = tiny_mae_state()
    execute(partial, 1)
    sampler = StatefulSampler(list(range(8)), shuffle=True, seed=4)
    checkpoint = tmp_path / "mae.pt"
    save_training_state(checkpoint, partial, sampler=sampler, config={"kind": "mae"})
    resumed = tiny_mae_state()
    resumed_sampler = StatefulSampler(list(range(8)), shuffle=True, seed=4)
    load_training_state(checkpoint, resumed, sampler=resumed_sampler, config={"kind": "mae"})
    execute(resumed, 2)
    for name, value in direct.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name]), name
        assert torch.equal(direct.ema[name], resumed.ema[name]), name

