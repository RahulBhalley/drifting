import json
from pathlib import Path

import torch
import torch.multiprocessing as mp

from drifting_torch.checkpointing.training import load_training_state, save_training_state
from drifting_torch.data.datasets import StatefulSampler
from drifting_torch.distributed.runtime import DistributedContext
from drifting_torch.distributed.strategies import unwrap_model, wrap_model
from drifting_torch.training.state import GeneratorTrainState


def _worker(rank: int, world_size: int, rendezvous: str, root: str) -> None:
    context = DistributedContext.initialize(
        {
            "strategy": "ddp",
            "device": "cpu",
            "precision": "fp32",
            "rank": rank,
            "local_rank": rank,
            "world_size": world_size,
            "distributed_backend": "gloo",
            "init_method": f"file://{rendezvous}",
        }
    )
    try:
        torch.manual_seed(30)
        model = wrap_model(torch.nn.Linear(2, 2), context)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        state = GeneratorTrainState.create(model, optimizer, ema_decay=0.9, seed=40 + rank)
        state.completed_steps = 7
        original = {
            name: value.clone() for name, value in unwrap_model(model).state_dict().items()
        }
        sampler = StatefulSampler(range(12), shuffle=True, seed=5, rank=rank, world_size=world_size)
        next(iter(sampler))
        sampler_state = sampler.state_dict()
        destination = Path(root) / "step-00000007.pt"
        save_training_state(
            destination,
            state,
            sampler=sampler,
            config={"trajectory": "distributed"},
            context=context,
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        state.completed_steps = 99
        load_training_state(
            destination,
            state,
            sampler=sampler,
            config={"trajectory": "distributed"},
            context=context,
        )
        assert state.completed_steps == 7
        assert sampler.state_dict() == sampler_state
        assert all(
            torch.equal(original[name], value)
            for name, value in unwrap_model(model).state_dict().items()
        )
        context.barrier()
        if context.is_main:
            manifest = json.loads(destination.read_text(encoding="utf-8"))
            shards = destination.with_name(manifest["shards"])
            assert len(list(shards.glob("rank-*.pt"))) == world_size
            assert (shards / manifest["dcp"] / ".metadata").is_file()
        Path(root, f"checkpoint-rank-{rank}.txt").write_text("ok", encoding="utf-8")
    finally:
        context.close()


def test_two_process_collective_checkpoint_and_restore(tmp_path: Path):
    mp.spawn(
        _worker,
        args=(2, str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    assert len(list(tmp_path.glob("checkpoint-rank-*.txt"))) == 2
