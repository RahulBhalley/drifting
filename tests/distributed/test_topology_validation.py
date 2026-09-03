import pytest
import torch

from drifting_torch.distributed.runtime import DistributedContext
from drifting_torch.distributed.strategies import strategy_topology


def test_hsdp_rejects_incompatible_mesh_before_initialization():
    with pytest.raises(ValueError, match="replicate_size.*shard_size.*world_size"):
        DistributedContext.initialize(
            {
                "strategy": "hsdp",
                "device": "cpu",
                "precision": "fp32",
                "rank": 0,
                "local_rank": 0,
                "world_size": 4,
                "replicate_size": 3,
                "shard_size": 2,
            }
        )


def test_distributed_strategy_requires_multiple_processes():
    with pytest.raises(ValueError, match="world_size greater than one"):
        DistributedContext.initialize(
            {"strategy": "ddp", "device": "cpu", "precision": "fp32", "world_size": 1}
        )


def test_fsdp_and_hsdp_mesh_dimensions_are_explicit():
    fsdp = DistributedContext("fsdp", torch.device("cuda", 0), "bf16", world_size=8)
    assert strategy_topology(fsdp)[:2] == ((8,), ("shard",))
    hsdp = DistributedContext(
        "hsdp",
        torch.device("cuda", 0),
        "bf16",
        world_size=8,
        replicate_size=2,
        shard_size=4,
    )
    assert strategy_topology(hsdp)[:2] == ((2, 4), ("replicate", "shard"))
