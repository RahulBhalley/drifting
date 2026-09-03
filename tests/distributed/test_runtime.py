import os

from drifting_torch.distributed.runtime import DistributedContext


def test_single_runtime_does_not_initialize_process_group(monkeypatch):
    monkeypatch.setenv("RANK", "7")
    context = DistributedContext.initialize(
        {"strategy": "single", "device": "cpu", "precision": "fp32"}
    )
    assert (context.rank, context.world_size, context.is_main) == (0, 1, True)
    assert not context.distributed
    context.close()


def test_runtime_reads_rank_topology_from_environment(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    values = DistributedContext.resolve(
        {"strategy": "ddp", "device": "cpu", "precision": "fp32"}
    )
    assert values["rank"] == 2
    assert values["local_rank"] == 0
    assert values["world_size"] == 4
