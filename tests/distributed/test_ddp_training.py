from pathlib import Path
import json

import torch
import torch.multiprocessing as mp

from drifting_torch.distributed.runtime import DistributedContext
from drifting_torch.distributed.strategies import unwrap_model, wrap_model
from drifting_torch.data.datasets import DataBatch
from drifting_torch.memory_bank import ClassMemoryBank
from drifting_torch.models.features import FrozenFeatureExtractor
from drifting_torch.models.generator import DitGen
from drifting_torch.training.generator import GeneratorStepOptions, generator_train_step
from drifting_torch.training.schedules import build_adamw
from drifting_torch.training.state import GeneratorTrainState
from drifting_common.config import compose_config
from drifting_torch.training.engine import train_generator


def _worker(rank: int, world_size: int, rendezvous: str, result_dir: str) -> None:
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
        torch.manual_seed(10)
        model = wrap_model(
            DitGen(
                cond_dim=16,
                num_classes=3,
                input_size=4,
                in_channels=1,
                patch_size=2,
                hidden_size=16,
                depth=1,
                num_heads=2,
                mlp_ratio=2,
                out_channels=1,
            ),
            context,
        )
        optimizer = build_adamw(model.parameters(), base_lr=1e-3, beta1=0.9, beta2=0.95)
        state = GeneratorTrainState.create(model, optimizer, ema_decay=0.9, seed=11 + rank)
        positive = ClassMemoryBank(num_classes=3, capacity=4, seed=5 + rank)
        negative = ClassMemoryBank(num_classes=1, capacity=8, seed=7 + rank)
        images = torch.linspace(-1, 1, 6 * 16).reshape(6, 1, 4, 4)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        positive.add(images, labels)
        negative.add(images, torch.zeros_like(labels))
        batch = DataBatch(torch.zeros(2, 1, 4, 4), torch.tensor([rank, (rank + 1) % 3]))
        generator_train_step(
            state,
            batch,
            (positive, negative),
            FrozenFeatureExtractor(),
            GeneratorStepOptions(
                gen_per_label=2,
                pos_per_sample=2,
                neg_per_sample=2,
                radii=(0.2,),
                base_lr=1e-3,
                total_steps=1,
            ),
        )
        assert state.completed_steps == 1
        weight = unwrap_model(model).class_embed.weight.detach()
        gathered = [torch.empty_like(weight) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, weight)
        assert all(torch.equal(gathered[0], item) for item in gathered[1:])
        Path(result_dir, f"rank-{rank}.txt").write_text("ok", encoding="utf-8")
    finally:
        context.close()


def test_two_process_ddp_step(tmp_path: Path):
    mp.spawn(
        _worker,
        args=(2, str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    assert sorted(path.read_text() for path in tmp_path.glob("rank-*.txt")) == ["ok", "ok"]


def _engine_worker(rank: int, world_size: int, rendezvous: str, root: str) -> None:
    config = compose_config(
        "configs/local/m1_fake_smoke.yaml", "configs/runtime/torch/cpu.yaml"
    )
    runtime = {
        "backend": "torch",
        "strategy": "ddp",
        "device": "cpu",
        "precision": "fp32",
        "rank": rank,
        "local_rank": rank,
        "world_size": world_size,
        "distributed_backend": "gloo",
        "init_method": f"file://{rendezvous}",
    }
    summary = train_generator(config, runtime, root)
    assert summary.completed_steps == 1
    Path(root, f"engine-rank-{rank}.txt").write_text("ok", encoding="utf-8")


def test_two_process_generator_engine_has_one_log_and_collective_checkpoint(tmp_path: Path):
    root = tmp_path / "run"
    mp.spawn(
        _engine_worker,
        args=(2, str(tmp_path / "engine-rendezvous"), str(root)),
        nprocs=2,
        join=True,
    )
    assert len(list(root.glob("engine-rank-*.txt"))) == 2
    assert len((root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    manifest = json.loads((root / "checkpoints/step-00000001.pt").read_text())
    assert manifest["world_size"] == 2
