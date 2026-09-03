# PyTorch Distributed Execution

Launch distributed runs with `torchrun`; rank, local rank, and world size are
read from its environment. All ranks enter checkpoint and evaluation
collectives. Only rank zero writes metrics and publishes EMA artifacts.

```bash
# Two-process CPU correctness/smoke run
torchrun --standalone --nproc-per-node=2 --no-python drifting-torch-train \
  --config configs/local/m1_fake_smoke.yaml \
  --runtime configs/runtime/torch/ddp_cpu.yaml \
  --workdir runs/ddp-smoke

# Eight-GPU FSDP or 2x4 HSDP
torchrun --standalone --nproc-per-node=8 --no-python drifting-torch-train \
  --config configs/gen/latent_sota_B.yaml \
  --runtime configs/runtime/torch/fsdp_cuda.yaml \
  --workdir runs/fsdp-latent-B

torchrun --standalone --nproc-per-node=8 --no-python drifting-torch-train \
  --config configs/gen/latent_sota_B.yaml \
  --runtime configs/runtime/torch/hsdp_cuda_8gpu.yaml \
  --workdir runs/hsdp-latent-B
```

`ddp` replicates parameters and keeps each process's memory banks local.
`fsdp` creates a one-dimensional full-shard device mesh. `hsdp` creates an
explicit `(replicate_size, shard_size)` mesh and rejects any product that does
not equal `WORLD_SIZE`. CPU uses Gloo; CUDA uses NCCL and selects the local CUDA
device before initializing collectives.

Distributed checkpoints use a manifest published only after every rank shard
exists. Each shard retains that rank's RNG, generator, sampler cursor, memory
banks, model, optimizer, scaler, and completed-step state. Resume rejects a
different strategy, world size, precision, or trajectory hash.

The two-process DDP generator and collective checkpoint paths are tested on
CPU. FSDP/HSDP mesh construction and validation are covered locally, but an
actual CUDA FSDP/HSDP run cannot be verified on the current non-CUDA Mac.
