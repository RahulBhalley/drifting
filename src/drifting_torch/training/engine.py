"""End-to-end native PyTorch generator training loop."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from drifting_torch.checkpointing import (
    load_torch_generator,
    load_torch_mae,
    load_training_state,
    save_torch_generator_artifact,
    save_training_state,
)
from drifting_torch.data import create_dataset_split
from drifting_torch.distributed import DistributedContext, unwrap_model, wrap_model
from drifting_torch.evaluation import ReleasedInception, compute_statistics, evaluate_generator
from drifting_torch.logging import JsonlLogger
from drifting_torch.memory_bank import ClassMemoryBank
from drifting_torch.models import ConvNeXtV2FeatureExtractor, FrozenFeatureExtractor
from drifting_torch.models.generator import build_generator
from .generator import GeneratorStepOptions, generator_train_step
from .schedules import build_adamw
from .state import GeneratorTrainState


@dataclass(frozen=True)
class TrainingSummary:
    completed_steps: int
    resumed_from: int
    checkpoint: Path
    ema_artifact: Path
    metrics: dict[str, float]


def _dict(value: Any) -> dict:
    return value.to_dict() if hasattr(value, "to_dict") else dict(value)


def _build_features(config, pipeline, device: torch.device) -> FrozenFeatureExtractor:
    values = dict(config.feature)
    mae = None
    convnext = None
    if values.get("use_mae", True):
        path = values.get("mae_path")
        if not path:
            raise ValueError("feature.mae_path is required when use_mae=true")
        mae = load_torch_mae(path, device=device).model
    if values.get("use_convnext", False):
        convnext = ConvNeXtV2FeatureExtractor.from_pretrained(
            values.get("convnext_model", "base"),
            cache_dir=values.get("convnext_cache_dir"),
            local_files_only=bool(values.get("local_files_only", False)),
        ).to(device)
    return FrozenFeatureExtractor(
        mae=mae,
        convnext=convnext,
        postprocess_fn=pipeline.postprocess,
        mae_options=values.get("mae_options"),
        convnext_options=values.get("convnext_options"),
    ).to(device)


def _artifact_ema(state: GeneratorTrainState) -> dict[str, torch.Tensor]:
    values = {}
    for name, value in state.ema.items():
        while name.startswith("module.") or name.startswith("_orig_mod."):
            name = name.split(".", 1)[1]
        values[name] = value
    return values


def should_evaluate(step: int, total_steps: int, every: int, enabled: bool) -> bool:
    return bool(enabled) and (step == 1 or step == total_steps or step % every == 0)


def _required_path(kwargs: dict[str, Any], key: str, environment: str) -> Path:
    value = kwargs.get(key) or os.environ.get(environment)
    if not value:
        raise ValueError(
            f"generator evaluation requires dataset.kwargs.{key} or {environment}"
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"generator evaluation file does not exist: {path}")
    return path


def _reference_statistics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as values:
        if "ref_mu" in values:
            return values["ref_mu"], values["ref_sigma"]
        return values["mu"], values["sigma"]


def _repeat_loader(loader):
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("evaluation loader produced no batches")


def _evaluate_state(
    config,
    state: GeneratorTrainState,
    eval_pipeline,
    device: torch.device,
    *,
    num_samples: int,
    cfg_scale: float,
):
    dataset_kwargs = dict(config.dataset.get("kwargs", {}))
    weights = _required_path(dataset_kwargs, "inception_weights", "DRIFTING_INCEPTION_WEIGHTS")
    fid_stats = _required_path(dataset_kwargs, "fid_stats_path", "DRIFTING_FID_STATS")
    reference_mu, reference_sigma = _reference_statistics(fid_stats)
    eval_options = dict(config.train.get("eval_forward_dict", {}))
    compute_pr = bool(eval_options.get("eval_prc_recall", False))
    reference_features = None
    if compute_pr:
        pr_path = _required_path(dataset_kwargs, "pr_reference_path", "DRIFTING_PR_REFERENCE")
        with np.load(pr_path) as values:
            if "features" not in values:
                raise ValueError("PyTorch PR reference NPZ must contain a 'features' array")
            reference_features = values["features"]

    evaluation_model = copy.deepcopy(unwrap_model(state.model)).to(device)
    evaluation_model.load_state_dict(_artifact_ema(state), strict=True)
    evaluation_model.eval()
    inception = ReleasedInception.from_pickle(weights).to(device)
    generator = torch.Generator(device="cpu").manual_seed(int(config.train.seed))
    base_model = unwrap_model(evaluation_model)

    def generate_batch(raw, _batch_index):
        labels = torch.as_tensor(raw[1], device=device, dtype=torch.long)
        noise = torch.randn(
            len(labels), base_model.in_channels, base_model.input_size, base_model.input_size,
            generator=generator,
        )
        noise_labels = torch.randint(
            max(1, base_model.noise_classes),
            (len(labels), max(1, base_model.noise_coords)),
            generator=generator,
        )
        with torch.inference_mode():
            generated = evaluation_model(
                labels,
                cfg_scale=cfg_scale,
                noise=noise.to(device),
                noise_labels=noise_labels.to(device),
            ).samples
            return eval_pipeline.postprocess(generated)

    eval_pipeline.sampler.epoch = 0
    eval_pipeline.sampler.cursor = 0
    return evaluate_generator(
        _repeat_loader(eval_pipeline.loader),
        generate_batch,
        inception,
        reference_mu=reference_mu,
        reference_sigma=reference_sigma,
        num_samples=num_samples,
        reference_features=reference_features,
        compute_is=bool(eval_options.get("eval_isc", True)),
        compute_pr=compute_pr,
        splits=int(eval_options.get("splits", 10)),
    )


def _train_generator(config, runtime, workdir: str | Path, context: DistributedContext) -> TrainingSummary:
    runtime_values = _dict(runtime or config.runtime or {})
    if runtime_values.get("backend", "torch") != "torch":
        raise ValueError("PyTorch training requires runtime.backend=torch")
    device = context.device
    torch.manual_seed(int(config.train.seed))
    model_config = dict(config.model)
    model_config["num_classes"] = int(config.dataset.num_classes)
    model_config["use_bf16"] = context.precision == "bf16"
    model_config["use_fp16"] = context.precision == "fp16"
    model = build_generator(model_config)
    if runtime_values.get("compile", False):
        model = torch.compile(model)
    model = wrap_model(model, context)
    optimizer_values = _dict(config.optimizer)
    schedule_values = dict(optimizer_values["lr_schedule"])
    optimizer = build_adamw(
        model.parameters(),
        base_lr=float(schedule_values["learning_rate"]),
        beta1=float(optimizer_values["adam_b1"]),
        beta2=float(optimizer_values["adam_b2"]),
        weight_decay=float(optimizer_values.get("weight_decay", 0.0)),
    )
    state = GeneratorTrainState.create(
        model,
        optimizer,
        ema_decay=float(config.train.get("ema_decay", 0.999)),
        seed=int(config.train.seed),
    )
    pipeline = create_dataset_split(config, runtime, "train")
    eval_pipeline = None
    if bool(config.train.get("enable_eval", True)):
        eval_pipeline = create_dataset_split(config, runtime, "val")
    feature_extractor = _build_features(config, pipeline, device)
    positive_bank = ClassMemoryBank(
        num_classes=int(config.dataset.num_classes),
        capacity=int(config.train.get("positive_bank_size", 64)),
        seed=int(config.train.seed) + 1,
    )
    negative_bank = ClassMemoryBank(
        num_classes=1,
        capacity=int(config.train.get("negative_bank_size", 512)),
        seed=int(config.train.seed) + 2,
    )
    banks = (positive_bank, negative_bank)
    output = Path(workdir).resolve()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    existing = sorted(checkpoints.glob("step-*.pt"))
    resumed_from = 0
    if existing:
        load_training_state(
            existing[-1], state, sampler=pipeline.sampler, banks=banks, config=config,
            context=context,
        )
        resumed_from = state.completed_steps
    elif config.train.get("init_from", ""):
        loaded = load_torch_generator(config.train["init_from"], device=device)
        unwrap_model(state.model).load_state_dict(loaded.model.state_dict(), strict=True)
        state.ema = {
            name: value.detach().clone()
            for name, value in state.model.state_dict().items()
        }

    forward = dict(config.train.get("forward_dict", {}))
    options = GeneratorStepOptions(
        gen_per_label=int(forward.get("gen_per_label", 8)),
        pos_per_sample=int(config.train.get("pos_per_sample", 32)),
        neg_per_sample=int(config.train.get("neg_per_sample", 16)),
        cfg_min=float(forward.get("cfg_min", 1.0)),
        cfg_max=float(forward.get("cfg_max", 4.0)),
        neg_cfg_pw=float(forward.get("neg_cfg_pw", 1.0)),
        no_cfg_frac=float(forward.get("no_cfg_frac", 0.0)),
        radii=tuple(float(value) for value in config.train.get("loss_kwargs", {}).get("R_list", (0.02, 0.05, 0.2))),
        max_grad_norm=float(config.train.get("max_grad_norm", 2.0)),
        base_lr=float(schedule_values["learning_rate"]),
        warmup_steps=int(schedule_values["warmup_steps"]),
        total_steps=int(schedule_values["total_steps"]),
        schedule=str(schedule_values["lr_schedule"]),
    )
    total_steps = int(config.train.total_steps)
    train_batch_size = int(config.train.get("train_batch_size", 0))
    save_every = int(config.train.save_per_step)
    iterator = iter(pipeline.loader)
    logger = JsonlLogger(output / "metrics.jsonl", enabled=context.is_main)
    final_metrics: dict[str, float] = {}
    latest_checkpoint = existing[-1] if existing else checkpoints / "step-00000000.pt"
    while state.completed_steps < total_steps:
        process_started = time.perf_counter()
        pushed = 0
        push_goal = int(config.train.get("push_per_step", 0))
        while pushed < max(1, push_goal):
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(pipeline.loader)
                raw = next(iterator)
            batch = pipeline.preprocess(raw)
            positive_bank.add(batch.images.detach().cpu(), batch.labels.detach().cpu())
            negative_bank.add(
                batch.images.detach().cpu(), torch.zeros_like(batch.labels.detach().cpu())
            )
            pushed += batch.images.shape[0]
        if train_batch_size:
            if batch.labels.shape[0] < train_batch_size:
                raise ValueError(
                    f"last pushed batch has {batch.labels.shape[0]} samples, "
                    f"smaller than train_batch_size={train_batch_size}"
                )
            selected = torch.randperm(
                batch.labels.shape[0], generator=state.generator
            )[:train_batch_size].to(batch.labels.device)
            batch = type(batch)(
                batch.images[selected], batch.labels[selected]
            )
        process_elapsed = time.perf_counter() - process_started
        started = time.perf_counter()
        result = generator_train_step(state, batch, banks, feature_extractor, options)
        elapsed = time.perf_counter() - started
        metrics = {
            **result.metrics,
            "process_time": torch.tensor(process_elapsed),
            "step_time": torch.tensor(elapsed),
            "kimg": torch.tensor(state.completed_steps * batch.images.shape[0] / 1000),
            "forward_kimg": torch.tensor(
                state.completed_steps
                * batch.images.shape[0]
                * options.gen_per_label
                / 1000
            ),
        }
        logger.log(state.completed_steps, metrics)
        final_metrics = {
            name: float(value.detach().cpu()) for name, value in metrics.items()
        }
        if should_evaluate(
            state.completed_steps,
            total_steps,
            int(config.train.eval_per_step),
            bool(config.train.get("enable_eval", True)),
        ):
            is_sanity = state.completed_steps == 1
            sample_count = 500 if is_sanity else int(config.train.get("eval_samples", 50_000))
            cfg_values = [float(value) for value in config.train.get("cfg_list", [1.0])]
            if is_sanity:
                cfg_values = cfg_values[:1]
            best_fid, best_cfg = float("inf"), cfg_values[0]
            for cfg_scale in cfg_values:
                result = _evaluate_state(
                    config,
                    state,
                    eval_pipeline,
                    device,
                    num_samples=sample_count,
                    cfg_scale=cfg_scale,
                )
                prefix = f"{'sanity' if is_sanity else 'CFG'}/{cfg_scale:g}/"
                logger.log(
                    state.completed_steps,
                    {prefix + name: value for name, value in result.metrics.items()},
                )
                if result.metrics.get("fid", float("inf")) < best_fid:
                    best_fid = result.metrics["fid"]
                    best_cfg = cfg_scale
            if not is_sanity:
                logger.log(
                    state.completed_steps,
                    {"best_fid": best_fid, "best_cfg": best_cfg},
                )
        if state.completed_steps % save_every == 0 or state.completed_steps == total_steps:
            latest_checkpoint = checkpoints / f"step-{state.completed_steps:08d}.pt"
            save_training_state(
                latest_checkpoint,
                state,
                sampler=pipeline.sampler,
                banks=banks,
                config=config,
                context=context,
            )

    artifact = output / "artifacts" / f"generator-ema-step-{state.completed_steps:08d}"
    context.barrier()
    if context.is_main and not artifact.exists():
        save_torch_generator_artifact(
            artifact,
            state_dict=_artifact_ema(state),
            model_config=model_config,
            step=state.completed_steps,
            ema_decay=state.ema_decay,
        )
    context.barrier()
    return TrainingSummary(
        completed_steps=state.completed_steps,
        resumed_from=resumed_from,
        checkpoint=latest_checkpoint,
        ema_artifact=artifact,
        metrics=final_metrics,
    )


def train_generator(config, runtime, workdir: str | Path) -> TrainingSummary:
    context = DistributedContext.initialize(runtime or config.runtime or {})
    try:
        return _train_generator(config, runtime, workdir, context)
    finally:
        context.close()


__all__ = ["TrainingSummary", "train_generator"]
