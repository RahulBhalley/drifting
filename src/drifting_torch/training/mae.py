"""Native PyTorch MAE optimization, evaluation, and exact-resume loop."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor

from drifting_torch.checkpointing import (
    load_torch_mae,
    load_training_state,
    save_torch_mae_artifact,
    save_training_state,
)
from drifting_torch.data import DataBatch, create_dataset_split
from drifting_torch.logging import JsonlLogger
from drifting_torch.models.mae import MAEResNet
from drifting_torch.runtime import resolve_device

from .generator import StepResult
from .schedules import assign_learning_rate, build_adamw, learning_rate
from .state import MAETrainState


@dataclass(frozen=True)
class MAEStepOptions:
    lambda_cls: float = 0.0
    mask_ratio_min: float = 0.75
    mask_ratio_max: float = 0.75
    max_grad_norm: float = 2.0
    base_lr: float = 2e-4
    warmup_steps: int = 0
    total_steps: int = 100_000
    schedule: str = "const"


@dataclass(frozen=True)
class MAETrainingSummary:
    completed_steps: int
    resumed_from: int
    checkpoint: Path
    ema_artifact: Path
    metrics: dict[str, float]


def classifier_weight(
    step: int,
    *,
    total: int,
    finetune_steps: int,
    warmup: int,
    target: float,
) -> float:
    start = total - finetune_steps
    if finetune_steps <= 0 or step < start:
        return 0.0
    return float(target * min(1.0, (step - start) / max(1, warmup)))


def _sample_mask(
    model: MAEResNet,
    images: Tensor,
    options: MAEStepOptions,
    generator: torch.Generator,
) -> Tensor:
    batch, _, height, width = images.shape
    height //= model.input_patch_size
    width //= model.input_patch_size
    ratios = torch.rand(batch, generator=generator)
    ratios = ratios * (options.mask_ratio_max - options.mask_ratio_min) + options.mask_ratio_min
    coarse_h, coarse_w = height // model.patch_size, width // model.patch_size
    noise = torch.rand(batch, 1, coarse_h, coarse_w, generator=generator)
    mask = (noise < ratios[:, None, None, None]).float()
    return mask.repeat_interleave(model.patch_size, 2).repeat_interleave(model.patch_size, 3)


def mae_train_step(
    state: MAETrainState,
    batch: DataBatch,
    mask: Tensor | None,
    options: MAEStepOptions,
) -> StepResult:
    state.model.train()
    device = next(state.model.parameters()).device
    images = batch.images.to(device)
    labels = batch.labels.to(device=device, dtype=torch.long)
    if mask is None:
        mask = _sample_mask(state.model, images, options, state.generator)
    mask = mask.to(device)
    rate = learning_rate(
        state.completed_steps,
        base_lr=options.base_lr,
        warmup_steps=options.warmup_steps,
        total_steps=options.total_steps,
        schedule=options.schedule,
    )
    assign_learning_rate(state.optimizer, rate)
    state.optimizer.zero_grad(set_to_none=True)
    output = state.model(
        images,
        labels,
        mask=mask,
        lambda_cls=options.lambda_cls,
        mask_ratio_min=options.mask_ratio_min,
        mask_ratio_max=options.mask_ratio_max,
        train=True,
    )
    loss = output.loss.mean()
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        state.model.parameters(), options.max_grad_norm
    )
    state.optimizer.step()
    state.update_ema()
    state.completed_steps += 1
    metrics = {
        "loss": loss.detach(),
        "cls_loss": output.cls_loss.mean().detach(),
        "recon_loss": output.recon_loss.mean().detach(),
        "accuracy": output.accuracy.mean().detach(),
        "mask_ratio": output.mask_ratio.mean().detach(),
        "lambda_cls": torch.tensor(options.lambda_cls, device=device),
        "g_norm": torch.as_tensor(gradient_norm).detach(),
        "lr": torch.tensor(rate, device=device),
    }
    return StepResult(loss=loss.detach(), metrics=metrics)


@torch.no_grad()
def evaluate_mae(
    state: MAETrainState,
    loader,
    options: MAEStepOptions,
    *,
    preprocess=lambda value: value,
    max_samples: int | None = None,
    use_ema: bool = False,
    no_mask: bool = False,
) -> dict[str, float]:
    original = None
    if use_ema:
        original = {name: value.detach().clone() for name, value in state.model.state_dict().items()}
        state.model.load_state_dict(state.ema, strict=True)
    state.model.eval()
    totals: dict[str, float] = {}
    count = 0
    eval_generator = torch.Generator(device="cpu").manual_seed(0)
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "epoch") and hasattr(sampler, "cursor"):
        sampler.epoch = 0
        sampler.cursor = 0
    try:
        for raw in loader:
            batch = preprocess(raw)
            remaining = batch.images.shape[0]
            if max_samples is not None:
                remaining = min(remaining, max_samples - count)
            if remaining <= 0:
                break
            images = batch.images[:remaining].to(next(state.model.parameters()).device)
            labels = batch.labels[:remaining].to(images.device)
            current = replace(options, mask_ratio_min=0.0, mask_ratio_max=0.0) if no_mask else options
            mask = _sample_mask(state.model, images, current, eval_generator).to(images.device)
            output = state.model(
                images, labels, mask=mask, lambda_cls=current.lambda_cls, train=False
            )
            values = {
                "loss": output.loss.mean(),
                "cls_loss": output.cls_loss.mean(),
                "recon_loss": output.recon_loss.mean(),
                "accuracy": output.accuracy.mean(),
                "mask_ratio": output.mask_ratio.mean(),
            }
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value) * remaining
            count += remaining
            if max_samples is not None and count >= max_samples:
                break
    finally:
        if original is not None:
            state.model.load_state_dict(original, strict=True)
    if count == 0:
        raise ValueError("MAE evaluation loader produced no samples")
    return {name: value / count for name, value in totals.items()}


def _dict(value: Any) -> dict:
    return value.to_dict() if hasattr(value, "to_dict") else dict(value)


def train_mae(config, runtime, workdir: str | Path) -> MAETrainingSummary:
    runtime_values = _dict(runtime or config.runtime or {})
    if runtime_values.get("backend", "torch") != "torch":
        raise ValueError("PyTorch MAE training requires runtime.backend=torch")
    device = resolve_device(runtime_values.get("device", "auto"))
    torch.manual_seed(int(config.train.seed))
    model_config = dict(config.model)
    model_config["num_classes"] = int(config.dataset.num_classes)
    model = MAEResNet(**model_config).to(device)
    optimizer_values = _dict(config.optimizer)
    schedule_values = dict(optimizer_values["lr_schedule"])
    optimizer = build_adamw(
        model.parameters(),
        base_lr=float(schedule_values["learning_rate"]),
        beta1=float(optimizer_values["adam_b1"]),
        beta2=float(optimizer_values["adam_b2"]),
        weight_decay=float(optimizer_values.get("weight_decay", 0.0)),
    )
    state = MAETrainState.create(
        model,
        optimizer,
        ema_decay=float(config.train.get("ema_decay", 0.999)),
        seed=int(config.train.seed),
    )
    train_pipeline = create_dataset_split(config, runtime, "train")
    eval_pipeline = create_dataset_split(config, runtime, "val")
    output = Path(workdir).resolve()
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(checkpoint_root.glob("step-*.pt"))
    resumed_from = 0
    if existing:
        load_training_state(
            existing[-1], state, sampler=train_pipeline.sampler, config=config
        )
        resumed_from = state.completed_steps
    elif config.train.get("init_from", ""):
        loaded = load_torch_mae(config.train["init_from"], device=device)
        state.model.load_state_dict(loaded.model.state_dict(), strict=True)
        state.ema = {name: value.detach().clone() for name, value in state.model.state_dict().items()}

    forward = dict(config.train.get("forward_dict", {}))
    base_options = MAEStepOptions(
        lambda_cls=float(forward.get("lambda_cls", 0.0)),
        mask_ratio_min=float(forward.get("mask_ratio_min", 0.75)),
        mask_ratio_max=float(forward.get("mask_ratio_max", 0.75)),
        max_grad_norm=float(config.train.get("max_grad_norm", 2.0)),
        base_lr=float(schedule_values["learning_rate"]),
        warmup_steps=int(schedule_values["warmup_steps"]),
        total_steps=int(schedule_values["total_steps"]),
        schedule=str(schedule_values["lr_schedule"]),
    )
    total_steps = int(config.train.total_steps)
    iterator = iter(train_pipeline.loader)
    logger = JsonlLogger(output / "metrics.jsonl")
    final_metrics: dict[str, float] = {}
    latest = existing[-1] if existing else checkpoint_root / "step-00000000.pt"
    while state.completed_steps < total_steps:
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(train_pipeline.loader)
            raw = next(iterator)
        batch = train_pipeline.preprocess(raw)
        weight = classifier_weight(
            state.completed_steps,
            total=total_steps,
            finetune_steps=int(config.train.get("finetune_last_steps", 0)),
            warmup=int(config.train.get("warmup_finetune", 1000)),
            target=float(config.train.get("finetune_cls", 0.5)),
        )
        options = replace(base_options, lambda_cls=weight or base_options.lambda_cls)
        started = time.perf_counter()
        result = mae_train_step(state, batch, None, options)
        metrics = {**result.metrics, "step_time": torch.tensor(time.perf_counter() - started)}
        logger.log(state.completed_steps, metrics)
        final_metrics = {name: float(value.detach().cpu()) for name, value in metrics.items()}
        if state.completed_steps % int(config.train.eval_per_step) == 0:
            eval_options = replace(
                base_options, **dict(config.train.get("eval_forward_dict", {}))
            )
            for use_ema in (False, True):
                for no_mask in (False, True):
                    evaluated = evaluate_mae(
                        state,
                        eval_pipeline.loader,
                        eval_options,
                        preprocess=eval_pipeline.preprocess,
                        max_samples=int(config.train.get("eval_samples", 5000)),
                        use_ema=use_ema,
                        no_mask=no_mask,
                    )
                    prefix = f"eval{'_ema' if use_ema else ''}{'_nomask' if no_mask else ''}/"
                    logger.log(state.completed_steps, {prefix + name: value for name, value in evaluated.items()})
        finetune_start = total_steps - int(config.train.get("finetune_last_steps", 0))
        should_save = (
            state.completed_steps == total_steps
            or state.completed_steps == finetune_start
            or (
                state.completed_steps % int(config.train.save_per_step) == 0
                and state.completed_steps < finetune_start
            )
        )
        if should_save:
            latest = checkpoint_root / f"step-{state.completed_steps:08d}.pt"
            save_training_state(
                latest, state, sampler=train_pipeline.sampler, config=config
            )
    artifact = output / "artifacts" / f"mae-ema-step-{state.completed_steps:08d}"
    if not artifact.exists():
        save_torch_mae_artifact(
            artifact,
            state_dict=state.ema,
            model_config=model_config,
            step=state.completed_steps,
            ema_decay=state.ema_decay,
        )
    return MAETrainingSummary(
        state.completed_steps, resumed_from, latest, artifact, final_metrics
    )


__all__ = [
    "MAEStepOptions",
    "MAETrainingSummary",
    "classifier_weight",
    "evaluate_mae",
    "mae_train_step",
    "train_mae",
]
