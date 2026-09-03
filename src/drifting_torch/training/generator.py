"""One-step native PyTorch implementation of Drifting generator training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from drifting_torch.data.datasets import DataBatch
from drifting_torch.loss import drift_loss
from drifting_torch.memory_bank import ClassMemoryBank

from .schedules import assign_learning_rate, learning_rate
from .state import GeneratorTrainState


@dataclass(frozen=True)
class GeneratorStepOptions:
    gen_per_label: int = 8
    pos_per_sample: int = 32
    neg_per_sample: int = 16
    cfg_min: float = 1.0
    cfg_max: float = 4.0
    neg_cfg_pw: float = 1.0
    no_cfg_frac: float = 0.0
    radii: tuple[float, ...] = (0.02, 0.05, 0.2)
    max_grad_norm: float = 2.0
    base_lr: float = 2e-4
    warmup_steps: int = 0
    total_steps: int = 100_000
    schedule: str = "const"
    cfg: Tensor | None = None
    noise: Tensor | None = None
    noise_labels: Tensor | None = None


@dataclass(frozen=True)
class StepResult:
    loss: Tensor
    metrics: dict[str, Tensor]


def _sample_cfg(
    count: int, options: GeneratorStepOptions, generator: torch.Generator
) -> Tensor:
    if options.cfg is not None:
        cfg = options.cfg.detach().cpu().float()
        if cfg.shape != (count,):
            raise ValueError(f"explicit cfg must have shape {(count,)}")
        return cfg
    fraction = torch.rand(count, generator=generator)
    power = 1 - options.neg_cfg_pw
    if abs(power) < 1e-6:
        cfg = torch.exp(
            torch.log(torch.tensor(options.cfg_min))
            + fraction
            * (torch.log(torch.tensor(options.cfg_max)) - torch.log(torch.tensor(options.cfg_min)))
        )
    else:
        cfg = (
            options.cfg_min**power
            + fraction * (options.cfg_max**power - options.cfg_min**power)
        ) ** (1 / power)
    if options.no_cfg_frac:
        no_cfg = torch.rand(count, generator=generator) < options.no_cfg_frac
        cfg = torch.where(no_cfg, torch.ones_like(cfg), cfg)
    return cfg


def _group_features(value: Tensor, batch: int, candidates: int) -> Tensor:
    if value.ndim != 3:
        raise ValueError("feature extractor outputs must have shape (batch, tokens, channels)")
    tokens, width = value.shape[1:]
    return (
        value.reshape(batch, candidates, tokens, width)
        .permute(0, 2, 1, 3)
        .reshape(batch * tokens, candidates, width)
    )


def generator_train_step(
    state: GeneratorTrainState,
    batch: DataBatch,
    banks: tuple[ClassMemoryBank, ClassMemoryBank],
    feature_extractor: nn.Module,
    options: GeneratorStepOptions,
) -> StepResult:
    """Apply one optimizer transition and then update EMA."""
    state.model.train()
    feature_extractor.eval()
    device = next(state.model.parameters()).device
    labels = batch.labels.to(device=device, dtype=torch.long)
    batch_size = labels.shape[0]
    positive_bank, negative_bank = banks
    positives = positive_bank.sample(labels.detach().cpu(), options.pos_per_sample).to(device)
    negatives = negative_bank.sample(
        torch.zeros(batch_size, dtype=torch.long), options.neg_per_sample
    ).to(device)
    reference_count = options.pos_per_sample + options.neg_per_sample
    references = torch.cat((positives, negatives), dim=1)
    reference_features = feature_extractor(
        references.reshape(batch_size * reference_count, *references.shape[2:]),
        detach=True,
    )

    cfg = _sample_cfg(batch_size, options, state.generator).to(device)
    repeated_labels = labels.repeat_interleave(options.gen_per_label)
    repeated_cfg = cfg.repeat_interleave(options.gen_per_label)
    generated_count = repeated_labels.shape[0]
    noise = options.noise
    if noise is None:
        noise = torch.randn(
            generated_count,
            state.model.in_channels,
            state.model.input_size,
            state.model.input_size,
            generator=state.generator,
        )
    noise_labels = options.noise_labels
    if noise_labels is None:
        noise_labels = torch.randint(
            max(1, state.model.noise_classes),
            (generated_count, max(1, state.model.noise_coords)),
            generator=state.generator,
        )
    generated = state.model(
        repeated_labels,
        cfg_scale=repeated_cfg,
        deterministic=False,
        noise=noise.to(device),
        noise_labels=noise_labels.to(device),
    ).samples
    generated_features = feature_extractor(generated)
    if set(reference_features) != set(generated_features):
        raise ValueError("reference and generated feature keys differ")

    unconditional_weight = (
        (cfg - 1)
        * (options.gen_per_label - 1)
        / max(1, options.neg_per_sample)
    )
    total_loss = torch.zeros((), device=device)
    metrics: dict[str, Tensor] = {}
    for name in sorted(generated_features):
        live = _group_features(
            generated_features[name], batch_size, options.gen_per_label
        )
        stopped = _group_features(reference_features[name], batch_size, reference_count)
        fixed_pos = stopped[:, : options.pos_per_sample]
        fixed_neg = stopped[:, options.pos_per_sample :]
        token_count = live.shape[0] // batch_size
        negative_weight = (
            unconditional_weight[:, None, None]
            .expand(batch_size, token_count, options.neg_per_sample)
            .reshape(batch_size * token_count, options.neg_per_sample)
        )
        feature_loss, feature_info = drift_loss(
            live,
            fixed_pos,
            fixed_neg,
            weight_gen=torch.ones_like(live[:, :, 0]),
            weight_pos=torch.ones_like(fixed_pos[:, :, 0]),
            weight_neg=negative_weight,
            R_list=options.radii,
        )
        total_loss = total_loss + feature_loss.mean()
        for metric_name, value in feature_info.items():
            metrics[f"{metric_name}/{name}"] = value.detach()

    rate = learning_rate(
        state.completed_steps,
        base_lr=options.base_lr,
        warmup_steps=options.warmup_steps,
        total_steps=options.total_steps,
        schedule=options.schedule,
    )
    assign_learning_rate(state.optimizer, rate)
    state.optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        state.model.parameters(), options.max_grad_norm
    )
    state.optimizer.step()
    state.update_ema()
    state.completed_steps += 1
    metrics.update(
        {
            "loss": total_loss.detach(),
            "g_norm": torch.as_tensor(gradient_norm).detach(),
            "lr": torch.tensor(rate, device=device),
            "cfg_mean": cfg.mean().detach(),
        }
    )
    return StepResult(loss=total_loss.detach(), metrics=metrics)


__all__ = ["GeneratorStepOptions", "StepResult", "generator_train_step"]
