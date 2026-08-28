from __future__ import annotations

from typing import Any

import torch
from torch import nn

from finetune_library.config import OptimizerConfig, OptimizerName, SchedulerConfig


def _trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("optimizer received no trainable parameters")
    return parameters


def _can_use_fused_adamw(parameters: list[nn.Parameter], requested: bool) -> bool:
    if not requested:
        return False
    return all(
        parameter.device.type == "cuda" and parameter.__class__.__name__ != "DTensor"
        for parameter in parameters
    )


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """Build an unmodified, standard optimizer over the BF16 LoRA parameters."""

    parameters = _trainable_parameters(model)
    common: dict[str, Any] = {
        "lr": config.learning_rate,
        "betas": (config.beta1, config.beta2),
        "eps": config.epsilon,
        "weight_decay": config.weight_decay,
    }
    if config.name == OptimizerName.ADAMW:
        return torch.optim.AdamW(
            parameters,
            fused=_can_use_fused_adamw(parameters, config.fused),
            **common,
        )
    if config.name == OptimizerName.ADAMW_8BIT:
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise RuntimeError(
                "optimizer.name=adamw_8bit requires the optional bitsandbytes dependency"
            ) from error
        return bnb.optim.AdamW8bit(parameters, **common)
    raise AssertionError(f"unhandled optimizer {config.name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    total_steps: int,
):
    from transformers import (
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_wsd_schedule,
    )

    warmup = config.warmup_steps or int(total_steps * config.warmup_ratio)
    if config.name == "constant":
        return get_constant_schedule_with_warmup(optimizer, warmup)
    if config.name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    if config.name == "wsd":
        assert config.stable_steps is not None
        assert config.decay_steps is not None
        return get_wsd_schedule(
            optimizer,
            num_warmup_steps=warmup,
            num_stable_steps=config.stable_steps,
            num_decay_steps=config.decay_steps,
        )
    raise AssertionError(f"unhandled scheduler {config.name}")
