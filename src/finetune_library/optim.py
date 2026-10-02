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
    # torch==2.11.0 (this project's pinned version) supports the fused AdamW
    # kernel over DTensor params directly (verified: FSDP2-sharded DTensor
    # params take the fused path and step without error, on both a single
    # shard and multiple shards). Earlier torch releases raised for DTensor
    # here, which is why this check originally excluded them; under FSDP2
    # that meant optimizer.fused=True (the default) was silently downgraded
    # to the foreach path. This restores fused=True actually taking effect.
    if not all(parameter.device.type == "cuda" for parameter in parameters):
        return False
    # A single fused call requires every tensor in it to be the same kind
    # ("aten._fused_adamw_.default: got mixed torch.Tensor and DTensor").
    # build_optimizer splits mixed Tensor/DTensor parameters into separate,
    # internally-homogeneous param groups before calling this, so by the
    # time we get here `parameters` is one such homogeneous group and this
    # check only needs to confirm it really is homogeneous.
    from torch.distributed.tensor import DTensor

    is_dtensor = [isinstance(parameter, DTensor) for parameter in parameters]
    if any(is_dtensor) and not all(is_dtensor):
        return False
    return True


def build_optimizer(model: nn.Module, config: OptimizerConfig, effective_lr: float | None = None) -> torch.optim.Optimizer:
    """Build an unmodified, standard optimizer over the trainable parameters (LoRA or full)."""

    parameters = _trainable_parameters(model)
    lr = config.learning_rate if config.learning_rate is not None else effective_lr
    if lr is None:
        raise ValueError("learning_rate must be set either in config or via effective_lr")
    common: dict[str, Any] = {
        "lr": lr,
        "betas": (config.beta1, config.beta2),
        "eps": config.epsilon,
        "weight_decay": config.weight_decay,
    }
    if config.name == OptimizerName.ADAMW:
        from torch.distributed.tensor import DTensor

        is_dtensor = [isinstance(parameter, DTensor) for parameter in parameters]
        if config.fused and any(is_dtensor) and not all(is_dtensor):
            # Mixed Tensor/DTensor params (EP+FSDP2: routed/shared-expert
            # LoRA stays a plain local Tensor while the rest is sharded as
            # DTensor). Split into two homogeneous param groups so each can
            # still take the fused path internally, instead of downgrading
            # every parameter in the model to the foreach path just because
            # a handful of expert-adapter tensors are not DTensors.
            dtensor_params = [p for p, flag in zip(parameters, is_dtensor) if flag]
            plain_params = [p for p, flag in zip(parameters, is_dtensor) if not flag]
            groups = [
                {
                    "params": group_params,
                    "fused": _can_use_fused_adamw(group_params, config.fused),
                }
                for group_params in (dtensor_params, plain_params)
                if group_params
            ]
            return torch.optim.AdamW(groups, **common)
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
        get_linear_schedule_with_warmup,
        get_wsd_schedule,
    )

    warmup = config.warmup_steps or int(total_steps * config.warmup_ratio)
    if config.name == "constant":
        return get_constant_schedule_with_warmup(optimizer, warmup)
    if config.name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    if config.name == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
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
