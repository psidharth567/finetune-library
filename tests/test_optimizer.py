from __future__ import annotations

import copy

import torch
from torch import nn

from finetune_library.config import OptimizerConfig
from finetune_library.optim import build_optimizer


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_weight = nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))


def optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        learning_rate=0.1,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.99,
        epsilon=1e-8,
        fused=False,
    )


def test_builds_standard_adamw_over_bf16_parameters() -> None:
    model = TinyModel()
    optimizer = build_optimizer(model, optimizer_config())
    assert type(optimizer) is torch.optim.AdamW

    gradient = torch.tensor([0.25, -0.5], dtype=torch.bfloat16)
    model.lora_weight.grad = gradient
    optimizer.step()

    state = optimizer.state[model.lora_weight]
    assert model.lora_weight.dtype == torch.bfloat16
    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg_sq"].dtype == torch.bfloat16


def test_standard_adamw_state_round_trip() -> None:
    model = TinyModel()
    optimizer = build_optimizer(model, optimizer_config())
    model.lora_weight.grad = torch.tensor([0.5, 0.25], dtype=torch.bfloat16)
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())

    restored_model = TinyModel()
    restored = build_optimizer(restored_model, optimizer_config())
    restored.load_state_dict(saved)
    assert restored.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    restored_state = next(iter(restored.state.values()))
    original_state = next(iter(optimizer.state.values()))
    assert torch.equal(restored_state["exp_avg"], original_state["exp_avg"])
    assert torch.equal(restored_state["exp_avg_sq"], original_state["exp_avg_sq"])
