from __future__ import annotations

import copy

import pytest
import torch
from peft.tuners.lora.layer import ParamWrapper
from transformers import (
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
)

from finetune_library.config import DistributedStrategy, LoraSettings
from finetune_library.lora import inject_lora
from finetune_library.loss import TrainingModel
from finetune_library.registry import ModelSpec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel parity")


def _spec(*, moe: bool = False) -> ModelSpec:
    return ModelSpec(
        key="tiny",
        repo_id="tiny",
        revision="0" * 40,
        architecture="moe" if moe else "dense",
        layer_class="Gemma4TextDecoderLayer" if moe else "LlamaDecoderLayer",
        preferred_strategy=DistributedStrategy.DDP,
        moe=moe,
    )


def _trainable_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().float().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }


def _global_gradient_norm(gradients: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([gradient.norm().square() for gradient in gradients.values()]).sum().sqrt()


def _assert_bf16_gradient_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    name: str,
    *,
    relative_rms_limit: float,
) -> None:
    difference = actual - expected
    relative_rms = float(
        (
            difference.square().mean().sqrt()
            / expected.square().mean().sqrt().clamp_min(1e-8)
        ).detach()
    )
    assert relative_rms < relative_rms_limit, (
        f"{name}: relative_rms={relative_rms:.6f} "
        f"max_abs={float(difference.abs().max()):.6f}"
    )


def test_fused_lora_linear_cross_entropy_matches_transformers() -> None:
    torch.manual_seed(7)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=257,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            tie_word_embeddings=False,
        )
    ).to(device="cuda", dtype=torch.bfloat16)
    reference, _ = inject_lora(
        base,
        _spec(),
        LoraSettings(rank=8, alpha=16, expert_rank=2),
    )
    for name, parameter in reference.named_parameters():
        if parameter.requires_grad and "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.01)
    optimized = copy.deepcopy(reference)

    input_ids = torch.randint(0, 257, (2, 64), device="cuda")
    labels = input_ids.clone()
    labels[:, :9] = -100
    denominator = (labels[:, 1:] != -100).sum().float()
    attention_mask = torch.ones_like(input_ids)

    reference_output = reference(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        num_items_in_batch=denominator,
    )
    reference_output.loss.backward()
    optimized_output = TrainingModel(
        optimized,
        "fused_linear_cross_entropy",
    )(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        num_items_in_batch=denominator,
    )
    optimized_output.loss.backward()

    assert torch.allclose(
        optimized_output.loss.float(),
        reference_output.loss.float(),
        rtol=3e-3,
        atol=3e-3,
    )
    expected = _trainable_gradients(reference)
    actual = _trainable_gradients(optimized)
    assert expected.keys() == actual.keys()
    for name in expected:
        _assert_bf16_gradient_close(
            actual[name],
            expected[name],
            name,
            relative_rms_limit=0.03,
        )


def test_grouped_gemma_experts_match_peft_materialization() -> None:
    torch.manual_seed(11)
    config = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        layer_types=["full_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        enable_moe_block=True,
        num_experts=8,
        top_k_experts=2,
        moe_intermediate_size=32,
        tie_word_embeddings=True,
        use_bidirectional_attention=None,
    )
    config.rope_parameters = {
        "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    }
    grouped, _ = inject_lora(
        Gemma4ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16),
        _spec(moe=True),
        LoraSettings(rank=8, alpha=16, expert_rank=2),
        expert_implementation="grouped_mm",
    )
    for name, parameter in grouped.named_parameters():
        if parameter.requires_grad and "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.01)
    reference, _ = inject_lora(
        Gemma4ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16),
        _spec(moe=True),
        LoraSettings(rank=8, alpha=16, expert_rank=2),
        expert_implementation="eager",
    )
    reference.load_state_dict(grouped.state_dict())
    for module in reference.modules():
        if module.__class__.__name__ == "EagerGemmaExpertWrapper":
            module.__class__ = ParamWrapper

    # Compare the optimized expert kernel with fixed routing first. Whole-model
    # BF16 perturbations can legitimately change a later router's discrete
    # top-k selection, which makes individual end-to-end gradients
    # discontinuous even when the expert operation itself is correct.
    grouped_expert = next(
        module
        for module in grouped.modules()
        if module.__class__.__name__ == "GroupedGemmaExpertWrapper"
    )
    reference_expert = next(
        module
        for module in reference.modules()
        if isinstance(module, ParamWrapper)
        and module.get_base_layer().__class__.__name__ == "Gemma4TextExperts"
    )
    # Gemma's residual stream can be FP32 under DDP even though the requested
    # expert compute and weights are BF16.
    hidden_states = torch.randn(32, config.hidden_size, device="cuda", dtype=torch.float32)
    first_expert = torch.arange(32, device="cuda") % config.num_experts
    top_k_index = torch.stack(
        (first_expert, (first_expert + 1) % config.num_experts),
        dim=-1,
    )
    top_k_weights = torch.tensor([0.625, 0.375], device="cuda").expand(32, -1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference_expert_output = reference_expert(
            hidden_states,
            top_k_index,
            top_k_weights,
        )
        grouped_expert_output = grouped_expert(
            hidden_states,
            top_k_index,
            top_k_weights,
        )
    _assert_bf16_gradient_close(
        grouped_expert_output.float(),
        reference_expert_output.float(),
        "fixed-routing expert output",
        relative_rms_limit=0.02,
    )
    upstream = torch.randn_like(grouped_expert_output)
    (reference_expert_output * upstream).sum().backward()
    (grouped_expert_output * upstream).sum().backward()
    expected_expert_gradients = _trainable_gradients(reference_expert)
    actual_expert_gradients = _trainable_gradients(grouped_expert)
    assert expected_expert_gradients.keys() == actual_expert_gradients.keys()
    for name in expected_expert_gradients:
        _assert_bf16_gradient_close(
            actual_expert_gradients[name],
            expected_expert_gradients[name],
            name,
            relative_rms_limit=0.08,
        )
    reference.zero_grad(set_to_none=True)
    grouped.zero_grad(set_to_none=True)

    input_ids = torch.randint(0, config.vocab_size, (2, 32), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    denominator = torch.tensor(62.0, device="cuda")
    reference_output = reference(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        num_items_in_batch=denominator,
    )
    reference_output.loss.backward()
    grouped_output = grouped(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        num_items_in_batch=denominator,
    )
    grouped_output.loss.backward()

    assert torch.allclose(
        grouped_output.loss.float(),
        reference_output.loss.float(),
        rtol=5e-3,
        atol=5e-3,
    )
    expected = _trainable_gradients(reference)
    actual = _trainable_gradients(grouped)
    assert expected.keys() == actual.keys()
    assert all(torch.isfinite(gradient).all() for gradient in actual.values())
    assert torch.allclose(
        _global_gradient_norm(actual),
        _global_gradient_norm(expected),
        rtol=0.02,
        atol=1e-4,
    )
