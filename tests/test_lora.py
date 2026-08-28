from __future__ import annotations

import copy
from pathlib import Path

import torch
from peft.tuners.lora.layer import LoraLayer, ParamWrapper
from transformers import (
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
)

from finetune_library.checkpoint import load_adapter
from finetune_library.config import DistributedStrategy, LoraSettings
from finetune_library.lora import consolidate_tied_lora_gradients, inject_lora
from finetune_library.registry import ModelSpec


def dense_spec() -> ModelSpec:
    return ModelSpec(
        key="tiny",
        repo_id="tiny",
        revision="0" * 40,
        architecture="dense",
        layer_class="LlamaDecoderLayer",
        preferred_strategy=DistributedStrategy.DDP,
    )


def moe_spec() -> ModelSpec:
    return ModelSpec(
        key="tiny-moe",
        repo_id="tiny-moe",
        revision="0" * 40,
        architecture="moe",
        layer_class="Gemma4TextDecoderLayer",
        preferred_strategy=DistributedStrategy.DDP,
        moe=True,
    )


def test_dense_lora_covers_linear_embedding_and_head() -> None:
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            tie_word_embeddings=False,
        )
    ).bfloat16()
    peft_model, audit = inject_lora(
        model,
        dense_spec(),
        LoraSettings(rank=4, alpha=8, expert_rank=2),
    )
    kinds = {target.kind for target in audit.targets}
    assert {"linear", "embedding", "lm_head"}.issubset(kinds)
    assert all(
        parameter.dtype == torch.bfloat16
        for parameter in peft_model.parameters()
        if parameter.requires_grad
    )
    assert all(
        "lora_" in name
        for name, parameter in peft_model.named_parameters()
        if parameter.requires_grad
    )
    assert any(isinstance(module, LoraLayer) for module in peft_model.modules())


def test_gemma_moe_uses_expert_rank_for_routed_and_shared_experts(
    tmp_path: Path,
) -> None:
    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        layer_types=["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=8,
        tie_word_embeddings=True,
        use_bidirectional_attention=None,
    )
    # Direct construction validates the generic root rope shape before Gemma's
    # per-attention dictionaries are installed.
    config.rope_parameters = {
        "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    }
    model = Gemma4ForCausalLM(config).bfloat16()
    restored_base = copy.deepcopy(model)
    peft_model, audit = inject_lora(
        model,
        moe_spec(),
        LoraSettings(rank=8, alpha=16, expert_rank=2),
    )
    routed = [target for target in audit.targets if target.kind == "routed_expert"]
    shared = [target for target in audit.targets if target.kind == "shared_expert"]
    router = [target for target in audit.targets if ".router.proj" in target.name]
    assert routed and all(target.rank == 2 for target in routed)
    assert shared and all(target.rank == 2 for target in shared)
    assert router and all(target.rank == 8 for target in router)
    assert any(target.kind == "embedding" for target in audit.targets)
    assert any(target.kind == "lm_head" for target in audit.targets)

    embedding = peft_model.get_input_embeddings()
    head = peft_model.get_output_embeddings()
    adapter = embedding.active_adapters[0]
    tied_pairs = (
        (
            embedding.lora_embedding_A[adapter],
            head.lora_B[adapter].weight,
        ),
        (
            embedding.lora_embedding_B[adapter],
            head.lora_A[adapter].weight,
        ),
    )
    for source, tied in tied_pairs:
        assert source.dtype == tied.dtype == torch.bfloat16
        assert source.untyped_storage().data_ptr() == tied.untyped_storage().data_ptr()
        assert source.shape == tied.T.shape

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    expert_wrappers = [
        module
        for module in peft_model.modules()
        if module.__class__.__name__ == "EagerGemmaExpertWrapper"
    ]
    assert len(expert_wrappers) == 1
    for name, parameter in peft_model.named_parameters():
        if ".experts." in name and ".lora_B." in name:
            torch.nn.init.normal_(parameter, std=0.01)
    with torch.inference_mode():
        active_logits = peft_model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        ).logits
        active_class = expert_wrappers[0].__class__
        expert_wrappers[0].__class__ = ParamWrapper
        materialized_logits = peft_model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        ).logits
        expert_wrappers[0].__class__ = active_class
    assert torch.allclose(
        active_logits.float(),
        materialized_logits.float(),
        rtol=2e-2,
        atol=2e-2,
    )

    output = peft_model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
        num_items_in_batch=torch.tensor(14.0),
    )
    output.loss.backward()
    gradients = [parameter.grad for parameter in peft_model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    expected_tied_gradients = [
        source.grad.detach().clone() + tied.grad.detach().T
        for source, tied in tied_pairs
    ]
    consolidate_tied_lora_gradients(peft_model)
    for (source, tied), expected in zip(
        tied_pairs,
        expected_tied_gradients,
        strict=True,
    ):
        assert tied.grad is None
        assert torch.equal(source.grad, expected)

    peft_model.save_pretrained(
        tmp_path,
        safe_serialization=True,
        save_embedding_layers=False,
    )
    restored, _ = inject_lora(
        restored_base,
        moe_spec(),
        LoraSettings(rank=8, alpha=16, expert_rank=2),
    )
    load_adapter(restored, tmp_path)
    restored.eval()
    with torch.inference_mode():
        restored_logits = restored(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        ).logits
    assert torch.allclose(
        active_logits.float(),
        restored_logits.float(),
        rtol=2e-2,
        atol=2e-2,
    )
    merged = restored.merge_and_unload(safe_merge=True)
    with torch.inference_mode():
        merged_logits = merged(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        ).logits
    assert torch.allclose(
        restored_logits.float(),
        merged_logits.float(),
        rtol=2e-2,
        atol=2e-2,
    )
