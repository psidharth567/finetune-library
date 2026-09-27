"""Packed forward == each example run on its own (GPU).

Covers the isolation mechanisms packed SFT relies on: softmax attention via
position_ids (flash varlen, or sdpa/eager block-diagonal mask) and Qwen3.5
gated-delta-net layers via packed_linear_attention. transformers only derives
the sdpa/eager packed mask without a KV cache, hence use_cache=False (the
trainer always passes it).
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from finetune_library.data import IGNORE_INDEX, CausalCollator, pack_whole_examples
from finetune_library.packed_linear_attention import patch_linear_attention_for_packing

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
LENGTHS = (7, 19, 5, 12)


class _Tok:
    pad_token_id = 0
    eos_token_id = 1


def _docs(vocab: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return [torch.randint(2, vocab, (n,), generator=generator).cuda() for n in LENGTHS]


def _packed_batch(docs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    rows = [{"input_ids": d.tolist(), "labels": d.tolist()} for d in docs]
    windows = pack_whole_examples(rows, max_length=32)
    assert len(windows) > 1 and any(len(w["input_ids"]) > max(LENGTHS) for w in windows)
    batch = CausalCollator(_Tok(), max_length=32, packing_isolation="attention")(windows)
    return {key: value.cuda() for key, value in batch.items()}


def _per_doc_logits(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run each packed example alone, in packed order, and concatenate."""
    ids, pos = batch["input_ids"][0], batch["position_ids"][0]
    starts = (pos == 0).nonzero().flatten().tolist() + [len(pos)]
    return torch.cat(
        [model(input_ids=ids[a:b][None], use_cache=False).logits[0] for a, b in zip(starts, starts[1:])]
    ).float()


def _llama(attn: str) -> torch.nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    model = LlamaForCausalLM(cfg).cuda().to(torch.bfloat16).eval()
    model.set_attn_implementation(attn)
    return model


@cuda
@pytest.mark.parametrize("attn", ["sdpa", "eager", "flash_attention_2"])
def test_dense_packed_matches_per_example(attn: str) -> None:
    if attn == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        pytest.skip("needs flash_attn")
    model = _llama(attn)
    batch = _packed_batch(_docs(128))
    with torch.no_grad():
        packed = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"], use_cache=False).logits[0].float()
        reference = _per_doc_logits(model, batch)
        leaky = model(input_ids=batch["input_ids"], use_cache=False).logits[0].float()  # no position_ids: one long doc
    torch.testing.assert_close(packed, reference, atol=3e-2, rtol=3e-2)
    assert (leaky - reference).abs().max() > 0.1  # the comparison is sensitive to leakage


def _qwen35(attn: str) -> torch.nn.Module:
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=256, hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, linear_conv_kernel_dim=4, linear_key_head_dim=32,
        linear_value_head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
        moe_intermediate_size=64, shared_expert_intermediate_size=64, num_experts=4,
        num_experts_per_tok=2, max_position_embeddings=128,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
    )
    cfg._attn_implementation = attn
    return Qwen3_5MoeForCausalLM(cfg).cuda().to(torch.bfloat16)


@cuda
@pytest.mark.skipif(
    importlib.util.find_spec("fla") is None or importlib.util.find_spec("causal_conv1d") is None,
    reason="needs flash-linear-attention + causal-conv1d",
)
@pytest.mark.parametrize("attn", ["sdpa", "flash_attention_3"])
def test_gated_delta_net_packed_matches_per_example(attn: str) -> None:
    if attn == "flash_attention_3" and importlib.util.find_spec("flash_attn_3") is None:
        pytest.skip("needs flash_attn_3")
    model = _qwen35(attn).eval()
    batch = _packed_batch(_docs(256))
    ids, pos = batch["input_ids"], batch["position_ids"]
    with torch.no_grad():
        reference = _per_doc_logits(model, batch)
        unpatched = model(input_ids=ids, position_ids=pos, use_cache=False).logits[0].float()
        assert patch_linear_attention_for_packing(model) == 2
        packed = model(input_ids=ids, position_ids=pos, use_cache=False).logits[0].float()
        single = model(input_ids=ids[:, :9], use_cache=False).logits[0].float()  # unpacked input passes through
    torch.testing.assert_close(packed, reference, atol=3e-2, rtol=3e-2)
    # Without the patch the GDN state leaks across examples (attention alone is isolated).
    assert (unpatched - reference).abs().max() > 0.1
    torch.testing.assert_close(single, _per_doc_logits(model, {"input_ids": ids[:, :9], "position_ids": torch.arange(9, device="cuda")[None]}), atol=3e-2, rtol=3e-2)


@cuda
@pytest.mark.skipif(
    importlib.util.find_spec("fla") is None or importlib.util.find_spec("causal_conv1d") is None,
    reason="needs flash-linear-attention + causal-conv1d",
)
def test_gated_delta_net_packed_gradients_with_checkpointing() -> None:
    model = _qwen35("sdpa").train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    patch_linear_attention_for_packing(model)
    batch = _packed_batch(_docs(256))
    ids, pos, labels = batch["input_ids"], batch["position_ids"], batch["labels"]
    assert labels[0][pos[0] == 0].eq(IGNORE_INDEX).all()

    def grads(loss: torch.Tensor) -> list[torch.Tensor]:
        model.zero_grad(set_to_none=True)
        loss.backward()
        return [p.grad.float().clone() for p in model.parameters() if p.grad is not None]

    def token_loss_sum(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            logits[:-1].float(), targets[1:], ignore_index=IGNORE_INDEX, reduction="sum"
        )

    packed = grads(token_loss_sum(model(input_ids=ids, position_ids=pos, use_cache=False).logits[0], labels[0]))
    starts = (pos[0] == 0).nonzero().flatten().tolist() + [pos.shape[1]]
    separate = grads(
        sum(
            token_loss_sum(model(input_ids=ids[:, a:b], use_cache=False).logits[0], labels[0, a:b])
            for a, b in zip(starts, starts[1:])
        )
    )
    assert len(packed) == len(separate) > 0
    for p, s in zip(packed, separate):
        torch.testing.assert_close(p, s, atol=5e-2, rtol=5e-2)
