"""Padding-free isolated packing: collator shape/label semantics + FA2 varlen parity."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from finetune_library.data import IGNORE_INDEX, CausalCollator


class _Tok:
    pad_token_id = 0
    eos_token_id = 1


def test_flattened_collator_rebases_and_masks_boundaries():
    collator = CausalCollator(_Tok(), max_length=16, packing_isolation="attention")
    rows = [
        # window fully containing two docs of len 3 and 2
        {"input_ids": [5, 6, 7, 8, 9], "labels": [5, 6, 7, 8, 9], "position_ids": [0, 1, 2, 0, 1]},
        # window starting mid-document (positions 4,5) then a new doc
        {"input_ids": [10, 11, 12, 13], "labels": [10, 11, 12, 13], "position_ids": [4, 5, 0, 1]},
    ]
    batch = collator(rows)
    assert set(batch) == {"input_ids", "labels", "position_ids"}
    assert batch["input_ids"].shape == (1, 9)
    assert batch["position_ids"][0].tolist() == [0, 1, 2, 0, 1, 0, 1, 0, 1]
    labels = batch["labels"][0].tolist()
    assert [i for i, v in enumerate(labels) if v == IGNORE_INDEX] == [0, 3, 5, 7]
    assert labels[1:3] == [6, 7]


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("flash_attn") is None,
    reason="needs CUDA + flash_attn",
)
def test_fa2_varlen_matches_per_document_sdpa():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
    )
    sdpa = LlamaForCausalLM(cfg).cuda().to(torch.bfloat16).eval()
    sdpa.config._attn_implementation = "sdpa"
    fa2 = LlamaForCausalLM(cfg).cuda().to(torch.bfloat16).eval()
    fa2.load_state_dict(sdpa.state_dict())
    fa2.config._attn_implementation = "flash_attention_2"
    fa2.set_attn_implementation("flash_attention_2")

    docs = [torch.randint(2, 128, (n,), device="cuda") for n in (7, 12, 5)]
    flat_ids = torch.cat(docs)[None]
    flat_pos = torch.cat([torch.arange(len(d), device="cuda") for d in docs])[None]
    with torch.no_grad():
        packed = fa2(input_ids=flat_ids, position_ids=flat_pos).logits[0].float()
        ref = torch.cat([sdpa(input_ids=d[None]).logits[0] for d in docs]).float()
    torch.testing.assert_close(packed, ref, atol=6e-2, rtol=6e-2)
