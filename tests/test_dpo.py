from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from test_data import FakeTokenizer
from test_lora import dense_spec
from transformers import LlamaConfig, LlamaForCausalLM

from finetune_library.config import ExperimentConfig, PreferenceConfig
from finetune_library.data import IGNORE_INDEX, cache_key
from finetune_library.dpo import (
    STAT_NAMES,
    PreferenceCollator,
    PreferenceModel,
    adapters_disabled,
    iter_preference_rows,
    preference_conversations,
    preference_losses,
    tokenize_response,
)
from finetune_library.lora import inject_lora


def dpo_raw(**updates: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "task": "dpo",
        "model": {"name": "qwen3-8b"},
        "data": {"format": "preference", "path": "unused.jsonl"},
        "training": {"output_dir": "outputs/test-dpo", "max_seq_length": 64},
    }
    raw.update(updates)
    return raw


def dpo_config(**updates: Any) -> ExperimentConfig:
    return ExperimentConfig.model_validate(dpo_raw(**updates))


# ------------------------------------------------------------------ config


def test_dpo_defaults_preference_block_and_isolates() -> None:
    config = dpo_config()
    assert config.preference == PreferenceConfig()
    assert config.packing_isolation() == "attention"
    assert "preference" in config.to_json_dict()


def test_dpo_rejects_wrong_format_fused_loss_compile_and_stray_block() -> None:
    with pytest.raises(ValueError, match="data.format=preference"):
        dpo_config(data={"format": "messages", "path": "x.jsonl"})
    with pytest.raises(ValueError, match="fused"):
        dpo_config(runtime={"loss": "fused_linear_cross_entropy"})
    with pytest.raises(ValueError, match="torch_compile"):
        dpo_config(runtime={"torch_compile": True})
    with pytest.raises(ValueError, match="label_smoothing"):
        dpo_config(preference={"loss_type": "ipo", "label_smoothing": 0.1})
    sft = {
        "task": "sft",
        "model": {"name": "qwen3-8b"},
        "data": {"format": "messages", "path": "x.jsonl"},
        "training": {"output_dir": "o"},
        "preference": {},
    }
    with pytest.raises(ValueError, match="task=dpo"):
        ExperimentConfig.model_validate(sft)


def test_cpt_resolved_config_omits_preference_block() -> None:
    config = ExperimentConfig.from_yaml("configs/models/qwen3-8b-cpt.yaml")
    assert config.preference is None
    assert "preference" not in config.to_json_dict()


def test_dpo_cache_key_tracks_preference_fields() -> None:
    config = dpo_config()
    renamed = dpo_config(preference={"chosen_field": "better"})
    rebeta = dpo_config(preference={"beta": 0.5})
    assert cache_key(config, FakeTokenizer()) != cache_key(renamed, FakeTokenizer())
    # beta does not change prepared data
    assert cache_key(config, FakeTokenizer()) == cache_key(rebeta, FakeTokenizer())


# ------------------------------------------------------------------ data


def test_preference_conversations_accept_olmo_and_string_layouts() -> None:
    config = dpo_config()
    olmo = {
        "prompt": "q",
        "chosen": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "good"}],
        "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "bad"}],
    }
    context, chosen, rejected = preference_conversations(olmo, config)
    assert context == [{"role": "user", "content": "q"}]
    assert (chosen["content"], rejected["content"]) == ("good", "bad")

    flat = {"prompt": "q", "chosen": "good", "rejected": "bad"}
    assert preference_conversations(flat, config) == (context, chosen, rejected)

    mismatch = {**olmo, "rejected": [{"role": "user", "content": "other"}, olmo["rejected"][1]]}
    with pytest.raises(ValueError, match="same prompt"):
        preference_conversations(mismatch, config)
    with pytest.raises(ValueError, match="prompt"):
        preference_conversations({"chosen": "a", "rejected": "b"}, config)


def test_preference_labels_cover_only_the_final_response() -> None:
    config = dpo_config()
    tokenizer = FakeTokenizer()
    context = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "earlier"},
        {"role": "user", "content": "again"},
    ]
    final = {"role": "assistant", "content": "final"}
    ids, labels = tokenize_response(tokenizer, context, final, config)
    context_length = len(tokenizer.apply_chat_template(context))
    assert all(label == IGNORE_INDEX for label in labels[:context_length])
    supervised = [token for token in labels if token != IGNORE_INDEX]
    final_turn = [70, *tokenizer("final")["input_ids"], tokenizer.eos_token_id]
    assert supervised == final_turn
    assert ids[context_length:] == final_turn


def test_preference_rows_drop_overlength_and_identical_pairs(tmp_path: Path) -> None:
    rows = [
        {"prompt": "q", "chosen": "good", "rejected": "bad"},
        {"prompt": "q", "chosen": "x" * 200, "rejected": "bad"},  # over 64 tokens
        {"prompt": "q", "chosen": "same", "rejected": "same"},
    ]
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = dpo_config(data={"format": "preference", "path": str(path)})
    stats: dict[str, Any] = {}
    prepared = list(iter_preference_rows(config, FakeTokenizer(), stats))
    assert len(prepared) == 1
    assert stats == {
        "raw_records": 3,
        "dropped_overlength": 1,
        "longest_dropped_tokens": stats["longest_dropped_tokens"],
        "dropped_identical": 1,
        "pairs": 1,
    }
    assert stats["longest_dropped_tokens"] > 64
    errors = dpo_config(data={"format": "preference", "path": str(path), "sft_overlength": "error"})
    with pytest.raises(ValueError, match="never truncated"):
        list(iter_preference_rows(errors, FakeTokenizer(), {}))


def test_preference_collator_interleaves_pairs_and_isolates_sequences() -> None:
    rows = [
        {
            "chosen_input_ids": [5, 6, 7],
            "chosen_labels": [5, 6, 7],
            "rejected_input_ids": [5, 8],
            "rejected_labels": [IGNORE_INDEX, 8],
        },
        {
            "chosen_input_ids": [9, 9],
            "chosen_labels": [9, 9],
            "rejected_input_ids": [4, 4, 4],
            "rejected_labels": [4, 4, 4],
        },
    ]
    batch = PreferenceCollator()(rows)
    assert batch["input_ids"].tolist() == [[5, 6, 7, 5, 8, 9, 9, 4, 4, 4]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1, 0, 1, 0, 1, 2]]
    assert batch["sequence_index"].tolist() == [[0, 0, 0, 1, 1, 2, 2, 3, 3, 3]]
    assert batch["labels"].tolist() == [[-100, 6, 7, -100, 8, -100, 9, -100, 4, 4]]
    assert int(batch["pair_count"]) == 2


# ------------------------------------------------------------------ loss


def test_losses_at_zero_margin() -> None:
    zero = torch.zeros(3)
    tokens = torch.full((3,), 4.0)
    for loss_type, expected in (("sigmoid", math.log(2)), ("hinge", 1.0), ("ipo", (1 / 0.2) ** 2)):
        losses, margin = preference_losses(
            zero, zero, zero, zero, tokens, tokens, PreferenceConfig(loss_type=loss_type)
        )
        torch.testing.assert_close(losses, torch.full((3,), expected))
        assert margin.abs().max() == 0


def _tiny_policy() -> torch.nn.Module:
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            tie_word_embeddings=False,
            attn_implementation="sdpa",
        )
    ).bfloat16()
    from finetune_library.config import LoraSettings

    peft_model, _ = inject_lora(model, dense_spec(), LoraSettings(rank=4, alpha=8, expert_rank=2))
    # inject_lora requires BF16; upcast afterwards so the comparison is exact to fp32.
    return peft_model.float()


def _pairs() -> list[dict[str, list[int]]]:
    generator = torch.Generator().manual_seed(1)

    def side(prompt: list[int], length: int) -> tuple[list[int], list[int]]:
        response = torch.randint(3, 64, (length,), generator=generator).tolist()
        return prompt + response, [IGNORE_INDEX] * len(prompt) + response

    rows = []
    for prompt_length, chosen_length, rejected_length in ((5, 7, 3), (2, 4, 9), (6, 1, 5)):
        prompt = torch.randint(3, 64, (prompt_length,), generator=generator).tolist()
        chosen_ids, chosen_labels = side(prompt, chosen_length)
        rejected_ids, rejected_labels = side(prompt, rejected_length)
        rows.append(
            {
                "chosen_input_ids": chosen_ids,
                "chosen_labels": chosen_labels,
                "rejected_input_ids": rejected_ids,
                "rejected_labels": rejected_labels,
            }
        )
    return rows


def _sequence_logp(peft_model: Any, ids: list[int], labels: list[int]) -> torch.Tensor:
    """Independent path: one unpadded sequence, full logits, log_softmax."""
    logits = peft_model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float()
    targets = torch.tensor(labels[1:])
    mask = targets.ne(IGNORE_INDEX)
    logps = torch.log_softmax(logits[:-1], dim=-1).gather(-1, targets.clamp(min=0)[:, None])[:, 0]
    return (logps * mask).sum()


def test_preference_model_matches_independent_per_sequence_path() -> None:
    peft_model = _tiny_policy()
    with torch.no_grad():  # move the policy away from the reference
        for name, parameter in peft_model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.2)
    preference = PreferenceConfig(beta=0.5, logprob_chunk_tokens=4)
    rows = _pairs()

    model = PreferenceModel(peft_model, preference)
    batch = PreferenceCollator()(rows)
    output = model(**batch, num_items_in_batch=torch.tensor(float(len(rows))))
    output.loss.backward()
    grads = {n: p.grad.clone() for n, p in peft_model.named_parameters() if p.requires_grad}
    peft_model.zero_grad(set_to_none=True)

    losses = []
    margins = []
    for row in rows:
        with torch.no_grad(), peft_model.disable_adapter():
            ref_c = _sequence_logp(peft_model, row["chosen_input_ids"], row["chosen_labels"])
            ref_r = _sequence_logp(peft_model, row["rejected_input_ids"], row["rejected_labels"])
        pol_c = _sequence_logp(peft_model, row["chosen_input_ids"], row["chosen_labels"])
        pol_r = _sequence_logp(peft_model, row["rejected_input_ids"], row["rejected_labels"])
        margin = 0.5 * ((pol_c - ref_c) - (pol_r - ref_r))
        margins.append(margin.detach())
        losses.append(-F.logsigmoid(margin))
    expected = torch.stack(losses).mean()
    expected.backward()

    torch.testing.assert_close(output.loss, expected, atol=1e-5, rtol=1e-5)
    assert torch.stack(margins).abs().min() > 1e-3  # the check is not vacuous
    stats = dict(zip(STAT_NAMES, model.last_stats.tolist(), strict=True))
    torch.testing.assert_close(
        torch.tensor(stats["rewards_margin"]), torch.stack(margins).sum(), atol=1e-4, rtol=1e-5
    )
    torch.testing.assert_close(torch.tensor(stats["loss"]), expected.detach() * len(rows))
    assert stats["tokens"] == batch["input_ids"].shape[1]
    for name, parameter in peft_model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(grads[name], parameter.grad, atol=1e-5, rtol=1e-4)


def test_fresh_adapter_starts_at_zero_margin() -> None:
    """LoRA B starts at zero, so the policy equals the reference: loss is ln 2."""
    peft_model = _tiny_policy()
    model = PreferenceModel(peft_model, PreferenceConfig())
    rows = _pairs()
    output = model(**PreferenceCollator()(rows), num_items_in_batch=torch.tensor(float(len(rows))))
    assert output.loss.item() == pytest.approx(math.log(2), abs=1e-6)
    stats = dict(zip(STAT_NAMES, model.last_stats.tolist(), strict=True))
    assert stats["rewards_margin"] == pytest.approx(0.0, abs=1e-6)


def _tiny_gemma_moe(experts: str) -> torch.nn.Module:
    from test_lora import moe_spec
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    from finetune_library.config import LoraSettings

    torch.manual_seed(0)
    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        layer_types=["full_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=8,
        tie_word_embeddings=True,
        use_bidirectional_attention=None,
        attn_implementation="sdpa",
    )
    config.rope_parameters = {
        "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    }
    model = Gemma4ForCausalLM(config).bfloat16()
    peft_model, _ = inject_lora(
        model,
        moe_spec(),
        LoraSettings(rank=8, alpha=16, expert_rank=2),
        expert_implementation=experts,
    )
    return peft_model


@pytest.mark.parametrize("experts", ["eager", "grouped_mm"])
def test_moe_reference_uses_policy_kernels_and_ignores_adapter(experts: str) -> None:
    """The reference pass ignores adapter weights and starts at a zero margin.

    Not a regression test for the expert-kernel mismatch itself: on this tiny
    CPU model PEFT's stock expert path and ours give identical bits. That bug
    (a ~21-nat per-sequence reference offset on Gemma4-26B, 8xH100) is checked
    by a fresh adapter's exact zero margin on real weights.
    """
    try:
        peft_model = _tiny_gemma_moe(experts)
    except (RuntimeError, NotImplementedError) as error:  # grouped_mm without kernel support
        pytest.skip(f"{experts} unavailable: {error}")
    model = PreferenceModel(peft_model, PreferenceConfig())
    rows = _pairs()
    batch = PreferenceCollator()(rows)
    count = torch.tensor(float(len(rows)))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        try:
            fresh = model(**batch, num_items_in_batch=count)
        except (RuntimeError, NotImplementedError) as error:
            pytest.skip(f"{experts} unavailable on this device: {error}")
        fresh_stats = dict(zip(STAT_NAMES, model.last_stats.tolist(), strict=True))
        shifted = batch["labels"][0, 1:]
        positions = shifted.ne(IGNORE_INDEX).nonzero().squeeze(-1)
        call = (
            batch["input_ids"],
            batch["position_ids"],
            shifted.index_select(0, positions),
            positions,
            batch["sequence_index"][0, 1:].index_select(0, positions),
            2 * len(rows),
        )
        with adapters_disabled(model._tuner_layers):
            ref_fresh = model._sequence_logps(*call)
        for name, parameter in peft_model.named_parameters():
            if "lora_B" in name or "lora_embedding_A" in name:
                parameter.normal_(std=0.5)
        with adapters_disabled(model._tuner_layers):
            ref_trained = model._sequence_logps(*call)
        policy_trained = model._sequence_logps(*call)
    assert fresh.loss.item() == pytest.approx(math.log(2), abs=1e-6)
    assert fresh_stats["rewards_margin"] == 0.0  # exact: same kernels, zero deltas
    assert torch.equal(ref_fresh, ref_trained)
    assert not torch.equal(policy_trained, ref_trained)  # the perturbation is visible
