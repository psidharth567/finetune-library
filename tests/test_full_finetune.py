"""Full fine-tuning (lora: null): config rules, linear schedule, fp32 master upcast, DPO with a frozen
reference copy, and the full-model checkpoint (HF export + per-rank resume state)."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from test_checkpoint import FakeTokenizer, context, tiny_spec
from test_dpo import _pairs, _sequence_logp
from transformers import LlamaConfig, LlamaForCausalLM

from finetune_library.checkpoint import load_full_model_state, policy_model, save_checkpoint
from finetune_library.config import ExperimentConfig, PreferenceConfig, SchedulerConfig
from finetune_library.distributed import wrap_model
from finetune_library.dpo import STAT_NAMES, PreferenceCollator, PreferenceModel
from finetune_library.loss import build_training_model
from finetune_library.optim import build_optimizer, build_scheduler


def raw(**updates: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "task": "sft",
        "model": {"name": "qwen3-8b"},
        "data": {"format": "messages", "path": "unused.jsonl"},
        "training": {"output_dir": "outputs/test-full", "max_seq_length": 64},
        "lora": None,
        "optimizer": {"learning_rate": 5.0e-6},
        "distributed": {"strategy": "fsdp"},
    }
    config.update(updates)
    return config


def tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(
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


def test_lora_null_means_full_finetune_and_lora_stays_default() -> None:
    assert ExperimentConfig.model_validate(raw()).full_finetune
    lora = raw()
    lora.pop("lora")
    lora.pop("optimizer")
    assert not ExperimentConfig.model_validate(lora).full_finetune
    resolved = ExperimentConfig.model_validate(raw()).to_json_dict()
    assert resolved["lora"] is None
    assert ExperimentConfig.model_validate(resolved).full_finetune


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"optimizer": {}}, "requires optimizer.learning_rate"),
        ({"distributed": {"strategy": "ddp"}}, "fsdp"),
        ({"runtime": {"loss": "fused_linear_cross_entropy"}}, "fused"),
    ],
)
def test_full_finetune_rejections(updates: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ExperimentConfig.model_validate(raw(**updates))


def test_linear_schedule_warms_up_then_decays_to_zero() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_scheduler(optimizer, SchedulerConfig(name="linear", warmup_steps=2), 10)
    rates = []
    for _ in range(10):
        rates.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    assert rates[:3] == pytest.approx([0.0, 0.5, 1.0])
    assert rates[-1] == pytest.approx(1 / 8)
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0)


def test_master_dtype_upcast_on_single_process_keeps_values() -> None:
    model = tiny_llama()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    wrapped = wrap_model(
        model, context(), tiny_spec(), ExperimentConfig.model_validate(raw()).distributed,
        master_dtype=torch.float32,
    )
    for name, parameter in wrapped.named_parameters():
        assert parameter.dtype == torch.float32
        torch.testing.assert_close(parameter.to(torch.bfloat16), before[name], rtol=0, atol=0)


def test_full_dpo_reference_copy_starts_at_ln2_and_matches_independent_path() -> None:
    policy = tiny_llama().float()
    reference = copy.deepcopy(policy).requires_grad_(False)
    rows = _pairs()
    batch = PreferenceCollator()(rows)
    count = torch.tensor(float(len(rows)))

    model = PreferenceModel(policy, PreferenceConfig(beta=0.5), reference_model=reference)
    model.train()
    assert not reference.training
    assert model(**batch, num_items_in_batch=count).loss.item() == pytest.approx(math.log(2), abs=1e-6)

    with torch.no_grad():  # move the policy away from the frozen reference
        for parameter in policy.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    output = model(**batch, num_items_in_batch=count)
    output.loss.backward()
    assert all(p.grad is None for p in reference.parameters())
    assert all(p.grad is not None for p in policy.parameters())

    margins = []
    for row in rows:
        with torch.no_grad():
            ref_c = _sequence_logp(reference, row["chosen_input_ids"], row["chosen_labels"])
            ref_r = _sequence_logp(reference, row["rejected_input_ids"], row["rejected_labels"])
            pol_c = _sequence_logp(policy, row["chosen_input_ids"], row["chosen_labels"])
            pol_r = _sequence_logp(policy, row["rejected_input_ids"], row["rejected_labels"])
        margins.append(0.5 * ((pol_c - ref_c) - (pol_r - ref_r)))
    expected = torch.stack([-torch.nn.functional.logsigmoid(m) for m in margins]).mean()
    torch.testing.assert_close(output.loss, expected, atol=1e-5, rtol=1e-5)
    stats = dict(zip(STAT_NAMES, model.last_stats.tolist(), strict=True))
    assert stats["rewards_margin"] == pytest.approx(float(torch.stack(margins).sum()), abs=1e-4)


def test_reference_model_must_be_frozen() -> None:
    with pytest.raises(ValueError, match="frozen"):
        PreferenceModel(tiny_llama(), PreferenceConfig(), reference_model=tiny_llama())


def test_full_checkpoint_exports_hf_model_and_resumes_shards(tmp_path: Path) -> None:
    config = ExperimentConfig.model_validate(raw(training={"output_dir": str(tmp_path), "max_seq_length": 64}))
    hf_model = tiny_llama()
    training_model = build_training_model(hf_model, "auto")
    model = wrap_model(training_model, context(), tiny_spec(), config.distributed, master_dtype=torch.float32)
    assert policy_model(model) is hf_model
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.scheduler, total_steps=4)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()

    checkpoint = save_checkpoint(
        model=model, tokenizer=FakeTokenizer(), optimizer=optimizer, scheduler=scheduler,
        config=config, context=context(), step=1, epoch=0, batch_in_epoch=1, metadata={"test": True},
    )
    assert not (checkpoint / "adapter_model.safetensors").exists()
    saved_config = json.loads((checkpoint / "config.json").read_text())
    assert saved_config.get("dtype", saved_config.get("torch_dtype")) == "bfloat16"
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as weights:
        assert set(weights.keys()) == set(hf_model.state_dict())
        name = "model.layers.0.mlp.down_proj.weight"
        assert weights.get_tensor(name).dtype == torch.bfloat16
        torch.testing.assert_close(
            weights.get_tensor(name), hf_model.state_dict()[name].to(torch.bfloat16), rtol=0, atol=0
        )
    reloaded = LlamaForCausalLM.from_pretrained(checkpoint)
    torch.testing.assert_close(
        reloaded.lm_head.weight, hf_model.lm_head.weight.to(torch.bfloat16).to(reloaded.dtype)
    )

    fresh = wrap_model(build_training_model(tiny_llama(), "auto"), context(), tiny_spec(),
                       config.distributed, master_dtype=torch.float32)
    load_full_model_state(fresh, checkpoint, context())
    for (name, a), (_, b) in zip(policy_model(fresh).named_parameters(), hf_model.named_parameters(), strict=True):
        assert a.dtype == torch.float32
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
