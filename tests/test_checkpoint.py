from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from transformers import LlamaConfig, LlamaForCausalLM

from finetune_library.checkpoint import (
    load_adapter,
    load_training_state,
    save_checkpoint,
)
from finetune_library.config import (
    DistributedStrategy,
    ExperimentConfig,
    LoraSettings,
)
from finetune_library.distributed import DistributedContext
from finetune_library.lora import inject_lora
from finetune_library.optim import build_optimizer, build_scheduler
from finetune_library.registry import ModelSpec


class FakeTokenizer:
    def save_pretrained(self, output: str | Path) -> None:
        Path(output, "tokenizer.json").write_text("{}\n", encoding="utf-8")


def tiny_spec() -> ModelSpec:
    return ModelSpec(
        key="tiny",
        repo_id="tiny",
        revision="0" * 40,
        architecture="dense",
        layer_class="LlamaDecoderLayer",
        preferred_strategy=DistributedStrategy.DDP,
    )


def tiny_model():
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
            tie_word_embeddings=False,
        )
    ).bfloat16()
    return inject_lora(
        model,
        tiny_spec(),
        LoraSettings(rank=2, alpha=4, expert_rank=1),
    )[0]


def context() -> DistributedContext:
    return DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        strategy=DistributedStrategy.DDP,
        shard_size=1,
        replicate_size=1,
        expert_parallel_size=1,
        data_parallel_rank=0,
        data_parallel_size=1,
    )


def test_adapter_and_adamw_checkpoint_round_trip(tmp_path: Path) -> None:
    config = ExperimentConfig.from_yaml("configs/models/qwen3-8b-cpt.yaml")
    config = config.model_copy(
        update={"training": config.training.model_copy(update={"output_dir": str(tmp_path)})}
    )
    model = tiny_model()
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.scheduler, total_steps=2)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()

    checkpoint = save_checkpoint(
        model=model,
        tokenizer=FakeTokenizer(),
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        context=context(),
        step=1,
        epoch=0,
        batch_in_epoch=1,
        metadata={"test": True},
    )
    with safe_open(
        checkpoint / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as adapter:
        keys = adapter.keys()
        assert keys
        assert all("lora_" in key for key in keys)
    rank_state = torch.load(
        checkpoint / "trainer_state_rank00000.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert rank_state["data_position"] == {"epoch": 0, "batch_in_epoch": 1}
    assert rank_state["sampler"] == {"epoch": 0}
    resolved = ExperimentConfig.model_validate_json(
        (checkpoint / "resolved_config.json").read_text(encoding="utf-8")
    )
    assert resolved.model.revision == "b968826d9c46dd6066d109eabc6255188de91218"

    restored_model = tiny_model()
    load_adapter(restored_model, checkpoint)
    restored_optimizer = build_optimizer(restored_model, config.optimizer)
    restored_scheduler = build_scheduler(restored_optimizer, config.scheduler, total_steps=2)
    position = load_training_state(
        checkpoint=checkpoint,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        context=context(),
    )
    assert position == {"step": 1, "epoch": 0, "batch_in_epoch": 1}
    expected = {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    actual = {
        name: parameter.detach()
        for name, parameter in restored_model.named_parameters()
        if parameter.requires_grad
    }
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[name], actual[name]) for name in expected)
