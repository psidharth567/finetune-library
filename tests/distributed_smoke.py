"""Two-process FSDP2/HSDP smoke; run with torchrun, not pytest."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from finetune_library.checkpoint import load_training_state, save_checkpoint
from finetune_library.config import (
    DistributedConfig,
    DistributedStrategy,
    ExperimentConfig,
    LoraSettings,
    OptimizerConfig,
)
from finetune_library.distributed import initialize_distributed, wrap_model
from finetune_library.lora import consolidate_tied_lora_gradients, inject_lora
from finetune_library.loss import build_training_model
from finetune_library.optim import build_optimizer, build_scheduler
from finetune_library.registry import ModelSpec


class FakeTokenizer:
    def save_pretrained(self, output: str | Path) -> None:
        Path(output, "tokenizer.json").write_text("{}\n", encoding="utf-8")


def main() -> None:
    spec = ModelSpec(
        key="tiny",
        repo_id="tiny",
        revision="0" * 40,
        architecture="dense",
        layer_class="LlamaDecoderLayer",
        preferred_strategy=DistributedStrategy.HSDP,
        shard_size=2,
        replicate_size=1,
    )
    distributed = DistributedConfig(
        strategy=DistributedStrategy.HSDP,
        shard_size=2,
        replicate_size=1,
    )
    context = initialize_distributed(distributed, spec)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            tie_word_embeddings=True,
        )
    ).to(device=context.device, dtype=torch.bfloat16)
    model, audit = inject_lora(
        model,
        spec,
        LoraSettings(rank=4, alpha=8, expert_rank=1),
    )
    model = build_training_model(model, "fused_linear_cross_entropy")
    model = wrap_model(model, context, spec, distributed)
    optimizer = build_optimizer(
        model,
        OptimizerConfig(learning_rate=1e-3, fused=False),
    )
    config = ExperimentConfig.from_yaml("configs/models/qwen3-8b-cpt.yaml")
    config = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={"output_dir": "outputs/validation-two-node-distributed-smoke"}
            ),
            "distributed": distributed,
        }
    )
    scheduler = build_scheduler(optimizer, config.scheduler, total_steps=2)
    input_ids = torch.randint(0, 64, (2, 16), device=context.device)
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
        num_items_in_batch=torch.tensor(30.0, device=context.device),
    )
    output.loss.backward()
    context.sync_replicated_gradients()
    consolidate_tied_lora_gradients(model)
    context.clip_grad_norm(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    checkpoint = save_checkpoint(
        model=model,
        tokenizer=FakeTokenizer(),
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        context=context,
        step=1,
        epoch=0,
        batch_in_epoch=1,
        metadata={"test": "two-node"},
    )
    position = load_training_state(
        checkpoint=checkpoint,
        optimizer=optimizer,
        scheduler=scheduler,
        context=context,
    )
    if context.is_main:
        print(
            json.dumps(
                {
                    "loss": float(output.loss.detach()),
                    "trainable_parameters": audit.trainable_parameters,
                    "optimizer": optimizer.__class__.__name__,
                    "resume_position": position,
                    "strategy": context.strategy.value,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    context.close()


if __name__ == "__main__":
    main()
