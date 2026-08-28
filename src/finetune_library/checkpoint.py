from __future__ import annotations

import json
import os
import random
import shutil
import warnings
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from finetune_library.config import ExperimentConfig
from finetune_library.distributed import DistributedContext


def find_peft_model(model: nn.Module):
    from peft import PeftModel

    if isinstance(model, PeftModel):
        return model
    for module in model.modules():
        if isinstance(module, PeftModel):
            return module
    raise TypeError("could not find a PEFT model inside the distributed wrapper")


def _full_tensor(parameter: torch.Tensor) -> torch.Tensor:
    if parameter.__class__.__name__ == "DTensor":
        return cast(Any, parameter).full_tensor().detach().cpu()
    return parameter.detach().cpu()


def _adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    peft_model = find_peft_model(model)
    return {
        name: _full_tensor(parameter)
        for name, parameter in peft_model.named_parameters(remove_duplicate=False)
        if parameter.requires_grad
    }


def save_adapter(
    model: nn.Module,
    tokenizer: Any,
    destination: Path,
    config: ExperimentConfig,
    context: DistributedContext,
    metadata: dict[str, Any],
) -> None:
    # DTensor.full_tensor is collective, so every rank must build this state.
    adapter_state = _adapter_state(model)
    if not context.is_main:
        return
    destination.mkdir(parents=True, exist_ok=True)
    peft_model = find_peft_model(model)
    with warnings.catch_warnings():
        # Tied Gemma adapters are intentional and verified by the LoRA audit;
        # PEFT's generic advisory does not account for our shared-storage
        # gradient consolidation and merge-parity gate.
        warnings.filterwarnings(
            "ignore",
            message=r"Model (?:has|with) `tie_word_embeddings=True`.*",
        )
        peft_model.save_pretrained(
            destination,
            state_dict=adapter_state,
            safe_serialization=True,
            save_embedding_layers=False,
        )
    tokenizer.save_pretrained(destination)
    revision_value = metadata.get("revision") or config.model.revision
    if revision_value is None:
        from finetune_library.registry import resolve_model

        revision_value = resolve_model(config.model.name).revision
    revision = str(revision_value)
    config.with_model_revision(revision).write_resolved(destination / "resolved_config.json")
    (destination / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_adapter(model: nn.Module, checkpoint: str | Path) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    path = Path(checkpoint)
    adapter_file = path / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(adapter_file)
    state = load_file(str(adapter_file), device="cpu")
    result = set_peft_model_state_dict(find_peft_model(model), state)
    unexpected = getattr(result, "unexpected_keys", ())
    if unexpected:
        raise RuntimeError(f"unexpected adapter keys: {unexpected[:8]}")


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"].cpu())
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])


def save_checkpoint(
    *,
    model: nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: ExperimentConfig,
    context: DistributedContext,
    step: int,
    epoch: int,
    batch_in_epoch: int,
    metadata: dict[str, Any],
) -> Path:
    destination = Path(config.training.output_dir) / f"checkpoint-{step:08d}"
    destination.mkdir(parents=True, exist_ok=True)
    rank_state = {
        "step": step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "data_position": {
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
        },
        "sampler": {"epoch": epoch},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": capture_rng_state(),
        "world_size": context.world_size,
        "strategy": context.strategy.value,
        "shard_size": context.shard_size,
        "replicate_size": context.replicate_size,
    }
    temporary = destination / f".trainer_state_rank{context.rank:05d}.tmp"
    final = destination / f"trainer_state_rank{context.rank:05d}.pt"
    torch.save(rank_state, temporary)
    os.replace(temporary, final)
    context.barrier()
    save_adapter(model, tokenizer, destination, config, context, metadata)
    context.barrier()
    if context.is_main:
        _prune_checkpoints(Path(config.training.output_dir), config.checkpoint.keep_last)
    return destination


def load_training_state(
    *,
    checkpoint: str | Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    context: DistributedContext,
) -> dict[str, int]:
    path = Path(checkpoint) / f"trainer_state_rank{context.rank:05d}.pt"
    state = torch.load(path, map_location=context.device, weights_only=False)
    expected = {
        "world_size": context.world_size,
        "strategy": context.strategy.value,
        "shard_size": context.shard_size,
        "replicate_size": context.replicate_size,
    }
    for key, value in expected.items():
        if state[key] != value:
            raise ValueError(
                f"resume layout mismatch for {key}: checkpoint={state[key]!r}, current={value!r}"
            )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(state["rng"])
    position = state.get("data_position", state)
    return {
        "step": int(state["step"]),
        "epoch": int(position["epoch"]),
        "batch_in_epoch": int(position["batch_in_epoch"]),
    }


def _prune_checkpoints(output_dir: Path, keep_last: int) -> None:
    checkpoints = sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: path.name,
    )
    for path in checkpoints[:-keep_last]:
        shutil.rmtree(path)
