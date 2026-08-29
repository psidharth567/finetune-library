from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from finetune_library.config import DataFormat, RuntimeBackend, Task
from finetune_library.registry import resolve_model


def _default_runtime_backend(model_name: str) -> str:
    spec = resolve_model(model_name)
    if spec.unsloth_compatible and spec.preferred_strategy.value == "ddp":
        return RuntimeBackend.UNSLOTH.value
    return RuntimeBackend.NATIVE.value


def _default_data_format(task: Task) -> DataFormat:
    return DataFormat.TEXT if task == Task.CPT else DataFormat.MESSAGES


def scaffold_project_config(
    *,
    project_root: Path,
    run_name: str,
    task: Task,
    model_name: str,
    data_path: Path,
    data_format: DataFormat | None = None,
    max_steps: int = 500,
    max_seq_length: int = 2048,
    wandb_project: str | None = None,
) -> Path:
    project_root = project_root.resolve()
    data_path = data_path.resolve()
    finetune_root = project_root / "finetune"
    config_dir = finetune_root / "configs"
    output_dir = finetune_root / "outputs" / run_name
    cache_dir = finetune_root / ".cache" / "prepared"
    hf_cache = project_root / ".cache" / "huggingface"

    for directory in (config_dir, output_dir, cache_dir, hf_cache):
        directory.mkdir(parents=True, exist_ok=True)

    resolved_format = str(data_format or _default_data_format(task))
    runtime_backend = _default_runtime_backend(model_name)

    spec = resolve_model(model_name)
    lr = spec.recommended_lr_cpt if task == Task.CPT else spec.recommended_lr_sft
    config: dict[str, Any] = {
        "version": 1,
        "task": task.value,
        "model": {
            "name": model_name,
            "cache_dir": str(hf_cache),
        },
        "data": {
            "format": resolved_format,
            "path": str(data_path),
            "cache_dir": str(cache_dir),
            "packing": task == Task.CPT,
            "drop_remainder": False,
            "packing_isolation": "none",
            "chunk_long_examples": False,
            "chunk_overlap": 0,
            "chunk_strategy": "truncate",
            "require_full_seq_length": False,
            "length_policy": "global",
        },
        "training": {
            "output_dir": str(output_dir),
            "max_seq_length": max_seq_length,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": max_steps,
            "log_every_steps": 10,
        },
        "lora": {"rank": 32, "alpha": 64, "dropout": 0.0, "expert_rank": 4},
        "optimizer": {
            "name": "adamw",
            "learning_rate": lr,
            "weight_decay": 0.01,
        },
        "scheduler": {"name": "cosine", "warmup_ratio": 0.03},
        "distributed": {"strategy": "ddp"},
        "runtime": {
            "backend": runtime_backend,
            "attention": "sdpa",
            "model_kernels": "auto",
            "loss": "fused_linear_cross_entropy",
            "experts": "auto",
            "torch_compile": False,
            "gradient_checkpointing": True,
        },
        "checkpoint": {"save_every_steps": 0, "save_final": True},
        "logging": {
            "level": "INFO",
            "file": True,
            "console": True,
            "metrics_jsonl": True,
            "events_jsonl": True,
            "wandb": {
                "enabled": False,
                "project": wandb_project or project_root.name,
            },
        },
    }

    config_path = config_dir / f"{run_name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path
