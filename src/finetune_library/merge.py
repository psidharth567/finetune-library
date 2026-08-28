from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from finetune_library.checkpoint import load_adapter
from finetune_library.config import DistributedStrategy, ExperimentConfig, RuntimeBackend
from finetune_library.distributed import DistributedContext
from finetune_library.lora import inject_lora
from finetune_library.registry import resolve_model
from finetune_library.runtime import load_runtime


def merge_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    validation_text: str = "The quick brown fox",
    max_shard_size: str = "5GB",
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    output_path = Path(output)
    resolved = checkpoint_path / "resolved_config.json"
    if not resolved.exists():
        raise FileNotFoundError(
            f"{resolved} is required so the exact base model revision can be restored"
        )
    config = ExperimentConfig.model_validate_json(resolved.read_text(encoding="utf-8"))
    spec = resolve_model(config.model.name)
    runtime_settings = config.runtime.model_copy(update={"backend": RuntimeBackend.NATIVE})
    distributed_settings = config.distributed.model_copy(
        update={
            "strategy": DistributedStrategy.DDP,
            "shard_size": None,
            "replicate_size": None,
        }
    )
    config = config.model_copy(
        update={"runtime": runtime_settings, "distributed": distributed_settings}
    )
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    context = DistributedContext(
        rank=0,
        local_rank=torch_device.index or 0,
        world_size=1,
        device=torch_device,
        strategy=DistributedStrategy.DDP,
        shard_size=1,
        replicate_size=1,
        data_parallel_rank=0,
        data_parallel_size=1,
    )
    runtime = load_runtime(config, spec, context)
    adapter_model, audit = inject_lora(runtime.model, spec, config.lora)
    load_adapter(adapter_model, checkpoint_path)
    adapter_model.eval()
    inputs = runtime.tokenizer(validation_text, return_tensors="pt")
    inputs = {name: tensor.to(torch_device) for name, tensor in inputs.items()}
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=torch_device.type,
            dtype=torch.bfloat16,
            enabled=torch_device.type == "cuda",
        ),
    ):
        adapter_logits = adapter_model(**inputs).logits[:, -1, :].float().cpu()

    merged = cast(Any, adapter_model).merge_and_unload(safe_merge=True)
    merged = cast(nn.Module, merged)
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=torch_device.type,
            dtype=torch.bfloat16,
            enabled=torch_device.type == "cuda",
        ),
    ):
        merged_logits = merged(**inputs).logits[:, -1, :].float().cpu()
    difference = adapter_logits - merged_logits
    maximum_error = float(difference.abs().max().item())
    rms_error = float(difference.square().mean().sqrt().item())
    reference_rms = float(adapter_logits.square().mean().sqrt().item())
    relative_rms_error = rms_error / max(reference_rms, 1.0e-12)
    reference_scale = float(adapter_logits.abs().max().item())
    maximum_tolerance = max(0.25, 0.02 * reference_scale)
    finite = bool(torch.isfinite(adapter_logits).all() and torch.isfinite(merged_logits).all())
    if not finite or maximum_error > maximum_tolerance or relative_rms_error > 0.02:
        raise RuntimeError(
            "merged-model BF16 logit validation failed; "
            f"max={maximum_error:.6g} allowed={maximum_tolerance:.6g} "
            f"relative_rms={relative_rms_error:.6g}"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    cast(Any, merged).save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    runtime.tokenizer.save_pretrained(output_path)
    report = {
        "base_model": spec.repo_id,
        "revision": runtime.revision,
        "adapter": str(checkpoint_path),
        "output": str(output_path),
        "validation_max_absolute_error": maximum_error,
        "validation_max_absolute_tolerance": maximum_tolerance,
        "validation_relative_rms_error": relative_rms_error,
        "lora_targets": len(audit.targets),
    }
    (output_path / "merge_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
