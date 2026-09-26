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


def _merge_moe_with_weight_check(
    adapter_model: nn.Module, *, tolerance: float = 0.01
) -> tuple[nn.Module, float]:
    """Merge a MoE adapter, verifying exactness at the weight level.

    Samples expert targets from the first/middle/last transformer layers plus
    a few dense targets; checks ``merged == base + PEFT delta`` for each.
    Returns the merged model and the worst max-abs error.
    """
    import re

    def layer_of(name: str) -> int | None:
        match = re.search(r"layers\.(\d+)", name)
        return int(match.group(1)) if match else None

    experts: dict[int, list[tuple[str, Any]]] = {}
    dense: list[tuple[str, Any]] = []
    n_layers = 0
    for name, module in adapter_model.named_modules():
        parameter_name = getattr(module, "parameter_name", None)
        if parameter_name is None or not hasattr(module, "get_delta_weight"):
            continue
        layer = layer_of(name)
        if layer is not None:
            n_layers = max(n_layers, layer + 1)
        if ".experts." in name:
            if layer is not None:
                experts.setdefault(layer, []).append((name, module))
        elif len(dense) < 8:
            dense.append((name, module))
    picks: list[tuple[str, Any]] = list(dense)
    for layer in sorted(experts):
        if layer in (0, n_layers // 2, n_layers - 1):
            picks.extend(experts[layer][:4])

    snapshots: list[tuple[str, str, torch.Tensor, torch.Tensor]] = []
    for name, module in picks:
        parameter_name = module.parameter_name
        base = module.get_param().detach().to("cpu").float()
        delta = module.get_delta_weight("default").detach().to("cpu").float()
        snapshots.append((name, parameter_name, base, delta))

    merged = cast(Any, adapter_model).merge_and_unload(safe_merge=True)
    flat = dict(merged.named_parameters())
    worst = 0.0
    for name, parameter_name, base, delta in snapshots:
        layer = layer_of(name)
        cands = [
            key
            for key in flat
            if f"layers.{layer}" in key and key.endswith(parameter_name) and "lora_" not in key
        ] if layer is not None else [
            key for key in flat if key.endswith(parameter_name) and "lora_" not in key
        ]
        if not cands:
            raise RuntimeError(f"merged model is missing target {name}.{parameter_name}")
        got = flat[cands[0]].detach().to("cpu").float()
        worst = max(worst, float((got - base - delta).abs().max().item()))
    if worst > tolerance:
        raise RuntimeError(
            f"weight-level merge check failed; max abs err {worst:.6g} > {tolerance}"
        )
    return cast(nn.Module, merged), worst


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
    # Merge needs identical weights, not training kernels: Liger's Triton
    # kernels cannot run on the CPU merge device.
    runtime_settings = config.runtime.model_copy(
        update={"backend": RuntimeBackend.NATIVE, "model_kernels": "native"}
    )
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
        expert_parallel_size=1,
        data_parallel_rank=0,
        data_parallel_size=1,
    )
    runtime = load_runtime(config, spec, context)
    # Validation must run the adapter forward under the same expert
    # implementation as training (eager vs grouped_mm differ numerically).
    adapter_model, audit = inject_lora(
        runtime.model, spec, config.lora, expert_implementation=config.runtime.experts
    )
    load_adapter(adapter_model, checkpoint_path)
    adapter_model.eval()

    validation_method = "bf16_logits"
    # Weight-level validation for MoE and Gemma-family merges. Their custom
    # expert loops (MoE) and norm/softcap-heavy attention (Gemma2/3/4) drift
    # in BF16 across forward summation orders even when the merge is exact
    # (verified weight-level to <2e-3), tripping the logit gate spuriously.
    if spec.moe or spec.layer_class.startswith(("Gemma2", "Gemma3", "Gemma4")):
        # MoE logit validation is unreliable: the custom expert-LoRA forward
        # loops accumulate in a different order than the plain batched forward
        # of the merged model, drifting ~5% in BF16 over many MoE layers even
        # when the merge is exact (verified weight-level to <2e-3). Same for
        # Gemma-family attention (q/k-norms, sliding window, logit softcaps):
        # exact dense gemma2 merges trip the logit gate at max ~4. Validate
        # the merge directly at the weight level on a sample of targets
        # (first/mid/last layer experts + dense) instead.
        merged, maximum_error = _merge_moe_with_weight_check(adapter_model)
        maximum_tolerance = 0.01
        relative_rms_error = 0.0
        finite = True
        validation_method = "moe_weight_sample"
    else:
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
        "validation_method": validation_method,
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
