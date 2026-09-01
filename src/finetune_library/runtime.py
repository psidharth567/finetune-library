from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from finetune_library.config import (
    DistributedStrategy,
    ExperimentConfig,
    RuntimeBackend,
)
from finetune_library.distributed import DistributedContext
from finetune_library.registry import ModelSpec


@dataclass(frozen=True, slots=True)
class LoadedRuntime:
    model: nn.Module
    tokenizer: Any
    backend: RuntimeBackend
    attention: str
    revision: str
    model_kernels: str = "native"


def _cached_snapshot(
    repo_id: str,
    revision: str,
    cache_dir: str,
) -> str | None:
    try:
        from huggingface_hub import try_to_load_from_cache

        cached_config = try_to_load_from_cache(
            repo_id,
            "config.json",
            cache_dir=cache_dir,
            revision=revision,
        )
    except (OSError, ValueError):
        return None
    if isinstance(cached_config, str):
        # Keep the snapshot directory. Resolving the config symlink would
        # incorrectly return the content-addressed ``blobs/`` directory.
        return str(Path(cached_config).absolute().parent)
    return None


def _apply_liger_kernels(model: nn.Module, spec: ModelSpec) -> None:
    from liger_kernel import transformers as liger  # type: ignore[import-untyped]

    common = {
        "model": model,
        "cross_entropy": False,
        "fused_linear_cross_entropy": False,
        "rms_norm": True,
    }
    if spec.key.startswith("qwen3-"):
        liger.apply_liger_kernel_to_qwen3(rope=True, swiglu=True, **common)
    elif spec.key == "deepseek-r1-distill-llama-8b":
        liger.apply_liger_kernel_to_llama(rope=True, swiglu=True, **common)
    elif spec.key == "olmo3-32b-think-dpo":
        liger.apply_liger_kernel_to_olmo3(rope=True, swiglu=True, **common)
    elif spec.key.startswith("qwen3.5-"):
        apply = getattr(liger, "apply_liger_kernel_to_qwen3_5", None)
        if callable(apply):
            # Qwen 3.5 hybrid GDN+attention blocks do not support Liger RoPE yet.
            apply(rope=False, swiglu=True, **common)
        else:
            raise ValueError(f"no Liger model-kernel profile for {spec.key}")
    elif spec.key.startswith("gemma4-"):
        liger.apply_liger_kernel_to_gemma4_text(rope=False, geglu=True, **common)
    else:
        raise ValueError(f"no Liger model-kernel profile for {spec.key}")


def select_attention(config: ExperimentConfig, spec: ModelSpec) -> str:
    requested = config.runtime.attention
    if requested != "auto":
        return requested
    for candidate in spec.attention_candidates:
        if candidate == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
            continue
        if candidate == "flash_attention_3" and importlib.util.find_spec("flash_attn_3") is None:
            continue
        return candidate
    return "sdpa"


def _load_native(
    config: ExperimentConfig,
    spec: ModelSpec,
    context: DistributedContext,
    revision: str,
    attention: str,
) -> tuple[nn.Module, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging

    previous_verbosity = transformers_logging.get_verbosity()
    if context.distributed:
        # Eight simultaneous progress bars and the expected multimodal-key
        # report obscure actionable errors. Exceptions still propagate.
        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()

    try:
        cached_snapshot = _cached_snapshot(
            spec.repo_id,
            revision,
            config.model.cache_dir,
        )
        if cached_snapshot is not None:
            snap = Path(cached_snapshot)
            has_weights = any(snap.glob("model*.safetensors")) or (snap / "pytorch_model.bin").exists() or (snap / "model.safetensors").exists()
            if not has_weights:
                import os
                toolkit_weights = os.environ.get("TOOLKIT_WEIGHTS", "/projects/data/llmteam/sidharth/toolkit/weights")
                for cand in [Path(toolkit_weights) / "Qwen3-8B", Path(toolkit_weights) / spec.repo_id.split("/")[-1]]:
                    if cand.exists() and any(cand.glob("model*.safetensors")):
                        cached_snapshot = str(cand)
                        break
                else:
                    cached_snapshot = None
        else:
            import os
            toolkit_weights = os.environ.get("TOOLKIT_WEIGHTS", "/projects/data/llmteam/sidharth/toolkit/weights")
            for cand in [Path(toolkit_weights) / spec.repo_id.split("/")[-1]]:
                if cand.exists() and any(cand.glob("model*.safetensors")):
                    cached_snapshot = str(cand)
                    break
        model_source = cached_snapshot or spec.repo_id
        tokenizer_kwargs: dict[str, Any] = {
            "revision": revision,
            "cache_dir": config.model.cache_dir,
            "trust_remote_code": config.model.trust_remote_code,
        }
        if cached_snapshot is not None:
            # Let huggingface_hub resolve tokenizer assets from the repository
            # cache. A snapshot containing config/model files need not contain
            # every tokenizer file or optional SentencePiece asset.
            tokenizer_kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(
            spec.repo_id,
            **tokenizer_kwargs,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer defines neither pad_token_id nor eos_token_id")
            tokenizer.pad_token = tokenizer.eos_token

        common: dict[str, Any] = {
            "revision": revision,
            "cache_dir": config.model.cache_dir,
            "trust_remote_code": config.model.trust_remote_code,
            "dtype": torch.bfloat16,
            "attn_implementation": attention,
            "low_cpu_mem_usage": True,
        }
        if context.device.type == "cuda":
            common["device_map"] = {"": context.device}

        model: nn.Module
        if spec.text_only and spec.key.startswith("gemma4-"):
            from transformers import Gemma4ForCausalLM

            root_config = AutoConfig.from_pretrained(
                model_source,
                revision=revision,
                cache_dir=config.model.cache_dir,
                trust_remote_code=config.model.trust_remote_code,
            )
            text_config = root_config.get_text_config()
            text_config.use_cache = config.runtime.use_cache
            model = cast(
                nn.Module,
                Gemma4ForCausalLM.from_pretrained(
                    model_source,
                    config=text_config,
                    key_mapping={"model.language_model.": "model."},
                    **common,
                ),
            )
            cast(Any, model).tie_weights()
        elif spec.text_only and spec.key.startswith("qwen3.5-"):
            from transformers import Qwen3_5MoeForCausalLM

            root_config = AutoConfig.from_pretrained(
                model_source,
                revision=revision,
                cache_dir=config.model.cache_dir,
                trust_remote_code=config.model.trust_remote_code,
            )
            text_config = root_config.get_text_config()
            text_config.use_cache = config.runtime.use_cache
            model = cast(
                nn.Module,
                Qwen3_5MoeForCausalLM.from_pretrained(
                    model_source,
                    config=text_config,
                    **common,
                ),
            )
            cast(Any, model).tie_weights()
        else:
            model = cast(
                nn.Module,
                AutoModelForCausalLM.from_pretrained(model_source, **common),
            )
    finally:
        if context.distributed:
            transformers_logging.set_verbosity(previous_verbosity)
            transformers_logging.enable_progress_bar()
    return model, tokenizer


def _load_unsloth(
    config: ExperimentConfig,
    spec: ModelSpec,
    context: DistributedContext,
    revision: str,
) -> tuple[nn.Module, Any]:
    if not spec.unsloth_compatible:
        raise ValueError(f"{spec.key} is intentionally native-only")
    from unsloth import FastLanguageModel  # type: ignore[import-untyped]

    # A local snapshot avoids eight ranks independently probing the Hub and
    # makes production launches immune to API rate limits.
    model_name = (
        _cached_snapshot(spec.repo_id, revision, config.model.cache_dir)
        or spec.repo_id
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        revision=revision,
        max_seq_length=config.training.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        device_map={"": context.device} if context.device.type == "cuda" else None,
        trust_remote_code=config.model.trust_remote_code,
        cache_dir=config.model.cache_dir,
        # Keep the registry's pinned upstream repository and revision. Unsloth
        # otherwise remaps to a mirror where that upstream commit does not
        # exist, invalidating a like-for-like benchmark.
        use_exact_model_name=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_runtime(
    config: ExperimentConfig,
    spec: ModelSpec,
    context: DistributedContext,
) -> LoadedRuntime:
    revision = config.model.revision or spec.revision
    attention = select_attention(config, spec)
    backend = config.runtime.backend
    if backend == RuntimeBackend.AUTO:
        # Native is the correctness baseline. A benchmark result may explicitly
        # promote Unsloth in a checked-in model profile.
        backend = RuntimeBackend.NATIVE

    model_kernels: str = config.runtime.model_kernels
    if backend == RuntimeBackend.UNSLOTH:
        if context.strategy in {
            DistributedStrategy.HSDP,
            DistributedStrategy.FSDP,
        }:
            raise ValueError(
                "Unsloth is not qualified with FSDP2/HSDP; use runtime.backend=native "
                "for sharded layouts"
            )
        model, tokenizer = _load_unsloth(config, spec, context, revision)
        if model_kernels == "liger":
            raise ValueError("Liger model kernels cannot be combined with the Unsloth backend")
        model_kernels = "unsloth"
    else:
        model, tokenizer = _load_native(config, spec, context, revision, attention)
        if model_kernels == "auto":
            liger_supported = (
                spec.key.startswith("qwen3-")
                or spec.key.startswith("qwen3.5-")
                or spec.key.startswith("gemma4-")
                or spec.key in ("deepseek-r1-distill-llama-8b", "olmo3-32b-think-dpo")
            )
            if liger_supported and importlib.util.find_spec("liger_kernel") is not None:
                try:
                    _apply_liger_kernels(model, spec)
                    model_kernels = "liger"
                except Exception:
                    model_kernels = "native"
            else:
                model_kernels = "native"
        elif model_kernels == "liger":
            _apply_liger_kernels(model, spec)

    cast(Any, model).config.use_cache = config.runtime.use_cache
    if config.runtime.gradient_checkpointing:
        cast(Any, model).gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        enable_inputs = getattr(model, "enable_input_require_grads", None)
        if callable(enable_inputs):
            enable_inputs()
    return LoadedRuntime(
        model=model,
        tokenizer=tokenizer,
        backend=backend,
        attention=attention,
        revision=revision,
        model_kernels=model_kernels,
    )

def _maybe_compile_grouped_moe(model: nn.Module, mode: str, *, expert_parallel_size: int = 1) -> None:
    if expert_parallel_size > 1:
        return
    try:
        import finetune_library.lora as lora_mod
        orig = getattr(lora_mod, "_grouped_linear", None)
        if orig is not None and not getattr(orig, "_is_compiled", False):
            try:
                compiled = torch.compile(orig, mode=mode, dynamic=False, fullgraph=False)
                compiled._is_compiled = True  # type: ignore[attr-defined]
                lora_mod._grouped_linear = compiled  # type: ignore[assignment]
            except Exception:
                pass
    except Exception:
        pass


def maybe_compile(
    model: nn.Module,
    config: ExperimentConfig,
    *,
    expert_parallel_size: int = 1,
) -> nn.Module:
    if not config.runtime.torch_compile:
        return model
    # Liger kernels use custom Triton ops that currently hang inductor's
    # cudagraph/block capture on H100 (swiglu tiling warning). For hard wins
    # we keep liger without inductor; compile path is reserved for native.
    try:
        is_liger = False
        # Detect via model_kernels attribute or liger-patched modules
        for m in model.modules():
            if m.__class__.__name__ in ("LigerRMSNorm", "LigerSwiGLUMLP", "LigerGEGLUMLP"):
                is_liger = True
                break
        if is_liger:
            return model
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        import torch._inductor.config as inductor_config  # type: ignore[import-untyped]
        inductor_config.triton.cudagraphs = True
        inductor_config.coordinate_descent_tuning = True
        inductor_config.epilogue_fusion = True
        inductor_config.triton.unique_kernel_names = True
        inductor_config.fx_graph_cache = True
        inductor_config.triton.max_tiles = 2
        inductor_config.benchmark_kernel = True  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch._dynamo.config.suppress_errors = False  # type: ignore[attr-defined]
        torch._dynamo.config.cache_size_limit = 64  # type: ignore[attr-defined]
    except Exception:
        pass
    scope = getattr(config.runtime, "compile_scope", "full")
    compile_kwargs = {
        "mode": config.runtime.compile_mode,
        "dynamic": False,
        "fullgraph": False,
    }
    if scope == "loss_only":
        if hasattr(model, "lm_head"):
            try:
                model.lm_head = torch.compile(
                    model.lm_head,
                    mode=config.runtime.compile_mode,
                    dynamic=False,
                    fullgraph=True,
                )
            except Exception:
                pass
        return model
    if scope == "blocks":
        from finetune_library.moe_parallel import is_expert_unit

        blocks = None
        for attr in ("model", "language_model", "model.layers", "layers"):
            try:
                candidate = model
                for part in attr.split("."):
                    candidate = getattr(candidate, part)
                if isinstance(candidate, torch.nn.ModuleList):
                    blocks = candidate
                    break
            except Exception:
                continue
        if blocks is None:
            for mod in model.modules():
                if isinstance(mod, torch.nn.ModuleList) and len(mod) > 4:
                    blocks = mod
                    break
        if blocks is not None:
            for i, block in enumerate(blocks):
                if expert_parallel_size > 1:
                    for child_name, child in block.named_children():
                        if is_expert_unit(child):
                            continue
                        try:
                            setattr(block, child_name, torch.compile(child, **compile_kwargs))
                        except Exception:
                            pass
                else:
                    try:
                        blocks[i] = torch.compile(block, **compile_kwargs)
                    except Exception:
                        pass
        _maybe_compile_grouped_moe(
            model,
            config.runtime.compile_mode,
            expert_parallel_size=expert_parallel_size,
        )
        return model
    compiled = cast(
        nn.Module,
        torch.compile(model, mode=config.runtime.compile_mode, dynamic=False, fullgraph=False),
    )
    _maybe_compile_grouped_moe(
        compiled,
        config.runtime.compile_mode,
        expert_parallel_size=expert_parallel_size,
    )
    return compiled
