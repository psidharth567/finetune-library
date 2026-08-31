from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from finetune_library.config import LoraSettings
from finetune_library.moe_parallel import MoeA2ABackend, MoeParallelLayout, expert_parallel_forward
from finetune_library.registry import ModelSpec

EXPERT_LAYER_NAMES = frozenset({"Gemma4TextExperts", "Qwen3_5MoeExperts"})


@dataclass(frozen=True, slots=True)
class LoraTarget:
    name: str
    kind: str
    rank: int
    shape: tuple[int, ...]
    adapter_parameters: int


@dataclass(frozen=True, slots=True)
class LoraAudit:
    targets: tuple[LoraTarget, ...]
    trainable_tensors: int
    trainable_parameters: int
    trainable_dtype: torch.dtype

    def as_dict(self) -> dict[str, Any]:
        return {
            "dtype": str(self.trainable_dtype),
            "targets": [
                {
                    "name": target.name,
                    "kind": target.kind,
                    "rank": target.rank,
                    "shape": list(target.shape),
                    "adapter_parameters": target.adapter_parameters,
                }
                for target in self.targets
            ],
            "trainable_tensors": self.trainable_tensors,
            "trainable_parameters": self.trainable_parameters,
        }

    def format(self, *, verbose: bool = False) -> str:
        lines = [
            "LoRA coverage:",
            f"  targets={len(self.targets)} trainable_tensors={self.trainable_tensors} "
            f"trainable_parameters={self.trainable_parameters:,} dtype={self.trainable_dtype}",
        ]
        counts = Counter(target.kind for target in self.targets)
        for kind in sorted(counts):
            selected = [target for target in self.targets if target.kind == kind]
            ranks = sorted({target.rank for target in selected})
            parameters = sum(target.adapter_parameters for target in selected)
            lines.append(
                f"  {kind:16s} targets={len(selected):<3d} ranks={ranks!s:<8s} "
                f"logical_parameters={parameters:,}"
            )
        if not verbose:
            lines.append("  full target audit: lora_audit.json")
            return "\n".join(lines)
        for target in self.targets:
            lines.append(
                f"  {target.kind:16s} rank={target.rank:<3d} "
                f"shape={str(target.shape):20s} "
                f"parameters={target.adapter_parameters:<10,d} {target.name}"
            )
        return "\n".join(lines)


def _adapter_parameter_count(shape: tuple[int, ...], rank: int) -> int:
    if len(shape) == 2:
        return rank * (shape[0] + shape[1])
    if len(shape) == 3:
        return shape[0] * rank * (shape[1] + shape[2])
    return 0


def _active_expert_delta(
    wrapper: Any,
    hidden_states: torch.Tensor,
    expert_index: torch.Tensor,
) -> torch.Tensor:
    adapter = wrapper.active_adapters[0]
    weight_a = wrapper.lora_A[adapter].weight
    weight_b = wrapper.lora_B[adapter].weight
    experts = wrapper.num_experts
    rank = weight_a.shape[0] // experts
    expert_a = weight_a.reshape(experts, rank, weight_a.shape[-1])[expert_index]
    expert_b = weight_b.reshape(weight_b.shape[0], rank, experts)[:, :, expert_index]
    return F.linear(F.linear(hidden_states, expert_a), expert_b) * wrapper.scaling[adapter]


def _expert_layers(wrapper: Any) -> tuple[Any, Any, Any]:
    from peft.tuners.lora.layer import ParamWrapper

    wrappers: list[ParamWrapper] = []
    layer: Any = wrapper
    while isinstance(layer, ParamWrapper):
        wrappers.append(layer)
        layer = layer.base_layer
    by_parameter = {item.parameter_name: item for item in wrappers}
    return layer, by_parameter["gate_up_proj"], by_parameter["down_proj"]


def _grouped_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    # ``torch._grouped_mm`` is not registered as an autocast operation. Gemma
    # can keep its residual stream in FP32 under DDP even though expert weights
    # and the requested compute dtype are BF16. Match ordinary ``F.linear``
    # autocast semantics explicitly at this kernel boundary.
    if hidden_states.dtype != weight.dtype:
        hidden_states = hidden_states.to(dtype=weight.dtype)
    return torch._grouped_mm(hidden_states, weight.transpose(-2, -1), offs=offsets)


def _grouped_expert_delta(
    wrapper: Any,
    hidden_states: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    adapter = wrapper.active_adapters[0]
    weight_a = wrapper.lora_A[adapter].weight
    weight_b = wrapper.lora_B[adapter].weight
    experts = wrapper.num_experts
    rank = weight_a.shape[0] // experts
    expert_a = weight_a.reshape(experts, rank, weight_a.shape[-1])
    expert_b = (
        weight_b.reshape(weight_b.shape[0], rank, experts)
        .permute(2, 0, 1)
        .contiguous()
    )
    # Hopper grouped GEMM requires byte-aligned matrix strides. Logical expert
    # ranks remain unchanged; zero-padding only supplies an aligned kernel view.
    padded_rank = ((rank + 7) // 8) * 8
    if padded_rank != rank:
        expert_a = F.pad(expert_a, (0, 0, 0, padded_rank - rank))
        expert_b = F.pad(expert_b, (0, padded_rank - rank))
    after_a = _grouped_linear(hidden_states, expert_a, offsets)
    return _grouped_linear(after_a, expert_b, offsets) * wrapper.scaling[adapter]


def _grouped_local_expert_compute(
    base: Any,
    gate_wrapper: Any,
    down_wrapper: Any,
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    global_offset: int,
) -> torch.Tensor:
    if hidden_states.numel() == 0:
        return hidden_states
    local_expert_ids = expert_ids - global_offset
    if torch.any(local_expert_ids < 0) or torch.any(local_expert_ids >= base.num_experts):
        raise RuntimeError(
            "received expert ids outside the local shard: "
            f"min={int(local_expert_ids.min())} max={int(local_expert_ids.max())} "
            f"local={base.num_experts}"
        )
    permutation = torch.argsort(local_expert_ids)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=hidden_states.device)
    sorted_experts = local_expert_ids[permutation]
    sorted_hidden = hidden_states[permutation]
    counts = torch.bincount(sorted_experts, minlength=base.num_experts)
    offsets = counts.cumsum(0, dtype=torch.int32)

    gate_up = _grouped_linear(sorted_hidden, base.gate_up_proj, offsets)
    gate_up = gate_up + _grouped_expert_delta(gate_wrapper, sorted_hidden, offsets)
    gate, up = gate_up.chunk(2, dim=-1)
    intermediate = base.act_fn(gate) * up
    current = _grouped_linear(intermediate, base.down_proj, offsets)
    current = current + _grouped_expert_delta(down_wrapper, intermediate, offsets)
    sorted_weights = weights[permutation]
    current = current * sorted_weights.unsqueeze(-1)
    return current[inverse]


def _install_active_expert_forward(
    peft_model: nn.Module,
    implementation: str,
    moe_layout: MoeParallelLayout | None = None,
) -> int:
    """Avoid PEFT's full `[experts, out, in]` delta materialization.

    PEFT's parameter-level wrapper is retained, including its state-dict names
    and merge implementation. Only the outer wrapper's forward is specialized
    to apply the two low-rank expert updates to routed tokens.
    """

    from peft.tuners.lora.layer import ParamWrapper

    class EagerGemmaExpertWrapper(ParamWrapper):
        def forward(
            self,
            x: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            if self.disable_adapters or self.merged:
                return super().forward(x, *args, **kwargs)
            if len(args) < 2:
                raise TypeError("Gemma expert LoRA requires indices and routing weights")
            top_k_index, top_k_weights, *remaining_args = args
            if remaining_args or kwargs:
                raise TypeError("unexpected arguments for Gemma expert LoRA")
            hidden_states = x
            base, gate_wrapper, down_wrapper = _expert_layers(self)

            final = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_index, num_classes=base.num_experts).permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_index in expert_hit:
                expert_index = expert_index[0]
                top_k_position, token_index = torch.where(expert_mask[expert_index])
                current = hidden_states[token_index]
                gate_up = F.linear(current, base.gate_up_proj[expert_index])
                gate_up = gate_up + _active_expert_delta(gate_wrapper, current, expert_index)
                gate, up = gate_up.chunk(2, dim=-1)
                intermediate = base.act_fn(gate) * up
                current = F.linear(intermediate, base.down_proj[expert_index])
                current = current + _active_expert_delta(
                    down_wrapper,
                    intermediate,
                    expert_index,
                )
                current = current * top_k_weights[token_index, top_k_position, None]
                final.index_add_(0, token_index, current.to(dtype=final.dtype))
            return final

    class GroupedGemmaExpertWrapper(ParamWrapper):
        def forward(
            self,
            x: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            if self.disable_adapters or self.merged:
                return super().forward(x, *args, **kwargs)
            if len(args) < 2:
                raise TypeError("Gemma expert LoRA requires indices and routing weights")
            top_k_index, top_k_weights, *remaining_args = args
            if remaining_args or kwargs:
                raise TypeError("unexpected arguments for Gemma expert LoRA")
            base, gate_wrapper, down_wrapper = _expert_layers(self)

            if moe_layout is not None and moe_layout.enabled:
                def local_forward(
                    recv_hidden: torch.Tensor,
                    recv_expert_ids: torch.Tensor,
                    recv_weights: torch.Tensor,
                ) -> torch.Tensor:
                    expert_offset = (
                        0
                        if moe_layout.a2a_backend == MoeA2ABackend.DEEPEP
                        else moe_layout.global_expert_offset
                    )
                    return _grouped_local_expert_compute(
                        base,
                        gate_wrapper,
                        down_wrapper,
                        recv_hidden,
                        recv_expert_ids,
                        recv_weights,
                        expert_offset,
                    )

                return expert_parallel_forward(
                    x,
                    top_k_index,
                    top_k_weights,
                    moe_layout,
                    local_forward,
                )

            num_tokens, hidden_dim = x.shape
            top_k = top_k_index.shape[-1]
            token_index = (
                torch.arange(num_tokens, device=x.device)
                .unsqueeze(1)
                .expand(-1, top_k)
                .reshape(-1)
            )
            expert_index = top_k_index.reshape(-1)
            permutation = torch.argsort(expert_index)
            inverse = torch.empty_like(permutation)
            inverse[permutation] = torch.arange(permutation.numel(), device=x.device)

            sorted_experts = expert_index[permutation]
            sorted_tokens = token_index[permutation]
            current = x[sorted_tokens]
            counts = torch.bincount(sorted_experts, minlength=base.num_experts)
            offsets = counts.cumsum(0, dtype=torch.int32)

            gate_up = _grouped_linear(current, base.gate_up_proj, offsets)
            gate_up = gate_up + _grouped_expert_delta(
                gate_wrapper,
                current,
                offsets,
            )
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = base.act_fn(gate) * up
            current = _grouped_linear(intermediate, base.down_proj, offsets)
            current = current + _grouped_expert_delta(
                down_wrapper,
                intermediate,
                offsets,
            )
            sorted_weights = top_k_weights.reshape(-1)[permutation]
            current = current * sorted_weights.unsqueeze(-1)
            current = current[inverse].reshape(num_tokens, top_k, hidden_dim)
            expert_order = torch.argsort(top_k_index, dim=-1)
            current = current.gather(
                1,
                expert_order.unsqueeze(-1).expand(-1, -1, hidden_dim),
            )
            # Match the eager implementation's ascending-expert BF16
            # accumulation without its atomic index_add nondeterminism.
            final = torch.zeros_like(current[:, 0])
            for top_k_position in range(top_k):
                final.add_(current[:, top_k_position].to(dtype=x.dtype))
            return final

    wrapper_class = (
        GroupedGemmaExpertWrapper
        if implementation == "grouped_mm"
        else EagerGemmaExpertWrapper
    )

    installed = 0
    for name, module in list(peft_model.named_modules()):
        if not isinstance(module, ParamWrapper):
            continue
        if isinstance(module.base_layer, ParamWrapper):
            raw = module.get_base_layer()
            if raw.__class__.__name__ in EXPERT_LAYER_NAMES:
                # Install an explicit specialized module at the same tree
                # location. Copying the Module registry retains PEFT's exact
                # parameter/state-dict names without rewriting Transformers
                # source or mutating the existing object's class.
                replacement = wrapper_class.__new__(wrapper_class)
                replacement.__dict__ = module.__dict__.copy()
                if moe_layout is not None and moe_layout.enabled:
                    replacement.forward = torch.compiler.disable(replacement.forward)  # type: ignore[method-assign]
                parent_name, _, child_name = name.rpartition(".")
                parent = (
                    peft_model.get_submodule(parent_name)
                    if parent_name
                    else peft_model
                )
                setattr(parent, child_name, replacement)
                installed += 1
    return installed


def _module_name(model: nn.Module, target: nn.Module | None) -> str | None:
    if target is None:
        return None
    for name, module in model.named_modules():
        if module is target:
            return name
    return None


def discover_lora_targets(
    model: nn.Module,
    spec: ModelSpec,
    settings: LoraSettings,
) -> tuple[list[str], list[str], dict[str, int], tuple[LoraTarget, ...]]:
    module_targets: list[str] = []
    parameter_targets: list[str] = []
    rank_pattern: dict[str, int] = {}
    audit: list[LoraTarget] = []

    input_name = _module_name(model, getattr(model, "get_input_embeddings", lambda: None)())
    output_name = _module_name(model, getattr(model, "get_output_embeddings", lambda: None)())

    for name, module in model.named_modules():
        is_required_embedding = name in {input_name, output_name}
        if not isinstance(module, (nn.Linear, nn.Embedding)) and not is_required_embedding:
            continue
        if isinstance(module, nn.Embedding) and not is_required_embedding:
            continue
        if not name:
            continue
        rank = settings.rank
        kind = "linear"
        if name == input_name:
            kind = "embedding"
        if name == output_name:
            kind = "lm_head"
        if spec.moe and ".mlp." in name and ".experts." not in name:
            rank = settings.expert_rank
            kind = "shared_expert"
            rank_pattern[name] = rank
        weight = getattr(module, "weight", None)
        shape = tuple(weight.shape) if weight is not None else ()
        module_targets.append(name)
        audit.append(
            LoraTarget(
                name=name,
                kind=kind,
                rank=rank,
                shape=shape,
                adapter_parameters=_adapter_parameter_count(shape, rank),
            )
        )

    if input_name is None:
        raise ValueError("model does not expose an input embedding module")
    if output_name is None:
        raise ValueError("model does not expose an LM-head module")

    if spec.moe:
        for name, parameter in model.named_parameters():
            if ".experts." not in name:
                continue
            if parameter.ndim != 3:
                continue
            parameter_targets.append(name)
            rank_pattern[name] = settings.expert_rank
            audit.append(
                LoraTarget(
                    name=name,
                    kind="routed_expert",
                    rank=settings.expert_rank,
                    shape=tuple(parameter.shape),
                    adapter_parameters=_adapter_parameter_count(
                        tuple(parameter.shape), settings.expert_rank
                    ),
                )
            )
        if not parameter_targets:
            raise ValueError("MoE model has no supported 3D expert parameters")

    if len(module_targets) != len(set(module_targets)):
        raise AssertionError("duplicate module targets discovered")
    return module_targets, parameter_targets, rank_pattern, tuple(audit)


def inject_lora(
    model: nn.Module,
    spec: ModelSpec,
    settings: LoraSettings,
    *,
    expert_implementation: str = "auto",
    moe_layout: MoeParallelLayout | None = None,
) -> tuple[nn.Module, LoraAudit]:
    from peft import LoraConfig, TaskType, get_peft_model

    module_targets, parameter_targets, rank_pattern, targets = discover_lora_targets(
        model, spec, settings
    )
    input_weight = getattr(cast(Any, model).get_input_embeddings(), "weight", None)
    output_weight = getattr(cast(Any, model).get_output_embeddings(), "weight", None)
    weights_are_tied = (
        isinstance(input_weight, torch.Tensor)
        and isinstance(output_weight, torch.Tensor)
        and input_weight.untyped_storage().data_ptr() == output_weight.untyped_storage().data_ptr()
    )
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        bias="none",
        target_modules=module_targets,
        target_parameters=parameter_targets or None,
        rank_pattern=rank_pattern,
        modules_to_save=None,
        ensure_weight_tying=weights_are_tied,
    )
    with warnings.catch_warnings():
        # PEFT emits this from architecture metadata for untied Qwen/Llama
        # heads and for Gemma's intentional tie. The latter is audited above,
        # preserved as transposed adapter storage, and consolidated before
        # AdamW, so neither variant needs an operator-facing warning.
        warnings.filterwarnings(
            "ignore",
            message=r"Model (?:has|with) `tie_word_embeddings=True`.*",
        )
        # PEFT otherwise upcasts BF16 adapters to FP32 after injection. For a
        # tied embedding/head it performs that cast independently for the two
        # transposed Parameter views, which also breaks their shared storage.
        peft_model = get_peft_model(
            cast(Any, model),
            config,
            autocast_adapter_dtype=False,
        )
    if spec.moe:
        if expert_implementation == "auto":
            expert_implementation = "eager"
        if moe_layout is not None and moe_layout.enabled and expert_implementation != "grouped_mm":
            raise ValueError(
                "expert_parallel_size > 1 requires runtime.experts=grouped_mm; "
                f"got {expert_implementation!r}"
            )
        installed = _install_active_expert_forward(peft_model, expert_implementation, moe_layout)
        expected = sum(
            1 for module in model.modules() if module.__class__.__name__ in EXPERT_LAYER_NAMES
        )
        if installed != expected:
            raise RuntimeError(
                f"installed {installed} active expert LoRA paths, expected {expected}"
            )

    trainable: list[tuple[str, nn.Parameter]] = []
    for name, parameter in peft_model.named_parameters():
        if parameter.requires_grad:
            if parameter.dtype != torch.bfloat16:
                raise RuntimeError(
                    f"PEFT did not preserve the BF16 adapter dtype for {name}: "
                    f"found {parameter.dtype}"
                )
            trainable.append((name, parameter))
    if not trainable:
        raise RuntimeError("PEFT created no trainable LoRA parameters")
    unexpected = [name for name, _ in trainable if "lora_" not in name]
    if unexpected:
        raise RuntimeError(
            "non-LoRA parameters unexpectedly remain trainable: " + ", ".join(unexpected[:8])
        )
    dtypes = {parameter.dtype for _, parameter in trainable}
    if dtypes != {torch.bfloat16}:
        raise RuntimeError(
            f"LoRA parameters must all be bfloat16, found {sorted(map(str, dtypes))}"
        )

    _verify_injected_targets(peft_model, targets)
    if weights_are_tied:
        _verify_tied_lora_storage(peft_model)
    return peft_model, LoraAudit(
        targets=targets,
        trainable_tensors=len(trainable),
        trainable_parameters=sum(parameter.numel() for _, parameter in trainable),
        trainable_dtype=torch.bfloat16,
    )


def _trainable_storage_groups(model: nn.Module) -> list[list[nn.Parameter]]:
    groups: dict[tuple[str, int, int], list[nn.Parameter]] = {}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        tensor = cast(torch.Tensor, parameter)
        if tensor.__class__.__name__ == "DTensor":
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        groups.setdefault(key, []).append(parameter)
    return [group for group in groups.values() if len(group) > 1]


def _verify_tied_lora_storage(model: nn.Module) -> None:
    input_embeddings = cast(Any, cast(Any, model).get_input_embeddings())
    output_embeddings = cast(Any, cast(Any, model).get_output_embeddings())
    adapter = input_embeddings.active_adapters[0]
    pairs = (
        (
            input_embeddings.lora_embedding_A[adapter],
            output_embeddings.lora_B[adapter].weight,
        ),
        (
            input_embeddings.lora_embedding_B[adapter],
            output_embeddings.lora_A[adapter].weight,
        ),
    )
    for source, tied in pairs:
        same_storage = (
            source.untyped_storage().data_ptr()
            == tied.untyped_storage().data_ptr()
        )
        if not same_storage or source.shape != tied.T.shape:
            raise RuntimeError("PEFT failed to preserve tied embedding/head LoRA storage")


@torch.no_grad()
def consolidate_tied_lora_gradients(model: nn.Module) -> None:
    """Sum gradients for transposed tied-LoRA views before standard AdamW.

    PEFT represents a tied embedding/head adapter with two Parameter objects
    that are transposed views over the same storage. Optimizers would otherwise
    update that storage twice with independent moments. This function leaves
    AdamW untouched: it accumulates both path gradients into the contiguous
    source Parameter and clears the transposed alias gradient.
    """

    for group in _trainable_storage_groups(model):
        contiguous = [parameter for parameter in group if parameter.is_contiguous()]
        if len(group) != 2 or len(contiguous) != 1:
            raise RuntimeError("unsupported trainable shared-storage parameter group")
        source = contiguous[0]
        tied = group[0] if group[1] is source else group[1]
        if source.ndim != 2 or source.shape != tied.T.shape:
            raise RuntimeError("tied LoRA parameters must be transposed matrices")
        if tied.grad is None:
            continue
        tied_gradient = tied.grad.T.contiguous()
        if source.grad is None:
            source.grad = tied_gradient
        else:
            source.grad.add_(tied_gradient)
        tied.grad = None


def _verify_injected_targets(model: nn.Module, targets: tuple[LoraTarget, ...]) -> None:
    from peft.tuners.lora.layer import LoraLayer, ParamWrapper

    lora_modules = {
        name.removeprefix("base_model.model."): module
        for name, module in model.named_modules()
        if isinstance(module, LoraLayer)
    }
    missing: list[str] = []
    wrong_rank: list[str] = []
    routed_by_parent: dict[str, list[LoraTarget]] = {}
    for target in targets:
        if target.kind == "routed_expert":
            routed_by_parent.setdefault(target.name.rsplit(".", 1)[0], []).append(target)

    for parent, parent_targets in routed_by_parent.items():
        wrappers = [
            module
            for name, module in model.named_modules()
            if isinstance(module, ParamWrapper)
            and name.removeprefix("base_model.model.").startswith(parent)
        ]
        if len(wrappers) < len(parent_targets):
            missing.extend(target.name for target in parent_targets[len(wrappers) :])
        for wrapper in wrappers[: len(parent_targets)]:
            wrapper_ranks: Any = getattr(wrapper, "r", {})
            rank = wrapper_ranks.get("default") if isinstance(wrapper_ranks, dict) else None
            expected = parent_targets[0].rank
            if rank is not None and int(rank) != expected:
                wrong_rank.append(f"{parent}: expected {expected}, found {rank}")

    for target in targets:
        if target.kind == "routed_expert":
            continue
        module = lora_modules.get(target.name)
        if module is None:
            # PEFT can add another model prefix for architecture-specific wrappers.
            module = next(
                (value for name, value in lora_modules.items() if name.endswith(target.name)),
                None,
            )
        if module is None:
            missing.append(target.name)
            continue
        ranks: Any = getattr(module, "r", {})
        rank = ranks.get("default") if isinstance(ranks, dict) else None
        if rank is not None and int(rank) != target.rank:
            wrong_rank.append(f"{target.name}: expected {target.rank}, found {rank}")
    if missing:
        raise RuntimeError("LoRA targets were not injected: " + ", ".join(missing[:12]))
    if wrong_rank:
        raise RuntimeError("LoRA rank mismatch: " + "; ".join(wrong_rank[:12]))
