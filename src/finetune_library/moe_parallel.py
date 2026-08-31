from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

EXPERT_MODULE_NAMES = frozenset({"Gemma4TextExperts", "Qwen3_5MoeExperts"})
EXPERT_WRAPPER_NAMES = frozenset({"GroupedGemmaExpertWrapper", "EagerGemmaExpertWrapper"})


def is_expert_module(module: nn.Module) -> bool:
    layer: Any = module
    if hasattr(module, "get_base_layer"):
        try:
            layer = module.get_base_layer()
        except Exception:
            layer = module
    return layer.__class__.__name__ in EXPERT_MODULE_NAMES


def is_expert_unit(module: nn.Module) -> bool:
    if module.__class__.__name__ in EXPERT_MODULE_NAMES | EXPERT_WRAPPER_NAMES:
        return True
    return is_expert_module(module)


def expert_parameters(model: nn.Module) -> set[nn.Parameter]:
    params: set[nn.Parameter] = set()
    for module in model.modules():
        if is_expert_module(module):
            params.update(module.parameters())
    return params


class MoeA2ABackend(StrEnum):
    AUTO = "auto"
    NATIVE = "native"
    DEEPEP = "deepep"


def resolve_a2a_backend(requested: str) -> MoeA2ABackend:
    if requested == MoeA2ABackend.AUTO:
        try:
            import deep_ep  # noqa: F401

            return MoeA2ABackend.DEEPEP
        except Exception:
            return MoeA2ABackend.NATIVE
    return MoeA2ABackend(requested)


@dataclass(frozen=True, slots=True)
class MoeParallelLayout:
    num_experts: int
    expert_parallel_size: int
    expert_parallel_rank: int
    global_expert_offset: int
    local_num_experts: int
    ep_group: Any
    dense_mesh: Any = None
    a2a_backend: MoeA2ABackend = MoeA2ABackend.NATIVE

    @property
    def enabled(self) -> bool:
        return self.expert_parallel_size > 1


def _all_to_all_exchange(
    tensor: torch.Tensor,
    *,
    perm: torch.Tensor,
    input_split_sizes: list[int],
    recv_sizes: list[int],
    ep_group: Any,
) -> torch.Tensor:
    ordered = tensor[perm]
    split_tensors = list(torch.split(ordered, input_split_sizes, dim=0))
    trailing = tuple(tensor.shape[1:])
    recv_buffers = [tensor.new_zeros((size, *trailing)) for size in recv_sizes]
    if split_tensors:
        dist.all_to_all(recv_buffers, split_tensors, group=ep_group)
    return torch.cat(recv_buffers, dim=0) if recv_buffers else tensor.new_zeros((0, *trailing))


class _AllToAllTokens(Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        token_ids: torch.Tensor,
        weights: torch.Tensor,
        ep_group: Any,
        ep_size: int,
        experts_per_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int], torch.Tensor]:
        device = hidden_states.device
        dest_rank = expert_ids // experts_per_rank
        perm = torch.argsort(dest_rank)
        input_split_sizes = torch.bincount(dest_rank, minlength=ep_size).tolist()
        input_splits = torch.tensor(input_split_sizes, device=device, dtype=torch.long)
        output_splits = torch.zeros(ep_size, device=device, dtype=torch.long)
        dist.all_to_all_single(output_splits, input_splits, group=ep_group)
        recv_sizes = output_splits.tolist()

        recv_hidden = _all_to_all_exchange(
            hidden_states[token_ids],
            perm=perm,
            input_split_sizes=input_split_sizes,
            recv_sizes=recv_sizes,
            ep_group=ep_group,
        )
        recv_expert_ids = _all_to_all_exchange(
            expert_ids,
            perm=perm,
            input_split_sizes=input_split_sizes,
            recv_sizes=recv_sizes,
            ep_group=ep_group,
        )
        recv_weights = _all_to_all_exchange(
            weights,
            perm=perm,
            input_split_sizes=input_split_sizes,
            recv_sizes=recv_sizes,
            ep_group=ep_group,
        )
        recv_token_ids = _all_to_all_exchange(
            token_ids,
            perm=perm,
            input_split_sizes=input_split_sizes,
            recv_sizes=recv_sizes,
            ep_group=ep_group,
        )

        ctx.ep_group = ep_group
        ctx.input_split_sizes = input_split_sizes
        ctx.recv_sizes = recv_sizes
        ctx.perm = perm
        ctx.token_ids = token_ids
        ctx.weights = weights
        ctx.hidden_dim = hidden_states.shape[-1]
        ctx.num_tokens = hidden_states.shape[0]
        return (
            recv_hidden,
            recv_expert_ids,
            recv_weights,
            recv_token_ids,
            input_split_sizes,
            recv_sizes,
            perm,
        )

    @staticmethod
    def backward(ctx: Any, grad_hidden: torch.Tensor, *_: Any) -> tuple[torch.Tensor | None, ...]:
        split_grad = list(torch.split(grad_hidden, ctx.recv_sizes, dim=0))
        recv_grad = [
            grad_hidden.new_zeros((size, ctx.hidden_dim)) for size in ctx.input_split_sizes
        ]
        if split_grad:
            dist.all_to_all(recv_grad, split_grad, group=ctx.ep_group)
        sorted_grad = torch.cat(recv_grad, dim=0) if recv_grad else grad_hidden.new_zeros(0, ctx.hidden_dim)
        grad_out = grad_hidden.new_zeros(ctx.num_tokens, ctx.hidden_dim)
        grad_out.index_add_(0, ctx.token_ids[ctx.perm], sorted_grad)
        return grad_out, None, None, None, None, None, None


class _CombineExpertOutputs(Function):
    @staticmethod
    def forward(
        ctx: Any,
        expert_output: torch.Tensor,
        token_ids: torch.Tensor,
        weights: torch.Tensor,
        ep_group: Any,
        input_split_sizes: list[int],
        recv_sizes: list[int],
        perm: torch.Tensor,
        original_token_ids: torch.Tensor,
        num_tokens: int,
        hidden_dim: int,
    ) -> torch.Tensor:
        split_output = list(torch.split(expert_output, recv_sizes, dim=0))
        recv_output = [expert_output.new_zeros(size, hidden_dim) for size in input_split_sizes]
        dist.all_to_all(recv_output, split_output, group=ep_group)
        sorted_output = torch.cat(recv_output, dim=0) if recv_output else expert_output.new_zeros(0, hidden_dim)
        # Local expert kernels already apply routing weights; only route outputs
        # back to their source tokens here.
        final = expert_output.new_zeros(num_tokens, hidden_dim)
        final.index_add_(0, original_token_ids[perm], sorted_output.to(final.dtype))
        ctx.ep_group = ep_group
        ctx.input_split_sizes = input_split_sizes
        ctx.recv_sizes = recv_sizes
        ctx.perm = perm
        ctx.token_ids = original_token_ids
        ctx.weights = weights
        return final

    @staticmethod
    def backward(ctx: Any, grad_final: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        sorted_grad = grad_final[ctx.token_ids[ctx.perm]]
        split_grad = list(torch.split(sorted_grad, ctx.input_split_sizes, dim=0))
        recv_grad = [sorted_grad.new_zeros(size, grad_final.shape[-1]) for size in ctx.recv_sizes]
        dist.all_to_all(recv_grad, split_grad, group=ctx.ep_group)
        expert_grad = torch.cat(recv_grad, dim=0) if recv_grad else sorted_grad.new_zeros(0, grad_final.shape[-1])
        return expert_grad, None, None, None, None, None, None, None, None, None


_deepep_buffer: Any = None
_deepep_buffer_key: tuple[int, int, int] | None = None


def _get_deepep_buffer(ep_group: Any, hidden_dim: int, param_bytes: int = 2) -> Any:
    global _deepep_buffer, _deepep_buffer_key
    import deep_ep

    hidden_bytes = hidden_dim * max(param_bytes, 2)
    key = (id(ep_group), hidden_bytes, ep_group.size())
    if _deepep_buffer is not None and _deepep_buffer_key == key:
        return _deepep_buffer

    num_nvl_bytes = 0
    num_rdma_bytes = 0
    for config in (
        deep_ep.Buffer.get_dispatch_config(ep_group.size()),
        deep_ep.Buffer.get_combine_config(ep_group.size()),
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, ep_group.size()),
            num_nvl_bytes,
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, ep_group.size()),
            num_rdma_bytes,
        )
    _deepep_buffer = deep_ep.Buffer(ep_group, num_nvl_bytes, num_rdma_bytes)
    _deepep_buffer_key = key
    return _deepep_buffer


def _flatten_deepep_assignments(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if expert_ids.ndim == 1:
        mask = expert_ids >= 0
        token_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)[mask]
        return (
            hidden_states[mask],
            expert_ids[mask],
            weights[mask],
            token_indices,
        )
    mask = expert_ids >= 0
    valid_ids = expert_ids[mask]
    valid_weights = weights[mask]
    token_indices = (
        torch.arange(hidden_states.shape[0], device=hidden_states.device)
        .unsqueeze(1)
        .expand_as(expert_ids)[mask]
    )
    flat_hidden = hidden_states.index_select(0, token_indices)
    return flat_hidden, valid_ids, valid_weights, token_indices


def _scatter_deepep_assignments(
    expert_output: torch.Tensor,
    token_indices: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    output = expert_output.new_zeros(num_tokens, expert_output.shape[-1])
    output.index_add_(0, token_indices, expert_output.to(output.dtype))
    return output


class _DeepEPDispatch(Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        topk_index: torch.Tensor,
        topk_weights: torch.Tensor,
        buffer: Any,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        topk_ids = topk_index.to(torch.int64).contiguous()
        weights = topk_weights.to(torch.float32).contiguous()
        dispatch_x = hidden_states.to(torch.bfloat16).contiguous()
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = buffer.get_dispatch_layout(topk_ids, num_experts)
        (
            recv_hidden,
            recv_topk_ids,
            recv_topk_weights,
            _,
            handle,
            _,
        ) = buffer.dispatch(
            dispatch_x,
            topk_idx=topk_ids,
            topk_weights=weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.input_dtype = hidden_states.dtype
        return recv_hidden, recv_topk_ids, recv_topk_weights, handle

    @staticmethod
    def backward(ctx: Any, grad_recv_hidden: torch.Tensor, *_: Any) -> tuple[torch.Tensor | None, ...]:
        if grad_recv_hidden is None:
            return None, None, None, None, None, None
        combined_grad, _, _ = ctx.buffer.combine(
            grad_recv_hidden.contiguous(),
            ctx.handle,
            async_finish=False,
        )
        grad_hidden = combined_grad.to(ctx.input_dtype)
        return grad_hidden, None, None, None, None


class _DeepEPCombine(Function):
    @staticmethod
    def forward(ctx: Any, expert_output: torch.Tensor, buffer: Any, handle: Any) -> torch.Tensor:
        combined, _, _ = buffer.combine(expert_output.contiguous(), handle, async_finish=False)
        ctx.buffer = buffer
        ctx.handle = handle
        return combined

    @staticmethod
    def backward(ctx: Any, grad_combined: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        if grad_combined is None:
            return None, None, None
        grad_output, _, _, _, _, _ = ctx.buffer.dispatch(
            grad_combined.contiguous(),
            handle=ctx.handle,
            async_finish=False,
        )
        return grad_output, None, None


def expert_parallel_forward_deepep(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    layout: MoeParallelLayout,
    local_forward: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    buffer = _get_deepep_buffer(layout.ep_group, hidden_states.shape[-1])
    recv_hidden, recv_expert_ids, recv_weights, handle = _DeepEPDispatch.apply(
        hidden_states,
        top_k_index,
        top_k_weights,
        buffer,
        layout.num_experts,
    )
    flat_hidden, flat_ids, flat_weights, token_indices = _flatten_deepep_assignments(
        recv_hidden,
        recv_expert_ids,
        recv_weights,
    )
    local_output = local_forward(flat_hidden, flat_ids, flat_weights)
    per_token_output = _scatter_deepep_assignments(
        local_output,
        token_indices,
        recv_hidden.shape[0],
    )
    combined = _DeepEPCombine.apply(per_token_output, buffer, handle)
    return combined.to(hidden_states.dtype)


def _local_grouped_experts_forward(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    global_offset: int,
) -> torch.Tensor:
    if hidden_states.numel() == 0:
        return hidden_states
    local_expert_ids = expert_ids - global_offset
    if torch.any(local_expert_ids < 0) or torch.any(local_expert_ids >= gate_up_proj.shape[0]):
        raise RuntimeError(
            "received expert ids outside the local shard: "
            f"min={int(local_expert_ids.min())} max={int(local_expert_ids.max())} "
            f"local={gate_up_proj.shape[0]}"
        )
    permutation = torch.argsort(local_expert_ids)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=hidden_states.device)
    sorted_experts = local_expert_ids[permutation]
    sorted_hidden = hidden_states[permutation]
    counts = torch.bincount(sorted_experts, minlength=gate_up_proj.shape[0])
    offsets = counts.cumsum(0, dtype=torch.int32)
    gate_up = torch._grouped_mm(
        sorted_hidden,
        gate_up_proj.transpose(-2, -1),
        offs=offsets,
    )
    gate, up = gate_up.chunk(2, dim=-1)
    intermediate = act_fn(gate) * up
    current = torch._grouped_mm(
        intermediate,
        down_proj.transpose(-2, -1),
        offs=offsets,
    )
    sorted_weights = weights[permutation]
    current = current * sorted_weights.unsqueeze(-1)
    return current[inverse]


def expert_parallel_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    layout: MoeParallelLayout,
    local_forward: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    if layout.a2a_backend == MoeA2ABackend.DEEPEP:
        return expert_parallel_forward_deepep(
            hidden_states,
            top_k_index,
            top_k_weights,
            layout,
            local_forward,
        )
    num_tokens, hidden_dim = hidden_states.shape
    top_k = top_k_index.shape[-1]
    device = hidden_states.device
    expert_ids = top_k_index.reshape(-1)
    token_ids = torch.arange(num_tokens, device=device).repeat_interleave(top_k)
    weights = top_k_weights.reshape(-1)

    (
        recv_hidden,
        recv_expert_ids,
        recv_weights,
        recv_token_ids,
        input_split_sizes,
        recv_sizes,
        perm,
    ) = _AllToAllTokens.apply(
        hidden_states,
        expert_ids,
        token_ids,
        weights,
        layout.ep_group,
        layout.expert_parallel_size,
        layout.local_num_experts,
    )
    local_output = local_forward(recv_hidden, recv_expert_ids, recv_weights)
    return _CombineExpertOutputs.apply(
        local_output,
        recv_token_ids,
        weights,
        layout.ep_group,
        input_split_sizes,
        recv_sizes,
        perm,
        token_ids,
        num_tokens,
        hidden_dim,
    )


def _ep_experts_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    layout: MoeParallelLayout,
) -> torch.Tensor:
    expert_offset = (
        0 if layout.a2a_backend == MoeA2ABackend.DEEPEP else layout.global_expert_offset
    )

    def local_forward(
        recv_hidden: torch.Tensor,
        recv_expert_ids: torch.Tensor,
        recv_weights: torch.Tensor,
    ) -> torch.Tensor:
        return _local_grouped_experts_forward(
            recv_hidden,
            module.gate_up_proj,
            module.down_proj,
            recv_expert_ids,
            recv_weights,
            module.act_fn,
            expert_offset,
        )

    return expert_parallel_forward(
        hidden_states,
        top_k_index,
        top_k_weights,
        layout,
        local_forward,
    )


def slice_expert_parameters(model: nn.Module, layout: MoeParallelLayout) -> int:
    sliced = 0
    start = layout.global_expert_offset
    end = start + layout.local_num_experts
    for module in model.modules():
        if module.__class__.__name__ not in EXPERT_MODULE_NAMES:
            continue
        module.gate_up_proj = nn.Parameter(module.gate_up_proj.data[start:end].contiguous())
        module.down_proj = nn.Parameter(module.down_proj.data[start:end].contiguous())
        module.num_experts = layout.local_num_experts
        module._global_num_experts = layout.num_experts
        module._global_expert_offset = start
        sliced += 1
    return sliced


def install_expert_parallel(model: nn.Module, layout: MoeParallelLayout) -> int:
    if not layout.enabled:
        return 0
    installed = 0
    for module in model.modules():
        if module.__class__.__name__ not in EXPERT_MODULE_NAMES:
            continue
        if getattr(module, "_ep_forward_installed", False):
            continue

        def forward(
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
            *,
            _module: nn.Module = module,
            _layout: MoeParallelLayout = layout,
        ) -> torch.Tensor:
            return _ep_experts_forward(_module, hidden_states, top_k_index, top_k_weights, _layout)

        module.forward = forward  # type: ignore[method-assign]
        module._ep_forward_installed = True
        installed += 1
    return installed


def resolve_num_experts(model: nn.Module, spec_num_experts: int | None) -> int:
    if spec_num_experts is not None:
        return spec_num_experts
    for module in model.modules():
        if module.__class__.__name__ in EXPERT_MODULE_NAMES:
            return int(module.num_experts)
    config = getattr(model, "config", None)
    if config is not None:
        text_config = getattr(config, "text_config", config)
        value = getattr(text_config, "num_experts", None)
        if value is not None:
            return int(value)
    raise ValueError("could not resolve MoE expert count")


def build_moe_layout(
    *,
    num_experts: int,
    expert_parallel_size: int,
    mesh: Any,
    a2a_backend: str = MoeA2ABackend.AUTO,
) -> MoeParallelLayout:
    if expert_parallel_size < 1:
        raise ValueError("expert_parallel_size must be >= 1")
    if num_experts % expert_parallel_size != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by expert_parallel_size={expert_parallel_size}"
        )
    experts_per_rank = num_experts // expert_parallel_size
    if mesh is None or expert_parallel_size == 1:
        ep_rank = 0
        ep_group = None
        dense_mesh = mesh
    else:
        ep_rank = int(mesh["ep"].get_local_rank())
        ep_group = mesh["ep"].get_group()
        if "dp_replicate" in mesh.mesh_dim_names and "dp_shard" in mesh.mesh_dim_names:
            dense_mesh = mesh["dp_replicate", "dp_shard"]
        elif "dp_shard" in mesh.mesh_dim_names:
            dense_mesh = mesh["dp_shard"]
        else:
            dense_mesh = mesh
    resolved_backend = resolve_a2a_backend(a2a_backend)
    if resolved_backend == MoeA2ABackend.DEEPEP:
        try:
            import deep_ep  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "runtime.moe_a2a_backend=deepep requires the deep_ep package"
            ) from exc
    return MoeParallelLayout(
        num_experts=num_experts,
        expert_parallel_size=expert_parallel_size,
        expert_parallel_rank=ep_rank,
        global_expert_offset=ep_rank * experts_per_rank,
        local_num_experts=experts_per_rank,
        ep_group=ep_group,
        dense_mesh=dense_mesh,
        a2a_backend=resolved_backend,
    )
