from __future__ import annotations

import logging
import os
import sys
import traceback
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from finetune_library.config import DistributedConfig, DistributedStrategy
from finetune_library.moe_parallel import MoeParallelLayout, expert_parameters, is_expert_unit
from finetune_library.registry import ModelSpec


@dataclass(slots=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    strategy: DistributedStrategy
    shard_size: int
    replicate_size: int
    expert_parallel_size: int
    data_parallel_rank: int
    data_parallel_size: int
    data_parallel_group: Any = None
    mesh: Any = None
    dense_mesh: Any = None
    moe_layout: MoeParallelLayout | None = None
    initialized_here: bool = False
    dp_shard_group: Any = None
    dp_shard_size: int = 1
    replicated_trainables: tuple[nn.Parameter, ...] = ()
    ep_replicated_trainables: tuple[nn.Parameter, ...] = ()

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def reduce_data_parallel(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.data_parallel_size > 1:
            dist.all_reduce(tensor, group=self.data_parallel_group)
        return tensor

    @torch.no_grad()
    def sync_replicated_gradients(self) -> None:
        """Average FSDP-ignored adapter gradients across data replicas.

        Tied embedding/head LoRA parameters include transposed views. FSDP2
        cannot shard those non-contiguous parameters, so ``wrap_model`` keeps
        the small tied adapter group replicated. Shard ranks consume the same
        batch and replication groups consume different batches; only the
        latter therefore require an all-reduce. Their two path gradients are
        consolidated after this synchronization and before clipping/AdamW.
        """

        if self.data_parallel_size > 1 and self.data_parallel_group is not None:
            for parameter in self.replicated_trainables:
                if parameter.grad is None:
                    continue
                synchronized = parameter.grad.contiguous()
                dist.all_reduce(synchronized, group=self.data_parallel_group)
                synchronized.div_(self.data_parallel_size)
                parameter.grad.copy_(synchronized)
        if self.expert_parallel_size > 1 and self.ep_replicated_trainables:
            # A local expert's gradient already sums the tokens of every rank
            # in its EP group (via the all-to-all backward), each scaled by the
            # per-rank loss normalization. Summing over the expert's replicas
            # therefore covers every rank's tokens exactly once, and dividing
            # by the full world size yields the same mean as DDP. (Dividing by
            # the replica count alone over-scaled expert grads by ep_size.)
            for parameter in self.ep_replicated_trainables:
                if parameter.grad is None:
                    continue
                synchronized = parameter.grad.contiguous()
                if self.dp_shard_size > 1 and self.dp_shard_group is not None:
                    dist.all_reduce(synchronized, group=self.dp_shard_group)
                synchronized.div_(self.world_size)
                parameter.grad.copy_(synchronized)

    @torch.no_grad()
    def clip_grad_norm(
        self,
        parameters: Any,
        max_norm: float,
    ) -> torch.Tensor:
        """Clip mixed local and FSDP2 gradients with one multi-tensor pass."""

        # Local (non-DTensor) gradients are replicated over different groups
        # depending on which FSDP-ignored bucket the parameter falls into:
        # ep_replicated_trainables (routed/shared-expert LoRA, ignored by
        # fully_shard) are only replicated across dp_shard_size ranks per ep
        # coordinate, *not* the whole world -- using world_size there (as
        # this used to, unconditionally) understates their gradient-norm
        # contribution whenever expert_parallel_size > 1. Every other local
        # gradient (plain DDP, or replicated_trainables/tied adapters, which
        # data_parallel_group now spans the full world) is replicated over
        # the whole world.
        ep_replicated_ids = {id(parameter) for parameter in self.ep_replicated_trainables}
        local_gradients: list[torch.Tensor] = []
        replication_factors: list[int] = []
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.__class__.__name__ == "DTensor":
                local = cast(Any, gradient).to_local()
                factor = 1
                mesh = cast(Any, gradient).device_mesh
                for dimension, placement in enumerate(cast(Any, gradient).placements):
                    if placement.__class__.__name__ == "Replicate":
                        factor *= int(mesh.size(dimension))
            else:
                local = gradient
                if id(parameter) in ep_replicated_ids and self.dp_shard_size > 1:
                    factor = self.dp_shard_size
                else:
                    factor = self.world_size if self.distributed else 1
            local_gradients.append(local)
            replication_factors.append(factor)

        if not local_gradients:
            return torch.zeros((), device=self.device, dtype=torch.float32)
        norms = torch._foreach_norm(local_gradients, 2.0)
        local_square = torch.stack(
            [
                norm.float().square() / factor
                for norm, factor in zip(norms, replication_factors, strict=True)
            ]
        ).sum()
        if self.distributed:
            dist.all_reduce(local_square)
        total_norm = local_square.sqrt()
        coefficient = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
        torch._foreach_mul_(local_gradients, coefficient)
        return total_norm

    def close(self, *, failed: bool = False) -> None:
        if self.initialized_here and dist.is_initialized():
            if failed and self.world_size > 1:
                # Any NCCL teardown can block forever once a rank has failed:
                # after a CUDA OOM inside an NCCL all-reduce (Qwen3.5-35B EP4),
                # destroy_process_group() and _abort_process_group() both hung
                # every rank with the exception never printed. Report the
                # exception now and exit this rank; torchrun sees the failed
                # child and stops the others.
                traceback.print_exc()
                logging.shutdown()
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(1)
            # Successful training already synchronizes before returning.
            # A barrier here can deadlock or mask the root exception when one
            # rank has failed (for example, after a CUDA OOM).
            dist.destroy_process_group()


def initialize_distributed(
    config: DistributedConfig,
    spec: ModelSpec,
    *,
    timeout_minutes: int = 30,
) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    expert_parallel_size = config.expert_parallel_size
    if expert_parallel_size > 1 and not spec.moe:
        raise ValueError(f"{spec.key} does not support expert_parallel_size > 1")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=timeout_minutes),
            device_id=device if device.type == "cuda" else None,
        )
        initialized_here = True

    strategy = config.strategy
    if strategy == DistributedStrategy.AUTO:
        strategy = spec.preferred_strategy
    if world_size == 1 and strategy == DistributedStrategy.DDP:
        # A registry DDP default should still permit a one-GPU development run.
        shard_size, replicate_size = 1, 1
        data_rank, data_size = 0, 1
        return DistributedContext(
            rank,
            local_rank,
            world_size,
            device,
            strategy,
            shard_size,
            replicate_size,
            expert_parallel_size,
            data_rank,
            data_size,
            initialized_here=initialized_here,
        )

    mesh = None
    dense_mesh = None
    dp_shard_group = None
    dp_shard_size = 1
    data_group = dist.group.WORLD if world_size > 1 else None
    if strategy == DistributedStrategy.DDP:
        shard_size, replicate_size = 1, world_size
        data_rank, data_size = rank, world_size
    elif strategy in {DistributedStrategy.FSDP, DistributedStrategy.HSDP}:
        from torch.distributed.device_mesh import init_device_mesh

        dense_world = world_size // expert_parallel_size if expert_parallel_size > 1 else world_size
        if expert_parallel_size > 1 and dense_world * expert_parallel_size != world_size:
            raise ValueError(
                "world_size must equal dense_parallel_size * expert_parallel_size, "
                f"got {world_size} and ep={expert_parallel_size}"
            )
        # Every rank consumes distinct data under FSDP/HSDP, with or without
        # EP: data_rank/data_size/data_group always describe the full dense
        # world (identical to DDP). FSDP2's own reduce-scatter/all-reduce
        # over `dense_mesh` (below) provides the gradient averaging that
        # replaces DDP's ring all-reduce, so the *sampler* partition and the
        # *gradient averaging* group must span the same set of ranks -- the
        # whole world -- for the update to match DDP's mean over the same
        # global batch. (Previously this branch pinned data_rank/data_size to
        # 0/1 -- or, for HSDP, to the replicate dimension only -- which made
        # shard-group ranks silently train on the same batch; see
        # DistributedContext.sync_replicated_gradients for the FSDP-ignored
        # tied-adapter parameters, which still need an explicit all-reduce
        # because fully_shard never sees them.)
        data_rank, data_size = rank, world_size
        if strategy == DistributedStrategy.FSDP:
            if dense_world < 1:
                raise ValueError("fsdp requires at least one dense parallel rank")
            if dense_world == 1 and expert_parallel_size == 1 and world_size < 2:
                raise ValueError("fsdp requires torchrun with at least two processes")
            shard_size, replicate_size = dense_world, 1
            if expert_parallel_size > 1:
                # Dense (non-expert) params are sharded on dp_shard alone but
                # must be *replicated* (and therefore gradient-averaged) over
                # the ep dimension too, since ep ranks now see different
                # tokens. torch 2.11's DeviceMesh.__getitem__ can only select
                # dims in their *original creation order* (it raises "Mesh
                # dim indices should be in ascending order" otherwise, despite
                # its own docstring example implying free reordering --
                # confirmed empirically, not just from the docstring), so the
                # mesh is built directly in (ep, dp_shard) order -- dim0 (ep)
                # an implicit all-reduce/replicate dimension, dim1 (dp_shard)
                # sharded -- rather than created as (dp_shard, ep) and
                # reordered after the fact. fully_shard then reads this
                # exactly like a 2D HSDP mesh, so the reduce-scatter +
                # all-reduce together average over all `world_size` ranks,
                # matching DDP.
                mesh = init_device_mesh(
                    device.type,
                    (expert_parallel_size, dense_world),
                    mesh_dim_names=("ep", "dp_shard"),
                )
                dense_mesh = mesh
                dp_shard_group = mesh.get_group("dp_shard")
                dp_shard_size = mesh["dp_shard"].size()
            else:
                mesh = init_device_mesh(device.type, (dense_world,), mesh_dim_names=("dp_shard",))
                dense_mesh = mesh
        else:
            shard_size = config.shard_size or spec.shard_size
            replicate_size = config.replicate_size or spec.replicate_size
            if shard_size * replicate_size != dense_world:
                if dense_world % shard_size:
                    raise ValueError(
                        "hsdp requires dense_world_size to be divisible by shard_size, "
                        f"got {dense_world} % {shard_size}"
                    )
                replicate_size = dense_world // shard_size
            if shard_size > 8:
                raise ValueError("HSDP shard groups must remain within one 8-GPU NVSwitch node")
            if expert_parallel_size > 1:
                # Mesh dims are created (dp_replicate, ep, dp_shard) -- with
                # dp_replicate and ep *adjacent* -- rather than the more
                # "natural" (dp_replicate, dp_shard, ep), because torch
                # 2.11's DeviceMesh._flatten only supports flattening
                # adjacent, contiguous-stride dims (confirmed empirically);
                # dp_replicate and dp_shard alone, skipping over ep, are not
                # adjacent and cannot be flattened this way.
                mesh = init_device_mesh(
                    device.type,
                    (replicate_size, expert_parallel_size, shard_size),
                    mesh_dim_names=("dp_replicate", "ep", "dp_shard"),
                )
                # Dense params must be replicated (and averaged) over both
                # dp_replicate and ep, sharded over dp_shard only. fully_shard
                # only understands a 2D (replicate, shard) mesh, so flatten
                # the two (now-adjacent) replicate dimensions into one before
                # combining with dp_shard -- the same pattern used for
                # HSDP+CP elsewhere (e.g. torchtitan).
                mesh["dp_replicate", "ep"]._flatten("dp_replicate_ep")
                dense_mesh = mesh["dp_replicate_ep", "dp_shard"]
                # Expert-adapter parameters are ignored by fully_shard, so
                # each ep-coordinate's copy is replicated verbatim across
                # *both* dp_replicate and dp_shard; the manual all-reduce in
                # sync_replicated_gradients must therefore cover that full
                # group. dp_replicate and dp_shard are not adjacent in this
                # mesh (ep sits between them), so DeviceMesh cannot flatten
                # them directly -- build the group explicitly instead, with
                # the same rank arithmetic every rank computes identically
                # and in the same enumeration order (required so every rank
                # makes the same sequence of dist.new_group calls).
                ep_coordinate = (rank // shard_size) % expert_parallel_size
                dp_shard_group = None
                for ep_index in range(expert_parallel_size):
                    ranks = [
                        dp_replicate_index * (expert_parallel_size * shard_size)
                        + ep_index * shard_size
                        + dp_shard_index
                        for dp_replicate_index in range(replicate_size)
                        for dp_shard_index in range(shard_size)
                    ]
                    group = dist.new_group(ranks)
                    if ep_index == ep_coordinate:
                        dp_shard_group = group
                dp_shard_size = replicate_size * shard_size
            else:
                mesh = init_device_mesh(
                    device.type,
                    (replicate_size, shard_size),
                    mesh_dim_names=("dp_replicate", "dp_shard"),
                )
                dense_mesh = mesh
    else:
        raise AssertionError(f"unhandled distributed strategy {strategy}")

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        strategy=strategy,
        shard_size=shard_size,
        replicate_size=replicate_size,
        expert_parallel_size=expert_parallel_size,
        data_parallel_rank=data_rank,
        data_parallel_size=data_size,
        data_parallel_group=data_group,
        mesh=mesh,
        dense_mesh=dense_mesh,
        dp_shard_group=dp_shard_group,
        dp_shard_size=dp_shard_size,
        initialized_here=initialized_here,
    )


def _layer_shard_units(layer: nn.Module, spec: ModelSpec) -> list[nn.Module]:
    for child in layer.children():
        if child.__class__.__name__ == "Qwen3_5MoeSparseMoeBlock":
            return list(child.children())
    if any(is_expert_unit(child) for child in layer.children()):
        return list(layer.children())
    return [layer]


def wrap_model(
    model: nn.Module,
    context: DistributedContext,
    spec: ModelSpec,
    config: DistributedConfig,
) -> nn.Module:
    if context.world_size == 1:
        return model
    if context.strategy == DistributedStrategy.DDP:
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            output_device=context.local_rank if context.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        if config.ddp_static_graph:
            ddp_model._set_static_graph()
        return ddp_model

    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    reduce_dtype = torch.float32 if config.reduce_dtype == "float32" else torch.bfloat16
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=reduce_dtype,
        output_dtype=torch.bfloat16,
    )
    ignored = _noncontiguous_storage_groups(model)
    ep_params: set[nn.Parameter] = set()
    if context.expert_parallel_size > 1:
        ep_params = expert_parameters(model)
        ignored |= ep_params
    context.replicated_trainables = tuple(
        parameter
        for parameter in model.parameters()
        if parameter in ignored and parameter.requires_grad and parameter not in ep_params
    )
    context.ep_replicated_trainables = tuple(
        parameter
        for parameter in model.parameters()
        if parameter in ep_params and parameter.requires_grad
    )
    shard_mesh = context.dense_mesh or context.mesh
    layers = [module for module in model.modules() if module.__class__.__name__ == spec.layer_class]
    if not layers:
        raise RuntimeError(f"could not find transformer layers named {spec.layer_class}")
    # None reproduces the original hardcoded True for per-layer units.
    layer_reshard_after_forward = (
        True if config.reshard_after_forward is None else config.reshard_after_forward
    )
    shard_units: list[nn.Module] = []
    for layer in layers:
        layer_parameters = set(layer.parameters())
        layer_ignored = ignored.intersection(layer_parameters)
        units = _layer_shard_units(layer, spec) if spec.moe and context.expert_parallel_size > 1 else [layer]
        for unit in units:
            if context.expert_parallel_size > 1 and is_expert_unit(unit):
                continue
            unit_parameters = set(unit.parameters())
            unit_ignored = ignored.intersection(unit_parameters)
            fully_shard(
                unit,
                mesh=shard_mesh,
                mp_policy=policy,
                reshard_after_forward=layer_reshard_after_forward,
                ignored_params=unit_ignored or None,
            )
            shard_units.append(unit)
    fully_shard(
        model,
        mesh=shard_mesh,
        mp_policy=policy,
        # Root-only embedding/head parameters remain materialized through
        # backward. This avoids a redundant all-gather and keeps fused
        # linear-loss local views valid; decoder layers still reshard eagerly.
        reshard_after_forward=False,
        ignored_params=ignored or None,
    )
    if config.fsdp_prefetch_layers > 0:
        _install_fsdp_prefetch(shard_units, config.fsdp_prefetch_layers)
    return model


def _install_fsdp_prefetch(shard_units: list[nn.Module], depth: int) -> None:
    """Opt-in explicit forward/backward prefetch across FSDP2 shard units.

    Each unit is hinted to prefetch the all-gather of the next `depth` units
    in the forward order (and the previous `depth` units in the reverse,
    backward order), overlapping communication with compute instead of
    relying on FSDP2's implicit one-step-ahead prefetch.
    """

    for index, unit in enumerate(shard_units):
        forward_targets = shard_units[index + 1 : index + 1 + depth]
        if forward_targets:
            unit.set_modules_to_forward_prefetch(forward_targets)
        backward_targets = shard_units[max(0, index - depth) : index][::-1]
        if backward_targets:
            unit.set_modules_to_backward_prefetch(backward_targets)


def _noncontiguous_storage_groups(model: nn.Module) -> set[nn.Parameter]:
    """Return complete alias groups containing a non-contiguous parameter.

    Ignoring the complete storage group is important: PEFT's tied
    embedding/head adapters are separate Parameter views over one storage.
    Sharding one view while retaining the transposed alias would invalidate
    both the tie and optimizer ownership.
    """

    parameters = list(model.parameters())
    storage_groups: dict[tuple[str, int, int], list[nn.Parameter]] = {}
    noncontiguous_keys: set[tuple[str, int, int]] = set()
    for parameter in parameters:
        tensor = cast(torch.Tensor, parameter)
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        storage_groups.setdefault(key, []).append(parameter)
        if not tensor.is_contiguous():
            noncontiguous_keys.add(key)
    return {parameter for key in noncontiguous_keys for parameter in storage_groups[key]}


@contextmanager
def maybe_no_sync(model: nn.Module, *, synchronize: bool):
    """Disable gradient synchronization for non-final accumulation microsteps."""

    if synchronize:
        with nullcontext():
            yield
        return
    no_sync = getattr(model, "no_sync", None)
    if callable(no_sync):
        with no_sync():
            yield
        return
    set_sync = getattr(model, "set_requires_gradient_sync", None)
    if callable(set_sync):
        set_sync(False)
        try:
            yield
        finally:
            set_sync(True)
        return
    with nullcontext():
        yield


def unwrap_ddp(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model
