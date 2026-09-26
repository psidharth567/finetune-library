"""CPU-only unit tests for initialize_distributed's data_rank/data_size.

These pin down the fix for the throughput bug: every rank must consume
distinct data under DDP, FSDP, and HSDP (data_rank == rank, data_size ==
world_size, data_parallel_group == the full world), since FSDP2's own
gradient averaging over `dense_mesh` -- not a separate DistributedSampler
partition -- is what makes the update mathematically match DDP's mean over
the same global batch. Before the fix, FSDP pinned data_rank/data_size to
(0, 1) and HSDP pinned data_size to `replicate_size`, so shard-group ranks
silently trained on the same batch (see distributed.py's initialize_distributed
docstring-style comment for the full argument).

Runs multiple gloo/CPU processes via torch.multiprocessing so DeviceMesh /
process-group construction is exercised for real, without requiring a GPU.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.multiprocessing as mp

from finetune_library.config import DistributedConfig, DistributedStrategy
from finetune_library.distributed import initialize_distributed
from finetune_library.registry import ModelSpec

_SPEC = ModelSpec(
    key="test-dense",
    repo_id="test/dense",
    revision="0" * 40,
    architecture="dense",
    layer_class="TestDecoderLayer",
    preferred_strategy=DistributedStrategy.DDP,
)

_MOE_SPEC = ModelSpec(
    key="test-moe",
    repo_id="test/moe",
    revision="0" * 40,
    architecture="moe",
    layer_class="TestMoeDecoderLayer",
    preferred_strategy=DistributedStrategy.FSDP,
    moe=True,
)


def _worker(
    rank: int,
    world_size: int,
    strategy: str,
    kwargs: dict,
    result_queue: "mp.Queue",
    spec: ModelSpec = _SPEC,
    port: int = 29511,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # Loopback only: inside `docker build` gloo otherwise picks the bridge NIC.
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.cuda.is_available = lambda: False  # force the gloo/CPU path
    config = DistributedConfig(strategy=DistributedStrategy(strategy), **kwargs)
    context = initialize_distributed(config, spec)
    try:
        result_queue.put(
            (
                rank,
                context.data_parallel_rank,
                context.data_parallel_size,
                context.world_size,
                context.data_parallel_group is not None,
                context.dp_shard_size,
                tuple(context.dense_mesh.mesh_dim_names) if context.dense_mesh is not None else (),
            )
        )
        # Keep every rank alive until all have finished building groups.
        torch.distributed.barrier()
    finally:
        context.close()


def _run(
    world_size: int, strategy: str, spec: ModelSpec = _SPEC, **kwargs: object
) -> list[tuple]:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(rank, world_size, strategy, kwargs, queue, spec, port))
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    results = [queue.get(timeout=60) for _ in procs]
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    return sorted(results)


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_every_rank_gets_distinct_data_no_ep(strategy: str) -> None:
    world_size = 4
    results = _run(world_size, strategy)
    for rank, data_rank, data_size, reported_world_size, has_group, _, _ in results:
        assert data_rank == rank, f"rank {rank}: expected data_rank==rank, got {data_rank}"
        assert data_size == world_size, f"rank {rank}: expected data_size==world_size, got {data_size}"
        assert reported_world_size == world_size
        assert has_group


def test_hsdp_every_rank_gets_distinct_data() -> None:
    world_size = 4
    results = _run(world_size, "hsdp", shard_size=2, replicate_size=2)
    for rank, data_rank, data_size, reported_world_size, has_group, _, _ in results:
        # Before the fix, HSDP set data_size=replicate_size (2), so the two
        # ranks sharing a shard group would see the same batch. The fix
        # makes every rank distinct, matching DDP over the same 4-way batch.
        assert data_rank == rank
        assert data_size == world_size
        assert has_group


def test_fsdp_ep_mesh_is_2d_replicate_shard_and_every_rank_distinct() -> None:
    """FSDP + expert_parallel_size>1: dense params must be replicated over ep
    and sharded over dp_shard only, so `dense_mesh` must be a 2D (ep,
    dp_shard) mesh that fully_shard reads as HSDP-style (dim0=replicate,
    dim1=shard) -- this also regression-tests that DeviceMesh is built with
    dims already in (ep, dp_shard) order, since torch 2.11's
    DeviceMesh.__getitem__ cannot reorder dims after the fact (it raises
    "Mesh dim indices should be in ascending order"), which is exactly the
    bug this test would have caught before the fix landed.
    """

    world_size = 8
    results = _run(world_size, "fsdp", spec=_MOE_SPEC, expert_parallel_size=4)
    for rank, data_rank, data_size, reported_world_size, has_group, dp_shard_size, mesh_dims in results:
        assert data_rank == rank
        assert data_size == world_size
        assert has_group
        assert dp_shard_size == 2  # world_size // expert_parallel_size
        assert mesh_dims == ("ep", "dp_shard")


def test_hsdp_ep_mesh_and_expert_replication_group() -> None:
    """HSDP + EP: dense params replicate over (dp_replicate, ep), shard over
    dp_shard; the expert-adapter replication group must span the full
    (dp_replicate, dp_shard) product for a fixed ep coordinate (previously
    this only covered dp_shard, under-averaging expert-adapter gradients
    whenever dp_replicate ranks hold different data, which is now always the
    case). world_size=8, replicate_size=2, expert_parallel_size=2 -> shard_size=2.
    """

    world_size = 8
    results = _run(
        world_size,
        "hsdp",
        spec=_MOE_SPEC,
        expert_parallel_size=2,
        shard_size=2,
        replicate_size=2,
    )
    for rank, data_rank, data_size, reported_world_size, has_group, dp_shard_size, mesh_dims in results:
        assert data_rank == rank
        assert data_size == world_size
        assert has_group
        # replicate_size * shard_size: the full non-expert-dim group.
        assert dp_shard_size == 4
        assert mesh_dims == ("dp_replicate_ep", "dp_shard")
