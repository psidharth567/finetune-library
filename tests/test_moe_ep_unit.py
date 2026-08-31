from __future__ import annotations

import pytest

from finetune_library.moe_parallel import build_moe_layout


def test_build_moe_layout_offsets() -> None:
    layout = build_moe_layout(num_experts=128, expert_parallel_size=1, mesh=None)
    assert layout.local_num_experts == 128
    assert layout.global_expert_offset == 0
    assert not layout.enabled

    layout_ep = build_moe_layout(num_experts=128, expert_parallel_size=8, mesh=None)
    assert layout_ep.local_num_experts == 16
    assert layout_ep.enabled


def test_qwen_expert_count_divisible() -> None:
    from finetune_library.registry import resolve_model

    qwen = resolve_model("qwen3.5-35b-a3b")
    gemma = resolve_model("gemma4-26b-a4b-it")
    assert qwen.num_experts == 256
    assert gemma.num_experts == 128
    assert 256 % 4 == 0
    assert 128 % 4 == 0


def test_all_to_all_exchange_buffer_shapes() -> None:
    import torch

    hidden = torch.randn(8, 16)
    trailing = tuple(hidden.shape[1:])
    recv_sizes = [3, 5]
    buffers = [hidden.new_zeros((size, *trailing)) for size in recv_sizes]
    assert buffers[0].shape == (3, 16)
    assert buffers[1].shape == (5, 16)
    split = list(torch.split(hidden, [4, 4], dim=0))
    assert split[0].shape == (4, 16)
