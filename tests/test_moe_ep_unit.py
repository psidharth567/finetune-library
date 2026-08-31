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
