"""SFT packing: whole examples only, isolated, never truncated."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from finetune_library.config import ExperimentConfig
from finetune_library.data import (
    IGNORE_INDEX,
    CausalCollator,
    cache_key,
    iter_prepared_rows,
    pack_whole_examples,
)
from finetune_library.packed_linear_attention import packed_boundaries
from test_data import FakeTokenizer

SFT_YAML = "configs/examples/qwen3-8b-sft-prompt-completion.yaml"


def sft_config(tmp_path: Path | None = None, rows: list[dict[str, str]] | None = None, **data: Any) -> ExperimentConfig:
    raw = yaml.safe_load(Path(SFT_YAML).read_text())
    raw["data"].pop("packing_isolation", None)
    raw["data"].update(data)
    if rows is not None:
        assert tmp_path is not None
        path = tmp_path / "sft.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        raw["data"]["path"] = str(path)
    return ExperimentConfig.model_validate(raw)


def example(index: int, length: int) -> dict[str, list[int]]:
    ids = [1000 * (index + 1) + offset for offset in range(length)]
    labels = [IGNORE_INDEX] * (length // 2) + ids[length // 2 :]
    return {"input_ids": ids, "labels": labels}


def split_window(window: dict[str, list[int]]) -> list[tuple[list[int], list[int]]]:
    """Recover (input_ids, labels) per example from position_ids restarts."""
    pieces: list[tuple[list[int], list[int]]] = []
    for token, label, position in zip(window["input_ids"], window["labels"], window["position_ids"]):
        if position == 0:
            pieces.append(([], []))
        pieces[-1][0].append(token)
        pieces[-1][1].append(label)
    return pieces


def test_whole_example_packing_keeps_every_example_intact() -> None:
    rng = random.Random(0)
    rows = [example(index, rng.randint(1, 64)) for index in range(300)]
    windows = pack_whole_examples(rows, max_length=64)

    recovered = []
    for window in windows:
        assert len(window["input_ids"]) <= 64
        assert len(window["input_ids"]) == len(window["labels"]) == len(window["position_ids"])
        for ids, labels in split_window(window):
            recovered.append((tuple(ids), tuple(labels)))
            # positions restart per example and run contiguously
        pos = window["position_ids"]
        assert all(b == a + 1 or b == 0 for a, b in zip(pos, pos[1:]))
    expected = sorted((tuple(r["input_ids"]), tuple(r["labels"])) for r in rows)
    assert sorted(recovered) == expected  # each example exactly once, unsplit, labels intact
    total = sum(len(r["input_ids"]) for r in rows)
    assert len(windows) <= total // 64 + 12  # best-fit decreasing packs tightly


def test_whole_example_packing_is_deterministic_and_never_splits() -> None:
    rows = [example(0, 5), example(1, 4), example(2, 3), example(3, 6)]
    first = pack_whole_examples(rows, max_length=8)
    assert first == pack_whole_examples(rows, max_length=8)
    # 6 and 5 cannot share a window of 8; 5 would be split by stream packing.
    lengths = sorted(sorted(len(ids) for ids, _ in split_window(w)) for w in first)
    assert lengths == [[3, 5], [4], [6]]
    assert all(len(w["input_ids"]) <= 8 for w in first)


def test_example_exactly_max_length_gets_its_own_window() -> None:
    windows = pack_whole_examples([example(0, 8), example(1, 2)], max_length=8)
    assert [len(w["input_ids"]) for w in windows] == [8, 2]


def test_packing_rejects_rows_that_do_not_fit() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        pack_whole_examples([example(0, 9)], max_length=8)


def _pc_rows(lengths: list[int]) -> list[dict[str, str]]:
    # FakeTokenizer: one token per character; eos appended by add_eos.
    return [{"prompt": "p" * (n // 2), "completion": "c" * (n - n // 2 - 1)} for n in lengths]


@pytest.mark.parametrize("packing", [True, False])
def test_sft_overlength_examples_are_dropped_never_truncated(tmp_path: Path, packing: bool) -> None:
    config = sft_config(tmp_path, _pc_rows([100, 700, 300, 513, 512]), packing=packing)
    assert config.training.max_seq_length == 512
    stats: dict[str, Any] = {}
    rows = list(iter_prepared_rows(config, FakeTokenizer(), stats=stats))
    if packing:
        examples = [piece for row in rows for piece in split_window(row)]
    else:
        examples = [(row["input_ids"], row["labels"]) for row in rows]
    assert sorted(len(ids) for ids, _ in examples) == [100, 300, 512]
    assert stats == {"raw_records": 5, "dropped_overlength": 2, "longest_dropped_tokens": 700, "examples": 3}
    for ids, labels in examples:
        assert labels[-1] == FakeTokenizer.eos_token_id  # target end survives


def test_sft_overlength_error_policy(tmp_path: Path) -> None:
    config = sft_config(tmp_path, _pc_rows([100, 700]), sft_overlength="error")
    with pytest.raises(ValueError, match="never truncated"):
        list(iter_prepared_rows(config, FakeTokenizer()))


def test_sft_packing_resolves_to_isolated() -> None:
    config = sft_config(packing=True)
    assert config.data.packing_isolation == "attention"
    assert config.packing_isolation() == "attention"
    assert config.runtime.attention == "sdpa"  # sdpa isolation is allowed now


def test_sft_packing_rejects_explicit_non_isolation() -> None:
    with pytest.raises(ValueError, match="always isolates"):
        sft_config(packing=True, packing_isolation="none")


def test_sft_unpacked_stays_unisolated_and_model_copy_cannot_bypass() -> None:
    unpacked = sft_config(packing=False)
    assert unpacked.data.packing_isolation == "none"
    assert unpacked.packing_isolation() == "none"
    bypass = unpacked.model_copy(update={"data": unpacked.data.model_copy(update={"packing": True})})
    assert bypass.packing_isolation() == "attention"


def test_sft_rejects_chunking_and_flex_isolation() -> None:
    with pytest.raises(ValueError, match="CPT-only"):
        sft_config(packing=False, chunk_long_examples=True)
    raw = yaml.safe_load(Path(SFT_YAML).read_text())
    raw["data"]["packing"] = True
    raw["runtime"]["attention"] = "flex_attention"
    with pytest.raises(ValueError, match="flex_attention"):
        ExperimentConfig.model_validate(raw)


def test_cpt_cache_key_ignores_sft_only_field() -> None:
    config = ExperimentConfig.from_yaml("configs/models/qwen3-8b-cpt.yaml")
    changed = config.model_copy(update={"data": config.data.model_copy(update={"sft_overlength": "error"})})
    assert cache_key(config, FakeTokenizer()) == cache_key(changed, FakeTokenizer())
    sft = sft_config(packing=True)
    sft_error = sft.model_copy(update={"data": sft.data.model_copy(update={"sft_overlength": "error"})})
    assert cache_key(sft, FakeTokenizer()) != cache_key(sft_error, FakeTokenizer())


def test_collator_flattens_packed_windows_and_masks_example_starts() -> None:
    windows = pack_whole_examples([example(0, 3), example(1, 2), example(2, 4)], max_length=5)
    batch = CausalCollator(FakeTokenizer(), max_length=5, packing_isolation="attention")(windows)
    assert set(batch) == {"input_ids", "labels", "position_ids"}  # no attention_mask: packed mask is derived
    pos = batch["position_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    starts = [i for i, p in enumerate(pos) if p == 0]
    assert len(starts) == 3
    assert all(labels[i] == IGNORE_INDEX for i in starts)  # never predict across examples
    assert batch["input_ids"].shape[1] == 9  # 3 + 2 + 4, no padding


def test_packed_boundaries() -> None:
    pos = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]])
    cu_seqlens, seq_idx = packed_boundaries(pos)
    assert cu_seqlens.tolist() == [0, 3, 5, 9]
    assert seq_idx.tolist() == [[0, 0, 0, 1, 1, 2, 2, 2, 2]]
    assert seq_idx.dtype == torch.int32
    assert packed_boundaries(torch.arange(6)[None]) is None
    assert packed_boundaries(torch.arange(6).repeat(2, 1)) is None
    assert packed_boundaries(None) is None
    with pytest.raises(ValueError, match="batch size 1"):
        packed_boundaries(torch.tensor([[0, 1, 0, 1], [0, 1, 2, 3]]))
