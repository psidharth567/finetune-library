from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_data import FakeTokenizer
from test_sft_packing import sft_config

from finetune_library.data import prepare_to_disk

ROWS = [{"prompt": "question ", "completion": f"answer {index}"} for index in range(6)]


def test_prepared_cache_rejects_a_source_file_rewritten_in_place(tmp_path: Path) -> None:
    config = sft_config(tmp_path, ROWS, packing=False, cache_dir=str(tmp_path / "cache"))
    first = prepare_to_disk(config, FakeTokenizer())
    metadata = json.loads((first / "finetune_library_metadata.json").read_text())
    assert metadata["source"]["size"] == Path(config.data.path).stat().st_size

    assert prepare_to_disk(config, FakeTokenizer()) == first  # unchanged file: reused

    source = Path(config.data.path)
    source.write_text(source.read_text() + json.dumps(ROWS[0]) + "\n")
    with pytest.raises(ValueError, match="changed since it was prepared"):
        prepare_to_disk(config, FakeTokenizer())


def test_prepared_cache_without_fingerprint_is_reused_as_before(tmp_path: Path) -> None:
    config = sft_config(tmp_path, ROWS, packing=False, cache_dir=str(tmp_path / "cache"))
    first = prepare_to_disk(config, FakeTokenizer())
    metadata_path = first / "finetune_library_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["source"]  # a cache written before fingerprints existed
    metadata_path.write_text(json.dumps(metadata))
    source = Path(config.data.path)
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert prepare_to_disk(config, FakeTokenizer()) == first
