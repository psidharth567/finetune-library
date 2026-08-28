from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from finetune_library.config import DistributedStrategy, ExperimentConfig, RuntimeBackend
from finetune_library.registry import resolve_model, unique_models
from finetune_library.runtime import _cached_snapshot


def test_all_model_profiles_are_strict_and_registered() -> None:
    profiles = sorted(Path("configs/models").glob("*.yaml"))
    assert len(profiles) == 7
    for profile in profiles:
        config = ExperimentConfig.from_yaml(profile)
        spec = resolve_model(config.model.name)
        assert config.model.revision is None
        assert len(spec.revision) == 40
        assert spec.chat_template == "tokenizer"
        assert {candidate.strategy.value for candidate in spec.distributed_candidates} == {
            "ddp",
            "hsdp",
            "fsdp",
        }
        forbidden = ("Qwen" + "3.5", "Nemo" + "tron")
        assert all(name not in spec.repo_id for name in forbidden)


def test_registry_contains_exact_production_set() -> None:
    assert {spec.repo_id for spec in unique_models()} == {
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "allenai/Olmo-3-32B-Think-DPO",
        "google/gemma-4-26B-A4B-it",
        "google/gemma-4-31B-it",
    }


def test_registry_auto_strategy_matches_qualified_profiles() -> None:
    assert resolve_model("qwen3-8b").preferred_strategy == DistributedStrategy.DDP
    assert (
        resolve_model("gemma4-26b-a4b-it").preferred_strategy
        == DistributedStrategy.DDP
    )
    for name in ("qwen3-32b", "olmo3-32b-think-dpo", "gemma4-31b-it"):
        spec = resolve_model(name)
        assert spec.preferred_strategy == DistributedStrategy.DDP
        assert (spec.shard_size, spec.replicate_size) == (1, 8)


def test_dense_32b_profiles_use_qualified_ddp_and_fused_loss() -> None:
    profiles = {
        "qwen3-32b-cpt.yaml": RuntimeBackend.NATIVE,
        "olmo3-32b-think-dpo-cpt.yaml": RuntimeBackend.NATIVE,
        "gemma4-31b-it-cpt.yaml": RuntimeBackend.NATIVE,
    }
    for filename, backend in profiles.items():
        config = ExperimentConfig.from_yaml(Path("configs/models") / filename)
        assert config.distributed.strategy == DistributedStrategy.DDP
        assert config.runtime.backend == backend
        assert config.runtime.loss == "fused_linear_cross_entropy"
        assert config.runtime.attention == "sdpa"
        assert config.optimizer.name == "adamw"
        assert config.optimizer.fused


def test_unknown_config_keys_fail() -> None:
    valid = ExperimentConfig.from_yaml("configs/models/qwen3-8b-cpt.yaml")
    raw = valid.model_dump(mode="json")
    raw["training"]["max_seq_lenght"] = 2048
    with pytest.raises(ValidationError, match="max_seq_lenght"):
        ExperimentConfig.model_validate(raw)


def test_task_and_data_format_must_match() -> None:
    valid = ExperimentConfig.from_yaml("configs/models/qwen3-8b-cpt.yaml")
    raw = valid.model_dump(mode="json")
    raw["data"]["format"] = "messages"
    with pytest.raises(ValidationError, match="invalid for task"):
        ExperimentConfig.model_validate(raw)


def test_cached_snapshot_keeps_snapshot_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "models--owner--model" / "snapshots" / ("a" * 40)
    blob = tmp_path / "models--owner--model" / "blobs" / "config-blob"
    snapshot.mkdir(parents=True)
    blob.parent.mkdir(parents=True)
    blob.write_text('{"model_type": "llama"}\n', encoding="utf-8")
    config_link = snapshot / "config.json"
    config_link.symlink_to(Path("../../blobs/config-blob"))
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda *_args, **_kwargs: str(config_link),
    )

    assert _cached_snapshot("owner/model", "a" * 40, str(tmp_path)) == str(
        snapshot.absolute()
    )
