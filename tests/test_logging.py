from __future__ import annotations

import os

from finetune_library.config import ExperimentConfig
from finetune_library.tracking import _wandb_should_run


def test_logging_defaults_are_backward_compatible() -> None:
    config = ExperimentConfig.from_yaml("configs/models/qwen3-8b-cpt.yaml")
    assert config.logging.level == "INFO"
    assert config.logging.file is True
    assert config.logging.metrics_jsonl is True
    assert config.logging.wandb.enabled is False


def test_wandb_auto_enables_with_env_key(monkeypatch) -> None:
    config = ExperimentConfig.from_yaml("configs/models/qwen3-8b-cpt.yaml")
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    assert _wandb_should_run(config.logging) is True


def test_wandb_stays_off_without_key(monkeypatch) -> None:
    config = ExperimentConfig.from_yaml("configs/models/qwen3-8b-cpt.yaml")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert _wandb_should_run(config.logging) is False
