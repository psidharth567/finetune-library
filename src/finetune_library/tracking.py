from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from finetune_library.config import ExperimentConfig, LoggingConfig


class TrackingSession:
    def __init__(self, config: ExperimentConfig, *, rank: int) -> None:
        self._active = False
        self._run = None
        self._watch_model = config.logging.wandb.watch_model
        self._log_every = (
            config.logging.wandb.log_every_steps or config.training.log_every_steps
        )
        if rank != 0 or not _wandb_should_run(config.logging):
            return
        import wandb

        api_key = config.logging.wandb.api_key or os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=True)
        output_name = Path(config.training.output_dir).name
        self._run = wandb.init(
            project=config.logging.wandb.project or "finetune-lib",
            entity=config.logging.wandb.entity,
            name=config.logging.wandb.run_name or output_name,
            group=config.logging.wandb.group,
            tags=list(config.logging.wandb.tags),
            config=config.to_json_dict(),
            dir=str(Path(config.training.output_dir)),
        )
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def watch(self, model: Any) -> None:
        if not self._active or not self._watch_model:
            return
        import wandb

        wandb.watch(model, log="gradients", log_freq=self._log_every)

    def log_step(self, step: int, metrics: dict[str, Any]) -> None:
        if not self._active or step % self._log_every != 0:
            return
        import wandb

        wandb.log(metrics, step=step)

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        if not self._active:
            return
        import wandb

        if summary:
            wandb.summary.update(summary)
        wandb.finish()
        self._active = False


def _wandb_should_run(logging_config: LoggingConfig) -> bool:
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return False
    has_key = bool(logging_config.wandb.api_key or os.environ.get("WANDB_API_KEY"))
    if logging_config.wandb.enabled:
        return has_key
    return has_key
