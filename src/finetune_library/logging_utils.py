from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from finetune_library.config import LoggingConfig


def setup_training_logger(
    output_dir: Path,
    config: LoggingConfig,
    *,
    rank: int,
) -> logging.Logger:
    logger = logging.getLogger("finetune_library")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, config.level))
    logger.propagate = False

    if rank != 0:
        logger.addHandler(logging.NullHandler())
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if config.console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)
    if config.file:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class EventLogger:
    def __init__(
        self,
        output_dir: Path,
        config: LoggingConfig,
        *,
        rank: int,
    ) -> None:
        self._enabled = rank == 0 and config.events_jsonl
        self._path = output_dir / "events.jsonl"
        if self._enabled:
            output_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        if not self._enabled:
            return
        payload = {
            "event": event,
            "timestamp": datetime.now(UTC).isoformat(),
            **fields,
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
