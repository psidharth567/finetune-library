"""One-step real Gemma-4-26B HSDP smoke for the H100 validation nodes."""

from __future__ import annotations

from finetune_library.config import ExperimentConfig
from finetune_library.trainer import run_training


def main() -> None:
    config = ExperimentConfig.from_yaml("configs/models/gemma4-26b-a4b-it-cpt.yaml")
    config = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "output_dir": "outputs/validation-gemma4-26b-hsdp",
                    "max_seq_length": 64,
                    "per_device_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "max_steps": 1,
                    "dataloader_workers": 0,
                    "log_every_steps": 1,
                }
            ),
            "checkpoint": config.checkpoint.model_copy(
                update={"save_every_steps": 0, "save_final": False}
            ),
        }
    )
    run_training(config)


if __name__ == "__main__":
    main()
