"""Manual long-duration benchmark using a checked-in qualified profile."""

from __future__ import annotations

import argparse
import importlib

from finetune_library.config import ExperimentConfig, RuntimeBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--measured-steps", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    if config.runtime.backend == RuntimeBackend.UNSLOTH:
        importlib.import_module("unsloth")
    from finetune_library.trainer import run_training

    config = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "output_dir": args.output_dir,
                    "benchmark_warmup_steps": 10,
                    "benchmark_steps": args.measured_steps,
                    "log_every_steps": 100,
                }
            )
        }
    )
    run_training(config, benchmark=True)


if __name__ == "__main__":
    main()
