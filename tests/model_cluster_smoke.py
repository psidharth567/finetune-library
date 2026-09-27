"""One-step real-model smoke for the H100 validation nodes.

This is intentionally a manual torchrun test because it loads production-size
checkpoints. It accepts any checked-in model profile and overrides only the
dataset-independent smoke limits.
"""

from __future__ import annotations

import argparse
import importlib

from finetune_library.config import ExperimentConfig, RuntimeBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HF cache_dir override (default: standard $HF_HOME/hub cache)",
    )
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the configured 10 warm-up and 50 measured steps",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Save a resumable step checkpoint and final PEFT adapter",
    )
    parser.add_argument("--resume-from")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    if config.runtime.backend == RuntimeBackend.UNSLOTH:
        # Unsloth must patch Transformers before the trainer imports model
        # classes. This mirrors the production CLI import order.
        importlib.import_module("unsloth")
    from finetune_library.trainer import run_training

    config = config.model_copy(
        update={
            "model": config.model.model_copy(update={"cache_dir": args.cache_dir}),
            "data": config.data.model_copy(update={"drop_remainder": False}),
            "training": config.training.model_copy(
                update={
                    "output_dir": (
                        args.output_dir
                        or f"outputs/validation-{config.model.name}"
                    ),
                    "max_seq_length": args.sequence_length,
                    "per_device_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "max_steps": args.max_steps,
                    "dataloader_workers": 0,
                    "log_every_steps": 1,
                }
            ),
            "checkpoint": config.checkpoint.model_copy(
                update={
                    "save_every_steps": 1 if args.checkpoint else 0,
                    "save_final": args.checkpoint,
                    "resume_from": args.resume_from,
                }
            ),
        }
    )
    run_training(config, benchmark=args.benchmark)


if __name__ == "__main__":
    main()
