from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from finetune_library.config import DataFormat, ExperimentConfig, RuntimeBackend, Task
from finetune_library.init_project import scaffold_project_config


def _config_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to a strict experiment YAML")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finetune-lib",
        description="Cross-project CPT, SFT, and DPO LoRA trainer",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    _config_parser(subcommands.add_parser("train", help="Run CPT, SFT, or DPO"))

    prepare = subcommands.add_parser("prepare-data", help="Tokenize and pack a dataset")
    _config_parser(prepare)
    prepare.add_argument("--output", help="Prepared Hugging Face dataset directory")

    benchmark = subcommands.add_parser(
        "benchmark", help="Run the configured warm-up and measured steps"
    )
    _config_parser(benchmark)

    init = subcommands.add_parser(
        "init", help="Scaffold a project-local finetune config and output directories"
    )
    init.add_argument("--project-root", required=True, type=Path)
    init.add_argument("--run-name", required=True)
    init.add_argument("--task", required=True, choices=[task.value for task in Task])
    init.add_argument("--model", required=True, help="Registry key or HF repo id")
    init.add_argument("--data", required=True, type=Path, help="Path to training data")
    init.add_argument(
        "--format",
        choices=[fmt.value for fmt in DataFormat],
        help="Data format; defaults to text for CPT and messages for SFT",
    )
    init.add_argument("--max-steps", type=int, default=500)
    init.add_argument("--max-seq-length", type=int, default=2048)
    init.add_argument("--wandb-project", help="Default wandb project name")

    merge = subcommands.add_parser("merge", help="Merge a PEFT adapter into its base model")
    merge.add_argument("--checkpoint", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument(
        "--device",
        default="cpu",
        help="Merge device; CPU is the memory-safe default for 31B/32B models",
    )
    merge.add_argument("--max-shard-size", default="5GB")
    merge.add_argument("--validation-text", default="The quick brown fox")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        config = ExperimentConfig.from_yaml(args.config)
        if config.runtime.backend == RuntimeBackend.UNSLOTH:
            importlib.import_module("unsloth")
        from finetune_library.trainer import run_training

        run_training(config)
        return 0
    if args.command == "benchmark":
        config = ExperimentConfig.from_yaml(args.config)
        if config.runtime.backend == RuntimeBackend.UNSLOTH:
            importlib.import_module("unsloth")
        from finetune_library.trainer import run_training

        report = run_training(config, benchmark=True)
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare-data":
        from transformers import AutoTokenizer

        from finetune_library.data import prepare_to_disk
        from finetune_library.registry import resolve_model

        config = ExperimentConfig.from_yaml(args.config)
        spec = resolve_model(config.model.name)
        tokenizer = AutoTokenizer.from_pretrained(
            spec.repo_id,
            revision=config.model.revision or spec.revision,
            cache_dir=config.model.cache_dir,
            trust_remote_code=config.model.trust_remote_code,
        )
        from finetune_library.runtime import apply_chat_template_override

        apply_chat_template_override(tokenizer, config)
        output = prepare_to_disk(config, tokenizer, args.output)
        print(output)
        return 0
    if args.command == "init":
        data_format = DataFormat(args.format) if args.format else None
        config_path = scaffold_project_config(
            project_root=args.project_root,
            run_name=args.run_name,
            task=Task(args.task),
            model_name=args.model,
            data_path=args.data,
            data_format=data_format,
            max_steps=args.max_steps,
            max_seq_length=args.max_seq_length,
            wandb_project=args.wandb_project,
        )
        print(config_path)
        return 0
    if args.command == "merge":
        from finetune_library.merge import merge_checkpoint

        report = merge_checkpoint(
            args.checkpoint,
            args.output,
            device=args.device,
            validation_text=args.validation_text,
            max_shard_size=args.max_shard_size,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
