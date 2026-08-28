# finetune-lib

Cross-project LoRA toolkit for continual pretraining (CPT) and supervised finetuning (SFT).

Agents should read [`AGENTS.md`](AGENTS.md) for the full workflow and request format.

## Quick start

```bash
cd /projects/data/llmteam/sidharth/toolkit/finetune-library
export HF_HOME="$(pwd)/.cache/huggingface"
uv sync --extra dev --extra unsloth --extra flash
source .venv/bin/activate
```

## Scaffold a project run

```bash
finetune-lib init \
  --project-root /path/to/my-project \
  --run-name qwen3-sft-v1 \
  --task sft \
  --model qwen3-8b \
  --data /path/to/my-project/data/train.jsonl \
  --format messages

export HF_HOME=/path/to/my-project/.cache/huggingface
finetune-lib prepare-data --config /path/to/my-project/finetune/configs/qwen3-sft-v1.yaml
scripts/production/launch-one-node.sh /path/to/my-project/finetune/configs/qwen3-sft-v1.yaml
```

## Commands

| Command | Purpose |
|---|---|
| `finetune-lib init` | Scaffold project config and directories |
| `finetune-lib prepare-data` | Tokenize and pack dataset |
| `finetune-lib train` | Run training |
| `finetune-lib benchmark` | Measure throughput |
| `finetune-lib merge` | Merge adapter into base weights |

## SLURM

```bash
export CONFIG_PATH=/path/to/project/finetune/configs/my-run.yaml
export PROJECT_ROOT=/path/to/project
sbatch scripts/slurm/train-one-node.sh
```

## Logging and Wandb

Every run writes `train.log`, `metrics.jsonl`, and `events.jsonl` under the output directory.
Wandb is off by default and activates when `WANDB_API_KEY` is set (env or YAML).
Install with `uv sync --extra tracking`.

## Supported models

Qwen3-8B/14B/32B, DeepSeek-R1-Distill-Llama-8B, OLMo-3-32B-Think-DPO, Gemma-4-26B-A4B-it,
Gemma-4-31B-it. See `configs/models/` for qualified profiles.

Benchmark history: [`docs/validation.md`](docs/validation.md).
