# finetune-lib — Agent instructions

Use this toolkit to run LoRA continual pretraining (CPT) or supervised finetuning (SFT)
for any project. The library lives at a fixed path; each project keeps its own data,
configs, and outputs under `<project>/finetune/`.

## Toolkit location

```bash
export FINETUNE_LIBRARY_ROOT=/projects/data/llmteam/sidharth/toolkit/finetune-library
export HF_HOME="${HF_HOME:-<project-root>/.cache/huggingface}"
```

`/home/` is out of space. Always set `HF_HOME` to the consuming project's cache directory.

## Environment setup

```bash
cd "${FINETUNE_LIBRARY_ROOT}"
uv sync --extra dev --extra unsloth --extra flash
# Optional experiment tracking:
uv sync --extra tracking
source .venv/bin/activate
```

Stack: Python 3.12, PyTorch 2.11+cu129, Transformers 5.5, PEFT 0.19.

## Agent request format

When the user asks to finetune a model, translate their request into this block first,
then execute the workflow below.

```yaml
finetune_request:
  project_root: /absolute/path/to/project
  run_name: descriptive-run-name
  task: cpt | sft
  model: <registry-key>            # e.g. qwen3-8b, gemma4-31b-it
  data:
    path: /absolute/path/to/data.jsonl
    format: text | messages | alpaca | prompt_completion | tokenized
  training:
    max_steps: 500
    max_seq_length: 2048
  logging:
    wandb:
      enabled: false               # auto-enables when WANDB_API_KEY is set
      project: my-project
  launch:
    gpus: 8
    nodes: 1
    slurm: false
```

### Data format guide

| Format | Task | Row shape | Loss on |
|---|---|---|---|
| `text` | CPT | `{"text": "..."}` | all tokens |
| `messages` | SFT | `{"messages": [{"role":"user","content":"..."}, ...]}` | assistant tokens only |
| `alpaca` | SFT | `{"instruction":"...", "input":"...", "output":"..."}` | `output` only |
| `prompt_completion` | SFT | `{"prompt":"...", "completion":"..."}` | `completion` only |
| `tokenized` | CPT/SFT | `{"input_ids": [...], "labels": [...]}` | labels not equal to -100 |

Default: use `messages` for chat-model SFT, `text` for CPT.

## Supported models

| Registry key | HF repo | Backend |
|---|---|---|
| `qwen3-8b` | Qwen/Qwen3-8B | unsloth |
| `qwen3-14b` | Qwen/Qwen3-14B | unsloth |
| `qwen3-32b` | Qwen/Qwen3-32B | native |
| `deepseek-r1-distill-llama-8b` | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | native |
| `olmo3-32b-think-dpo` | allenai/Olmo-3-32B-Think-DPO | native |
| `gemma4-26b-a4b-it` | google/gemma-4-26B-A4B-it | native (grouped MoE) |
| `gemma4-31b-it` | google/gemma-4-31B-it | native |

## Agent workflow

1. Parse the user request into `finetune_request`.
2. Verify `data.path` exists and matches `task` + `format`.
3. Scaffold config:
   ```bash
   finetune-lib init \
     --project-root <project_root> \
     --run-name <run_name> \
     --task <cpt|sft> \
     --model <model> \
     --data <data.path> \
     --format <format> \
     --max-steps <n>
   ```
   This creates `<project_root>/finetune/configs/<run_name>.yaml` and output dirs.
4. Set caches:
   ```bash
   export HF_HOME=<project_root>/.cache/huggingface
   ```
5. Prepare data once:
   ```bash
   finetune-lib prepare-data --config <project_root>/finetune/configs/<run_name>.yaml
   ```
6. Launch training (one node, 8 GPUs):
   ```bash
   "${FINETUNE_LIBRARY_ROOT}/scripts/production/launch-one-node.sh" \
     <project_root>/finetune/configs/<run_name>.yaml
   ```
7. Monitor:
   - `<output_dir>/train.log` — human-readable log
   - `<output_dir>/metrics.jsonl` — per-step JSON metrics
   - `<output_dir>/events.jsonl` — lifecycle events
8. On success, merge adapter:
   ```bash
   finetune-lib merge \
     --checkpoint <output_dir>/final \
     --output <output_dir>/merged
   ```
9. Report to user: final loss, peak memory, adapter path, merged model path.

## SLURM launch

```bash
export CONFIG_PATH=/path/to/project/finetune/configs/my-run.yaml
export PROJECT_ROOT=/path/to/project
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
sbatch "${FINETUNE_LIBRARY_ROOT}/scripts/slurm/train-one-node.sh"
```

Override `COMMAND=benchmark` for throughput measurement.

## Wandb

Off by default. Automatically enabled when `WANDB_API_KEY` is set in the environment
or `logging.wandb.api_key` is set in the YAML. Explicit opt-in: `logging.wandb.enabled: true`.

```yaml
logging:
  wandb:
    enabled: true
    api_key: null          # or set WANDB_API_KEY env
    project: my-project
    run_name: my-run-v1
```

Install tracking support: `uv sync --extra tracking`.

## Output artifacts

Every run writes to `training.output_dir`:

| File | Purpose |
|---|---|
| `resolved_config.json` | Exact config used |
| `lora_audit.json` | LoRA target coverage |
| `train.log` | Structured text log |
| `metrics.jsonl` | Per-step metrics |
| `events.jsonl` | run_start / step / checkpoint / run_end |
| `summary.json` | Final run summary |
| `final/` | BF16 PEFT adapter checkpoint |

## Resume

Set `checkpoint.resume_from` to a prior checkpoint directory, keeping the same world size.

## Common user phrases → actions

| User says | Action |
|---|---|
| "Finetune Qwen3-8B on my SFT data" | `task: sft`, `model: qwen3-8b`, `format: messages` |
| "CPT Gemma-4-31B on corpus.jsonl" | `task: cpt`, `model: gemma4-31b-it`, `format: text` |
| "Train with wandb" | set `WANDB_API_KEY` or yaml `logging.wandb` |
| "Benchmark throughput" | launch with `benchmark` command |

## Do not

- Do not put project outputs inside `toolkit/finetune-library/outputs/` for production runs.
- Do not use `/home/llmteam/.cache/huggingface` for model downloads.
- Do not change pinned model revisions unless the user explicitly requests it.
