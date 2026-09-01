# finetune-lib

Cross-project LoRA toolkit for continual pretraining (CPT) and supervised finetuning (SFT).

Agents should read [`AGENTS.md`](AGENTS.md) for the full workflow.

## Quick start (venv)

```bash
cd /projects/data/llmteam/sidharth/toolkit/finetune-library
export HF_HOME=/projects/data/llmteam/sidharth/toolkit/grpo-library
uv sync --extra dev --extra unsloth --extra flash
source .venv/bin/activate
bash scripts/install_fla.sh   # Qwen 3.5 GDN only
```

## Quick start (Docker — recommended on GPU nodes)

```bash
export HF_HOME=/projects/data/llmteam/sidharth/toolkit/grpo-library
export FLA_SKIP_TRITON_AUTOTUNE=1

docker run --rm --gpus all --ipc=host --shm-size=16g \
  --entrypoint bash \
  -e HF_HOME -e FLA_SKIP_TRITON_AUTOTUNE \
  -v /projects/data/llmteam/sidharth/toolkit:/workspace \
  ghcr.io/psidharth567/toolkit/finetune:12.8-cu129 -lc '
    cd /workspace/finetune-library
    uv pip install -e . --no-deps -q
    scripts/production/launch-one-node.sh configs/models/qwen3-8b-cpt.yaml train
  '
```

Or use the node wrapper:

```bash
bash scripts/run-on-gpu-node.sh configs/models/gemma4-26b-a4b-it-cpt.yaml
```

## Commands

| Command | Purpose |
|---|---|
| `finetune-lib init` | Scaffold project config and directories |
| `finetune-lib prepare-data` | Tokenize and pack dataset |
| `finetune-lib train` | Run training |
| `finetune-lib benchmark` | Measure throughput (warmup + measured steps) |
| `finetune-lib merge` | Merge adapter into base weights |

## Supported models

| Key | Notes |
|---|---|
| `qwen3-8b`, `qwen3-14b`, `qwen3-32b` | DDP/HSDP; Unsloth optional on 8B/14B |
| `qwen3.5-35b-a3b` | EP4+FSDP, FA3+FLA, DeepEP; see `configs/models/qwen35-35b-a3b-cpt.yaml` |
| `gemma4-26b-a4b-it` | DDP MoE, Liger, grouped_mm |
| `gemma4-31b-it` | DDP/HSDP dense |
| `deepseek-r1-distill-llama-8b`, `olmo3-32b-think-dpo` | DDP native |

Production profiles: `configs/models/*.yaml`. Benchmark history: [`docs/validation.md`](docs/validation.md).

## Kernel stack (container)

| Kernel | Purpose |
|---|---|
| FA2 + FA3 | Flash attention (Qwen 3.5 uses FA3) |
| Liger | Fused RMSNorm / SwiGLU / GEGLU |
| FLA + TileLang | Qwen 3.5 GDN layers |
| DeepEP | MoE expert-parallel dispatch |
| grouped_mm | MoE expert matmuls (Gemma, Qwen) |

Verify: `bash scripts/verify-finetune-image.sh` inside the image.

## SLURM

```bash
export CONFIG_PATH=/path/to/config.yaml
export HF_HOME=/path/to/.cache/huggingface
sbatch scripts/slurm/train-one-node.sh
```

## Important runtime rules

- Set `HF_HOME` to a project cache under `/projects/...` (not `~/`).
- Do **not** use `torch_compile` with Liger kernels (auto-skipped).
- For MoE + `grouped_mm`, block-level compile is auto-downgraded to `loss_only`.
- Qwen 3.5: export `FLA_SKIP_TRITON_AUTOTUNE=1` on memory-tight EP runs.

## Publish

Build and push container:

```bash
bash containers/push_ghcr_finetune.sh
```
