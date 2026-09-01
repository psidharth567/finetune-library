# finetune-lib — Agent instructions

LoRA CPT/SFT toolkit. Each project keeps data, configs, and outputs under
`<project>/finetune/`; the library lives at a fixed path.

## Toolkit location

```bash
export FINETUNE_LIBRARY_ROOT=/projects/data/llmteam/sidharth/toolkit/finetune-library
export HF_HOME="${HF_HOME:-/projects/data/llmteam/sidharth/toolkit/grpo-library}"
```

`/home/` is out of space. **Always** set `HF_HOME` under `/projects/...`.

## Environment

### Option A — Docker (recommended on GPU nodes)

```bash
export HF_HOME=/projects/data/llmteam/sidharth/toolkit/grpo-library
export FLA_SKIP_TRITON_AUTOTUNE=1

bash "${FINETUNE_LIBRARY_ROOT}/scripts/run-on-gpu-node.sh" \
  configs/models/qwen35-35b-a3b-cpt.yaml
```

Image: `ghcr.io/psidharth567/toolkit/finetune:12.8-cu129` (local alias: `toolkit/finetune:12.8-cu129`).

All kernels (FA2, FA3, Liger, FLA, TileLang, DeepEP) are baked into the image — no runtime install scripts needed.

### Option B — venv

```bash
cd "${FINETUNE_LIBRARY_ROOT}"
uv sync --extra dev --extra unsloth --extra flash
source .venv/bin/activate
bash scripts/install_fla.sh        # Qwen 3.5 only
pip install --no-deps containers/wheels/extra/deep_ep-*.whl  # optional EP
```

Stack: Python 3.12, PyTorch 2.11+cu129, Transformers 5.5, PEFT 0.19, Liger 0.8.1.

## Agent request format

```yaml
finetune_request:
  project_root: /absolute/path/to/project
  run_name: descriptive-run-name
  task: cpt | sft
  model: <registry-key>
  data:
    path: /absolute/path/to/data.jsonl
    format: text | messages | alpaca | prompt_completion | tokenized
  training:
    max_steps: 500
    max_seq_length: 2048
  launch:
    gpus: 8
    nodes: 1
```

## Supported models

| Registry key | HF repo | Layout | Production config |
|---|---|---|---|
| `qwen3-8b` | Qwen/Qwen3-8B | DDP | `configs/models/qwen3-8b-cpt.yaml` |
| `qwen3-14b` | Qwen/Qwen3-14B | DDP | `configs/models/qwen3-14b-cpt.yaml` |
| `qwen3-32b` | Qwen/Qwen3-32B | DDP/HSDP | `configs/models/qwen3-32b-cpt.yaml` |
| `qwen3.5-35b-a3b` | Qwen/Qwen3.5-35B-A3B | EP4+FSDP | `configs/models/qwen35-35b-a3b-cpt.yaml` |
| `gemma4-26b-a4b-it` | google/gemma-4-26B-A4B-it | DDP MoE | `configs/models/gemma4-26b-a4b-it-cpt.yaml` |
| `gemma4-31b-it` | google/gemma-4-31B-it | DDP/HSDP | `configs/models/gemma4-31b-it-cpt.yaml` |
| `deepseek-r1-distill-llama-8b` | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | DDP | `configs/models/deepseek-r1-distill-llama-8b-cpt.yaml` |
| `olmo3-32b-think-dpo` | allenai/Olmo-3-32B-Think-DPO | DDP/HSDP | `configs/models/olmo3-32b-think-dpo-cpt.yaml` |

New models require a `ModelSpec` entry in `src/finetune_library/registry.py` — upgrading Transformers alone is not enough.

## Qualified runtime defaults

| Model class | attention | model_kernels | experts | compile | notes |
|---|---|---|---|---|---|
| Qwen3 dense | sdpa / fa2 | liger (auto) | auto | **off** | Unsloth ok on 8B/14B |
| Gemma4 26B MoE | sdpa | liger | grouped_mm | **off** | Liger+compile incompatible |
| Qwen3.5 35B MoE | fa3 | native | grouped_mm | **off** | FLA+TileLang, DeepEP, EP4+FSDP |

## Agent workflow

1. Parse request → verify `data.path` exists.
2. `finetune-lib init ...` or copy a `configs/models/*.yaml` profile.
3. `export HF_HOME=<project>/.cache/huggingface`
4. `finetune-lib prepare-data --config <yaml>`
5. Launch:
   ```bash
   scripts/production/launch-one-node.sh <config.yaml> train
   # or benchmark for throughput:
   scripts/production/launch-one-node.sh <config.yaml> benchmark
   ```
6. Monitor `<output_dir>/train.log`, `metrics.jsonl`, `events.jsonl`.
7. `finetune-lib merge --checkpoint <output_dir>/final --output <output_dir>/merged`

## SLURM

```bash
export CONFIG_PATH=/path/to/config.yaml
export HF_HOME=/path/to/.cache/huggingface
sbatch "${FINETUNE_LIBRARY_ROOT}/scripts/slurm/train-one-node.sh"
```

## Output artifacts

| File | Purpose |
|---|---|
| `resolved_config.json` | Exact config used |
| `lora_audit.json` | LoRA target coverage |
| `train.log` | Text log |
| `metrics.jsonl` / `events.jsonl` | Per-step metrics |
| `summary.json` / `benchmark.json` | Final summary |
| `final/` | BF16 PEFT adapter |

## Do not

- Put production outputs in `toolkit/finetune-library/outputs/` (gitignored scratch).
- Use `~/` for `HF_HOME` or model caches.
- Enable `torch_compile` with Liger (skipped automatically; no benefit).
- Use `compile_scope: blocks` with MoE `grouped_mm` (auto-downgraded to `loss_only`).
- Bump pinned model revisions without re-running smoke + benchmark gates.

## Adding a new model

Minimum checklist:

1. `ModelSpec` in `registry.py` (revision pin, `layer_class`, `moe`, attention candidates).
2. Loader branch in `runtime.py` if multimodal / non-standard.
3. MoE expert class names in `lora.py` + `moe_parallel.py` if MoE.
4. Liger wiring in `_apply_liger_kernels()` if using Liger.
5. `configs/models/<name>-cpt.yaml` + smoke on 8×GPU.
6. Benchmark before promoting to production defaults.

## Container publish

```bash
cd /projects/data/llmteam/sidharth/toolkit
bash containers/push_ghcr_finetune.sh
```

Requires `docker login ghcr.io` with a GitHub PAT (`write:packages`).
