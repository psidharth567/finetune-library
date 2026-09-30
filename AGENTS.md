# finetune-lib — Agent instructions

LoRA CPT/SFT/DPO toolkit. This is a standalone repo — all paths below are
relative to the repo root.

## Toolkit location

```bash
export FINETUNE_LIBRARY_ROOT=/path/to/finetune-library   # repo root
export HF_HOME="${HF_HOME:-${FINETUNE_LIBRARY_ROOT}/.cache/huggingface}"
```

Always set `HF_HOME` under a project/data volume, not `~/`.

## Environment

### Option A — Docker (standalone image, recommended on GPU nodes)

The image is self-contained: build context is the repo root, no dependency
on any sibling project.

```bash
export HF_HOME="${FINETUNE_LIBRARY_ROOT}/.cache/huggingface"
export FLA_SKIP_TRITON_AUTOTUNE=1

"${FINETUNE_LIBRARY_ROOT}/scripts/run.sh" configs/models/qwen35-35b-a3b-cpt.yaml train
```

Image: `ghcr.io/psidharth567/finetune-library:latest` — the only finetune
image (the `scripts/run.sh` default; `scripts/build.sh TAG` builds
`ghcr.io/psidharth567/finetune-library:TAG`). `NODE=<host>` on `build.sh`/`run.sh` builds/runs
over ssh; unset runs on the current host.

Before building, run `scripts/fetch_wheels.sh` to download the cp312/cu129/
torch-2.11 wheels into `wheels/` (gitignored): `flash_attn`, `flash_attn_3`,
`deep_ep`, `causal_conv1d`, `mamba_ssm`. A missing wheel is a hard build failure unless
`ALLOW_MISSING_WHEELS=1`. `unsloth` is not installed by default (broken under
torch 2.11); opt in with `--build-arg INSTALL_UNSLOTH=1` if you specifically
need it — no production config requires it.

Baked into the image at `/opt/toolkit/finetune-library`:
- `src/` + `finetune-lib` CLI
- `configs/models/`, `configs/benchmarks/`, `configs/examples/`
- `scripts/production/`, `scripts/slurm/`
- `tests/` (full CPU unit test suite runs at image build time; GPU-only
  tests self-skip)
- All kernels (FA2, FA3, Liger, FLA, TileLang, DeepEP), installed from
  `wheels/` at build time

`DEV=1 scripts/run.sh ...` bind-mounts live source over
`/opt/toolkit/finetune-library` for iterating on library code without
rebuilding.

`scripts/run.sh` bind-mounts the repo's `outputs/` and `.cache/` over the
baked tree, so run outputs land on the host. It runs as the invoking host uid:gid by default
(`RUN_AS_ROOT=1` to skip that), forwards NCCL env vars and every `FINETUNE_*`
env var set on the host, and sets `HF_HUB_OFFLINE=1` inside the container by
default (fails loudly on a missing pinned revision instead of downloading).

### Option B — venv

```bash
cd "${FINETUNE_LIBRARY_ROOT}"
uv sync --extra dev --extra flash   # add --extra unsloth only if you need it (broken under torch 2.11; opt-in)
source .venv/bin/activate
bash scripts/install_fla.sh                 # Qwen 3.5 only
pip install --no-deps wheels/deep_ep-*.whl  # optional EP
```

Stack: Python 3.12, PyTorch 2.11+cu129, Transformers 5.5, PEFT 0.19, Liger 0.8.1.

## Agent request format

```yaml
finetune_request:
  project_root: /absolute/path/to/project
  run_name: descriptive-run-name
  task: cpt | sft | dpo
  model: <registry-key>
  data:
    path: /absolute/path/to/data.jsonl
    format: text | messages | alpaca | prompt_completion | tokenized | preference
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
| `olmo3-1125-32b` | allenai/Olmo-3-1125-32B | DDP/HSDP | `configs/models/olmo3-1125-32b-cpt.yaml` |
| `olmo3-1025-7b` | allenai/Olmo-3-1025-7B | DDP | `configs/models/olmo3-1025-7b-cpt.yaml` |
| `olmo2-1124-13b` | allenai/OLMo-2-1124-13B | DDP | `configs/models/olmo2-1124-13b-cpt.yaml` |
| `olmo2-1124-7b` | allenai/OLMo-2-1124-7B | DDP | `configs/models/olmo2-1124-7b-cpt.yaml` |

New models require a `ModelSpec` entry in `src/finetune_library/registry.py` — upgrading Transformers alone is not enough.

## Qualified runtime defaults

| Model class | attention | model_kernels | experts | compile | notes |
|---|---|---|---|---|---|
| Qwen3 dense | sdpa / fa2 | liger (auto) | auto | **off** | `backend: native`; Unsloth is opt-in, not required |
| Gemma4 26B MoE | sdpa | liger | grouped_mm | **off** | Liger+compile incompatible |
| OLMo 2/3 (incl. Think-DPO) | sdpa | liger (auto) | auto | **off** | Liger +14-32% vs native, loss parity checked |
| Qwen3.5 35B MoE | fa3 | native | grouped_mm | **off** | FLA+TileLang, DeepEP or native all-to-all, EP4+FSDP |

`backend: native` is the default everywhere, including all example SFT
configs under `configs/examples/`.

## Measured throughput (8xH100, seq 2048, LoRA r32, production configs)

| Model | tok/s | Peak GiB |
|---|---|---|
| qwen3-8b | 97.9k | 58 |
| deepseek-r1-distill-llama-8b | 112k | 52 |
| qwen3-14b | 58.2k | 60 |
| qwen3-32b (DDP) | 17.7k | 65 |
| gemma4-26b-a4b-it | 34.7k | 75 |
| gemma4-31b-it | 11.5k | 64 |
| olmo3-32b-think-dpo | 18.0k | 64 |
| olmo3-1125-32b | 18.0k | 64 |
| olmo3-1025-7b | 97.1k | 57 |
| olmo2-1124-13b | 57.2k | 60 |
| olmo2-1124-7b | 102k | 57 |
| qwen3.5-35b-a3b (FSDP+EP4, native all-to-all) | 18.0k | 75 |

Qwen3-32B under FSDP/HSDP: ~13.8k tok/s at 23-29 GiB (vs. DDP 17.7k/65 GiB) —
prefer DDP when it fits, FSDP/HSDP when memory-bound.

## Distributed correctness (verified)

DDP, FSDP, HSDP, and FSDP/HSDP+expert-parallel all give every rank distinct
data and per-step losses matching DDP within bf16 noise (Qwen3-8B, 20 steps:
final loss 1.3598/1.3594/1.3594 for DDP/FSDP/HSDP; step-1 per-parameter
gradients agree within 0.7%). LoRA init is seeded identically across ranks;
per-rank seeds apply afterwards.

MoE expert-parallel (Qwen3.5-35B-A3B): `moe_a2a_backend: native` is exact
(bitwise-identical EP vs non-EP forward) and faster than `deepep` at EP4
(18.0k vs 16.3k tok/s); `deepep` combines in bf16 (~2% drift over 4 layers).
Production config uses `native`. Verify with:

```bash
torchrun --nproc-per-node 8 tests/multigpu/ep_gradcheck.py 4 native
```

## Model weights and caching

`model.cache_dir` defaults to `None` — the standard HF hub cache
(`$HF_HOME/hub`). Only set it explicitly if you need the flat
`<cache_dir>/models--org--name` layout (note: **not** `<cache_dir>/hub/...`).
Registry models resolve at their pinned revision; `HF_HUB_OFFLINE=1` inside
the container fails loudly on a missing revision instead of downloading. An
optional `TOOLKIT_WEIGHTS` env var can point at a flat dir of `<ModelName>/`
weights instead, only used if set.

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
7. Merge: `scripts/run.sh _ merge --checkpoint <output_dir>/final --output <output_dir>/merged`
   (or, outside Docker: `finetune-lib merge --checkpoint ... --output ...`).
   Note `merge` takes `--checkpoint`/`--output`/`--device`/`--max-shard-size`/
   `--validation-text`, not `--config`.

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

## Opt-in knobs (off by default)

- `distributed.reshard_after_forward`, `distributed.fsdp_prefetch_layers`,
  `distributed.ddp_static_graph` — measured no win (or slower) on our
  workloads.
- `data.packing_isolation: attention` (CPT) — padding-free isolated packing,
  any attention except `flex_attention`; ~4% slower on packed data but avoids
  cross-document attention.

## SFT packing

`task: sft` + `data.packing: true` bin-packs **whole** examples into windows
of <= `max_seq_length` (never split), always isolated (`packing_isolation`
resolves to `attention`; Qwen3.5 GDN layers get example boundaries via
`packed_linear_attention.py`). SFT examples longer than `max_seq_length` are
never truncated: dropped and reported (`data.sft_overlength: drop`, default)
or rejected (`error`) — packed or not. Example:
`configs/examples/qwen3-8b-sft-packed.yaml`. Check `prep_stats.json`
(`dropped_overlength`, `examples_per_window`, `avg_fill_pct`) after
`prepare-data`.

## OLMo base models

`olmo3-1125-32b`, `olmo3-1025-7b`, `olmo2-1124-*` have no chat template:
use `text` / `prompt_completion` / `alpaca`, or set `model.chat_template` for
`messages` SFT and DPO (otherwise they fail at load time). OLMo 2 context is
4,096 tokens.

## DPO

`task: dpo` + `data.format: preference` runs offline DPO on
`{"prompt", "chosen", "rejected"}` rows (strings or message lists; Olmo/Tulu
layouts load as-is). See README "DPO" and `configs/examples/qwen3-8b-dpo.yaml`.
The reference is the base model (adapter disabled, no grad); a fresh adapter
must log step-1 `loss` = 0.6931472 and `rewards_margin` = 0. Treat any other
value as a bug. Pairs are never truncated (`dropped_overlength` in
`prep_stats.json`). Unsupported: `torch_compile`,
`loss: fused_linear_cross_entropy`, `flex_attention`. Build pairs from
`inference batch` output with `scripts/build_pairs.py` (stdlib only).

## Do not

- Put production outputs in `outputs/` at the repo root (gitignored scratch).
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
5. `configs/models/<name>-cpt.yaml` + smoke on 8xGPU.
6. Benchmark before promoting to production defaults.

## Container publish

```bash
cd "${FINETUNE_LIBRARY_ROOT}"
PUSH_GHCR=1 scripts/build.sh 1.0.0 && PUSH_GHCR=1 scripts/build.sh latest
```

Requires `docker login ghcr.io` with a GitHub PAT (`write:packages`).
