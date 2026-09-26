# finetune-lib

LoRA toolkit for continual pretraining (CPT) and supervised finetuning (SFT).

Agents should read [`AGENTS.md`](AGENTS.md) for the full workflow.

This repo is standalone: everything below is relative to the repo root (the
directory this README lives in).

## Quick start (local venv)

```bash
export HF_HOME="$(pwd)/.cache/huggingface"
uv sync --extra dev --extra flash   # add --extra unsloth only if you need it (see below)
source .venv/bin/activate
bash scripts/install_fla.sh         # Qwen 3.5 GDN only
pytest -q                           # CPU-safe unit tests; GPU tests self-skip
```

Try it against the checked-in dummy data (`data/dummy_cpt.jsonl`,
`data/dummy_sft.jsonl`) with `configs/models/qwen3-8b-cpt.yaml`:

```bash
finetune-lib prepare-data --config configs/models/qwen3-8b-cpt.yaml
scripts/production/launch-one-node.sh configs/models/qwen3-8b-cpt.yaml train
```

## Docker (standalone image, recommended on GPU nodes)

Build context is this repo's root only — no dependency on any sibling
project. Configs, launch scripts, and tests are baked into the image at
`/opt/toolkit/finetune-library`.

**Wheels required.** Populate `wheels/` (gitignored, not part of the repo)
with the cp312 / cu129 / torch 2.11 wheels before building:
`flash_attn`, `flash_attn_3`, `deep_ep`, `causal_conv1d`, `mamba_ssm`. See the
header of `docker/Dockerfile` and `docker/build_deepep_wheel.sh` (for building
`deep_ep` from source). A missing wheel fails the build loudly by default;
pass `ALLOW_MISSING_WHEELS=1` to intentionally build a reduced image (e.g.
SDPA-only attention, no DeepEP).

`unsloth` is **not** installed by default — `unsloth==2026.7.6` crashes at
import under torch 2.11 ("Artifact of type=inductor already registered").
Opt in with `--build-arg INSTALL_UNSLOTH=1` if you specifically need it; every
production config in this repo uses `backend: native` and does not require it.

Build (~4 min; runs the full CPU pytest suite as part of the build):

```bash
scripts/build.sh latest                          # -> toolkit/finetune:latest
NODE=<host> scripts/build.sh standalone-test      # build on a remote host via ssh
PUSH_GHCR=1 scripts/build.sh 12.8-cu129           # also push to GHCR
```

Run:

```bash
scripts/run.sh configs/models/qwen3-8b-cpt.yaml train
scripts/run.sh configs/models/qwen3-8b-cpt.yaml benchmark
NODE=<host> GPUS=0,1,2,3 scripts/run.sh configs/models/qwen3-8b-cpt.yaml train
DEV=1 scripts/run.sh configs/models/qwen3-8b-cpt.yaml train   # live source, bind-mounted
scripts/run.sh _ merge --checkpoint outputs/qwen3-8b-cpt/final --output outputs/qwen3-8b-cpt/merged
```

`scripts/run.sh` env vars:

| Var | Default | Purpose |
|---|---|---|
| `NODE` | unset (run locally) | ssh to this host and run there |
| `IMAGE` | `toolkit/finetune:latest` | image to run |
| `HF_HOME` | `<repo>/.cache/huggingface` | HF cache mounted at `/cache` inside the container |
| `DATA_DIR` | unset | extra host dir bind-mounted at `/data` |
| `EXTRA_MOUNTS` | unset | extra `-v host:container[:ro]` args, space separated |
| `DEV` | `0` | `1` = bind-mount live source over `/opt/toolkit/finetune-library` |
| `GPUS` | unset (all GPUs) | comma list, e.g. `0,1,2,3` |
| `HF_HUB_OFFLINE` | `1` (inside container) | `1` fails loudly instead of silently downloading a missing revision; `0` allows downloads |
| `LOG` | `<repo>/logs/run-<timestamp>.log` | log file path |
| `RUN_AS_ROOT` | `0` | `1` skips `--user`, runs as root inside the container |

`scripts/run.sh` mounts `/projects` and the repo's `HF_HOME`, runs as your
invoking uid:gid by default (so outputs aren't root-owned), forwards NCCL env
vars and every `FINETUNE_*` env var that's set on the host, and uses
`--gpus all --ipc=host --ulimit memlock=-1 --shm-size=64g` (or `GPUS=...` for
a subset).

## Commands

| Command | Purpose |
|---|---|
| `finetune-lib init` | Scaffold project config and directories |
| `finetune-lib prepare-data` | Tokenize and pack dataset |
| `finetune-lib train` | Run training |
| `finetune-lib benchmark` | Measure throughput (warmup + measured steps) |
| `finetune-lib merge` | Merge adapter into base weights |

`train`/`benchmark` run distributed via `torchrun` — use
`scripts/production/launch-one-node.sh <config.yaml> [train|benchmark]`
(`NPROC_PER_NODE`, default 8) rather than invoking `finetune-lib` directly.
`merge` takes `--checkpoint`/`--output`/`--device`/`--max-shard-size`/
`--validation-text` instead of `--config`.

## Model weights and caching

`model.cache_dir` in a config defaults to `None`, which means the standard
Hugging Face hub cache (`$HF_HOME/hub`) — there is no toolkit-specific cache
layout to reason about. `scripts/run.sh` mounts the host `HF_HOME` (default
`<repo>/.cache/huggingface`) at `/cache` inside the container. Registry models
are resolved at their pinned revision; hub network calls are disallowed by
default inside the container (`HF_HUB_OFFLINE=1`) so a missing revision fails
loudly instead of silently downloading. Set `HF_HUB_OFFLINE=0` to allow
downloads. An optional `TOOLKIT_WEIGHTS` env var can point at a flat directory
of `<ModelName>/` weight folders instead — only used if you set it.

## Supported models

| Key | Notes |
|---|---|
| `qwen3-8b`, `qwen3-14b`, `qwen3-32b` | DDP/HSDP; Unsloth is opt-in and not required |
| `qwen3.5-35b-a3b` | EP4+FSDP, FA3+FLA, DeepEP or native all-to-all; see `configs/models/qwen35-35b-a3b-cpt.yaml` |
| `gemma4-26b-a4b-it` | DDP MoE, Liger, grouped_mm |
| `gemma4-31b-it` | DDP/HSDP dense |
| `deepseek-r1-distill-llama-8b`, `olmo3-32b-think-dpo` | DDP native |

Production profiles: `configs/models/*.yaml`. Example SFT configs (messages /
prompt-completion formats): `configs/examples/`.

## Measured throughput (8xH100, seq 2048, LoRA r32, production configs)

Median steady-state tokens/s and peak memory, from the current production
configs:

| Model | tok/s | Peak GiB |
|---|---|---|
| qwen3-8b | 97.9k | 58 |
| deepseek-r1-distill-llama-8b | 112k | 52 |
| qwen3-14b | 58.2k | 60 |
| qwen3-32b (DDP) | 17.7k | 65 |
| gemma4-26b-a4b-it | 34.7k | 75 |
| gemma4-31b-it | 11.5k | 64 |
| olmo3-32b-think-dpo | 15.8k | 65 |
| qwen3.5-35b-a3b (FSDP+EP4, native all-to-all) | 18.0k | 75 |

Qwen3-32B under FSDP/HSDP runs ~13.8k tok/s at 23-29 GiB peak (vs. DDP's
17.7k/65 GiB) — prefer DDP when it fits in memory, FSDP/HSDP when memory-bound.

## Distributed correctness

Verified this cycle across DDP, FSDP, HSDP, and FSDP/HSDP+expert-parallel:
every rank consumes distinct data, per-step losses match DDP within bf16
noise (Qwen3-8B, 20 steps: DDP/FSDP/HSDP final loss 1.3598/1.3594/1.3594;
step-1 per-parameter gradients agree within 0.7%), and LoRA init is seeded
identically across ranks (per-rank seeds apply afterwards).

For MoE expert-parallel (Qwen3.5-35B-A3B): `moe_a2a_backend: native` is exact
(EP forward is bitwise-identical to the non-EP path) and faster than
`deepep` at EP4 (18.0k vs 16.3k tok/s); `deepep` combines in bf16 (~2% drift
over 4 layers). The production config uses `native`. Verify with:

```bash
torchrun --nproc-per-node 8 tests/multigpu/ep_gradcheck.py 4 native
```

## Opt-in knobs (off by default)

- `distributed.reshard_after_forward`, `distributed.fsdp_prefetch_layers`,
  `distributed.ddp_static_graph` — measured no win (or slower) on our
  workloads; left off by default.
- `data.packing_isolation: attention` — requires `flash_attention_2` or
  `flash_attention_3` (padding-free varlen attention). ~4% slower on packed
  data, but avoids cross-document attention.

## Kernel stack (container)

| Kernel | Purpose |
|---|---|
| FA2 + FA3 | Flash attention (Qwen 3.5 uses FA3) |
| Liger | Fused RMSNorm / SwiGLU / GEGLU |
| FLA + TileLang | Qwen 3.5 GDN layers |
| DeepEP | MoE expert-parallel dispatch (opt-in backend; `native` is default) |
| grouped_mm | MoE expert matmuls (Gemma, Qwen) |

## SLURM

```bash
export CONFIG_PATH=/path/to/config.yaml
export HF_HOME=/path/to/.cache/huggingface
sbatch scripts/slurm/train-one-node.sh
```

## Important runtime rules

- Set `HF_HOME` under a project/data volume, not `~/` (home directories are
  often small or ephemeral). Default: `<repo>/.cache/huggingface`.
- `model.cache_dir`, if set explicitly, is passed straight through as the
  `huggingface_hub`/`transformers` `cache_dir=` kwarg — snapshots land at
  `<cache_dir>/models--org--name` directly, **not**
  `<cache_dir>/hub/models--org--name`. Leave it `None` (the default) to use
  the standard `$HF_HOME/hub` cache layout instead.
- Do **not** use `torch_compile` with Liger kernels (auto-skipped).
- For MoE + `grouped_mm`, block-level compile is auto-downgraded to `loss_only`.
- Qwen 3.5: export `FLA_SKIP_TRITON_AUTOTUNE=1` on memory-tight EP runs.

## Publish

```bash
PUSH_GHCR=1 scripts/build.sh 12.8-cu129
```

Requires `docker login ghcr.io` with a GitHub PAT (`write:packages`).
