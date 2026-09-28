# finetune-lib

LoRA toolkit for continual pretraining (CPT), supervised finetuning (SFT), and
offline direct preference optimization (DPO) on a single 8×GPU node: DDP, FSDP/HSDP, and MoE expert parallelism, with
pinned model revisions and a prebuilt kernel stack (FA2/FA3, Liger, FLA,
DeepEP).

Agents should read [`AGENTS.md`](AGENTS.md) for the full workflow. All paths
below are relative to the repo root.

## Requirements

- Linux x86_64, NVIDIA **Hopper** GPUs (H100/H200; the kernels are built for
  `sm_90a`), 8 GPUs per node for the production configs
- NVIDIA driver + NVIDIA Container Toolkit (the image carries `cuda-compat`
  for CUDA 12.8, so older 535+ datacenter drivers work)
- Docker, or Python 3.12 + [uv](https://docs.astral.sh/uv/) for a local venv

## Quick start (prebuilt image)

```bash
git clone https://github.com/psidharth567/finetune-library.git
cd finetune-library
docker pull ghcr.io/psidharth567/finetune-library:latest

# Download model weights into the cache the container mounts (it runs offline by default):
HF_HUB_OFFLINE=0 scripts/run.sh configs/models/qwen3-8b-cpt.yaml train
# Afterwards, runs stay offline and fail loudly on a missing pinned revision:
scripts/run.sh configs/models/qwen3-8b-cpt.yaml train
```

For gated models, export `HF_TOKEN` before the first (downloading) run.

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

## Docker image

Published at `ghcr.io/psidharth567/finetune-library` (tags: `latest`, and the
package version, e.g. `1.0.0`). Configs, launch scripts, and tests are baked
into the image at `/opt/toolkit/finetune-library`.

### Building it yourself

Build context is the repo root. The image installs five prebuilt kernel
wheels from `wheels/` (gitignored): `flash_attn`, `flash_attn_3`, `deep_ep`,
`causal_conv1d`, `mamba_ssm` (cp312 / torch 2.11 / CUDA 12). Download the
exact set from the
[`wheels-torch2.11-cu129`](https://github.com/psidharth567/finetune-library/releases/tag/wheels-torch2.11-cu129)
release (sha256-verified):

```bash
scripts/fetch_wheels.sh
```

`docker/build_deepep_wheel.sh` rebuilds `deep_ep` from source. A missing
wheel fails the build loudly by default; pass `ALLOW_MISSING_WHEELS=1` to
intentionally build a reduced image (e.g. SDPA-only attention, no DeepEP).

`unsloth` is **not** installed by default — `unsloth==2026.7.6` crashes at
import under torch 2.11 ("Artifact of type=inductor already registered").
Opt in with `--build-arg INSTALL_UNSLOTH=1` if you specifically need it; every
production config in this repo uses `backend: native` and does not require it.

Build (runs the full CPU pytest suite as part of the build):

```bash
scripts/build.sh latest                  # -> ghcr.io/psidharth567/finetune-library:latest
NODE=<host> scripts/build.sh mytag       # build on a remote host via ssh
```

### Running

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
| `IMAGE` | `ghcr.io/psidharth567/finetune-library:latest` | image to run (pulled if missing) |
| `HF_HOME` | `<repo>/.cache/huggingface` | HF cache mounted at `/cache` inside the container |
| `DATA_DIR` | unset | extra host dir bind-mounted at `/data` |
| `EXTRA_MOUNTS` | unset | extra `-v host:container[:ro]` args, space separated |
| `DEV` | `0` | `1` = bind-mount live source over `/opt/toolkit/finetune-library` |
| `GPUS` | unset (all GPUs) | comma list, e.g. `0,1,2,3` |
| `HF_HUB_OFFLINE` | `1` (inside container) | `1` fails loudly instead of silently downloading a missing revision; `0` allows downloads |
| `LOG` | `<repo>/logs/run-<timestamp>.log` | log file path |
| `RUN_AS_ROOT` | `0` | `1` skips `--user`, runs as root inside the container |
| `HF_TOKEN` | unset | forwarded if set (gated models); local runs only, not over `NODE=` |

`scripts/run.sh` mounts the repo's `HF_HOME` at `/cache`, the repo's `outputs/`
and `.cache/` over the baked tree (so repo-relative `output_dir` and prepared
data land on the host), plus `/projects` at the same path if the host has it
(for shared-filesystem clusters). It runs as your
invoking uid:gid by default (so outputs aren't root-owned), forwards NCCL env
vars and every `FINETUNE_*` env var that's set on the host, and uses
`--gpus all --ipc=host --ulimit memlock=-1 --shm-size=64g` (or `GPUS=...` for
a subset).

## Commands

| Command | Purpose |
|---|---|
| `finetune-lib init` | Scaffold project config and directories |
| `finetune-lib prepare-data` | Tokenize and pack dataset |
| `finetune-lib train` | Run training (CPT, SFT, or DPO) |
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

## SFT packing

With `task: sft` and `data.packing: true` (see
`configs/examples/qwen3-8b-sft-packed.yaml`):

- **Whole examples only.** Examples are bin-packed (best-fit decreasing)
  into windows of at most `training.max_seq_length` tokens. An example is
  never split across windows; if it does not fit in any open window it starts
  a new one. `drop_remainder` does not apply.
- **Isolated.** `packing_isolation` resolves to `attention` (an explicit
  `none` is rejected). Batches are padding-free with `position_ids`
  restarting at 0 per example: flash attention runs varlen, sdpa/eager get a
  block-diagonal causal mask, and Qwen3.5's gated-delta-net layers receive
  the example boundaries (`seq_idx` / `cu_seqlens`). The label at each
  example start is ignored, so no example predicts the next one.
- **Never truncated.** SFT examples longer than `max_seq_length` are dropped
  (`data.sft_overlength: drop`, the default; counted and reported by
  `prepare-data`) or fail preparation (`sft_overlength: error`). This applies
  to unpacked SFT too. `chunk_long_examples` is CPT-only.

CPT packing is unchanged (stream packing, truncation/chunking as configured).

Verified on real weights (8xH100, bf16 unless noted), packing 37 chat
examples into 4k-token windows: randomizing every *other* example in a
window changes an example's per-token loss by exactly 0 — Qwen3-8B (sdpa,
FA2), Gemma4-26B (sdpa, eager), Qwen3.5-35B-A3B (FA3 + GDN) — versus up to
36 nats/token without isolation. In fp32, packed vs each example alone agrees
to 2e-4 (Qwen3-8B sdpa/eager). GPU tests: `tests/test_packed_isolation_gpu.py`;
real-model check: `tests/multigpu/packed_sft_parity.py`.

Under sdpa/eager the packed mask is a dense L x L boolean over the flattened
batch (L = per_device_batch_size x max_seq_length), so prefer flash attention
for long SFT sequences.

## DPO (offline preference optimization)

`task: dpo` trains a LoRA adapter on fixed (prompt, chosen, rejected) pairs
(Rafailov et al. 2023). No generation happens during training. Example:
`configs/examples/qwen3-8b-dpo.yaml`.

- **Data** (`data.format: preference`, JSONL): `prompt` is a string or a
  message list; `chosen`/`rejected` are response strings or message lists.
  Tulu/Olmo preference sets, whose `chosen`/`rejected` repeat the prompt
  turns as `[user, assistant]`, load as-is. Both sides must share the same
  prompt. Only the final assistant turn is supervised.
- **Never truncated.** A pair with either side longer than
  `training.max_seq_length` is dropped (`data.sft_overlength: drop`) or
  rejected (`error`). Pairs whose two sides tokenize identically are dropped.
  Check `prep_stats.json` after `prepare-data`.
- **Reference model = the frozen base.** The adapter is switched off for a
  no-grad reference pass, so no second copy of the weights is held. The
  reference runs the same kernels as the policy minus the LoRA deltas, so a
  fresh adapter starts at loss exactly ln 2 with margin 0.
- **Batches** are padding-free: each micro-batch of
  `per_device_batch_size` pairs becomes one row with per-sequence
  `position_ids` (the isolated layout packed SFT uses). The LM head only
  sees supervised positions, in `preference.logprob_chunk_tokens` chunks.
- **Loss** (`preference:`): `beta` (0.1), `loss_type` `sigmoid | ipo | hinge`,
  `label_smoothing` (sigmoid only), and `sft_weight` (NLL on chosen, off).
  The loss is averaged over pairs across all ranks.
- **Metrics** (`metrics.jsonl`) are pair means: `rewards_chosen`,
  `rewards_rejected`, `rewards_margin`, `rewards_accuracy`, `logps_chosen`,
  `logps_rejected`, plus `pairs`/`pairs_per_second`. `tokens` counts chosen
  and rejected input tokens; each gets a reference and a policy forward.
- **Not supported yet:** `runtime.torch_compile`,
  `runtime.loss: fused_linear_cross_entropy` (the fused kernel assumes
  uniform token weights), and `flex_attention`.

Pairs can come from any source. `scripts/build_pairs.py` (standard library
only) turns `inference batch` outputs into pairs, either best vs worst of N
samples under a reward file (the same file grpo-library takes as
`reward.path`), or strong vs weak model responses (Olmo 3 "delta learning").

Measured on 8xH100, Qwen3-8B DDP, FA2, gradient checkpointing, LoRA r32. One
epoch was 8,024 Olmo-2 13B preference pairs (the first 8,192 rows of one
shuffled shard), `max_seq_length` 4096, 128 pairs per step, lr 5e-5,
beta 0.1, `logprob_chunk_tokens` 1024. It ran at 38k tok/s median with a
24.7 GiB peak. Held-out reward accuracy on 1,009 unseen pairs from the same
shard was 76.4%, and loss went from 0.693 to 0.501. The base-model control is
exactly ln 2 by construction.

Qwen3.5-35B-A3B FSDP+EP4 (FA3, gradient checkpointing, 1024 tokens per side)
fits at 74 GiB with the default `logprob_chunk_tokens: 256`; 1024 ran out of
memory. It is close to the 80 GiB limit, so lower `max_seq_length` or the
chunk if it fails.

## Opt-in knobs (off by default)

- `distributed.reshard_after_forward`, `distributed.fsdp_prefetch_layers`,
  `distributed.ddp_static_graph` — measured no win (or slower) on our
  workloads; left off by default.
- `data.packing_isolation: attention` (CPT) — padding-free isolated packing
  (any attention except `flex_attention`). ~4% slower on packed data, but
  avoids cross-document attention. Always on for packed SFT.

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

## Publish (maintainers)

```bash
PUSH_GHCR=1 scripts/build.sh 1.0.0
PUSH_GHCR=1 scripts/build.sh latest
```

Requires `docker login ghcr.io` with a GitHub PAT (`write:packages`).
`IMAGE_NAME` overrides the registry path for a fork.

## License

[Apache-2.0](LICENSE). The prebuilt wheels and the image bundle third-party
software under their own licenses (PyTorch, FlashAttention, DeepEP, mamba,
FLA, Liger, NVIDIA CUDA base image).
