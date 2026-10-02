#!/usr/bin/env bash
# One node's share of a multi-node torchrun job (static rendezvous on MASTER_ADDR:MASTER_PORT).
# Run once per node with the same NNODES/MASTER_ADDR/MASTER_PORT and that node's NODE_RANK (scripts/run.sh does this
# when NNODES > 1). Use distributed.strategy=hsdp with shard_size=8 (shard within a node over NVLink, replicate across
# nodes) for full fine-tuning; fsdp over all ranks also works but all-gathers every layer over the network.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FINETUNE_LIBRARY_ROOT="${FINETUNE_LIBRARY_ROOT:-$(cd "${script_dir}/../.." && pwd)}"

config_path="${1:?usage: launch-multi-node.sh CONFIG [train|benchmark]}"
command_name="${2:-train}"
: "${NNODES:?NNODES is required}" "${NODE_RANK:?NODE_RANK is required}" "${MASTER_ADDR:?MASTER_ADDR is required}"

cd "${FINETUNE_LIBRARY_ROOT}"
exec torchrun \
  --nnodes="${NNODES}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT:-29500}" \
  --nproc-per-node="${NPROC_PER_NODE:-8}" \
  -m finetune_library "${command_name}" --config "${config_path}"
