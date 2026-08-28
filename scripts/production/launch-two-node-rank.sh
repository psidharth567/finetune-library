#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FINETUNE_LIBRARY_ROOT="${FINETUNE_LIBRARY_ROOT:-$(cd "${script_dir}/../.." && pwd)}"

config_path="${1:?usage: launch-two-node-rank.sh CONFIG [train|benchmark]}"
command_name="${2:-train}"
master_address="${MASTER_ADDR:?set MASTER_ADDR to the rendezvous host}"
node_rank="${NODE_RANK:?set NODE_RANK to 0 or 1}"
master_port="${MASTER_PORT:-29500}"
process_count="${NPROC_PER_NODE:-8}"

cd "${FINETUNE_LIBRARY_ROOT}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec torchrun \
  --nnodes=2 \
  --nproc-per-node="${process_count}" \
  --node-rank="${node_rank}" \
  --master-addr="${master_address}" \
  --master-port="${master_port}" \
  -m finetune_library "${command_name}" --config "${config_path}"
