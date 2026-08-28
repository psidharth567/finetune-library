#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FINETUNE_LIBRARY_ROOT="${FINETUNE_LIBRARY_ROOT:-$(cd "${script_dir}/../.." && pwd)}"

config_path="${1:?usage: launch-one-node.sh CONFIG [train|benchmark]}"
command_name="${2:-train}"
process_count="${NPROC_PER_NODE:-8}"

cd "${FINETUNE_LIBRARY_ROOT}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${process_count}" \
  -m finetune_library "${command_name}" --config "${config_path}"
