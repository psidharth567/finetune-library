#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FINETUNE_LIBRARY_ROOT="${FINETUNE_LIBRARY_ROOT:-$(cd "${script_dir}/../.." && pwd)}"

config_path="${1:?usage: launch-one-node.sh CONFIG [train|benchmark]}"
command_name="${2:-train}"
process_count="${NPROC_PER_NODE:-8}"

cd "${FINETUNE_LIBRARY_ROOT}"
# Only source a host .venv when we are actually running on the host. Inside
# the container (DEV=1 scripts/run.sh bind-mounts live source over
# /opt/toolkit/finetune-library, which is the same host path as
# FINETUNE_LIBRARY_ROOT), /projects is mounted read/write, so a .venv built
# for the *host* python/torch install is visible here too -- sourcing it
# would silently swap the container's own python/torch for an
# incompatible host build. /.dockerenv only exists inside a container.
if [[ -f .venv/bin/activate && ! -f /.dockerenv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${process_count}" \
  -m finetune_library "${command_name}" --config "${config_path}"
