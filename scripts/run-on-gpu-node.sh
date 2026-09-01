#!/usr/bin/env bash
# Run a finetune config on a GPU node via Docker (mounts workspace, editable install).
set -euo pipefail

export HF_HOME="${HF_HOME:-/projects/data/llmteam/sidharth/toolkit/grpo-library}"
export FLA_SKIP_TRITON_AUTOTUNE="${FLA_SKIP_TRITON_AUTOTUNE:-1}"

CONFIG="${1:?usage: run-on-gpu-node.sh CONFIG [LOG] [COMMAND]}"
LOG="${2:-/projects/data/llmteam/sidharth/toolkit/finetune-library/logs/run.log}"
COMMAND="${3:-train}"
NODE="${NODE:-bodhanai-node043}"
IMAGE="${IMAGE:-toolkit-finetune:12.8-cu129-fa3}"

mkdir -p "$(dirname "$LOG")"

ssh -o StrictHostKeyChecking=no "${NODE}" bash -s -- "${CONFIG}" "${LOG}" "${IMAGE}" "${COMMAND}" <<'REMOTE'
set -euo pipefail
CONFIG="$1"
LOG="$2"
IMAGE="$3"
COMMAND="$4"
export HF_HOME=/projects/data/llmteam/sidharth/toolkit/grpo-library
export FLA_SKIP_TRITON_AUTOTUNE=1

docker run --rm --gpus all --ipc=host --shm-size=16g \
  --entrypoint bash \
  -e HF_HOME -e FLA_SKIP_TRITON_AUTOTUNE \
  -v /projects/data/llmteam/sidharth/toolkit:/workspace \
  "${IMAGE}" -lc "
    set -euo pipefail
    cd /workspace/finetune-library
    uv pip install -e . --no-deps -q
    if ! python -c 'import fla' 2>/dev/null; then bash scripts/install_fla.sh; fi
    if ! python -c 'import deep_ep' 2>/dev/null; then
      pip install --no-deps /workspace/containers/wheels/extra/deep_ep-*.whl 2>/dev/null || bash scripts/install_deepep.sh
    fi
    scripts/production/launch-one-node.sh ${CONFIG} ${COMMAND}
  " 2>&1 | tee "${LOG}"
REMOTE

echo "Done. Log: ${LOG}"
