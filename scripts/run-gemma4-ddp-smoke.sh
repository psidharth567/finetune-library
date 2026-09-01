#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/projects/data/llmteam/sidharth/toolkit/grpo-library

CONFIG="${1:?usage: run-gemma4-ddp-smoke.sh CONFIG}"
LOG="${2:-/projects/data/llmteam/sidharth/toolkit/finetune-library/logs/gemma4-smoke.log}"
COMMAND="${3:-train}"
IMAGE="${IMAGE:-ghcr.io/psidharth567/toolkit/finetune:12.8-cu129}"
NODE="${NODE:-bodhanai-node043}"

mkdir -p "$(dirname "$LOG")"

ssh -o StrictHostKeyChecking=no "${NODE}" bash -s -- "${CONFIG}" "${LOG}" "${IMAGE}" "${COMMAND}" <<'REMOTE'
set -euo pipefail
CONFIG="$1"
LOG="$2"
IMAGE="$3"
COMMAND="$4"
export HF_HOME=/projects/data/llmteam/sidharth/toolkit/grpo-library

docker run --rm --gpus all --ipc=host --shm-size=16g \
  -e HF_HOME \
  -v /projects/data/llmteam/sidharth/toolkit:/workspace \
  "${IMAGE}" -lc "
    set -euo pipefail
    cd /workspace/finetune-library
    uv pip install -e . --no-deps -q
    timeout 600 scripts/production/launch-one-node.sh ${CONFIG} ${COMMAND}
  " 2>&1 | tee "${LOG}"
REMOTE

echo "Done. Log: ${LOG}"
