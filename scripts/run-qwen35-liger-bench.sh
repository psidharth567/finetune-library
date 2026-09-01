#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/projects/data/llmteam/sidharth/toolkit/grpo-library
export FLA_SKIP_TRITON_AUTOTUNE=1

CONFIG="${1:?usage: run-qwen35-liger-bench.sh CONFIG [LOG] [COMMAND]}"
LOG="${2:-/projects/data/llmteam/sidharth/toolkit/finetune-library/logs/liger-bench.log}"
COMMAND="${3:-benchmark}"
NODE="${NODE:-bodhanai-node043}"
IMAGE="${IMAGE:-ghcr.io/psidharth567/toolkit/finetune:12.8-cu129}"

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
  -e HF_HOME -e FLA_SKIP_TRITON_AUTOTUNE \
  -v /projects/data/llmteam/sidharth/toolkit:/workspace \
  "${IMAGE}" -lc "
    set -euo pipefail
    cd /workspace/finetune-library
    uv pip install -e . --no-deps -q
    scripts/production/launch-one-node.sh ${CONFIG} ${COMMAND}
  " 2>&1 | tee "${LOG}"
REMOTE

echo "Done. Log: ${LOG}"
