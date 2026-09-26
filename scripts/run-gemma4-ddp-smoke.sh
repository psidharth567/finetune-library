#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/projects/data/llmteam/sidharth/toolkit/finetune-library/.cache/huggingface}"

CONFIG="${1:?usage: run-gemma4-ddp-smoke.sh CONFIG}"
LOG="${2:-/projects/data/llmteam/sidharth/toolkit/finetune-library/logs/gemma4-smoke.log}"
COMMAND="${3:-train}"
IMAGE="${IMAGE:-toolkit/finetune:latest}"
# NODE is optional: unset runs locally instead of ssh-ing to a fixed node.
NODE="${NODE:-}"
MOUNT_WORKSPACE="${MOUNT_WORKSPACE:-0}"

mkdir -p "$(dirname "$LOG")"

remote_script() {
cat <<'REMOTE'
set -euo pipefail
CONFIG="$1"
LOG="$2"
IMAGE="$3"
COMMAND="$4"
MOUNT_WORKSPACE="$5"
export HF_HOME="${HF_HOME:-/projects/data/llmteam/sidharth/toolkit/finetune-library/.cache/huggingface}"

MOUNT_ARGS=()
if [[ "${MOUNT_WORKSPACE}" == "1" ]]; then
  MOUNT_ARGS=(-v /projects/data/llmteam/sidharth/toolkit:/workspace)
fi

docker run --rm --gpus all --ipc=host --shm-size=16g \
  "${MOUNT_ARGS[@]}" \
  -e HF_HOME \
  "${IMAGE}" bash -lc "
    set -euo pipefail
    if [[ -f /workspace/finetune-library/pyproject.toml ]]; then
      cd /workspace/finetune-library
      uv pip install -e . --no-deps -q
    else
      cd \"\${FINETUNE_LIBRARY_ROOT:-/opt/toolkit/finetune-library}\"
    fi
    timeout 600 scripts/production/launch-one-node.sh ${CONFIG} ${COMMAND}
  " 2>&1 | tee "${LOG}"
REMOTE
}

if [[ -n "${NODE}" ]]; then
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${NODE}" bash -s -- "${CONFIG}" "${LOG}" "${IMAGE}" "${COMMAND}" "${MOUNT_WORKSPACE}" < <(remote_script)
else
  bash -s -- "${CONFIG}" "${LOG}" "${IMAGE}" "${COMMAND}" "${MOUNT_WORKSPACE}" < <(remote_script)
fi

echo "Done. Log: ${LOG}"
