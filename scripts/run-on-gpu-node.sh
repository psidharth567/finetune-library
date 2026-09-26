#!/usr/bin/env bash
# Run a finetune config on a GPU node via Docker.
# Uses baked image content by default; set MOUNT_WORKSPACE=1 to pick up live repo changes.
set -euo pipefail

export HF_HOME="${HF_HOME:-/projects/data/llmteam/sidharth/toolkit/finetune-library/.cache/huggingface}"
export FLA_SKIP_TRITON_AUTOTUNE="${FLA_SKIP_TRITON_AUTOTUNE:-1}"

CONFIG="${1:?usage: run-on-gpu-node.sh CONFIG [LOG] [COMMAND]}"
LOG="${2:-/projects/data/llmteam/sidharth/toolkit/finetune-library/logs/run.log}"
COMMAND="${3:-train}"
# NODE is optional: unset (or empty) runs the docker command on this host
# instead of ssh-ing to a fixed default node.
NODE="${NODE:-}"
IMAGE="${IMAGE:-toolkit/finetune:latest}"
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
export FLA_SKIP_TRITON_AUTOTUNE="${FLA_SKIP_TRITON_AUTOTUNE:-1}"

MOUNT_ARGS=()
if [[ "${MOUNT_WORKSPACE}" == "1" ]]; then
  MOUNT_ARGS=(-v /projects/data/llmteam/sidharth/toolkit:/workspace)
fi

docker run --rm --gpus all --ipc=host --shm-size=16g \
  "${MOUNT_ARGS[@]}" \
  -e HF_HOME -e FLA_SKIP_TRITON_AUTOTUNE \
  "${IMAGE}" bash -lc "
    set -euo pipefail
    if [[ -f /workspace/finetune-library/pyproject.toml ]]; then
      cd /workspace/finetune-library
      uv pip install -e . --no-deps -q
    else
      cd \"\${FINETUNE_LIBRARY_ROOT:-/opt/toolkit/finetune-library}\"
    fi
    scripts/production/launch-one-node.sh ${CONFIG} ${COMMAND}
  " 2>&1 | tee "${LOG}"
REMOTE
}

if [[ -n "${NODE}" ]]; then
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${NODE}" bash -s -- "${CONFIG}" "${LOG}" "${IMAGE}" "${COMMAND}" "${MOUNT_WORKSPACE}" < <(remote_script)
else
  bash -s -- "${CONFIG}" "${LOG}" "${IMAGE}" "${COMMAND}" "${MOUNT_WORKSPACE}" < <(remote_script)
fi

echo "Done. Log: ${LOG}"
