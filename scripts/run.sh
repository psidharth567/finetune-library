#!/usr/bin/env bash
# Run a finetune-library command in the standalone Docker image.
#
# Usage:
#   scripts/run.sh CONFIG [train|benchmark|prepare-data|merge ...]
#
# Env:
#   NODE              Optional. ssh to this host and run there. Unset = run
#                      on the current host. The repo path must be the same on
#                      that host (shared filesystem).
#   IMAGE             Docker image (default: ghcr.io/psidharth567/finetune-library:latest;
#                      pulled automatically if not present locally)
#   HF_HOME           HF cache root on the host (default: <repo>/.cache/huggingface)
#   DATA_DIR          Extra host dir to bind-mount at /data (optional)
#   EXTRA_MOUNTS      Extra `-v host:container[:ro]` args, space separated (optional)
#   DEV               1 = bind-mount the live finetune-library source tree over
#                      /opt/toolkit/finetune-library inside the image (default: 0)
#   GPUS              Comma list, e.g. "0,1,2,3" (default: all GPUs)
#   HF_HUB_OFFLINE    1 = forbid HF hub network calls, fail loudly instead of
#                      silently downloading if a pinned revision is missing
#                      from cache (default: 1). Set 0 to allow downloads.
#   LOG               Log file path (default: <repo>/logs/run-<ts>.log)
#   RUN_AS_ROOT       1 = skip --user (default: 0, runs as invoking uid:gid)
#   HF_TOKEN          Forwarded if set (gated models; local runs only, not over NODE=)
set -euo pipefail

FINETUNE_LIBRARY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="${1:?usage: run.sh CONFIG [train|benchmark|prepare-data|merge ...]}"
shift || true
SUBCOMMAND="${1:-train}"
shift || true
EXTRA_ARGS=("$@")

NODE="${NODE:-}"
IMAGE="${IMAGE:-ghcr.io/psidharth567/finetune-library:latest}"
HF_HOME="${HF_HOME:-${FINETUNE_LIBRARY_ROOT}/.cache/huggingface}"
DATA_DIR="${DATA_DIR:-}"
EXTRA_MOUNTS="${EXTRA_MOUNTS:-}"
DEV="${DEV:-0}"
GPUS="${GPUS:-}"
TS="$(date +%Y%m%d-%H%M%S)"
LOG="${LOG:-${FINETUNE_LIBRARY_ROOT}/logs/run-${TS}.log}"
RUN_AS_ROOT="${RUN_AS_ROOT:-0}"

mkdir -p "$(dirname "${LOG}")" "${HF_HOME}"

# Writable per-invoker cache dirs for triton/inductor/uv/etc when running as
# non-root: containers have no home dir for an arbitrary host uid, so point
# every cache env var at a host-mounted, already-writable location.
RUNTIME_HOME="${FINETUNE_LIBRARY_ROOT}/.cache/container-home"
mkdir -p "${RUNTIME_HOME}/triton" "${RUNTIME_HOME}/inductor" "${RUNTIME_HOME}/uv"

INVOKER_UID="$(id -u)"
INVOKER_GID="$(id -g)"
# `--user uid:gid` below runs as an arbitrary uid the container's /etc/passwd
# has no entry for. That's normally harmless, but torch._inductor.codecache
# computes a module-level `_HEADER_DIR` constant at import time via
# default_cache_dir(), which unconditionally calls Python's
# getpass.getuser() -- and that function falls back to pwd.getpwuid(uid)
# only when none of LOGNAME/USER/LNAME/USERNAME are already set, raising
# `KeyError: getpwuid(): uid not found` otherwise. This crashes any run that
# imports torch dynamo/inductor internals (observed on Qwen3.5 EP+FSDP+FA3),
# even though TORCHINDUCTOR_CACHE_DIR is already set below -- that env var
# is honored by the *runtime* cache path, not by this particular
# import-time constant. Setting USER/LOGNAME makes getpass.getuser() return
# immediately from the environment, so it never reaches pwd.getpwuid at all.
INVOKER_USER="$(id -un)"

GPU_ARGS=(--gpus all)
if [[ -n "${GPUS}" ]]; then
  GPU_ARGS=(--gpus "device=${GPUS}")
fi

MOUNT_ARGS=(
  -v "${HF_HOME}:/cache"
  -v "${RUNTIME_HOME}:/runtime_home"
)
# Shared-filesystem convenience: if the host has /projects, mount it at the
# same path so absolute data/output paths in configs resolve unchanged.
if [[ -d /projects ]]; then
  MOUNT_ARGS+=(-v "/projects:/projects")
fi
if [[ -n "${DATA_DIR}" ]]; then
  MOUNT_ARGS+=(-v "${DATA_DIR}:/data")
fi
if [[ "${DEV}" == "1" ]]; then
  MOUNT_ARGS+=(-v "${FINETUNE_LIBRARY_ROOT}:/opt/toolkit/finetune-library")
fi
if [[ -n "${EXTRA_MOUNTS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${EXTRA_MOUNTS})
  for m in "${EXTRA_ARR[@]}"; do
    MOUNT_ARGS+=(-v "${m}")
  done
fi

USER_ARGS=()
if [[ "${RUN_AS_ROOT}" != "1" ]]; then
  USER_ARGS=(--user "${INVOKER_UID}:${INVOKER_GID}")
fi

# NCCL / distributed env pass-through (only forwarded if actually set on host).
NCCL_ENV_ARGS=()
for var in NCCL_SOCKET_IFNAME NCCL_IB_DISABLE NCCL_IB_HCA NCCL_NVLS_ENABLE \
           NCCL_DEBUG NCCL_P2P_DISABLE NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC \
           MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK; do
  if [[ -n "${!var:-}" ]]; then
    NCCL_ENV_ARGS+=(-e "${var}=${!var}")
  fi
done
# Forward every FINETUNE_* debug/feature flag (e.g. FINETUNE_DEBUG_EXPERT_IDS).
while IFS='=' read -r var _; do
  NCCL_ENV_ARGS+=(-e "${var}=${!var}")
done < <(env | grep -E '^FINETUNE_[A-Z0-9_]+=' || true)
# HF_TOKEN (gated models) is forwarded by name only, so its value never lands
# in the generated command file. Local runs only: it is not carried over ssh.
if [[ -n "${HF_TOKEN:-}" ]]; then
  NCCL_ENV_ARGS+=(-e HF_TOKEN)
fi

CONFIG_ARG="${CONFIG}"

# Number of processes for torchrun (train/benchmark): derived from GPUS if set,
# else NPROC_PER_NODE, else 8 (matches scripts/production/launch-one-node.sh).
if [[ -n "${GPUS}" ]]; then
  NPROC_PER_NODE="${NPROC_PER_NODE:-$(( $(tr ',' '\n' <<<"${GPUS}" | wc -l) ))}"
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
fi

EXTRA_ARGS_STR="${EXTRA_ARGS[*]:-}"

# train/benchmark run distributed via torchrun (scripts/production/launch-one-node.sh);
# prepare-data/init are single-process finetune-lib invocations that take
# --config. merge is also single-process but its CLI (cli.py's "merge"
# subparser) takes --checkpoint/--output/--device/--max-shard-size/
# --validation-text instead and has no --config option at all, so it cannot
# go through the generic --config passthrough below -- doing so used to fail
# every merge invocation with "unrecognized arguments: --config". The
# required CONFIG positional is kept only for a uniform call signature
# (`run.sh <placeholder> merge --checkpoint ... --output ...`); its value is
# ignored for merge.
case "${SUBCOMMAND}" in
  train|benchmark)
    INNER_CMD="NPROC_PER_NODE=${NPROC_PER_NODE} scripts/production/launch-one-node.sh \"${CONFIG_ARG}\" ${SUBCOMMAND}"
    ;;
  merge)
    INNER_CMD="finetune-lib merge ${EXTRA_ARGS_STR}"
    ;;
  *)
    INNER_CMD="finetune-lib ${SUBCOMMAND} --config \"${CONFIG_ARG}\" ${EXTRA_ARGS_STR}"
    ;;
esac

# Built as a single physical line per statement (no backslash-newline
# continuations): those get consumed unpredictably once this text is re-read
# from a heredoc/temp file, silently mangling the docker invocation.
DOCKER_ARGS_LINE="--rm ${GPU_ARGS[*]} --ipc=host --ulimit memlock=-1 --shm-size=64g ${USER_ARGS[*]} ${MOUNT_ARGS[*]} -e HOME=/runtime_home -e USER=${INVOKER_USER} -e LOGNAME=${INVOKER_USER} -e TRITON_CACHE_DIR=/runtime_home/triton -e TORCHINDUCTOR_CACHE_DIR=/runtime_home/inductor -e UV_CACHE_DIR=/runtime_home/uv -e XDG_CACHE_HOME=/runtime_home -e HF_HOME=/cache -e HUGGINGFACE_HUB_CACHE=/cache/hub -e HF_HUB_OFFLINE=\"\${HF_HUB_OFFLINE:-1}\" -e FLA_SKIP_TRITON_AUTOTUNE=\"\${FLA_SKIP_TRITON_AUTOTUNE:-1}\" ${NCCL_ENV_ARGS[*]}"

INNER_SCRIPT="set -euo pipefail; cd \"\${FINETUNE_LIBRARY_ROOT:-/opt/toolkit/finetune-library}\"; if [[ -f pyproject.toml ]]; then uv pip install -e . --no-deps -q 2>/dev/null || true; fi; ${INNER_CMD}"

run_cmd=$(cat <<CMD
set -euo pipefail
mkdir -p "${HF_HOME}" "${RUNTIME_HOME}/triton" "${RUNTIME_HOME}/inductor" "${RUNTIME_HOME}/uv"
docker run ${DOCKER_ARGS_LINE} "${IMAGE}" bash -lc '${INNER_SCRIPT}' 2>&1 | tee "${LOG}"
CMD
)

echo "[run] image=${IMAGE} node=${NODE:-<local:$(hostname)>} config=${CONFIG_ARG} subcommand=${SUBCOMMAND}"
echo "[run] HF_HOME(host)=${HF_HOME}  log=${LOG}"

# Write the generated command to a file and pipe that file's bytes verbatim
# (avoids double heredoc/quote-escaping issues from nesting a script-with-quotes
# inside another heredoc).
RUN_SCRIPT="$(mktemp)"
trap 'rm -f "${RUN_SCRIPT}"' EXIT
printf '%s\n' "${run_cmd}" > "${RUN_SCRIPT}"

if [[ -n "${NODE}" ]]; then
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${NODE}" bash -s < "${RUN_SCRIPT}"
else
  bash "${RUN_SCRIPT}"
fi

echo "[run] done. Log: ${LOG}"
