#!/usr/bin/env bash
# Build the standalone finetune-library Docker image.
#
# Build context is finetune-library/ only (not the monorepo) — see docker/Dockerfile.
#
# Usage:
#   scripts/build.sh [TAG]
#
# Env:
#   NODE            Optional. Build on this host via ssh instead of locally.
#   TAG             Image tag (default: latest) -> toolkit/finetune:<TAG>
#   PUSH_GHCR       1 to also tag + push to GHCR (default: 0)
#   REGISTRY        GHCR path (default: ghcr.io/psidharth567/toolkit)
#   ALLOW_MISSING_WHEELS  Passed through as a Docker build ARG (default: 0)
set -euo pipefail

FINETUNE_LIBRARY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAG="${1:-${TAG:-latest}}"
LOCAL_TAG="toolkit/finetune:${TAG}"
NODE="${NODE:-}"
PUSH_GHCR="${PUSH_GHCR:-0}"
REGISTRY="${REGISTRY:-ghcr.io/psidharth567/toolkit}"
ALLOW_MISSING_WHEELS="${ALLOW_MISSING_WHEELS:-0}"

build_cmd=$(cat <<CMD
set -euo pipefail
cd "${FINETUNE_LIBRARY_ROOT}"
echo "[build] context: \$(du -sh . 2>/dev/null | cut -f1) (before .dockerignore filtering)"
time docker build \
  --build-arg ALLOW_MISSING_WHEELS=${ALLOW_MISSING_WHEELS} \
  -t "${LOCAL_TAG}" \
  -f docker/Dockerfile \
  .
echo "[build] image size:"
docker images "${LOCAL_TAG}" --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.ID}}'
CMD
)

if [[ -n "${NODE}" ]]; then
  echo "[build] building on ${NODE} via ssh"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${NODE}" bash -s <<REMOTE
${build_cmd}
REMOTE
else
  echo "[build] building locally on $(hostname)"
  bash -c "${build_cmd}"
fi

if [[ "${PUSH_GHCR}" == "1" ]]; then
  REMOTE_TAG="${REGISTRY}/finetune:${TAG}"
  push_cmd=$(cat <<CMD
set -euo pipefail
docker tag "${LOCAL_TAG}" "${REMOTE_TAG}"
docker push "${REMOTE_TAG}"
echo "[push] pushed ${REMOTE_TAG}"
CMD
)
  if [[ -n "${NODE}" ]]; then
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${NODE}" bash -s <<REMOTE
${push_cmd}
REMOTE
  else
    bash -c "${push_cmd}"
  fi
fi

echo "[build] done: ${LOCAL_TAG}"
