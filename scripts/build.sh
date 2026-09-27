#!/usr/bin/env bash
# Build the standalone finetune-library Docker image.
#
# Build context is the repo root — see docker/Dockerfile. Needs the kernel
# wheels in wheels/ first: scripts/fetch_wheels.sh downloads them.
#
# Usage:
#   scripts/build.sh [TAG]
#
# Env:
#   NODE            Optional. Build on this host via ssh instead of locally.
#   TAG             Image tag (default: latest) -> ${IMAGE_NAME}:<TAG>
#   IMAGE_NAME      Registry image name (default: ghcr.io/psidharth567/finetune-library)
#   PUSH_GHCR       1 to also push ${IMAGE_NAME}:<TAG> (default: 0; needs docker login ghcr.io)
#   ALLOW_MISSING_WHEELS  Passed through as a Docker build ARG (default: 0)
set -euo pipefail

FINETUNE_LIBRARY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAG="${1:-${TAG:-latest}}"
IMAGE_NAME="${IMAGE_NAME:-ghcr.io/psidharth567/finetune-library}"
REMOTE_TAG="${IMAGE_NAME}:${TAG}"
GIT_REV="$(git -C "${FINETUNE_LIBRARY_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
NODE="${NODE:-}"
PUSH_GHCR="${PUSH_GHCR:-0}"
ALLOW_MISSING_WHEELS="${ALLOW_MISSING_WHEELS:-0}"

build_cmd=$(cat <<CMD
set -euo pipefail
cd "${FINETUNE_LIBRARY_ROOT}"
echo "[build] context: \$(du -sh . 2>/dev/null | cut -f1) (before .dockerignore filtering)"
time docker build \
  --build-arg ALLOW_MISSING_WHEELS=${ALLOW_MISSING_WHEELS} \
  --label org.opencontainers.image.version=${TAG} \
  --label org.opencontainers.image.revision=${GIT_REV} \
  -t "${REMOTE_TAG}" \
  -f docker/Dockerfile \
  .
echo "[build] image size:"
docker images "${REMOTE_TAG}" --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.ID}}'
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
  push_cmd=$(cat <<CMD
set -euo pipefail
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

echo "[build] done: ${REMOTE_TAG}"
