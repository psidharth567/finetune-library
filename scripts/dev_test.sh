#!/usr/bin/env bash
# Run pytest (CPU unit suite; GPU tests self-skip without GPUs) in the image against the LIVE source tree on THIS host.
# usage: scripts/dev_test.sh [pytest args...]   env: IMAGE (default ghcr.io/psidharth567/finetune-library:1.4.0)
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; RH=$R/.cache/container-home
docker run --rm --ipc=host --user "$(id -u):$(id -g)" -v /projects:/projects -v "$R":/opt/toolkit/finetune-library \
  -e HOME="$RH" -e XDG_CACHE_HOME="$RH" -e TRITON_CACHE_DIR="$RH/triton" -e UV_CACHE_DIR="$RH/uv" -e TORCHINDUCTOR_CACHE_DIR="$RH/inductor" \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" -e HF_HUB_OFFLINE=1 \
  -w /opt/toolkit/finetune-library "${IMAGE:-ghcr.io/psidharth567/finetune-library:1.4.0}" \
  bash -lc 'uv pip install -e . --no-deps -q 2>/dev/null || true; python -m pytest "$@"' _ "${@:--q}"
