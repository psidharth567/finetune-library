#!/usr/bin/env bash
# Download the prebuilt kernel wheels that docker/Dockerfile installs into
# wheels/ (gitignored), from this repo's GitHub release, and verify sha256.
#
# Only needed to build the image yourself; to just run, pull the published
# image instead: docker pull ghcr.io/psidharth567/finetune-library:latest
#
# All wheels are cp312 / torch 2.11 / CUDA 12.x / x86_64 builds.
#
# Usage:
#   scripts/fetch_wheels.sh
#
# Env:
#   WHEELS_RELEASE  Release tag to download from (default: wheels-torch2.11-cu129)
#   WHEEL_DIR       Destination (default: <repo>/wheels)
set -euo pipefail

FINETUNE_LIBRARY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELS_RELEASE="${WHEELS_RELEASE:-wheels-torch2.11-cu129}"
WHEEL_DIR="${WHEEL_DIR:-${FINETUNE_LIBRARY_ROOT}/wheels}"
BASE_URL="https://github.com/psidharth567/finetune-library/releases/download/${WHEELS_RELEASE}"

# <sha256> <wheel filename (= release asset name)>
WHEELS=(
  "c16c1c48d4fa63415cc797e02d69f97248c57c04627d99e394d5bb0ef266e288 causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl"
  "92f8b04f700a1ff8d220968d98b980d688f0b58c379d88dd36e8934e23c66516 deep_ep-1.2.1+9af0e0d-cp312-cp312-linux_x86_64.whl"
  "3d0c8e60f820321eedd7166e79c33cb816263d8be6e35c3f5ba8fe2df6fea697 flash_attn-2.8.3+cu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
  "3798659e5b7e776c874c77677cc126217e8ca4909676ebdb3ad9807db233caab flash_attn_3-3.0.0+cu128torch2.11gite2743ab-cp39-abi3-linux_x86_64.whl"
  "7a67070c1e7e99c95abd1319623f044e8a1b3fb46f774bfdea949f0a4fc79638 mamba_ssm-2.3.2.post1-cp312-cp312-linux_x86_64.whl"
)

mkdir -p "${WHEEL_DIR}"
for entry in "${WHEELS[@]}"; do
  read -r sha local_name <<<"${entry}"
  dest="${WHEEL_DIR}/${local_name}"
  if [[ -f "${dest}" ]] && echo "${sha}  ${dest}" | sha256sum --check --status; then
    echo "[wheels] ok (cached)  ${local_name}"
    continue
  fi
  echo "[wheels] downloading  ${local_name}"
  curl -fL --retry 3 --progress-bar -o "${dest}.part" "${BASE_URL}/${local_name//+/%2B}"
  echo "${sha}  ${dest}.part" | sha256sum --check --quiet
  mv "${dest}.part" "${dest}"
done
echo "[wheels] all $(( ${#WHEELS[@]} )) wheels present in ${WHEEL_DIR}"
