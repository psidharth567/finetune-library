#!/usr/bin/env bash
# Build a DeepEP wheel using the standalone finetune image (needs docker + GPU optional).
# Writes into finetune-library/wheels/ (gitignored) so docker/Dockerfile can COPY it in.
set -euo pipefail

FINETUNE_LIBRARY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL_DIR="${WHEEL_DIR:-${FINETUNE_LIBRARY_ROOT}/wheels}"
IMAGE="${FINETUNE_IMAGE:-toolkit/finetune:latest}"
DEEPEP_COMMIT="${DEEPEP_COMMIT:-9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee}"
HF_HOME="${HF_HOME:-${FINETUNE_LIBRARY_ROOT}/.cache/huggingface}"

mkdir -p "${WHEEL_DIR}"

docker run --rm --gpus all --ipc=host \
  -v "${FINETUNE_LIBRARY_ROOT}:/workspace/finetune-library" \
  -e HF_HOME=/workspace/finetune-library/.cache/huggingface \
  -e DEEPEP_COMMIT="${DEEPEP_COMMIT}" \
  -e WHEEL_DIR=/workspace/finetune-library/wheels \
  -e TORCH_CUDA_ARCH_LIST=9.0a \
  --entrypoint bash "${IMAGE}" -c '
set -euo pipefail
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
bash /workspace/finetune-library/scripts/install_deepep.sh
'

echo "Wheel(s):"
ls -lh "${WHEEL_DIR}"/deep_ep*.whl
