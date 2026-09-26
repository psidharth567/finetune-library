#!/usr/bin/env bash
# Build and install DeepEP into the active Python environment (finetune container venv).
set -euo pipefail

DEEPEP_COMMIT="${DEEPEP_COMMIT:-9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee}"
DEEPEP_DIR="${DEEPEP_DIR:-/tmp/DeepEP}"
WHEEL_DIR="${WHEEL_DIR:-/projects/data/llmteam/sidharth/toolkit/finetune-library/wheels}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
LOG_PATH="${LOG_PATH:-${WHEEL_DIR}/deepep-build.log}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export GDRCOPY_HOME="${GDRCOPY_HOME:-/usr/src/gdrdrv-2.5.1/}"
export HF_HOME="${HF_HOME:-/projects/data/llmteam/sidharth/toolkit/finetune-library/.cache/huggingface}"

if python3 -c "import deep_ep" >/dev/null 2>&1; then
  echo "deep_ep already installed: $(python3 -c 'import deep_ep; print(deep_ep.__file__)')"
  exit 0
fi

if command -v apt-get >/dev/null 2>&1 && [ ! -f /usr/include/infiniband/mlx5dv.h ]; then
  if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    sudo apt-get update -qq || true
    sudo apt-get install -y --no-install-recommends \
      build-essential cmake git curl wget \
      libibverbs-dev libibverbs1 ibverbs-providers libibumad3 \
      librdmacm1 libnl-3-200 libnl-route-3-200 libfabric-dev \
      rdma-core infiniband-diags perftest || true
  else
    echo "WARN: mlx5dv.h missing and passwordless sudo unavailable; continuing"
  fi
  ARCH="$(uname -m)"
  LIB_PATH="/usr/lib/${ARCH}-linux-gnu"
  if [ -e "${LIB_PATH}/libmlx5.so.1" ] && [ ! -e "${LIB_PATH}/libmlx5.so" ]; then
    if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
      sudo ln -sf "${LIB_PATH}/libmlx5.so.1" "${LIB_PATH}/libmlx5.so"
    else
      ln -sf "${LIB_PATH}/libmlx5.so.1" "${LIB_PATH}/libmlx5.so" 2>/dev/null || true
    fi
  fi
fi

if ! python3 -m pip --version >/dev/null 2>&1; then
  python3 -m ensurepip --upgrade || true
fi

python3 -m pip install -q "nvidia-nvshmem-cu12" || python3 -m pip install -q "nvidia-nvshmem-cu13" || true

rm -rf "${DEEPEP_DIR}"
git clone https://github.com/deepseek-ai/DeepEP.git "${DEEPEP_DIR}"
git -C "${DEEPEP_DIR}" checkout "${DEEPEP_COMMIT}"
sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' "${DEEPEP_DIR}/csrc/kernels/configs.cuh"
sed -i 's/#define NUM_TIMEOUT_CYCLES 200000000000ull/#define NUM_TIMEOUT_CYCLES 2000000000000ull/' "${DEEPEP_DIR}/csrc/kernels/configs.cuh"

mkdir -p "${WHEEL_DIR}"
cd "${DEEPEP_DIR}"
{
  echo "Building DeepEP @ ${DEEPEP_COMMIT} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    python3 -m pip install --no-build-isolation . 2>&1
  python3 -c "import deep_ep; print('deep_ep ok', deep_ep.__file__)"
  python3 setup.py bdist_wheel -d "${WHEEL_DIR}"
  ls -lh "${WHEEL_DIR}"/deep_ep*.whl
} | tee "${LOG_PATH}"
