#!/usr/bin/env bash
#SBATCH --job-name=finetune-lib
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Required: absolute path to the experiment YAML.
CONFIG_PATH="${CONFIG_PATH:?set CONFIG_PATH to an experiment YAML}"

# Optional overrides.
# sbatch copies this script to a spool dir, so BASH_SOURCE is not the repo:
# default to the submit dir (run `sbatch` from the repo root) or set it explicitly.
FINETUNE_LIBRARY_ROOT="${FINETUNE_LIBRARY_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
COMMAND="${COMMAND:-train}"

export HF_HOME
export FINETUNE_LIBRARY_ROOT
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

mkdir -p "${PROJECT_ROOT}/logs"
cd "${FINETUNE_LIBRARY_ROOT}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

srun --ntasks=1 --gpus-per-task="${NPROC_PER_NODE}" \
  scripts/production/launch-one-node.sh "${CONFIG_PATH}" "${COMMAND}"
