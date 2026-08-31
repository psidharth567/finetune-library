#!/usr/bin/env bash
# Install flash-linear-attention for Qwen 3.5 GDN fast path.
set -euo pipefail

export HF_HOME="${HF_HOME:-/projects/data/llmteam/sidharth/toolkit/grpo-library}"

if python3 -c "from transformers.utils.import_utils import is_flash_linear_attention_available; import sys; sys.exit(0 if is_flash_linear_attention_available() else 1)" 2>/dev/null; then
  echo "flash linear attention already available"
  exit 0
fi

python3 -m pip install -q fla-core flash-linear-attention einops
python3 -m pip install -q --no-deps "triton>=3.7.1" || true
python3 -c "
from transformers.utils.import_utils import is_flash_linear_attention_available, is_causal_conv1d_available
assert is_causal_conv1d_available(), 'causal_conv1d required for Qwen3.5 GDN'
assert is_flash_linear_attention_available(), 'fla install failed'
print('fla ok')
"

echo "Tip: export FLA_SKIP_TRITON_AUTOTUNE=1 and PYTHONSTARTUP=\$(pwd)/scripts/patch_fla_triton_autotune.py for EP+FSDP smokes"
