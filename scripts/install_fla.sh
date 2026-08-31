#!/usr/bin/env bash
# Install flash-linear-attention for Qwen 3.5 GDN fast path.
# On Hopper, use TileLang for GDN backward (FLA #640) and keep Triton 3.6.0.
set -euo pipefail

export HF_HOME="${HF_HOME:-/projects/data/llmteam/sidharth/toolkit/grpo-library}"

if python3 -c "from transformers.utils.import_utils import is_flash_linear_attention_available; import sys; sys.exit(0 if is_flash_linear_attention_available() else 1)" 2>/dev/null; then
  python3 -c "
import triton
from packaging import version
assert version.parse(triton.__version__) < version.parse('3.7.0'), (
    f'expected base Triton <3.7 (got {triton.__version__}); reinstall image or avoid triton upgrade'
)
import tilelang  # noqa: F401
print('fla already available (triton', triton.__version__ + ', tilelang ok)')
"
  exit 0
fi

python3 -m pip install -q fla-core flash-linear-attention einops
# mamba-ssm pins tilelang==0.1.8 which fails to import here; 0.1.13 works on Hopper.
python3 -m pip install -q "tilelang==0.1.13"

python3 -c "
import triton
import torch
from packaging import version
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

assert is_causal_conv1d_available(), 'causal_conv1d required for Qwen3.5 GDN'
assert is_flash_linear_attention_available(), 'fla install failed'
assert version.parse(triton.__version__) < version.parse('3.7.0'), (
    'Triton was upgraded; expected base 3.6.x with TileLang backend, got ' + triton.__version__
)
import tilelang  # noqa: F401

from fla.ops.gated_delta_rule import chunk_gated_delta_rule

B, H, L, D = 1, 4, 32, 64
q = torch.randn(B, L, H, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, L, H, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, L, H, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
g = torch.randn(B, L, H, device='cuda', dtype=torch.bfloat16, requires_grad=True)
beta = torch.rand(B, L, H, device='cuda', dtype=torch.bfloat16, requires_grad=True)
out, _ = chunk_gated_delta_rule(
    q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True,
)
out.sum().backward()
print('fla ok (triton', triton.__version__ + ', tilelang GDN backward ok)')
"

echo "Tip: export FLA_SKIP_TRITON_AUTOTUNE=1 for memory-tight EP+FSDP Qwen smokes"
