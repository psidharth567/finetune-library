#!/usr/bin/env bash
# Verify the finetune container has the expected kernel stack.
set -euo pipefail

python - <<'PY'
import importlib.util as u

checks = {
    "torch": "torch",
    "triton": "triton",
    "transformers": "transformers",
    "peft": "peft",
    "liger_kernel": "liger_kernel",
    "flash_attn (FA2)": "flash_attn",
    "flash_attn_3 (FA3)": "flash_attn_3",
    "deep_ep": "deep_ep",
    "fla": "fla",
    "tilelang": "tilelang",
}
missing = []
for label, mod in checks.items():
    ok = u.find_spec(mod) is not None
    print(f"{'OK' if ok else 'MISSING':7s} {label}")
    if not ok:
        missing.append(label)

import torch
print(f"torch {torch.__version__} cuda {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"gpus {torch.cuda.device_count()}")

if missing:
    raise SystemExit(f"kernel check failed: {', '.join(missing)}")
print("all kernels present")
PY

finetune-lib --help >/dev/null
echo "finetune-lib CLI ok"
