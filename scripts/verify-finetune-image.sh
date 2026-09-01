#!/usr/bin/env bash
# Verify the finetune container has the expected kernel stack and baked library tree.
set -euo pipefail

ROOT="${FINETUNE_LIBRARY_ROOT:-/opt/toolkit/finetune-library}"

for path in \
  "${ROOT}/scripts/production/launch-one-node.sh" \
  "${ROOT}/configs/models/qwen35-35b-a3b-cpt.yaml" \
  "${ROOT}/tests/test_config_registry.py"; do
  if [[ ! -e "${path}" ]]; then
    echo "MISSING baked path: ${path}"
    exit 1
  fi
  echo "OK      ${path#${ROOT}/}"
done

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

cd "${ROOT}"
python -m pytest tests/test_config_registry.py \
  tests/test_liger_compile_guard.py \
  tests/test_moe_ep_unit.py -q
echo "unit tests ok"
