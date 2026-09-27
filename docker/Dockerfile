# Standalone finetune-library image.
#
# Build context is the repo root:
#   docker build -f docker/Dockerfile -t ghcr.io/psidharth567/finetune-library:latest .
# or via scripts/build.sh.
#
# All wheels this Dockerfile installs live under wheels/ (gitignored).
# scripts/fetch_wheels.sh downloads the exact set from the GitHub release;
# otherwise populate it with (cp312 / cu129 / torch2.11 variants only):
#   flash_attn-*-cp312-*.whl
#   flash_attn_3-*-abi3*.whl        (or cp312)
#   deep_ep-*-cp312-*.whl
#   causal_conv1d-*-cp312-*.whl
#   mamba_ssm-*-cp312-*.whl
# See docker/build_deepep_wheel.sh for building deep_ep from source.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
LABEL org.opencontainers.image.source="https://github.com/psidharth567/finetune-library" \
    org.opencontainers.image.title="finetune-library" \
    org.opencontainers.image.licenses="Apache-2.0" \
    org.opencontainers.image.description="Standalone finetune-library: LoRA CPT/SFT, torch2.11 cu129, FA2+FA3, FLA, Liger, DeepEP, sm_90a" \
    org.opencontainers.image.base.name="nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git git-lfs rsync \
    cuda-compat-12-8 \
    libibverbs-dev libibverbs1 ibverbs-providers libibumad3 \
    libfabric-dev libnl-3-dev libnl-route-3-dev \
    librdmacm1 libnuma1 hwloc numactl pciutils kmod \
    libtcmalloc-minimal4 \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so \
 && git lfs install || true

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH
RUN uv python install 3.12 && uv python install 3.11

ENV CUDA_HOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    NCCL_NVLS_ENABLE=1 \
    TORCH_CUDA_ARCH_LIST=9.0a \
    TOOLKIT_WEIGHTS=/weights \
    HF_HOME=/cache \
    HF_XET_CACHE=/cache/.cache/xet \
    HUGGINGFACE_HUB_CACHE=/cache/hub \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

# ALLOW_MISSING_WHEELS=0 (default): a missing local wheel for a kernel this
# Dockerfile is supposed to install is a hard build failure, except flash_attn
# which additionally tries a URL fallback before failing. Set to 1 only to
# intentionally build a reduced image (e.g. SDPA-only, no DeepEP).
ARG ALLOW_MISSING_WHEELS=0

WORKDIR /opt/toolkit

RUN uv venv --python 3.12 /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv

COPY pyproject.toml uv.lock ./finetune-library/
COPY src ./finetune-library/src
COPY README.md AGENTS.md ./finetune-library/
COPY configs ./finetune-library/configs
COPY scripts ./finetune-library/scripts
COPY tests ./finetune-library/tests
# Only the small smoke fixtures (data/dummy_*.jsonl) pass .dockerignore; the
# production configs point at data/dummy_cpt.jsonl, so bake them in.
COPY data ./finetune-library/data
COPY wheels/ /wheels/

# Host source permissions can be arbitrarily restrictive (e.g. 700 dirs, 750
# scripts) and are not preserved meaningfully by COPY anyway (owner becomes
# root). scripts/run.sh runs the container as the invoking host uid:gid by
# default, so normalize to world-readable/executable here rather than relying
# on host perms or forcing every caller to run as root.
RUN chmod -R a+rX /opt/toolkit/finetune-library && \
    find /opt/toolkit/finetune-library/scripts -name '*.sh' -exec chmod a+rx {} + && \
    # /root defaults to 700: `uv python install` puts the actual cpython
    # binary under /root/.local/share/uv/python/..., which /opt/venv/bin/python
    # symlinks to. A non-root --user can't traverse into /root at all
    # otherwise ("bad interpreter: Permission denied" running any venv binary).
    chmod 755 /root

ENV FINETUNE_LIBRARY_ROOT=/opt/toolkit/finetune-library

RUN uv pip install --upgrade pip setuptools wheel && \
    uv pip install \
      "torch==2.11.0" "torchvision==0.26.0" "torchao==0.17.0" \
      --index-url https://download.pytorch.org/whl/cu129 && \
    uv pip install "triton==3.6.0" && \
    uv pip install fla-core flash-linear-attention einops "tilelang==0.1.13" && \
    uv pip install \
      "transformers==5.5.0" "peft==0.19.1" "accelerate==1.13.0" "datasets==4.3.0" \
      "liger-kernel==0.8.1" "einops" "pydantic==2.13.4" "pyyaml==6.0.3" \
      "huggingface-hub==1.26.0" "safetensors==0.8.0"

# --- flash_attn (FA2): local wheel -> URL fallback -> fail unless ALLOW_MISSING_WHEELS=1 ---
RUN set -eu; \
    if ls /wheels/flash_attn-*.whl >/dev/null 2>&1; then \
      uv pip install --no-deps --no-build-isolation /wheels/flash_attn-*.whl; \
    elif uv pip install --no-deps --no-build-isolation \
          "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"; then \
      echo "flash_attn: installed from URL fallback"; \
    elif [ "${ALLOW_MISSING_WHEELS}" = "1" ]; then \
      echo "WARN: flash_attn wheel missing and URL fallback failed; ALLOW_MISSING_WHEELS=1, continuing with SDPA fallback"; \
    else \
      echo "FATAL: flash_attn wheel missing under /wheels and URL fallback failed. Put flash_attn-*cp312*.whl in finetune-library/wheels/ or set ALLOW_MISSING_WHEELS=1." >&2; \
      exit 1; \
    fi

# --- flash_attn_3 (FA3): local wheel required unless ALLOW_MISSING_WHEELS=1 ---
RUN set -eu; \
    if ls /wheels/flash_attn_3-*.whl >/dev/null 2>&1; then \
      uv pip install --no-deps /wheels/flash_attn_3-*.whl; \
    elif [ "${ALLOW_MISSING_WHEELS}" = "1" ]; then \
      echo "WARN: flash_attn_3 wheel missing; ALLOW_MISSING_WHEELS=1, FA3 disabled"; \
    else \
      echo "FATAL: flash_attn_3 wheel missing under /wheels. Put flash_attn_3-*.whl in finetune-library/wheels/ or set ALLOW_MISSING_WHEELS=1." >&2; \
      exit 1; \
    fi

# --- mamba_ssm + causal_conv1d (Qwen3.5 GDN deps) ---
RUN set -eu; \
    for pkg in mamba_ssm causal_conv1d; do \
      if ls /wheels/${pkg}-*.whl >/dev/null 2>&1; then \
        uv pip install --no-deps /wheels/${pkg}-*.whl; \
      elif [ "${ALLOW_MISSING_WHEELS}" = "1" ]; then \
        echo "WARN: ${pkg} wheel missing; ALLOW_MISSING_WHEELS=1, continuing without it"; \
      else \
        echo "FATAL: ${pkg} wheel missing under /wheels. Put ${pkg}-*cp312*.whl in finetune-library/wheels/ or set ALLOW_MISSING_WHEELS=1." >&2; \
        exit 1; \
      fi; \
    done

# --- deep_ep (MoE expert-parallel dispatch) ---
RUN set -eu; \
    if ls /wheels/deep_ep-*.whl >/dev/null 2>&1; then \
      uv pip install --no-deps /wheels/deep_ep-*.whl; \
    elif [ "${ALLOW_MISSING_WHEELS}" = "1" ]; then \
      echo "WARN: deep_ep wheel missing; ALLOW_MISSING_WHEELS=1, continuing without it"; \
    else \
      echo "FATAL: deep_ep wheel missing under /wheels. Build it with finetune-library/docker/build_deepep_wheel.sh or set ALLOW_MISSING_WHEELS=1." >&2; \
      exit 1; \
    fi

# --index-url (rather than --extra-index-url) here previously made uv resolve
# unsloth's *entire* dependency tree (xformers, trl, diffusers, bitsandbytes,
# sentencepiece, ...) against the PyTorch cu129 wheel index alone, which does
# not host those packages -> "No solution found when resolving dependencies"
# on every build, silently swallowed by the `|| echo` fallback so the image
# built successfully with unsloth simply absent. --extra-index-url keeps PyPI
# as the primary index (those deps resolve normally) while still exposing
# cu129 for anything torch-related unsloth might request.
# unsloth 2026.7.6 crashes at import under torch 2.11 ("Artifact of type=inductor
# already registered"), so it is opt-in: --build-arg INSTALL_UNSLOTH=1.
ARG INSTALL_UNSLOTH=0
RUN if [ "${INSTALL_UNSLOTH}" = "1" ]; then \
      uv pip install "unsloth==2026.7.6" --extra-index-url https://download.pytorch.org/whl/cu129; \
    fi && \
    uv pip install "pytest==8.4.2" && \
    uv pip install -e ./finetune-library --no-deps || uv pip install -e ./finetune-library

# Full CPU-safe unit test suite (GPU-only tests self-skip via
# `pytest.mark.skipif(not torch.cuda.is_available(), ...)`).
RUN test -f finetune-library/scripts/production/launch-one-node.sh && \
    test -d finetune-library/configs/models && \
    test -d finetune-library/tests && \
    cd finetune-library && python -m pytest -q

RUN python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)" && \
    python -c "import importlib.util as u; mods=['flash_attn','flash_attn_3','liger_kernel','deep_ep','fla','tilelang']; missing=[m for m in mods if u.find_spec(m) is None]; print('kernels', {m: u.find_spec(m) is not None for m in mods}); print('WARN: missing (allowed only if ALLOW_MISSING_WHEELS=1):', missing) if missing else None" && \
    nvcc --version && ls -lh /usr/local/cuda/compat/ && ldconfig -p 2>/dev/null | grep -i compat || true

# /opt/venv's own files (site-packages, bin/*) are already world-readable —
# uv/pip install with a sane umask. The one blocker for a non-root --user is
# /root itself (700, from the base image), which the venv's python symlinks
# through (uv installs the interpreter under /root/.local/...). Re-assert
# this at the very end in case a later RUN somehow reset it; `chmod -R` over
# /opt/venv is deliberately avoided here — it would copy-up the entire
# (multi-GB) venv into this layer for a metadata-only change.
RUN chmod 755 /root

WORKDIR /opt/toolkit/finetune-library
CMD ["bash"]
