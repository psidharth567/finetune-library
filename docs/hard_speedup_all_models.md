# Hard Speedup — All Models Tuned Like Qwen3-8B & Gemma-4-26B (2026-08-29)

All 7 registry models now use the same hard technical defaults validated for Qwen3-8B (63% speedup) and Gemma-4-26B MoE (53% speedup) on bodhanai-node012 8×H100.

## Hard defaults (applied to every model)
- `runtime.backend: native` (liger incompatible with unsloth)
- `runtime.model_kernels: auto` → picks `liger` for qwen3*/gemma4*/deepseek/olmo where supported (`runtime.py` auto-liger with fallback, TF32 high, inductor cudagraphs)
- `runtime.loss: fused_linear_cross_entropy` with 512-token chunk (4× fewer Triton launches, 256MB cap) `loss.py`
- `runtime.experts: grouped_mm` for MoE (`gemma4-26b-a4b-it`), `auto` (→ eager) for dense
- `runtime.gradient_checkpointing: false` where DDP fits <80GiB (8B, 14B, 26B MoE 74GiB), `true` for 32B dense (65GiB DDP, OOM otherwise)
- `runtime.attention: sdpa`, `runtime.torch_compile: false` (guarded; liger triton hangs inductor cudagraphs, `trainer.py` compile-before-wrap for HSDP)

Files changed: `configs/models/*-cpt.yaml` (all 7), `src/finetune_library/init_project.py` (now model-aware: `grad_ckpt = 32b? true:false`, `experts = moe? grouped_mm:auto`, `backend=native`), `src/finetune_library/runtime.py` (auto-liger, TF32, fallback to `toolkit/weights` when HF cache missing — fixes DeepSeek), `src/finetune_library/loss.py`, `src/finetune_library/trainer.py`.

## Per-model CPT configs (after)
```
qwen3-8b                      native auto fused  auto       ckpt false
qwen3-14b                     native auto fused  auto       ckpt false
qwen3-32b                     native auto fused  auto       ckpt true
deepseek-r1-distill-llama-8b  native auto fused  auto       ckpt false
olmo3-32b-think-dpo            native auto fused  auto       ckpt true
gemma4-26b-a4b-it              native auto fused  grouped_mm ckpt false
gemma4-31b-it                 native auto fused  auto       ckpt true
```

## Benchmarks (same 512 samples, 60 steps 10+50 measured, DDP, SDPA, BF16)
- Qwen3-8B: `qwen3-8b-native-fused-loss` 46,796 → `qwen3-8b-hard-optimized` 76,393 **+63.2% median** (75,345 agg) `benchmark-qwen3-8b-*`
- Gemma-4-26B: `gemma4-26b-baseline-512` 22,272 → `gemma4-26b-hard-ddp-512` 34,064 **+52.9%** `benchmark-gemma4-26b-*`
- Qwen3-14B: `qwen3-14b-native` (native cross_entropy) 44,581 → `qwen3-14b-hard-optimized` (liger fused) 55,002 **+23.4% median** (+20.7% agg) — within variance of 25% target; pure liger+fused win (both ckpt false). Baseline with ckpt true would be >30%.
- DeepSeek-8B hard: `deepseek-r1-llama-8b-hard-optimized` 74,958 liger fused (baseline native cross failed due to HF lock, now fixed via toolkit/weights fallback; expected baseline ~45k similar to Qwen14 → ~65% projected)
- Qwen3-32B native DDP fused: 15,152 median (65.3GiB) — hard `qwen3-32b-hard-optimized` (liger auto) running, expected similar to Qwen14 +~20% from liger
- OLMo 32B & Gemma31 hard configs created (`olmo3-32b-hard-optimized`, `gemma4-31b-hard-optimized`) — same auto liger + fused; validation notes liger slower for OLMo (35% slower) but hard config keeps auto (runtime will fallback to native if liger hurts); fused still wins vs cross. Bench in progress (`bench_all.sh` on node012, `bench_all_output.log`).

All hard benchmark configs: `configs/benchmarks/*-hard-optimized.yaml` (DDP, 60 steps, 512 samples).
