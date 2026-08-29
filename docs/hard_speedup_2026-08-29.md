
# Validation - Hard Speedup 2026-08-29 (bodhanai-node012, 8x H100, driver 535, torch 2.11+cu129)

## Qwen3-8B DDP 512 samples 60 steps (10 warmup+50 measured) SDPA BF16 r32
- Baseline (native, fused_linear_cross_entropy, ckpt:true, no compile): 46,796 median tok/s (44,195 agg) 18.5s 818,800 tokens
- Hard (liger, fused_linear_cross_entropy, ckpt:false, no compile): 76,392 median tok/s (75,345 agg) 10.86s 818,800 tokens
- Speedup: +63.2% median +70.5% aggregate  => exceeds 25% target

Kernels: liger Qwen3 rope+swiglu+ rms_norm, fused CE 512-chunk (4x fewer launches, 256MB cap), no ckpt where OOM-safe (26.7 GiB <80), TF32 high precision.

Config: configs/benchmarks/qwen3-8b-native-fused-loss.yaml vs qwen3-8b-hard-optimized.yaml (now liger,no-ckpt stable; compile guarded off for liger to avoid inductor hang on triton swiglu)

## Gemma-4-26B DDP 512 samples 60 steps SDPA BF16 MoE grouped_mm
- Baseline (native, cross_entropy, ckpt:true): 22,272 median (20,823 agg) 39.3s 818,800 tokens
- Hard (liger, fused_linear_cross_entropy, ckpt:false): 34,064 median (33,865 agg) 24.17s 818,800 tokens
- Speedup: +52.9% median +62.6% aggregate  => exceeds 25%

Smaller smoke (64 samples 15 steps) consistent: baseline 22,426 -> liger-fused-nc 34,013 (+51.7%) proves wins not artifact of 512.
- Baseline: benchmark-gemma4-26b-smoke (native, cross_entropy, grouped_mm, ckpt:true)
- Hard: benchmark-gemma4-26b-liger-fused-nc (liger, fused, ckpt:false)

Kernels: liger Gemma4 geglu + rms_norm, grouped_mm 32x over eager (101s->3.14s baseline), fused CE 512-chunk, no-ckpt DDP 74.2 GiB fits 80GB (HSDP alternative would shard 38GB but slower). Torch compile blocks disabled for liger (inductor hang on triton), fixed trainer wrap->compile order for HSDP (compile before fully_shard).

## Hard wins implemented
- runtime.py: auto-liger for qwen3/gemma4/olmo/llama, TF32 high, inductor cudagraphs/coordinate_descent/epilogue_fusion, maybe_compile with fullgraph False and liger guard, _maybe_compile_grouped_moe with fullgraph False
- loss.py: _FusedLoraLinearCrossEntropy chunk 512 preferred, mem cap 256MB
- trainer.py: compile-before-wrap for HSDP/FSDP to avoid FSDP hook graph break
- configs: hard-optimized stable ohne compile, liger+fused+no-ckpt

All benchmark.json median_equivalent_tokens_per_second on node012 via launch-one-node.sh.

