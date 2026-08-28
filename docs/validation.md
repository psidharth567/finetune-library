# H100 validation

Validation was run on `bodhanai-node006`, `bodhanai-node024`, and
`bodhanai-node025`, each with 8×H100 80 GiB GPUs, using the versions frozen
in `uv.lock`.

## What the production backend is

The correctness reference is native PyTorch 2.11, Transformers 5.5, PEFT
0.19, SDPA, ordinary model cross-entropy, eager expert LoRA, and standard
`torch.optim.AdamW`. There is one trainer and one data/checkpoint path for CPT
and SFT.

Optimizations are individually gated against that reference:

- Qwen3-8B and Qwen3-14B: Unsloth model kernels and the exact LoRA-aware
  chunked LM-head cross-entropy in this package.
- Qwen3-32B, OLMo-3-32B, and Gemma-4-31B: native Transformers model kernels, the exact
  LoRA-aware chunked LM-head cross-entropy, DDP, and fused stock AdamW.
- Gemma-4-26B-A4B-it: native Transformers plus grouped expert LoRA using
  `torch._grouped_mm`.
- DeepSeek-R1-Distill-Llama-8B: native Transformers/PEFT and DDP.

Model weights, forward computation, adapters, gradients, and AdamW moments are
BF16. AdamW is not wrapped and has no master-parameter layer; PyTorch retains
its ordinary FP32 step counter and saves its standard state directly. The
optional 8-bit optimizer is never selected implicitly.

## Kernel and runtime benchmarks

The short qualification comparisons below used sequence length 2048, 10
warm-up steps, and 50 measured optimizer steps. Rates count global non-padding
loss tokens. Losses and gradient norms remained finite.

### Long stability gates

Every selected production profile then ran for at least 1,800 measured GPU
step-seconds at sequence length 2048, excluding loading, warm-up, data time,
and metric pauses. All used standard AdamW and finished with finite, nonzero
loss and gradients.

| Model/profile | Measured seconds | Median tokens/s | Reserved GiB | Final loss / grad norm |
|---|---:|---:|---:|---:|
| Qwen3-8B, Unsloth DDP | 3,788 | 79,929 | 27.40 | 0.001801 / 0.01996 |
| Qwen3-14B, Unsloth DDP | 3,725 | 53,308 | 45.47 | 0.001710 / 0.01711 |
| DeepSeek-R1-Llama-8B, native DDP | 1,841 | 80,305 | 32.58 | 0.001995 / 0.06683 |
| Gemma-4-26B, grouped native DDP | 1,900 | 23,130 | 60.52 | 0.006096 / 0.03092 |
| Qwen3-32B, native DDP | 2,106 | 13,581 | 65.61 | 0.004623 / 0.05687 |
| OLMo-3-32B, native DDP | 1,969 | 14,609 | 64.57 | 0.004989 / 0.02636 |
| Gemma-4-31B, native DDP | 1,911 | 11,133 | 63.87 | 0.006350 / 0.11963 |

These are repeated-data stability runs, not model-quality evaluations; the
late-run losses are expected to be low.

### Qwen3-8B, DDP, batch one per GPU

| Runtime | Attention/loss | Median tokens/s | Reserved GiB |
|---|---|---:|---:|
| Native | SDPA, ordinary CE | 68,587 | 35.54 |
| Native | SDPA, chunked fused CE | 63,695 | 29.42 |
| Unsloth | exact chunked fused CE | **80,331** | **27.40** |

The promoted Unsloth path is 17.1% faster than the equal-batch native
reference. A larger native batch can reach 77,502 tokens/s, but it processes
four times as many tokens per optimizer step and reserves 69.81 GiB, so it is
not the equal-token comparison.

With native gradient checkpointing and ordinary CE, SDPA measured 50,702
tokens/s versus 45,813 for the installed external FlashAttention 2 wheel.
SDPA therefore remains the H100 default. The fused CE kernel was still kept
because it substantially reduces memory and is useful to the winning Unsloth
path.

### DeepSeek-R1-Distill-Llama-8B, DDP, batch one per GPU

| Runtime | Median tokens/s | Reserved GiB |
|---|---:|---:|
| Native SDPA/ordinary CE | **79,945** | 32.58 |
| Unsloth/exact fused CE | 40,101 | 26.27 |

This model remains native; backend selection is empirical rather than based on
the model family.

### Qwen3-14B, DDP, batch one per GPU

| Runtime | Median tokens/s | Reserved GiB |
|---|---:|---:|
| Native SDPA/ordinary CE | 20,213 | 54.02 |
| Unsloth/exact fused CE | **53,863** | **45.47** |

The Unsloth profile is 2.66× faster at the same global tokens per optimizer
step and is promoted for Qwen3-14B.

### Gemma-4-26B-A4B-it

The traditional eager-expert implementation and grouped implementation were
run from the same packed-text batch:

- eager: loss 6.28125, gradient norm 90.6501, 101.08 seconds
- grouped: loss 6.375, gradient norm 90.5899, 3.143 seconds
- relative loss difference 1.49%; relative gradient-norm difference 0.07%

The grouped kernel avoids materializing a huge 3-D expert delta. Logical expert
rank remains 4; a zero-padded rank-8 kernel view is used only to meet Hopper
grouped-GEMM alignment. The grouped boundary explicitly converts a possible
FP32 Gemma residual stream to the BF16 expert-weight dtype because
`torch._grouped_mm` is not an autocast operation.

The final layout comparison used equal sequence length and measured global
non-padding tokens:

| Layout | Global tokens/step | Median tokens/s | Reserved GiB |
|---|---:|---:|---:|
| DDP × 8 | 16,376 | **22,838** | 60.52 |
| 4 replicas × 2 shards, batch 2 | 16,376 | 11,049 | 60.88 |
| 4 replicas × 2 shards, batch 1 | 8,188 | 8,034 | 52.88 |
| 2 replicas × 4 shards, accumulation 2 | 8,188 | 4,160 | 52.53 |

DDP is the production path. Its 60-step run had finite loss and gradients and
remained more than 15 GiB below the memory gate.

### Dense 31B/32B topology and kernel comparisons

The exact chunked LM-head loss avoids retaining the full FP32 vocabulary
logits. That memory reduction lets all three dense 31B/32B checkpoints use a
full BF16 replica on each H100. DDP then avoids FSDP2's decoder-layer parameter
all-gathers and lets ordinary PyTorch AdamW use its fused CUDA implementation.

The first gate used 10 warm-up and 10 measured steps. Every topology row below
processes 16,376 global non-padding tokens per optimizer step:

| Model | Native 2-way HSDP | Native DDP | Alternate kernels |
|---|---:|---:|---:|
| Qwen3-32B | 6,369 tok/s, 43.85 GiB | 15,165 tok/s, 65.44 GiB | **Unsloth DDP: 17,269 tok/s, 64.07 GiB** |
| OLMo-3-32B-Think-DPO | 4,564 tok/s, 41.19 GiB | **15,509 tok/s, 64.31 GiB** | Liger DDP: 10,034 tok/s, 64.17 GiB |
| Gemma-4-31B-it | 5,110 tok/s, 45.92 GiB | **11,091 tok/s, 63.95 GiB** | Liger DDP: 9,980 tok/s, 63.38 GiB |

The leading short-run candidates then completed the standard
10-warm-up/50-measured-step gate:

| Model | Candidate runtime | Median tokens/s | Aggregate tokens/s | Reserved GiB |
|---|---|---:|---:|---:|
| Qwen3-32B | Unsloth DDP, SDPA/fused loss | 16,403 | 16,193 | 64.07 |
| OLMo-3-32B-Think-DPO | Native DDP, SDPA/fused loss | **15,489** | 14,588 | 64.57 |
| Gemma-4-31B-it | Native DDP, SDPA/fused loss | **7,350** | 6,877 | 63.87 |

The 50-step Qwen sample was misleading for sustained training. Unsloth's
offloaded-gradient path slowed after repeated epochs, so native DDP was run for
the full stability horizon:

| Qwen3-32B runtime | Measured seconds | Median tokens/s | Aggregate tokens/s | Reserved GiB |
|---|---:|---:|---:|---:|
| Unsloth DDP | 4,603 | 6,391 | 6,404 | 64.07 |
| **Native DDP** | 2,106 | **13,581** | **12,832** | 65.61 |
| Earlier native 2-way HSDP | 1,872 | 5,625 | 5,600 | 47.46 |

Native DDP is therefore the Qwen3-32B production backend: 2.12× the sustained
Unsloth median and 2.41× the earlier long HSDP median. This is why backend
promotion uses the long gate rather than the warm benchmark alone.

All selected DDP paths retain at least 10 GiB below the 76 GiB gate. Across
the completed long gates, OLMo DDP is 2.54× faster than its earlier HSDP
profile and Gemma DDP is 2.45× faster. The final production medians are
13,581 tokens/s for Qwen3-32B, 14,609 for OLMo-32B, and 11,133 for
Gemma-4-31B.

Liger is not promoted. It reduced OLMo throughput by 35% with effectively no
memory saving. Gemma's Liger run was 10% slower in the equal-step gate and
materially changed the loss/gradient trajectory, failing numerical parity.
The installed prebuilt FlashAttention-2 wheel reduced OLMo throughput from
15,489 to 6,069 tokens/s at the same memory, and Gemma cannot use it because
its head dimension exceeds the kernel's 256-element limit. A prebuilt
FlashAttention-3 Hub binary is available for this Hopper/Torch ABI, but
Transformers 5.5's attention integration is incompatible with the current
versioned Kernel Hub interface. It is not enabled through a source patch.

Qwen3-32B without activation checkpointing exceeded the memory gate. Unsloth
with checkpointing failed its FSDP2 backward because its storage/offload patch
is incompatible with sharded DTensors, so Unsloth is qualified only with DDP.
Full-model `torch.compile` was also rejected because it incurred excessive
per-rank shape autotuning.

## Seven-model stock-AdamW smoke

Each row is a real eight-GPU forward, backward, gradient clip, and ordinary
AdamW update. This short sequence-64 smoke is a correctness/memory check, not a
throughput comparison.

| Model | Layout | BF16 LoRA parameters | Loss | Grad norm | Alloc./reserved GiB |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | DDP×8 | 97,280,000 | 3.7270 | 44.812 | 16.06/16.28 |
| Qwen3-14B | DDP×8 | 138,502,144 | 3.5176 | 10.829 | 28.64/28.66 |
| Qwen3-32B | 4 replicas × 2 shards | 278,487,040 | 3.4648 | 16.817 | 38.47/40.68 |
| DeepSeek-R1-Distill-Llama-8B | DDP×8 | 92,356,608 | 5.1065 | 198.123 | 15.72/15.89 |
| OLMo-3-32B-Think-DPO | 4 replicas × 2 shards | 275,180,928 | 4.0430 | 23.101 | 36.00/38.77 |
| Gemma-4-26B-A4B-it | DDP×8 | 163,484,160 | 6.8618 | 67.092 | 48.27/48.52 |
| Gemma-4-31B-it | 4 replicas × 2 shards | 261,980,160 | 10.1719 | 19.719 | 35.93/38.41 |

Every base parameter remained frozen. The startup audit records every target,
rank, logical shape, parameter count, and dtype. It aborts if a required
linear, embedding, head, routed expert, shared expert, or router target is
missing.

## Numerical and distributed gates

- Traditional and fused LM-head losses, hidden gradients, LoRA-A gradients,
  and LoRA-B gradients pass a CUDA parity test.
- Gemma eager and grouped expert forward/backward paths pass fixed-routing
  CUDA output/gradient parity. The end-to-end MoE test separately checks loss,
  finite adapter gradients, and global gradient-norm parity; it does not demand
  per-parameter identity after tiny BF16 changes cross a discrete top-k router.
- Gemma routed and shared experts audit at rank 4; router, attention,
  non-expert MLP, embedding, and head audit at rank 32.
- Gemma's embedding/head adapter storage tie survives BF16 injection. Gradients
  from both uses are consolidated before the unchanged AdamW step.
- A two-GPU FSDP2/HSDP test passes mixed sharded and replicated tied-LoRA
  parameters through forward, backward, clipping, and AdamW.
- An independent 16-GPU/two-node HSDP forward/backward/clip/AdamW smoke passed
  with two-GPU shard groups kept within each NVSwitch node and eight replicas
  distributed across the nodes. It also saved and reloaded the distributed
  adapter, rank-local AdamW/scheduler/RNG state, and exact data position
  (`step=1`, `epoch=0`, `batch_in_epoch=1`).
- CPT text/pretokenized packing and SFT messages, prompt/completion, Alpaca,
  assistant-only masking, and scheduler behavior are covered by unit tests.
- The production Qwen3-8B Unsloth path completed a real two-step, eight-GPU SFT
  messages run with assistant-only loss and saved a standard BF16 adapter.

## Checkpoint, exact resume, and merge

A full Qwen3-8B DDP checkpoint was saved after a real optimizer step:

- standard BF16 PEFT adapter: 186 MiB
- standard rank-local AdamW/scheduler/RNG/data state: 372 MiB per rank
- exact pinned model revision and resolved strict configuration

Resuming the step-1 checkpoint and running step 2 produced loss
3.3909406662. An uninterrupted two-step run produced the identical step-2
loss.

The generic merge wrote four safetensor shards and compared merged logits with
the adapter-loaded model. Maximum absolute error was 0.203125 (scale-aware
BF16 tolerance 0.495), and relative RMS error was 1.149%, below the 2% gate.

The promoted Qwen3-8B Unsloth profile separately passed an eight-GPU
forward/backward/AdamW/checkpoint run with all 254 audited targets active.

The promoted Gemma-4-26B DDP profile also saved a real checkpoint after its
grouped-expert update. Its 327,048,304-byte adapter contains all 594 expected
tensors, all BF16, covering 163,484,160 parameters; ordinary rank-local AdamW
state is approximately 621 MB per rank.

The legacy `finetune` stack was archived to `../archive/finetune` on 2026-08-25.
