# Features needed

Tracked improvements for `finetune-library`. Not implemented yet.

## Sequence packing and length handling

### Attention-isolated packing

**Problem:** With `data.packing: true`, multiple tokenized examples are concatenated into one
causal sequence. The model can attend across former document/example boundaries. EOS tokens
are soft separators only; they do not block cross-document attention.

**Needed:**
- Pack multiple examples into one `max_seq_length` window without cross-example attention.
- Typical approaches: document-boundary attention masks, `position_ids` resets, or
  sample-index / segment-id masks compatible with SDPA / fused loss.
- Preserve current label masking behavior for SFT (prompt tokens stay `-100`).

**Config sketch:**
```yaml
data:
  packing: true
  packing_isolation: attention  # none | attention
```

---

### Sliding-window chunking for long documents

**Problem:** A single example longer than `max_seq_length` is head-truncated when
`packing: false`. Tail content is discarded.

**Needed:**
- Split one long example into multiple training rows (overlapping or non-overlapping windows).
- Apply to CPT `text` and long SFT examples.
- Carry labels correctly per chunk (CPT: all tokens; SFT: preserve prompt/completion mask
  boundaries where possible).

**Config sketch:**
```yaml
data:
  chunk_long_examples: true
  chunk_overlap: 0          # or e.g. 128 for CPT
  chunk_strategy: truncate  # truncate | sliding_window
```

---

### Per-example or per-run custom sequence lengths

**Problem:** One global `training.max_seq_length` applies to every example. No way to
request different lengths per dataset, task type, or example metadata.

**Needed:**
- Optional per-example `max_length` field in raw / tokenized data.
- Optional dataset-level override in config.
- Cache key and prepared-data identity must include length policy.

**Config sketch:**
```yaml
training:
  max_seq_length: 2048
data:
  length_policy: global     # global | per_example | per_dataset
  default_max_length: 2048
```

---

### `require_full_seq_length`

**Problem:** The legacy `finetune` stack supported dropping examples shorter than
`max_seq_length`. This toolkit always keeps short examples and pads at batch time.

**Needed:**
- Option to skip examples shorter than `max_seq_length` during data prep.
- Useful for fixed-shape throughput experiments or kernels that prefer full-length rows.

**Config sketch:**
```yaml
data:
  require_full_seq_length: false
```

---

## Training defaults and ergonomics

### Per-model learning-rate presets

**Problem:** All registry models default to `learning_rate: 2e-4`. No model-family or
task-type (CPT vs SFT) guidance in the registry or `finetune-lib init`.

**Needed:**
- Registry fields for recommended CPT/SFT learning rates.
- `finetune-lib init` and AGENTS.md should use them when scaffolding configs.
- Still overridable in YAML.

**Config sketch:**
```yaml
# registry metadata (not user YAML)
recommended_lr:
  cpt: 2.0e-4
  sft: 1.0e-4
```

---

### `torch.compile` qualification per model

**Problem:** `runtime.torch_compile` exists but is off by default. Full-model compile was
rejected after benchmarks due to per-rank autotuning overhead.

**Needed:**
- Re-benchmark compile per model profile (possibly compile only loss head or decoder blocks).
- Promote to checked-in configs only after parity + throughput gates pass.

**Config sketch:**
```yaml
runtime:
  torch_compile: false
  compile_scope: full       # full | loss_only | blocks
```

---

## Observability

### Packing / length stats in prepare-data output

**Problem:** Hard to see how many examples were truncated, packed, or dropped without
reading code or manually inspecting prepared data.

**Needed:**
- `prepare-data` summary: row counts, truncated count, packed windows, remainder dropped,
  token utilization (% of `max_seq_length` filled on average).
- Write summary next to prepared dataset metadata.

---

## Priority suggestion

| Priority | Feature | Why |
|---|---|---|
| P0 | Attention-isolated packing | Correctness for CPT packing at scale |
| P0 | Sliding-window chunking | Avoid losing long-document tail |
| P1 | `require_full_seq_length` | Simple; existed in legacy stack |
| P1 | Per-model LR presets | Better defaults for agents and `init` |
| P2 | Per-example custom lengths | Niche; global length covers most cases |
| P2 | `torch.compile` re-qualification | Perf win if scoped compile works |
| P3 | Prepare-data length/packing stats | Debugging and agent visibility |
