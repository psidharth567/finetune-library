"""Real-model packed-SFT isolation check (the README's parity numbers).

Packs chat examples from data/dummy_sft.jsonl into 4k-token windows and, for
each example, randomizes every *other* example in its window: with isolation
the example's per-token loss must not change (exactly 0 on every model we
ship). The same perturbation without position_ids is the leaky control. With
``fp32`` it also compares packed vs each example run alone.

usage (inside the image, one GPU):
  torchrun --standalone --nproc-per-node 1 tests/multigpu/packed_sft_parity.py \
      configs/models/qwen3-8b-cpt.yaml sdpa [n_examples] [fp32]
"""
import json
import sys

import torch
import yaml

from finetune_library.config import DataConfig, ExperimentConfig, Task
from finetune_library.data import IGNORE_INDEX, CausalCollator, load_records, pack_whole_examples, tokenize_record
from finetune_library.distributed import initialize_distributed
from finetune_library.packed_linear_attention import patch_linear_attention_for_packing
from finetune_library.registry import resolve_model
from finetune_library.runtime import load_runtime

path, attention = sys.argv[1], sys.argv[2]
n_examples = int(sys.argv[3]) if len(sys.argv) > 3 else 40
raw = yaml.safe_load(open(path))
raw["task"] = "sft"
raw["data"] = {"format": "messages", "path": "data/dummy_sft.jsonl", "packing": True}
raw["distributed"] = {"strategy": "ddp"}
raw["runtime"]["attention"] = attention
raw["runtime"]["torch_compile"] = False
raw["training"]["max_seq_length"] = 4096
config = ExperimentConfig.model_validate(raw)
spec = resolve_model(config.model.name)
context = initialize_distributed(config.distributed, spec)
runtime = load_runtime(config, spec, context)
model = runtime.model.eval()
if "fp32" in sys.argv[4:]:
    model = model.float()
patched = patch_linear_attention_for_packing(model)

examples = []
for record in load_records(DataConfig(format="messages", path="data/dummy_sft.jsonl", max_samples=200)):
    row = tokenize_record(record, tokenizer=runtime.tokenizer, task=Task.SFT, data=config.data)
    if 32 <= len(row["input_ids"]) <= 1500:
        examples.append(row)
    if len(examples) == n_examples:
        break
windows = pack_whole_examples(examples, max_length=4096)
collate = CausalCollator(runtime.tokenizer, 4096, packing_isolation="attention")


def nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits[:-1].float(), labels[1:], ignore_index=IGNORE_INDEX, reduction="none")



generator = torch.Generator().manual_seed(0)
vocab = len(runtime.tokenizer)
res = {"iso": {"max_tok": 0.0, "max_ex_mean": 0.0}, "leaky": {"max_tok": 0.0, "max_ex_mean": 0.0}}
alone_vs_packed = {"max_tok": 0.0, "max_ex_mean": 0.0}
n_checked = 0
with torch.no_grad():
    for window in windows:
        batch = {k: v.cuda() for k, v in collate([window]).items()}
        ids, pos, labels = batch["input_ids"], batch["position_ids"], batch["labels"][0]
        starts = (pos[0] == 0).nonzero().flatten().tolist() + [ids.shape[1]]
        if len(starts) < 3:
            continue
        base_iso = nll(model(input_ids=ids, position_ids=pos, use_cache=False).logits[0], labels)
        base_leak = nll(model(input_ids=ids, use_cache=False).logits[0], labels)
        for a, b in zip(starts, starts[1:]):
            other = torch.ones_like(ids, dtype=torch.bool)
            other[:, a:b] = False
            noise = torch.randint(100, vocab - 100, ids.shape, generator=generator).cuda()
            perturbed = torch.where(other, noise, ids)
            iso = nll(model(input_ids=perturbed, position_ids=pos, use_cache=False).logits[0], labels)
            leak = nll(model(input_ids=perturbed, use_cache=False).logits[0], labels)
            mask = labels[a + 1 : b].ne(IGNORE_INDEX)
            for name, x, y in (("iso", iso, base_iso), ("leaky", leak, base_leak)):
                d = x[a : b - 1][mask] - y[a : b - 1][mask]
                res[name]["max_tok"] = max(res[name]["max_tok"], d.abs().max().item())
                res[name]["max_ex_mean"] = max(res[name]["max_ex_mean"], d.mean().abs().item())
            if "fp32" in sys.argv[4:]:
                alone = nll(model(input_ids=ids[:, a:b], use_cache=False).logits[0], labels[a:b])[mask]
                d = base_iso[a : b - 1][mask] - alone
                alone_vs_packed["max_tok"] = max(alone_vs_packed["max_tok"], d.abs().max().item())
                alone_vs_packed["max_ex_mean"] = max(alone_vs_packed["max_ex_mean"], d.mean().abs().item())
            n_checked += 1
out = {"model": config.model.name, "attention": runtime.attention, "dtype": str(next(model.parameters()).dtype),
       "gdn_patched": patched, "examples_perturbed": n_checked,
       "isolated: change when OTHER examples are randomized": {k: round(v, 5) for k, v in res["iso"].items()},
       "leaky control: same perturbation, no position_ids": {k: round(v, 4) for k, v in res["leaky"].items()}}
if "fp32" in sys.argv[4:]:
    out["fp32 packed vs alone"] = {k: round(v, 5) for k, v in alone_vs_packed.items()}
print("RESULT " + json.dumps(out))
