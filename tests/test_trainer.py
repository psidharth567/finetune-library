from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import LlamaConfig, LlamaForCausalLM

from finetune_library.config import DataFormat, ExperimentConfig, RuntimeBackend
from finetune_library.runtime import LoadedRuntime
from finetune_library.trainer import run_training


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    name_or_path = "tiny-tokenizer"

    def __len__(self) -> int:
        return 64

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str | None = None,
        **_: Any,
    ):
        import torch

        ids = [3 + (ord(character) % 61) for character in text]
        if return_tensors == "pt":
            tensor = torch.tensor([ids], dtype=torch.long)
            return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}
        return {"input_ids": ids}

    def save_pretrained(self, output: str | Path) -> None:
        Path(output, "tokenizer.json").write_text("{}\n", encoding="utf-8")


def test_one_loop_handles_cpt_optimizer_metrics_and_adapter(tmp_path: Path, monkeypatch) -> None:
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(
        "\n".join(
            json.dumps({"text": f"sample number {index} with enough tokens"}) for index in range(8)
        )
        + "\n",
        encoding="utf-8",
    )
    base = ExperimentConfig.from_yaml("configs/models/qwen3-8b-cpt.yaml")
    config = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={
                    "format": DataFormat.TEXT,
                    "path": str(data_path),
                    "cache_dir": str(tmp_path / "cache"),
                    "packing": False,
                    "drop_remainder": False,
                }
            ),
            "training": base.training.model_copy(
                update={
                    "output_dir": str(tmp_path / "output"),
                    "max_seq_length": 32,
                    "per_device_batch_size": 2,
                    "gradient_accumulation_steps": 2,
                    "max_steps": 1,
                    "dataloader_workers": 0,
                }
            ),
            "lora": base.lora.model_copy(update={"rank": 2, "alpha": 4, "expert_rank": 1}),
            "runtime": base.runtime.model_copy(
                update={
                    "backend": RuntimeBackend.NATIVE,
                    "model_kernels": "native",
                    "loss": "cross_entropy",
                    "gradient_checkpointing": False,
                }
            ),
        }
    )
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
            tie_word_embeddings=False,
        )
    ).bfloat16()
    tokenizer = TinyTokenizer()

    def fake_runtime(*_args, **_kwargs) -> LoadedRuntime:
        return LoadedRuntime(
            model=model,
            tokenizer=tokenizer,
            backend=config.runtime.backend,
            attention="eager",
            revision="0" * 40,
        )

    monkeypatch.setattr("finetune_library.trainer.load_runtime", fake_runtime)
    summary = run_training(config)
    assert summary["steps"] == 1
    assert summary["last_loss"] > 0
    assert (tmp_path / "output" / "metrics.jsonl").exists()
    assert (tmp_path / "output" / "final" / "adapter_model.safetensors").exists()
