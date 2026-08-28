from __future__ import annotations

from typing import Any

import torch

from finetune_library.config import DataConfig, DataFormat, ExperimentConfig, Task
from finetune_library.data import (
    IGNORE_INDEX,
    CausalCollator,
    cache_key,
    pack_rows,
    tokenize_record,
)


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    name_or_path = "fake"

    def __len__(self) -> int:
        return 256

    def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
        return {"input_ids": [10 + (ord(character) % 50) for character in text]}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        **_: Any,
    ):
        ids = [self.bos_token_id]
        mask = [0]
        for message in messages:
            role = 70 if message["role"] == "assistant" else 60
            content = self(message["content"])["input_ids"]
            turn = [role, *content, self.eos_token_id]
            ids.extend(turn)
            mask.extend([1 if message["role"] == "assistant" else 0] * len(turn))
        if return_dict:
            return {"input_ids": ids, "assistant_masks": mask}
        return ids


class ZeroMaskTokenizer(FakeTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        **kwargs: Any,
    ):
        ids = super().apply_chat_template(messages, **kwargs)
        if return_dict:
            return {"input_ids": ids, "assistant_masks": [0] * len(ids)}
        return {"input_ids": ids}


class BoundarylessTokenizer(FakeTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        **_: Any,
    ):
        ids: list[int] = []
        mask: list[int] = []
        for message in messages:
            turn = [70 if message["role"] == "assistant" else 60]
            turn.extend(self(message["content"])["input_ids"])
            ids.extend(turn)
            mask.extend([1 if message["role"] == "assistant" else 0] * len(turn))
        if return_dict:
            return {"input_ids": ids, "assistant_masks": mask}
        return ids


def data_config(format_: DataFormat, **updates: Any) -> DataConfig:
    return DataConfig(
        format=format_,
        path="unused.jsonl",
        packing=False,
        drop_remainder=False,
        **updates,
    )


def test_cpt_text_labels_every_token_and_adds_eos() -> None:
    tokenizer = FakeTokenizer()
    row = tokenize_record(
        {"text": "abc"},
        tokenizer=tokenizer,
        task=Task.CPT,
        data=data_config(DataFormat.TEXT),
    )
    assert row["input_ids"] == row["labels"]
    assert row["input_ids"][-1] == tokenizer.eos_token_id


def test_prompt_completion_masks_prompt() -> None:
    tokenizer = FakeTokenizer()
    row = tokenize_record(
        {"prompt": "question", "completion": "answer"},
        tokenizer=tokenizer,
        task=Task.SFT,
        data=data_config(DataFormat.PROMPT_COMPLETION),
    )
    prompt_length = len(tokenizer("question")["input_ids"])
    assert row["labels"][:prompt_length] == [IGNORE_INDEX] * prompt_length
    assert all(label != IGNORE_INDEX for label in row["labels"][prompt_length:])


def test_messages_use_assistant_mask() -> None:
    tokenizer = FakeTokenizer()
    row = tokenize_record(
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        },
        tokenizer=tokenizer,
        task=Task.SFT,
        data=data_config(DataFormat.MESSAGES),
    )
    first_assistant = row["input_ids"].index(70)
    assert all(label == IGNORE_INDEX for label in row["labels"][:first_assistant])
    assert row["labels"][first_assistant:] == row["input_ids"][first_assistant:]


def test_messages_mask_added_boundaries_outside_assistant_turns() -> None:
    tokenizer = BoundarylessTokenizer()
    row = tokenize_record(
        {
            "messages": [
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "follow-up"},
            ]
        },
        tokenizer=tokenizer,
        task=Task.SFT,
        data=data_config(
            DataFormat.MESSAGES,
            add_bos=True,
            add_eos=True,
        ),
    )
    assert row["input_ids"][0] == tokenizer.bos_token_id
    assert row["labels"][0] == IGNORE_INDEX
    assert row["input_ids"][-1] == tokenizer.eos_token_id
    assert row["labels"][-1] == IGNORE_INDEX


def test_messages_fall_back_when_template_returns_an_all_zero_mask() -> None:
    tokenizer = ZeroMaskTokenizer()
    row = tokenize_record(
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "long answer"},
            ]
        },
        tokenizer=tokenizer,
        task=Task.SFT,
        data=data_config(DataFormat.MESSAGES),
    )
    supervised = [label for label in row["labels"] if label != IGNORE_INDEX]
    assert len(supervised) > 2
    first_assistant = row["input_ids"].index(70)
    assert all(label == IGNORE_INDEX for label in row["labels"][:first_assistant])


def test_alpaca_masks_instruction_and_trains_response() -> None:
    tokenizer = FakeTokenizer()
    row = tokenize_record(
        {"instruction": "Solve", "input": "2+2", "output": "4"},
        tokenizer=tokenizer,
        task=Task.SFT,
        data=data_config(DataFormat.ALPACA),
    )
    first_trainable = next(
        index for index, label in enumerate(row["labels"]) if label != IGNORE_INDEX
    )
    assert first_trainable > 0
    assert row["labels"][first_trainable:] == row["input_ids"][first_trainable:]


def test_packing_never_decodes_or_changes_labels() -> None:
    rows = [
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5, 6], "labels": [-100, 5, 6]},
    ]
    packed = pack_rows(rows, max_length=4, packing=True, drop_remainder=False)
    assert packed == [
        {"input_ids": [1, 2, 3, 4], "labels": [-100, 2, 3, -100]},
        {"input_ids": [5, 6], "labels": [5, 6]},
    ]


def test_collator_masks_padding() -> None:
    collated = CausalCollator(FakeTokenizer(), max_length=8)(
        [
            {"input_ids": [1, 2], "labels": [1, 2]},
            {"input_ids": [3, 4, 5], "labels": [3, 4, 5]},
        ]
    )
    assert collated["input_ids"].shape == (2, 8)
    assert torch.equal(collated["attention_mask"][0], torch.tensor([1, 1, 0, 0, 0, 0, 0, 0]))
    assert collated["labels"][0, 2:].eq(IGNORE_INDEX).all()


def test_cache_key_includes_pinned_tokenizer_revision() -> None:
    config = ExperimentConfig.from_yaml("configs/models/qwen3-8b-cpt.yaml")
    overridden = config.model_copy(
        update={"model": config.model.model_copy(update={"revision": "1" * 40})}
    )
    assert cache_key(config, FakeTokenizer()) != cache_key(overridden, FakeTokenizer())
