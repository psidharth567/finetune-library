from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from finetune_library.config import DataConfig, DataFormat, ExperimentConfig, Task

IGNORE_INDEX = -100
PREPARATION_VERSION = 3


class TokenDataset(Dataset[dict[str, list[int]]]):
    """Small in-memory view used for raw JSONL and unit-test datasets."""

    def __init__(self, rows: Sequence[dict[str, list[int]]]) -> None:
        if not rows:
            raise ValueError("prepared dataset is empty")
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def _read_json_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                yield value
        return

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("data", [value])
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON object or array")
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(f"{path} contains a non-object row")
        yield row


def load_records(data: DataConfig) -> Iterable[Mapping[str, Any]]:
    if data.path is not None:
        path = Path(data.path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file() and path.suffix in {".json", ".jsonl"}:
            records: Iterable[Mapping[str, Any]] = _read_json_records(path)
        else:
            from datasets import load_from_disk  # type: ignore[import-untyped]

            dataset = load_from_disk(str(path))
            records = (
                dataset[data.split]
                if hasattr(dataset, "keys") and data.split in dataset
                else dataset
            )
    else:
        from datasets import load_dataset

        records = load_dataset(
            data.dataset_name,
            data.dataset_config,
            split=data.split,
        )

    limit = data.max_samples
    if limit is None:
        return records

    def limited() -> Iterator[Mapping[str, Any]]:
        for index, row in enumerate(records):
            if index >= limit:
                break
            yield row

    return limited()


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids]


def _special_id(tokenizer: Any, name: str) -> int | None:
    value = getattr(tokenizer, name, None)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return int(value[0]) if value else None
    return int(value) if value is not None else None


def _add_boundaries(
    ids: list[int],
    labels: list[int],
    tokenizer: Any,
    data: DataConfig,
    *,
    bos_is_trainable: bool,
    eos_is_trainable: bool,
) -> tuple[list[int], list[int]]:
    bos = _special_id(tokenizer, "bos_token_id")
    eos = _special_id(tokenizer, "eos_token_id")
    if data.add_bos and bos is not None and (not ids or ids[0] != bos):
        ids.insert(0, bos)
        labels.insert(0, bos if bos_is_trainable else IGNORE_INDEX)
    if data.add_eos and eos is not None and (not ids or ids[-1] != eos):
        ids.append(eos)
        labels.append(eos if eos_is_trainable else IGNORE_INDEX)
    return ids, labels


def _chat_with_assistant_mask(
    tokenizer: Any, messages: list[dict[str, str]]
) -> tuple[list[int], list[int]]:
    has_assistant = any(message["role"] == "assistant" for message in messages)
    if not has_assistant:
        raise ValueError("messages SFT rows require at least one assistant turn")
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        ids = encoded["input_ids"]
        mask = encoded.get("assistant_masks", encoded.get("assistant_tokens_mask"))
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if mask and isinstance(mask[0], list):
            mask = mask[0]
        if mask is not None and len(mask) == len(ids) and any(mask):
            labels = [
                int(token) if bool(active) else IGNORE_INDEX
                for token, active in zip(ids, mask, strict=True)
            ]
            return [int(token) for token in ids], labels
    except (TypeError, ValueError):
        pass

    # Templates without `{% generation %}` cannot return an assistant mask.
    # Re-render each prefix and preserve the longest common prefix so multi-turn
    # assistant spans are still masked deterministically.
    previous_ids: list[int] = []
    previous_labels: list[int] = []
    for index, message in enumerate(messages):
        current_ids = tokenizer.apply_chat_template(
            messages[: index + 1],
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(current_ids, Mapping):
            current_ids = current_ids["input_ids"]
        if current_ids and isinstance(current_ids[0], list):
            current_ids = current_ids[0]
        current_ids = [int(token) for token in current_ids]
        common = 0
        for old, new in zip(previous_ids, current_ids, strict=False):
            if old != new:
                break
            common += 1
        current_labels = previous_labels[:common] + [IGNORE_INDEX] * (len(current_ids) - common)
        if message["role"] == "assistant":
            current_labels[common:] = current_ids[common:]
        previous_ids, previous_labels = current_ids, current_labels
    if all(label == IGNORE_INDEX for label in previous_labels):
        raise ValueError("chat template produced no assistant supervision")
    return previous_ids, previous_labels


def tokenize_record(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    task: Task,
    data: DataConfig,
) -> dict[str, list[int]]:
    if data.format == DataFormat.TOKENIZED:
        ids = [int(token) for token in row["input_ids"]]
        if "labels" in row:
            labels = [int(token) for token in row["labels"]]
        elif task == Task.CPT:
            labels = ids.copy()
        else:
            raise ValueError("tokenized SFT rows must contain labels")
        if len(ids) != len(labels):
            raise ValueError("tokenized input_ids and labels must have equal length")
        return {"input_ids": ids, "labels": labels}

    if data.format == DataFormat.TEXT:
        text = str(row.get(data.text_field, "")).strip()
        if not text:
            raise ValueError(f"missing non-empty {data.text_field!r}")
        ids = _encode(tokenizer, text)
        labels = ids.copy()
        ids, labels = _add_boundaries(
            ids,
            labels,
            tokenizer,
            data,
            bos_is_trainable=True,
            eos_is_trainable=True,
        )
        return {"input_ids": ids, "labels": labels}

    if data.format == DataFormat.ALPACA:
        instruction = str(row.get(data.instruction_field, "")).strip()
        context = str(row.get(data.input_field, "")).strip()
        completion = str(row.get(data.output_field, "")).strip()
        if not instruction or not completion:
            raise ValueError("Alpaca rows require non-empty instruction and output")
        prompt = f"### Instruction:\n{instruction}\n\n"
        if context:
            prompt += f"### Input:\n{context}\n\n"
        prompt += "### Response:\n"
    elif data.format == DataFormat.PROMPT_COMPLETION:
        prompt = str(row.get(data.prompt_field, ""))
        completion = str(row.get(data.completion_field, ""))
        if not completion:
            raise ValueError("prompt/completion rows require a non-empty completion")
    elif data.format == DataFormat.MESSAGES:
        raw_messages = row.get(data.messages_field)
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("messages rows require a non-empty list")
        messages: list[dict[str, str]] = []
        for message in raw_messages:
            if not isinstance(message, Mapping):
                raise ValueError("each message must be an object")
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"unsupported message role {role!r}")
            messages.append({"role": role, "content": content})
        ids, labels = _chat_with_assistant_mask(tokenizer, messages)
        ids, labels = _add_boundaries(
            ids,
            labels,
            tokenizer,
            data,
            bos_is_trainable=False,
            eos_is_trainable=messages[-1]["role"] == "assistant",
        )
        return {"input_ids": ids, "labels": labels}
    else:
        raise AssertionError(f"unhandled data format {data.format}")

    prompt_ids = _encode(tokenizer, prompt)
    completion_ids = _encode(tokenizer, completion)
    ids = prompt_ids + completion_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids.copy()
    ids, labels = _add_boundaries(
        ids,
        labels,
        tokenizer,
        data,
        bos_is_trainable=False,
        eos_is_trainable=True,
    )
    return {"input_ids": ids, "labels": labels}


def pack_rows(
    rows: Iterable[dict[str, list[int]]],
    *,
    max_length: int,
    packing: bool,
    drop_remainder: bool,
) -> list[dict[str, list[int]]]:
    return list(
        iter_packed_rows(
            rows,
            max_length=max_length,
            packing=packing,
            drop_remainder=drop_remainder,
        )
    )


def iter_packed_rows(
    rows: Iterable[dict[str, list[int]]],
    *,
    max_length: int,
    packing: bool,
    drop_remainder: bool,
) -> Iterator[dict[str, list[int]]]:
    if not packing:
        for row in rows:
            if not row["input_ids"]:
                continue
            yield {
                "input_ids": row["input_ids"][:max_length],
                "labels": row["labels"][:max_length],
            }
        return

    ids_buffer: list[int] = []
    labels_buffer: list[int] = []
    for row in rows:
        ids_buffer.extend(row["input_ids"])
        labels_buffer.extend(row["labels"])
        while len(ids_buffer) >= max_length:
            yield {
                "input_ids": ids_buffer[:max_length],
                "labels": labels_buffer[:max_length],
            }
            del ids_buffer[:max_length]
            del labels_buffer[:max_length]
    if ids_buffer and not drop_remainder:
        yield {"input_ids": ids_buffer, "labels": labels_buffer}


def iter_prepared_rows(
    config: ExperimentConfig,
    tokenizer: Any,
) -> Iterator[dict[str, list[int]]]:
    tokenized = (
        tokenize_record(row, tokenizer=tokenizer, task=config.task, data=config.data)
        for row in load_records(config.data)
    )
    yield from iter_packed_rows(
        tokenized,
        max_length=config.training.max_seq_length,
        packing=config.data.packing,
        drop_remainder=config.data.drop_remainder,
    )


def build_token_dataset(config: ExperimentConfig, tokenizer: Any) -> TokenDataset:
    rows = list(iter_prepared_rows(config, tokenizer))
    return TokenDataset(rows)


def cache_key(config: ExperimentConfig, tokenizer: Any) -> str:
    from finetune_library.registry import resolve_model

    spec = resolve_model(config.model.name)
    identity = {
        "preparation_version": PREPARATION_VERSION,
        "task": config.task.value,
        "data": config.data.model_dump(mode="json"),
        "max_seq_length": config.training.max_seq_length,
        "tokenizer": getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__),
        "tokenizer_revision": config.model.revision or spec.revision,
        "tokenizer_commit": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "tokenizer_vocab_size": len(tokenizer),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def prepare_to_disk(
    config: ExperimentConfig,
    tokenizer: Any,
    output: str | Path | None = None,
) -> Path:
    from datasets import Dataset as HFDataset

    destination = (
        Path(output)
        if output is not None
        else Path(config.data.cache_dir) / cache_key(config, tokenizer)
    )
    marker = destination / "dataset_info.json"
    if marker.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = HFDataset.from_generator(lambda: iter_prepared_rows(config, tokenizer))
    if len(dataset) == 0:
        raise ValueError("prepared dataset is empty")
    dataset.save_to_disk(str(destination))
    metadata = {
        "cache_key": cache_key(config, tokenizer),
        "rows": len(dataset),
        "max_seq_length": config.training.max_seq_length,
    }
    (destination / "finetune_library_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_prepared(path: str | Path):
    from datasets import load_from_disk

    dataset = load_from_disk(str(path))
    required = {"input_ids", "labels"}
    columns = set(getattr(dataset, "column_names", ()))
    if not required.issubset(columns):
        raise ValueError(f"prepared dataset is missing columns: {sorted(required - columns)}")
    return dataset


class CausalCollator:
    def __init__(self, tokenizer: Any, max_length: int, pad_to_multiple_of: int = 8) -> None:
        pad = _special_id(tokenizer, "pad_token_id")
        if pad is None:
            pad = _special_id(tokenizer, "eos_token_id")
        if pad is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")
        self.pad_id = pad
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, rows: list[Mapping[str, Sequence[int]]]) -> dict[str, torch.Tensor]:
        longest = min(max(len(row["input_ids"]) for row in rows), self.max_length)
        target = min(
            self.max_length,
            ((longest + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of)
            * self.pad_to_multiple_of,
        )
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        masks: list[list[int]] = []
        for row in rows:
            ids = [int(token) for token in row["input_ids"][:target]]
            row_labels = [int(token) for token in row["labels"][:target]]
            padding = target - len(ids)
            input_ids.append(ids + [self.pad_id] * padding)
            labels.append(row_labels + [IGNORE_INDEX] * padding)
            masks.append([1] * len(ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def count_loss_tokens(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Count labels consumed by next-token prediction (the first label is never used)."""

    return batch["labels"][:, 1:].ne(IGNORE_INDEX).sum()
