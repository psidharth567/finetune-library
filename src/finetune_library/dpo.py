"""Direct preference optimization (task=dpo): data preparation, batching, loss.

Offline DPO over (prompt, chosen, rejected) triples. The reference policy is
the frozen base model, obtained by disabling the LoRA adapter for a no-grad
forward pass, so no second copy of the weights is held.

Batches are padding-free: every chosen and rejected sequence of a micro-batch
is concatenated into one row with ``position_ids`` restarting at 0 per
sequence, the same isolated layout packed SFT uses. Sequence ``2 * i`` is the
chosen response of pair ``i`` and ``2 * i + 1`` its rejected response.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from finetune_library.config import ExperimentConfig, PreferenceConfig
from finetune_library.data import (
    IGNORE_INDEX,
    _add_boundaries,
    _chat_with_assistant_mask,
    load_records,
    source_fingerprint,
)

# Bumped whenever preference preparation output changes; part of DPO cache keys only.
PREFERENCE_PREPARATION_VERSION = 1
PREFERENCE_COLUMNS = (
    "chosen_input_ids",
    "chosen_labels",
    "rejected_input_ids",
    "rejected_labels",
)
# Order of PreferenceModel.last_stats: fp32 sums over pairs (``loss`` is the sum
# of per-pair losses), except ``tokens``, which sums the input tokens of every
# chosen and rejected sequence.
STAT_NAMES = (
    "loss",
    "rewards_chosen",
    "rewards_rejected",
    "rewards_margin",
    "rewards_accuracy",
    "logps_chosen",
    "logps_rejected",
    "tokens",
)
_ROLES = {"system", "user", "assistant", "tool"}


# --------------------------------------------------------------------------- data


def _normalize_messages(value: Any, *, where: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a list of messages")
    messages: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, Mapping):
            raise ValueError(f"each message in {where} must be an object")
        role = str(message.get("role", ""))
        if role not in _ROLES:
            raise ValueError(f"unsupported message role {role!r} in {where}")
        messages.append({"role": role, "content": str(message.get("content", ""))})
    return messages


def _split_response(
    value: Any, *, where: str
) -> tuple[list[dict[str, str]] | None, dict[str, str]]:
    """Return (context turns or None, final assistant message)."""

    if isinstance(value, str):
        if not value:
            raise ValueError(f"{where} is empty")
        return None, {"role": "assistant", "content": value}
    messages = _normalize_messages(value, where=where)
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError(f"{where} must end with an assistant message")
    return messages[:-1], messages[-1]


def preference_conversations(
    row: Mapping[str, Any],
    config: ExperimentConfig,
) -> tuple[list[dict[str, str]], dict[str, str], dict[str, str]]:
    """Split a preference row into (shared context, chosen turn, rejected turn).

    ``chosen``/``rejected`` may be response strings or message lists, with or
    without the prompt turns (the Tulu/Olmo layout repeats the prompt inside
    both lists). Only the final assistant turn of each side is trained; both
    sides must share the same context.
    """

    preference = cast(PreferenceConfig, config.preference)
    chosen_context, chosen = _split_response(
        row.get(preference.chosen_field), where=preference.chosen_field
    )
    rejected_context, rejected = _split_response(
        row.get(preference.rejected_field), where=preference.rejected_field
    )
    contexts = [c for c in (chosen_context, rejected_context) if c]
    if len(contexts) == 2 and contexts[0] != contexts[1]:
        raise ValueError("chosen and rejected must share the same prompt turns")
    if contexts:
        context = contexts[0]
    else:
        prompt = row.get(config.data.prompt_field)
        if isinstance(prompt, str):
            context = [{"role": "user", "content": prompt}] if prompt else []
        elif prompt is None:
            context = []
        else:
            context = _normalize_messages(prompt, where=config.data.prompt_field)
    if not context:
        raise ValueError("preference rows require a prompt")
    return context, chosen, rejected


def _render_ids(
    tokenizer: Any, messages: list[dict[str, str]], *, generation_prompt: bool
) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation_prompt,
    )
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids]


def tokenize_response(
    tokenizer: Any,
    context: list[dict[str, str]],
    response: dict[str, str],
    config: ExperimentConfig,
) -> tuple[list[int], list[int]]:
    """Tokenize context + response with labels on the final response only."""

    ids, labels = _chat_with_assistant_mask(tokenizer, context + [response])
    # Earlier assistant turns in the context carry labels in the SFT mask; the
    # preference signal is the final response alone. Everything the rendered
    # generation prompt covers is context.
    prompt_ids = _render_ids(tokenizer, context, generation_prompt=True)
    common = 0
    for prompt_token, token in zip(prompt_ids, ids, strict=False):
        if prompt_token != token:
            break
        common += 1
    labels[:common] = [IGNORE_INDEX] * common
    ids, labels = _add_boundaries(
        ids,
        labels,
        tokenizer,
        config.data,
        bos_is_trainable=False,
        eos_is_trainable=True,
    )
    if all(label == IGNORE_INDEX for label in labels[1:]):
        raise ValueError("preference response produced no supervised tokens")
    return ids, labels


def iter_preference_rows(
    config: ExperimentConfig,
    tokenizer: Any,
    stats: dict[str, Any],
) -> Iterator[dict[str, list[int]]]:
    """Tokenized pairs; never truncated. Over-length pairs follow data.sft_overlength."""

    max_length = config.training.max_seq_length
    raw = dropped = identical = longest = 0
    for row in load_records(config.data):
        raw += 1
        context, chosen, rejected = preference_conversations(row, config)
        chosen_ids, chosen_labels = tokenize_response(tokenizer, context, chosen, config)
        rejected_ids, rejected_labels = tokenize_response(tokenizer, context, rejected, config)
        length = max(len(chosen_ids), len(rejected_ids))
        if length > max_length:
            if config.data.sft_overlength == "error":
                raise ValueError(
                    f"preference pair #{raw - 1} has a {length}-token side > "
                    f"training.max_seq_length={max_length}; pairs are never truncated. "
                    "Raise max_seq_length, filter the data, or set data.sft_overlength=drop"
                )
            dropped += 1
            longest = max(longest, length)
            continue
        if chosen_ids == rejected_ids:
            # Zero preference signal: the DPO margin is identically 0.
            identical += 1
            continue
        yield {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
        }
    stats.update(
        {
            "raw_records": raw,
            "dropped_overlength": dropped,
            "longest_dropped_tokens": longest,
            "dropped_identical": identical,
            "pairs": raw - dropped - identical,
        }
    )


def prepare_preference_to_disk(
    config: ExperimentConfig,
    tokenizer: Any,
    destination: Path,
    key: str,
) -> Path:
    from datasets import Dataset as HFDataset

    stats: dict[str, Any] = {}
    rows = list(iter_preference_rows(config, tokenizer, stats))
    if not rows:
        raise ValueError("prepared preference dataset is empty")
    chosen_tokens = sum(len(row["chosen_input_ids"]) for row in rows)
    rejected_tokens = sum(len(row["rejected_input_ids"]) for row in rows)
    stats.update(
        {
            "chosen_tokens": chosen_tokens,
            "rejected_tokens": rejected_tokens,
            "mean_chosen_tokens": round(chosen_tokens / len(rows), 1),
            "mean_rejected_tokens": round(rejected_tokens / len(rows), 1),
        }
    )
    if stats["dropped_overlength"]:
        print(
            f"WARNING: dropped {stats['dropped_overlength']} of {stats['raw_records']} "
            f"preference pairs with a side longer than max_seq_length="
            f"{config.training.max_seq_length} (longest {stats['longest_dropped_tokens']} "
            "tokens); pairs are never truncated"
        )
    HFDataset.from_list(rows).save_to_disk(str(destination))
    metadata = {
        "cache_key": key,
        "source": source_fingerprint(config.data),
        "rows": len(rows),
        "max_seq_length": config.training.max_seq_length,
        "stats": stats,
    }
    (destination / "finetune_library_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "prep_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"prepare-data stats: {json.dumps(stats, sort_keys=True)}")
    return destination


class PreferenceCollator:
    """Flatten a micro-batch of pairs into one padding-free, isolated row."""

    def __call__(self, rows: list[Mapping[str, Sequence[int]]]) -> dict[str, torch.Tensor]:
        flat_ids: list[int] = []
        flat_labels: list[int] = []
        flat_positions: list[int] = []
        flat_sequences: list[int] = []
        for pair_index, row in enumerate(rows):
            for side, name in enumerate(("chosen", "rejected")):
                ids = [int(token) for token in row[f"{name}_input_ids"]]
                labels = [int(token) for token in row[f"{name}_labels"]]
                if len(ids) != len(labels):
                    raise ValueError("preference input_ids and labels must have equal length")
                # The first label is never predicted; ignoring it also keeps the
                # shifted loss from predicting across sequence boundaries.
                labels[0] = IGNORE_INDEX
                flat_ids.extend(ids)
                flat_labels.extend(labels)
                flat_positions.extend(range(len(ids)))
                flat_sequences.extend([2 * pair_index + side] * len(ids))
        return {
            "input_ids": torch.tensor([flat_ids], dtype=torch.long),
            "labels": torch.tensor([flat_labels], dtype=torch.long),
            "position_ids": torch.tensor([flat_positions], dtype=torch.long),
            "sequence_index": torch.tensor([flat_sequences], dtype=torch.long),
            "pair_count": torch.tensor(len(rows), dtype=torch.long),
        }


def count_pairs(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return batch["pair_count"]


# --------------------------------------------------------------------------- loss


def _chunk_token_logps(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    lm_head: nn.Module,
    softcap: float | None,
) -> torch.Tensor:
    logits = lm_head(hidden).float()
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    picked = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return picked - torch.logsumexp(logits, dim=-1)


def token_logps(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    lm_head: nn.Module,
    *,
    softcap: float | None,
    chunk_tokens: int,
) -> torch.Tensor:
    """Per-token log p(target) without holding [tokens, vocab] logits.

    Each chunk is recomputed in backward (activation checkpointing), so peak
    logits memory is one chunk regardless of sequence length.
    """

    pieces = []
    for start in range(0, hidden.shape[0], chunk_tokens):
        h = hidden[start : start + chunk_tokens]
        t = targets[start : start + chunk_tokens]
        if torch.is_grad_enabled():
            pieces.append(
                checkpoint(_chunk_token_logps, h, t, lm_head, softcap, use_reentrant=False)
            )
        else:
            pieces.append(_chunk_token_logps(h, t, lm_head, softcap))
    if not pieces:
        return hidden.new_zeros(0, dtype=torch.float32)
    return torch.cat(pieces)


def preference_losses(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    chosen_tokens: torch.Tensor,
    rejected_tokens: torch.Tensor,
    preference: PreferenceConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pair losses and the scaled margin beta * (log-ratio difference)."""

    beta = preference.beta
    chosen_ratio = policy_chosen - reference_chosen
    rejected_ratio = policy_rejected - reference_rejected
    margin = beta * (chosen_ratio - rejected_ratio)
    if preference.loss_type == "sigmoid":
        smoothing = preference.label_smoothing
        losses = -F.logsigmoid(margin) * (1 - smoothing) - F.logsigmoid(-margin) * smoothing
    elif preference.loss_type == "hinge":
        losses = torch.relu(1 - margin)
    elif preference.loss_type == "ipo":
        normalized = chosen_ratio / chosen_tokens - rejected_ratio / rejected_tokens
        losses = (normalized - 1 / (2 * beta)) ** 2
    else:
        raise AssertionError(f"unhandled loss_type {preference.loss_type}")
    if preference.sft_weight:
        losses = losses + preference.sft_weight * (-policy_chosen / chosen_tokens)
    return losses, margin


@contextmanager
def adapters_disabled(layers: Sequence[Any]) -> Iterator[None]:
    """Run the base model by setting only the flag LoRA forwards read.

    PEFT's ``disable_adapter()`` also flips ``requires_grad`` on adapter
    parameters and, on re-enable, sets it on every active-adapter parameter;
    under DDP/FSDP that would touch the parameters the wrappers track.
    """

    previous = [layer._disable_adapters for layer in layers]
    for layer in layers:
        layer._disable_adapters = True
    try:
        yield
    finally:
        for layer, value in zip(layers, previous, strict=True):
            layer._disable_adapters = value


@dataclass(slots=True)
class PreferenceOutput:
    loss: torch.Tensor


def causal_lm_of(model: nn.Module) -> Any:
    """The Hugging Face causal LM inside a PEFT model, or the model itself."""

    base_model = getattr(model, "base_model", None)
    inner = getattr(base_model, "model", None)
    if inner is not None and model.__class__.__name__.startswith("Peft"):
        return inner
    return model


class PreferenceModel(nn.Module):
    """Reference (no grad) and policy passes plus the DPO loss.

    LoRA: the reference is the same model with its adapters disabled. Full
    fine-tuning: ``reference_model`` is a separate frozen copy of the initial
    weights (a submodule, so FSDP shards it too; it has no trainable
    parameters, so the optimizer, clipping and saves never see it).

    Per-step statistics are kept on ``last_stats`` rather than returned: FSDP2's
    ``output_dtype=bfloat16`` casts every floating tensor a root module returns,
    which would round the logged metrics (training is unaffected; the cast's
    backward is the identity).
    """

    loss_implementation = "dpo_chunked_logprob"

    def __init__(
        self,
        peft_model: nn.Module,
        preference: PreferenceConfig,
        reference_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.peft_model = peft_model
        self.preference = preference
        self.last_stats: torch.Tensor | None = None
        self.reference_model = reference_model
        self._tuner_layers: list[Any] = []
        if reference_model is not None:
            if any(parameter.requires_grad for parameter in reference_model.parameters()):
                raise ValueError("the DPO reference model must be frozen")
            return
        from peft.tuners.tuners_utils import BaseTunerLayer

        # Plain list, not a ModuleList: these are already submodules.
        self._tuner_layers = [m for m in peft_model.modules() if isinstance(m, BaseTunerLayer)]
        if not self._tuner_layers:
            raise ValueError(
                "preference training requires a LoRA-injected model or a reference_model"
            )

    def train(self, mode: bool = True) -> PreferenceModel:
        super().train(mode)
        if self.reference_model is not None:
            self.reference_model.eval()  # the reference never trains (dropout off)
        return self

    def _sequence_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        targets: torch.Tensor,
        target_positions: torch.Tensor,
        target_sequences: torch.Tensor,
        num_sequences: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        causal_lm = causal_lm_of(model)
        outputs = causal_lm.model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            **kwargs,
        )
        # Only supervised positions reach the LM head (prompt tokens never do).
        hidden = outputs.last_hidden_state[0].index_select(0, target_positions)
        logps = token_logps(
            hidden,
            targets,
            causal_lm.get_output_embeddings(),
            softcap=getattr(causal_lm.config, "final_logit_softcapping", None),
            chunk_tokens=self.preference.logprob_chunk_tokens,
        )
        # Targets are grouped by sequence in order, so a split-and-sum is exact
        # and deterministic (a CUDA index_add accumulates atomically, in
        # varying order).
        lengths = torch.bincount(target_sequences, minlength=num_sequences).tolist()
        return torch.stack([piece.sum() for piece in logps.split(lengths)])

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        position_ids: torch.Tensor,
        sequence_index: torch.Tensor,
        pair_count: torch.Tensor,
        use_cache: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> PreferenceOutput:
        del use_cache
        if num_items_in_batch is None:
            raise ValueError("preference loss requires num_items_in_batch (global pair mean)")
        if input_ids.shape[0] != 1:
            raise ValueError("preference batches must be flattened into one row")
        num_pairs = int(pair_count.item())
        num_sequences = 2 * num_pairs
        shifted = labels[0, 1:]
        target_positions = shifted.ne(IGNORE_INDEX).nonzero().squeeze(-1)
        targets = shifted.index_select(0, target_positions)
        target_sequences = sequence_index[0, 1:].index_select(0, target_positions)
        counts = torch.bincount(target_sequences, minlength=num_sequences).float()

        args = (input_ids, position_ids, targets, target_positions, target_sequences, num_sequences)
        if self.reference_model is not None:
            with torch.no_grad():
                reference = self._sequence_logps(self.reference_model, *args, **kwargs)
        else:
            with torch.no_grad(), adapters_disabled(self._tuner_layers):
                reference = self._sequence_logps(self.peft_model, *args, **kwargs)
        policy = self._sequence_logps(self.peft_model, *args, **kwargs)

        losses, margin = preference_losses(
            policy[0::2],
            policy[1::2],
            reference[0::2],
            reference[1::2],
            counts[0::2],
            counts[1::2],
            self.preference,
        )
        beta = self.preference.beta
        with torch.no_grad():
            self.last_stats = torch.stack(
                [
                    losses.detach().sum(),
                    (beta * (policy[0::2] - reference[0::2])).sum(),
                    (beta * (policy[1::2] - reference[1::2])).sum(),
                    margin.sum(),
                    margin.gt(0).float().sum(),
                    policy[0::2].sum(),
                    policy[1::2].sum(),
                    torch.tensor(float(input_ids.shape[1]), device=input_ids.device),
                ]
            ).float()
        return PreferenceOutput(loss=losses.sum() / num_items_in_batch)
