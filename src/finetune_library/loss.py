from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
import triton  # type: ignore[import-untyped]
from liger_kernel.ops.cross_entropy import (  # type: ignore[import-untyped]
    liger_cross_entropy_kernel,
)
from liger_kernel.ops.fused_linear_cross_entropy import (  # type: ignore[import-untyped]
    MAX_FUSED_SIZE,
)
from liger_kernel.ops.utils import is_hip  # type: ignore[import-untyped]
from torch import nn


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.__class__.__name__ == "DTensor":
        return cast(Any, tensor).to_local()
    return tensor


class _FusedLoraLinearCrossEntropy(torch.autograd.Function):
    """Exact fused LM-head LoRA and cross-entropy without persistent logits."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        base_weight: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        target: torch.Tensor,
        scaling: float,
        softcap: float | None,
    ) -> torch.Tensor:
        tokens, hidden_size = hidden_states.shape
        vocab_size = base_weight.shape[0]
        block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(vocab_size))
        increase_factor = triton.cdiv(vocab_size, hidden_size)
        chunk_size = triton.next_power_of_2(triton.cdiv(tokens, increase_factor))
        chunks = triton.cdiv(tokens, chunk_size)

        grad_hidden = torch.zeros_like(hidden_states)
        grad_a = torch.zeros_like(lora_a, dtype=torch.float32)
        grad_b = torch.zeros_like(lora_b, dtype=torch.float32)
        losses = torch.zeros(tokens, dtype=torch.float32, device=hidden_states.device)

        for chunk_index in range(chunks):
            start = chunk_index * chunk_size
            end = min((chunk_index + 1) * chunk_size, tokens)
            hidden = hidden_states[start:end]
            after_a = torch.nn.functional.linear(hidden, lora_a)
            logits = torch.nn.functional.linear(hidden, base_weight)
            logits = torch.addmm(
                logits,
                after_a,
                lora_b.T,
                beta=1.0,
                alpha=scaling,
            ).contiguous()
            labels = target[start:end].contiguous()
            loss_slice = losses[start:end]

            liger_cross_entropy_kernel[(end - start,)](
                X_ptr=logits,
                X_stride=logits.stride(-2),
                Y_ptr=labels,
                Y_stride=labels.stride(-1),
                weight_ptr=None,
                loss_ptr=loss_slice,
                z_loss_ptr=None,
                loss_stride=loss_slice.stride(-1),
                token_accuracy_ptr=None,
                token_accuracy_stride=0,
                predicted_tokens_ptr=None,
                predicted_tokens_stride=0,
                n_cols=vocab_size,
                n_non_ignore=1,
                sum_non_ignore_weight=1,
                weight_sum=0.0,
                ignore_index=-100,
                lse_square_scale=0.0,
                label_smoothing=0.0,
                reduction="sum",
                softcap=softcap,
                RETURN_Z_LOSS=False,
                RETURN_TOKEN_ACCURACY=False,
                RETURN_PREDICTED_TOKENS=False,
                HAS_WEIGHT=False,
                HAS_SOFTCAPPING=softcap is not None,
                HAS_GRADIENTS=True,
                BLOCK_SIZE=block_size,
                num_warps=16 if is_hip() else 32,
            )

            # The cross-entropy kernel replaces logits with d(loss)/d(logits).
            grad_logits = logits
            grad_after_a = torch.mm(grad_logits, lora_b) * scaling
            grad_hidden[start:end] = torch.addmm(
                torch.mm(grad_logits, base_weight),
                grad_after_a,
                lora_a,
            )
            grad_a.add_(torch.mm(grad_after_a.T, hidden).float())
            grad_b.add_(torch.mm(grad_logits.T, after_a).float(), alpha=scaling)

        ctx.save_for_backward(
            grad_hidden,
            grad_a.to(dtype=lora_a.dtype),
            grad_b.to(dtype=lora_b.dtype),
        )
        return losses.sum()

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        grad_hidden, grad_a, grad_b = ctx.saved_tensors
        return (
            grad_hidden * grad_output,
            None,
            grad_a * grad_output,
            grad_b * grad_output,
            None,
            None,
            None,
        )


def fused_lora_linear_cross_entropy(
    hidden_states: torch.Tensor,
    lm_head: nn.Module,
    labels: torch.Tensor,
    denominator: torch.Tensor,
    *,
    softcap: float | None,
) -> torch.Tensor:
    """Compute shifted causal loss while preserving both LM-head LoRA gradients."""

    from peft.tuners.lora.layer import Linear

    if not isinstance(lm_head, Linear):
        raise TypeError("fused linear cross-entropy requires a PEFT LoRA Linear LM head")
    if len(lm_head.active_adapters) != 1:
        raise ValueError("fused linear cross-entropy supports exactly one active adapter")
    adapter = lm_head.active_adapters[0]
    dropout = lm_head.lora_dropout[adapter]
    if not isinstance(dropout, nn.Identity):
        raise ValueError("fused linear cross-entropy requires LoRA dropout 0")

    base_weight = _local_tensor(cast(torch.Tensor, lm_head.get_base_layer().weight))
    lora_a = _local_tensor(cast(torch.Tensor, lm_head.lora_A[adapter].weight))
    lora_b = _local_tensor(cast(torch.Tensor, lm_head.lora_B[adapter].weight))
    shifted_hidden = hidden_states[:, :-1, :].reshape(-1, hidden_states.shape[-1]).contiguous()
    shifted_labels = labels[:, 1:].reshape(-1).contiguous()
    loss_sum = _FusedLoraLinearCrossEntropy.apply(
        shifted_hidden,
        base_weight,
        lora_a,
        lora_b,
        shifted_labels,
        float(lm_head.scaling[adapter]),
        softcap,
    )
    return loss_sum / denominator


@dataclass(slots=True)
class TrainingOutput:
    loss: torch.Tensor
    logits: torch.Tensor | None = None


class TrainingModel(nn.Module):
    """Select optimized training semantics while retaining a reference path."""

    def __init__(self, peft_model: nn.Module, loss_implementation: str) -> None:
        super().__init__()
        self.peft_model = peft_model
        self.loss_implementation = loss_implementation

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if self.loss_implementation == "cross_entropy" or labels is None:
            return self.peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=use_cache,
                num_items_in_batch=num_items_in_batch,
                **kwargs,
            )
        if num_items_in_batch is None:
            raise ValueError("optimized loss requires num_items_in_batch")

        causal_lm = cast(Any, cast(Any, self.peft_model).base_model.model)
        outputs = causal_lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs,
        )
        softcap = getattr(causal_lm.config, "final_logit_softcapping", None)
        loss = fused_lora_linear_cross_entropy(
            outputs.last_hidden_state,
            causal_lm.get_output_embeddings(),
            labels,
            num_items_in_batch,
            softcap=softcap,
        )
        return TrainingOutput(loss=loss)


def build_training_model(
    peft_model: nn.Module,
    loss_implementation: str,
) -> TrainingModel:
    if loss_implementation == "auto":
        loss_implementation = "cross_entropy"
    return TrainingModel(peft_model, loss_implementation)
