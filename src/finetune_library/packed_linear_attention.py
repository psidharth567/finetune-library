"""Packed-sequence isolation for gated-delta-net (linear attention) layers.

Isolated packing (``data.packing_isolation=attention``, always on for packed
SFT) feeds one padding-free sequence whose ``position_ids`` restart at 0 for
every example. Softmax attention layers already honour that (flash varlen /
block-diagonal mask). Qwen3.5's ``GatedDeltaNet`` layers do not: stock
transformers calls ``causal_conv1d_fn(..., seq_idx=None)`` and
``chunk_gated_delta_rule`` without ``cu_seqlens``, so the short convolution
and the recurrent state would carry over from one packed example into the
next.

``patch_linear_attention_for_packing`` fixes that without copying the
transformers forward: a pre-hook on each linear-attention decoder layer derives
the example boundaries from the layer's ``position_ids`` kwarg (it also fires
on gradient-checkpoint recompute), and the layer's two kernel entry points are
wrapped to receive ``seq_idx`` / ``cu_seqlens``. Unpacked batches (a single
sequence per row) pass through unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

_DECODER_LAYER_CLASSES = {"Qwen3_5MoeDecoderLayer", "Qwen3_5DecoderLayer"}

# (position_ids tensor, boundaries) for the most recent forward: every layer of
# one forward (and its checkpoint recompute) sees the same position_ids object,
# so boundaries are computed once rather than once per layer.
_last: tuple[torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None] = (None, None)


def packed_boundaries(position_ids: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return ``(cu_seqlens int64, seq_idx int32 [1, L])`` for a packed row, else None.

    A new example starts wherever consecutive positions do not increase by
    exactly 1 (same rule as transformers' ``find_packed_sequence_indices``).
    """
    global _last
    if position_ids is None:
        return None
    if _last[0] is position_ids:
        return _last[1]
    if position_ids.ndim != 2:
        raise ValueError(f"expected 2-D position_ids, got shape {tuple(position_ids.shape)}")
    first = position_ids[:, :1] - 1
    starts = torch.diff(position_ids, prepend=first, dim=-1) != 1
    seq_idx = starts.cumsum(-1)
    if not bool(seq_idx[:, -1].gt(0).any()):
        result = None
    else:
        if position_ids.shape[0] != 1:
            raise ValueError(
                "packed linear-attention batches must be flattened to batch size 1 "
                "(the padding-free collator does this)"
            )
        # The diff rule never flags token 0; every sequence starts there too.
        start_index = starts[0, 1:].nonzero().flatten() + 1
        zero = torch.zeros(1, dtype=start_index.dtype, device=position_ids.device)
        length = torch.full_like(zero, position_ids.shape[1])
        cu_seqlens = torch.cat([zero, start_index, length]).to(torch.long)
        result = (cu_seqlens, seq_idx.to(torch.int32).contiguous())
    _last = (position_ids, result)
    return result


def _stash_boundaries(layer: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    layer.linear_attn._packed_boundaries = packed_boundaries(kwargs.get("position_ids"))


def _wrap_kernels(gdn: nn.Module) -> None:
    conv_fn = gdn.causal_conv1d_fn
    delta_rule = gdn.chunk_gated_delta_rule

    def causal_conv1d_fn(*args: Any, **kwargs: Any) -> torch.Tensor:
        boundaries = getattr(gdn, "_packed_boundaries", None)
        if boundaries is not None:
            kwargs["seq_idx"] = boundaries[1]
        return conv_fn(*args, **kwargs)

    def chunk_gated_delta_rule(*args: Any, **kwargs: Any) -> Any:
        boundaries = getattr(gdn, "_packed_boundaries", None)
        if boundaries is not None:
            kwargs["cu_seqlens"] = boundaries[0]
        return delta_rule(*args, **kwargs)

    gdn.causal_conv1d_fn = causal_conv1d_fn
    gdn.chunk_gated_delta_rule = chunk_gated_delta_rule


def patch_linear_attention_for_packing(model: nn.Module) -> int:
    """Make every gated-delta-net layer in ``model`` respect packed boundaries.

    Returns the number of layers patched (0 for models without linear
    attention). Raises if a layer lacks the fused kernels: the torch fallback
    has no varlen support, so packing would silently leak across examples.
    """
    patched = 0
    for layer in model.modules():
        if type(layer).__name__ not in _DECODER_LAYER_CLASSES:
            continue
        if getattr(layer, "layer_type", None) != "linear_attention":
            continue
        gdn = layer.linear_attn
        if getattr(gdn, "_packing_patched", False):
            continue
        if gdn.causal_conv1d_fn is None or getattr(gdn.chunk_gated_delta_rule, "__name__", "").startswith("torch_"):
            raise RuntimeError(
                "packed sequences on gated-delta-net layers need the causal-conv1d and "
                "flash-linear-attention kernels (the torch fallback cannot isolate examples)"
            )
        _wrap_kernels(gdn)
        layer.register_forward_pre_hook(_stash_boundaries, with_kwargs=True)
        gdn._packing_patched = True
        patched += 1
    return patched
