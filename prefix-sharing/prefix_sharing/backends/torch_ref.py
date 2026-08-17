"""Pure PyTorch reference backend."""

from __future__ import annotations

import math
from typing import Any

import torch

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.kv_builder import apply_rope_with_plan, build_prefix_expanded_kv
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan


class TorchReferenceBackend:
    capabilities = BackendCapabilities(
        name="torch_ref",
        supports_cpu=True,
        supports_cuda=True,
        supports_cann=True,
        supports_different_q_kv_lengths=True,
        supports_prefix_last_restore=True,
    )

    def validate(self, config: PrefixSharingConfig, model_config: Any | None = None) -> None:
        config.validate(model_config=model_config)

    def apply_rope(
        self,
        query: Any,
        key: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        rope_fn: Any | None = None,
        **_: Any,
    ) -> tuple[Any, Any]:
        return apply_rope_with_plan(query, key, prefix_sharing_plan, rope_fn=rope_fn)

    def build_kv(
        self,
        key: Any,
        value: Any,
        store: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        layer_id: int,
        tp_rank: int = 0,
        stats: Any | None = None,
    ) -> tuple[Any, Any]:
        return build_prefix_expanded_kv(
            key,
            value,
            store,
            prefix_sharing_plan,
            packed_batch_layout=packed_batch_layout,
            layer_id=layer_id,
            tp_rank=tp_rank,
            stats=stats,
        )

    def attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        **_: Any,
    ) -> Any:
        batch_layout = packed_batch_layout or PackedBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q)
        
        # Split QKV from packed batch into individual sequences for precision debugging
        query_rows = _split_packed(query, batch_layout.padded_lengths)
        key_rows = _split_packed(key, prefix_sharing_plan.expanded_lengths_kv)
        value_rows = _split_packed(value, prefix_sharing_plan.expanded_lengths_kv)

        # Compute attention per sequence
        outputs = []
        for batch_index, (q_row, k_row, v_row) in enumerate(zip(query_rows, key_rows, value_rows)):
            valid_length = batch_layout.valid_lengths[batch_index]
            # Padding does not participate in attention computation
            q_valid = q_row[:valid_length]
            prefix_len = prefix_sharing_plan.q_position_offsets[batch_index]
            if valid_length == 0:
                outputs.append(torch.zeros_like(q_row))
                continue
            mask = _causal_q_kv_mask(
                q_len=q_valid.shape[0],
                kv_len=k_row.shape[0],
                q_start=prefix_len,
                device=q_valid.device,
            )
            valid_output = _attention_row(q_valid, k_row, v_row, mask)
            if valid_length == q_row.shape[0]:
                outputs.append(valid_output)
                continue
            padded_output = torch.zeros_like(q_row)
            padded_output[:valid_length] = valid_output
            outputs.append(padded_output)
        return torch.cat(outputs, dim=0)


def _split_packed(tensor: Any, lengths: list[int]) -> list[Any]:
    if not lengths:
        return []
    if sum(lengths) != tensor.shape[0]:
        raise ValueError("packed tensor first dimension does not match lengths")
    return list(torch.split(tensor, lengths, dim=0))


def _causal_q_kv_mask(q_len: int, kv_len: int, q_start: int, device: Any) -> Any:
    q_positions = torch.arange(q_start, q_start + q_len, device=device).unsqueeze(1)
    kv_positions = torch.arange(0, kv_len, device=device).unsqueeze(0)
    return kv_positions <= q_positions


def _attention_row(q_row: Any, k_row: Any, v_row: Any, mask: Any) -> Any:
    # Use ``F.scaled_dot_product_attention`` instead of manual einsum+softmax:
    # under bf16 autocast, ``torch.einsum`` gets downcast to bf16 (even when
    # inputs are .float()), causing severe softmax precision degradation that
    # amplifies through residual streams layer by layer. SDPA is unaffected by
    # autocast downcasting (accumulates in fp32 internally), matches HF
    # attention numerically, and is faster. Q/K/V retain their original dtype.
    import torch.nn.functional as _F

    scale = 1.0 / math.sqrt(q_row.shape[-1])

    if q_row.dim() == 2:
        q4 = q_row.unsqueeze(0)                     # [1, Lq, D]
        k4 = k_row.unsqueeze(0)                     # [1, Lk, D]
        v4 = v_row.unsqueeze(0)
        m4 = mask.unsqueeze(0)                      # [1, Lq, Lk]
        out = _F.scaled_dot_product_attention(q4, k4, v4, attn_mask=m4, scale=scale)
        return out.squeeze(0)

    if q_row.dim() != 3:
        raise ValueError("TorchReferenceBackend attention expects packed rows with 2 or 3 dims")

    # GQA adaptation: replicate KV heads to match Q head count
    # (older SDPA versions do not support native GQA).
    q_heads = q_row.shape[1]
    kv_heads = k_row.shape[1]
    if q_heads != kv_heads:
        if q_heads % kv_heads != 0:
            raise ValueError("query heads must be a multiple of kv heads for grouped-query attention")
        repeat = q_heads // kv_heads
        k_row = k_row.repeat_interleave(repeat, dim=1)
        v_row = v_row.repeat_interleave(repeat, dim=1)

    # Row layout [L, H, D] -> SDPA expected [B=1, H, L, D]
    q4 = q_row.transpose(0, 1).unsqueeze(0).contiguous()
    k4 = k_row.transpose(0, 1).unsqueeze(0).contiguous()
    v4 = v_row.transpose(0, 1).unsqueeze(0).contiguous()
    m4 = mask.unsqueeze(0).unsqueeze(0)             # [1, 1, Lq, Lk] bool
    out = _F.scaled_dot_product_attention(q4, k4, v4, attn_mask=m4, scale=scale)
    return out.squeeze(0).transpose(0, 1)           # Back to [Lq, H, D]
