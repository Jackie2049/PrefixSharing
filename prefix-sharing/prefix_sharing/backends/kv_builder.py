"""Shared prefix-expanded K/V construction for attention backends."""

from __future__ import annotations

from typing import Any

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.observability import PrefixSharingStats
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_ATTENTION_KV,
    PrefixActivationSlotId,
    PrefixAttentionStore,
)


def apply_rope_with_plan(
    query: Any,
    key: Any,
    prefix_sharing_plan: PrefixSharingPlan,
    *,
    rope_fn: Any | None = None,
    **_: Any,
) -> tuple[Any, Any]:
    """Apply an optional RoPE callback using PrefixSharing position offsets."""

    if rope_fn is None:
        return query, key
    return rope_fn(query, key, prefix_sharing_plan.q_position_offsets, prefix_sharing_plan.kv_position_offsets)


def build_prefix_expanded_kv(
    key: Any,
    value: Any,
    store: PrefixAttentionStore,
    prefix_sharing_plan: PrefixSharingPlan,
    *,
    packed_batch_layout: Any | None = None,
    layer_id: int,
    tp_rank: int = 0,
    stats: PrefixSharingStats | None = None,
) -> tuple[Any, Any]:
    """Build prefix-expanded K/V rows for attention backends.

    The framework produces all rows' Q/K/V tensors before this function runs.
    This builder then assembles the logical K/V rows in provider-before-reuser
    order. That order is required because reusers load provider KV from
    ``store``; do not parallelize this loop without replacing it with an
    explicit dependency-aware build phase.
    """

    layout = packed_batch_layout or PackedBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q)
    key_rows = _split_packed(key, layout.padded_lengths)
    value_rows = _split_packed(value, layout.padded_lengths)
    expanded_offsets = _cumsum(prefix_sharing_plan.expanded_lengths_kv)
    expanded_total = expanded_offsets[-1]
    expanded_key = key.new_empty((expanded_total, *key.shape[1:]))
    expanded_value = value.new_empty((expanded_total, *value.shape[1:]))
    store_count = 0
    reuse_count = 0
    reuse_hit_count = 0
    reuse_miss_count = 0
    stored_tokens = 0
    reused_prefix_tokens = 0

    for batch_index, (key_row, value_row) in enumerate(zip(key_rows, value_rows)):
        valid_length = layout.valid_lengths[batch_index]
        valid_key_row = key_row[:valid_length]
        valid_value_row = value_row[:valid_length]
        expanded_start = expanded_offsets[batch_index]
        expanded_end = expanded_offsets[batch_index + 1]
        expanded_key_row = expanded_key[expanded_start:expanded_end]
        expanded_value_row = expanded_value[expanded_start:expanded_end]
        if not prefix_sharing_plan.is_reuser(batch_index):
            expanded_key_row.copy_(valid_key_row)
            expanded_value_row.copy_(valid_value_row)
            slot_id = PrefixActivationSlotId(
                prefix_sharing_plan.forward_id,
                prefix_sharing_plan.micro_batch_id,
                layer_id,
                batch_index,
                PREFIX_STATE_TYPE_ATTENTION_KV,
                tp_rank,
            )
            # Publish provider KV so later reusers in this micro-batch can load it.
            store.store(
                slot_id,
                key_tensor=expanded_key_row,
                value_tensor=expanded_value_row,
                prefix_len=expanded_key_row.shape[0],
                overwrite=True,
            )
            store_count += 1
            stored_tokens += int(expanded_key_row.shape[0])
            continue

        provider = prefix_sharing_plan.provider_index[batch_index]
        provider_slot_id = PrefixActivationSlotId(
            prefix_sharing_plan.forward_id,
            prefix_sharing_plan.micro_batch_id,
            layer_id,
            provider,
            PREFIX_STATE_TYPE_ATTENTION_KV,
            tp_rank,
        )
        # Load the provider KV published earlier, then prepend its prefix to this suffix KV.
        reuse_count += 1
        try:
            entry = store.load(provider_slot_id)
        except KeyError:
            reuse_miss_count += 1
            _record_stats(
                stats,
                layer_id=layer_id,
                store_count=store_count,
                reuse_count=reuse_count,
                reuse_hit_count=reuse_hit_count,
                reuse_miss_count=reuse_miss_count,
                stored_tokens=stored_tokens,
                reused_prefix_tokens=reused_prefix_tokens,
                expanded_kv_tokens=expanded_total,
                valid_q_tokens=layout.total_valid_length,
                padded_q_tokens=layout.total_padded_length,
            )
            raise
        reuse_hit_count += 1
        prefix_len = prefix_sharing_plan.prefix_lens[batch_index]
        reused_prefix_tokens += int(prefix_len)
        expanded_key_row[:prefix_len].copy_(entry.key_tensor[:prefix_len])
        expanded_key_row[prefix_len:].copy_(valid_key_row)
        expanded_value_row[:prefix_len].copy_(entry.value_tensor[:prefix_len])
        expanded_value_row[prefix_len:].copy_(valid_value_row)
        own_slot_id = PrefixActivationSlotId(
            prefix_sharing_plan.forward_id,
            prefix_sharing_plan.micro_batch_id,
            layer_id,
            batch_index,
            PREFIX_STATE_TYPE_ATTENTION_KV,
            tp_rank,
        )
        # Publish expanded reuser KV so later rows can reuse this longer prefix.
        store.store(
            own_slot_id,
            key_tensor=expanded_key_row,
            value_tensor=expanded_value_row,
            prefix_len=expanded_key_row.shape[0],
            overwrite=True,
        )
        store_count += 1
        stored_tokens += int(expanded_key_row.shape[0])

    _record_stats(
        stats,
        layer_id=layer_id,
        store_count=store_count,
        reuse_count=reuse_count,
        reuse_hit_count=reuse_hit_count,
        reuse_miss_count=reuse_miss_count,
        stored_tokens=stored_tokens,
        reused_prefix_tokens=reused_prefix_tokens,
        expanded_kv_tokens=expanded_total,
        valid_q_tokens=layout.total_valid_length,
        padded_q_tokens=layout.total_padded_length,
    )
    return expanded_key, expanded_value


def _record_stats(stats: PrefixSharingStats | None, **kwargs: Any) -> None:
    if stats is not None:
        stats.record_attention_kv_build(**kwargs)


def _split_packed(tensor: Any, lengths: list[int]) -> list[Any]:
    if not lengths:
        return []
    if sum(lengths) != tensor.shape[0]:
        raise ValueError("packed tensor first dimension does not match lengths")
    import torch

    return list(torch.split(tensor, lengths, dim=0))


def _cumsum(lengths: list[int]) -> list[int]:
    offsets = [0]
    total = 0
    for length in lengths:
        total += int(length)
        offsets.append(total)
    return offsets
