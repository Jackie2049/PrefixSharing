"""Unit tests for shared prefix-expanded KV construction."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.kv_builder import apply_rope_with_plan, build_prefix_expanded_kv
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_ATTENTION_KV,
    PrefixActivationSlotId,
    PrefixAttentionStore,
)


def _make_plan(batch_sizes: list[int], prefix_lens: list[int]):
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1)
    planner = PrefixSharingPlanner(config)
    sequences = []
    next_token = 100
    provider_seqs = {}
    for index, (size, prefix_len) in enumerate(zip(batch_sizes, prefix_lens)):
        if prefix_len == 0:
            seq = list(range(next_token, next_token + size))
            next_token += size
            provider_seqs[index] = seq
            sequences.append(seq)
            continue
        provider_idx = max(j for j in range(index) if prefix_lens[j] == 0)
        provider_seq = provider_seqs[provider_idx]
        suffix = list(range(next_token, next_token + size - prefix_len))
        next_token += size - prefix_len
        sequences.append(provider_seq[:prefix_len] + suffix)
    return planner.plan(sequences)


def test_apply_rope_with_plan_delegates_offsets():
    plan = _make_plan([4, 3], [0, 2])
    query = torch.randn(sum(plan.kept_lengths_q), 2, 8)
    key = torch.randn(sum(plan.kept_lengths_q), 2, 8)
    calls = []

    def rope_fn(q, k, q_offsets, kv_offsets):
        calls.append((q_offsets, kv_offsets))
        return q + 1, k + 2

    out_q, out_k = apply_rope_with_plan(query, key, plan, rope_fn=rope_fn)

    assert calls == [(plan.q_position_offsets, plan.kv_position_offsets)]
    assert torch.allclose(out_q, query + 1)
    assert torch.allclose(out_k, key + 2)


def test_apply_rope_with_plan_without_fn_returns_inputs():
    plan = _make_plan([4], [0])
    query = torch.randn(4, 2, 8)
    key = torch.randn(4, 2, 8)

    out_q, out_k = apply_rope_with_plan(query, key, plan, rope_fn=None)

    assert out_q is query
    assert out_k is key


def test_build_prefix_expanded_kv_no_sharing_matches_valid_tokens():
    plan = _make_plan([4, 3], [0, 0])
    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
    key = torch.randn(layout.total_padded_length, 2, 8, requires_grad=True)
    value = torch.randn(layout.total_padded_length, 2, 8, requires_grad=True)
    store = PrefixAttentionStore()

    expanded_key, expanded_value = build_prefix_expanded_kv(
        key,
        value,
        store,
        plan,
        packed_batch_layout=layout,
        layer_id=0,
    )

    assert torch.allclose(expanded_key, key)
    assert torch.allclose(expanded_value, value)
    assert store.size == 2


def test_build_prefix_expanded_kv_reuser_loads_provider_prefix_and_keeps_grad():
    plan = _make_plan([6, 5], [0, 3])
    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
    key = torch.randn(layout.total_padded_length, 2, 8, requires_grad=True)
    value = torch.randn(layout.total_padded_length, 2, 8, requires_grad=True)
    store = PrefixAttentionStore()

    expanded_key, expanded_value = build_prefix_expanded_kv(
        key,
        value,
        store,
        plan,
        packed_batch_layout=layout,
        layer_id=2,
        tp_rank=3,
    )

    provider_len = plan.kept_lengths_q[0]
    prefix_len = plan.prefix_lens[1]
    reuser_start = plan.expanded_lengths_kv[0]
    reuser_end = reuser_start + plan.expanded_lengths_kv[1]
    provider_prefix = key[:prefix_len]
    reuser_suffix = key[provider_len:]

    assert torch.allclose(expanded_key[reuser_start : reuser_start + prefix_len], provider_prefix)
    assert torch.allclose(expanded_key[reuser_start + prefix_len : reuser_end], reuser_suffix)
    assert expanded_value.requires_grad

    loss = expanded_key[reuser_start : reuser_start + prefix_len].sum()
    loss.backward()
    assert key.grad is not None
    assert key.grad[:prefix_len].abs().sum() > 0

    provider_slot = PrefixActivationSlotId(
        plan.forward_id,
        plan.micro_batch_id,
        2,
        0,
        PREFIX_STATE_TYPE_ATTENTION_KV,
        3,
    )
    assert store.load(provider_slot).key_tensor.shape[0] == plan.expanded_lengths_kv[0]


def test_build_prefix_expanded_kv_ignores_tp_padding_slots():
    plan = _make_plan([5, 4], [0, 3])
    rows = [torch.arange(length) for length in plan.kept_lengths_q]
    layout = PackedBatchLayout.from_kept_position_rows(rows, align_size=4)
    key = torch.randn(layout.total_padded_length, 2, 8, requires_grad=True)
    value = torch.randn(layout.total_padded_length, 2, 8, requires_grad=True)
    store = PrefixAttentionStore()

    expanded_key, expanded_value = build_prefix_expanded_kv(
        key,
        value,
        store,
        plan,
        packed_batch_layout=layout,
        layer_id=0,
    )

    assert expanded_key.shape[0] == sum(plan.expanded_lengths_kv)
    assert expanded_value.shape[0] == sum(plan.expanded_lengths_kv)
    provider_slot = PrefixActivationSlotId(
        plan.forward_id,
        plan.micro_batch_id,
        0,
        0,
        PREFIX_STATE_TYPE_ATTENTION_KV,
        0,
    )
    assert store.load(provider_slot).key_tensor.shape[0] == layout.valid_lengths[0]
