"""Tests for s_packed plan construction, custom causal mask, and updated backends.

s_packed mode: KV storage is deduplicated (shared prefixes stored once).
Each input maps to one or more position ranges in s_packed via s_packed_kv_ranges.
A global custom causal mask encodes visibility: Q[i] can only see KV from its own input.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.backends.torch_ref import TorchReferenceBackend, _attention_row
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner, PrefixSharingPlan


# ------------------------------------------------------------------
# build_s_packed
# ------------------------------------------------------------------

def test_build_s_packed_no_sharing():
    """No shared prefix: each input gets its own range."""
    input_ids = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    prefix_lens = [0, 0, 0]
    original_lengths = [3, 3, 3]

    ranges, length = PrefixSharingPlan.build_s_packed(input_ids, prefix_lens, original_lengths)

    # s_packed = [1,2,3,4,5,6,7,8,9], length=9
    assert length == 9
    assert ranges[0] == [(0, 3)]
    assert ranges[1] == [(3, 6)]
    assert ranges[2] == [(6, 9)]


def test_build_s_packed_partial_sharing():
    """Two inputs share a prefix: prefix is deduplicated."""
    input_ids = [[1, 2, 3, 4], [1, 2, 5, 6]]
    prefix_lens = [0, 2]
    original_lengths = [4, 4]

    ranges, length = PrefixSharingPlan.build_s_packed(input_ids, prefix_lens, original_lengths)

    # input0: [1,2,3,4] -> s_packed[0:4], range=(0,4)
    # input1 prefix [1,2] matches s_packed[0:2], suffix [5,6] appended -> s_packed[4:6], range=(0,2)+(4,6)
    assert length == 6
    assert ranges[0] == [(0, 4)]
    assert ranges[1] == [(0, 2), (4, 6)]


def test_build_s_packed_three_inputs():
    """Three inputs with overlapping prefixes."""
    input_ids = [[1, 2, 3, 4], [1, 2, 5, 6], [1, 2, 3, 7]]
    prefix_lens = [0, 2, 3]
    original_lengths = [4, 4, 4]

    ranges, length = PrefixSharingPlan.build_s_packed(input_ids, prefix_lens, original_lengths)

    # s_packed = [1,2,3,4,5,6,7]
    # input0: (0,4)
    # input1: prefix [1,2] matches s_packed[0:2], suffix [5,6] -> (0,2)+(4,6)
    # input2: prefix [1,2,3] matches s_packed[0:3], suffix [7] -> (0,3)+(6,7)
    assert length == 7
    assert ranges[0] == [(0, 4)]
    assert ranges[1] == [(0, 2), (4, 6)]
    assert ranges[2] == [(0, 3), (6, 7)]


def test_build_s_packed_reuser_no_prefix():
    """Reuser with prefix_len=0: entire sequence is suffix."""
    input_ids = [[1, 2, 3, 4], [1, 2, 3, 5]]
    prefix_lens = [0, 0]
    original_lengths = [4, 4]

    ranges, length = PrefixSharingPlan.build_s_packed(input_ids, prefix_lens, original_lengths)

    # Both treated as providers (no prefix), each appends full sequence
    assert length == 8
    assert ranges[0] == [(0, 4)]
    assert ranges[1] == [(4, 8)]


def test_build_s_packed_all_same():
    """All inputs are identical: s_packed stores one copy."""
    input_ids = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    prefix_lens = [0, 3, 3]
    original_lengths = [3, 3, 3]

    ranges, length = PrefixSharingPlan.build_s_packed(input_ids, prefix_lens, original_lengths)

    # s_packed = [1,2,3], length=3
    assert length == 3
    assert ranges[0] == [(0, 3)]
    assert ranges[1] == [(0, 3)]
    assert ranges[2] == [(0, 3)]


# ------------------------------------------------------------------
# s_packed fields in plan
# ------------------------------------------------------------------

def test_plan_includes_s_packed_fields():
    """PrefixSharingPlanner.plan() populates s_packed fields."""
    input_ids = [[1, 2, 3, 4], [1, 2, 5, 6]]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1))
    plan = planner.plan(input_ids)

    assert plan.s_packed_length > 0
    assert len(plan.s_packed_kv_ranges) == 2
    assert len(plan.s_packed_q_lengths) == 2
    assert len(plan.s_packed_q_starts) == 2
    # Q lengths = suffix lengths
    assert plan.s_packed_q_lengths[0] == 4  # provider: full sequence
    assert plan.s_packed_q_lengths[1] == 2   # reuser: suffix [5,6]


def test_plan_s_packed_q_starts_are_sequential():
    """s_packed_q_starts are monotonically increasing by q_length."""
    input_ids = [[1, 2, 3, 4], [1, 2, 5, 6], [1, 2, 3, 7]]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1))
    plan = planner.plan(input_ids)

    for i in range(1, plan.batch_size):
        assert plan.s_packed_q_starts[i] == plan.s_packed_q_starts[i - 1] + plan.s_packed_q_lengths[i - 1]


# ------------------------------------------------------------------
# build_global_custom_mask
# ------------------------------------------------------------------

def test_mask_no_sharing_is_standard_causal():
    """No sharing: mask is standard causal on concatenated Q/KV (True = visible)."""
    input_ids = [[1, 2, 3], [4, 5, 6]]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1))
    plan = planner.plan(input_ids)

    mask = plan.build_global_custom_mask(device="cpu")

    # total_q = 3+3=6, s_packed_length = 3+3=6
    assert mask.shape == (6, 6)
    # Cross-input visibility should be False (cannot see other input's KV)
    for q_i in range(3):
        for kv_i in range(3, 6):
            assert not mask[q_i, kv_i], f"Q[{q_i}] should not see KV[{kv_i}]"

    for q_i in range(3, 6):
        for kv_i in range(0, 3):
            assert not mask[q_i, kv_i], f"Q[{q_i}] should not see KV[{kv_i}]"


def test_mask_prefix_sharing_input_sees_own_prefix():
    """Reuser's Q can see the shared prefix's KV (True = visible)."""
    input_ids = [[1, 2, 3, 4], [1, 2, 5, 6]]  # prefix [1,2] shared
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1))
    plan = planner.plan(input_ids)

    mask = plan.build_global_custom_mask(device="cpu")

    # Reuser Q[0] at relative position 0: visible KV = original positions ≤ 0
    assert mask[4, 0], "Reuser Q[0] should see KV[0] (prefix position 0)"
    assert not mask[4, 1], "Reuser Q[0] should NOT see KV[1] (causal: 1 > 0)"
    assert not mask[4, 4], "Reuser Q[0] should NOT see KV[4] (suffix starts at position 2 > Q pos 0)"

    # Reuser Q[1] at relative position 1: visible KV = original positions ≤ 1
    assert mask[5, 0], "Reuser Q[1] should see KV[0]"
    assert mask[5, 1], "Reuser Q[1] should see KV[1]"
    assert not mask[5, 4], "Reuser Q[1] should NOT see KV[4] (suffix starts at position 2 > Q pos 1)"
    assert not mask[5, 5], "Reuser Q[1] should NOT see KV[5] (causal: 3 > 1)"


def test_mask_provider_respects_causal():
    """Provider's Q follows standard causal within its KV range (True = visible)."""
    input_ids = [[1, 2, 3, 4], [1, 2, 5, 6]]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1))
    plan = planner.plan(input_ids)

    mask = plan.build_global_custom_mask(device="cpu")

    # Standard causal within provider's KV range
    for q_i in range(4):
        for kv_j in range(4):
            if kv_j <= q_i:
                assert mask[q_i, kv_j], f"Provider Q[{q_i}] should see KV[{kv_j}]"
            else:
                assert not mask[q_i, kv_j], f"Provider Q[{q_i}] should NOT see KV[{kv_j}]"


def test_mask_all_inputs_same():
    """All inputs identical: s_packed dedupes to one copy, reusers have 0 suffix."""
    input_ids = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1))
    plan = planner.plan(input_ids)

    # All inputs are identical [1,2,3].
    # Provider (input0): prefix_len=0, suffix_len=3
    # Reusers (1,2): prefix_len=3, suffix_len=0
    assert plan.prefix_lens[0] == 0
    assert plan.prefix_lens[1] == 3
    assert plan.prefix_lens[2] == 3
    assert plan.suffix_lens[0] == 3
    assert plan.suffix_lens[1] == 0
    assert plan.suffix_lens[2] == 0

    # s_packed = [1,2,3], length=3
    assert plan.s_packed_length == 3
    # Q length = only provider's suffix = 3
    assert sum(plan.s_packed_q_lengths) == 3
    assert plan.s_packed_kv_ranges[0] == [(0, 3)]  # provider: full range

    mask = plan.build_global_custom_mask(device="cpu")
    # shape = (total_q=3, s_packed_length=3)
    assert mask.shape == (3, 3)
    # Standard causal: Q[i] sees KV[j] iff j <= i (True = visible)
    for qi in range(3):
        for kv in range(3):
            if kv <= qi:
                assert mask[qi, kv], f"Q[{qi}] should see KV[{kv}]"
            else:
                assert not mask[qi, kv], f"Q[{qi}] should NOT see KV[{kv}]"


# ------------------------------------------------------------------
# torch_ref.build_kv with s_packed
# ------------------------------------------------------------------

def test_build_kv_s_packed_returns_concatenated_suffix():
    """build_kv returns K/V with one row per input (suffix only)."""
    input_ids = [[1, 2, 3, 4], [1, 2, 5, 6]]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1))
    plan = planner.plan(input_ids)
    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

    backend = TorchReferenceBackend()

    num_heads, head_dim = 2, 8
    key = torch.randn(layout.total_padded_length, num_heads, head_dim)
    value = torch.randn(layout.total_padded_length, num_heads, head_dim)

    k_out, v_out = backend.build_kv(
        key, value, store=None,
        prefix_sharing_plan=plan,
        packed_batch_layout=layout,
        layer_id=0, tp_rank=0,
    )

    # s_packed KV length = 6 (suffix tokens only: [3,4] + [5,6])
    assert k_out.shape[0] == plan.s_packed_length
    assert v_out.shape[0] == plan.s_packed_length


def test_build_kv_s_packed_provider_own_kv():
    """Provider's KV in output equals its own suffix K/V."""
    input_ids = [[1, 2, 3, 4], [1, 2, 5, 6]]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1))
    plan = planner.plan(input_ids)
    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

    backend = TorchReferenceBackend()

    num_heads, head_dim = 2, 8
    key = torch.randn(layout.total_padded_length, num_heads, head_dim)
    value = torch.randn(layout.total_padded_length, num_heads, head_dim)

    k_out, v_out = backend.build_kv(
        key, value, store=None,
        prefix_sharing_plan=plan,
        packed_batch_layout=layout,
        layer_id=0, tp_rank=0,
    )

    # Provider (input0): kept q_len=4, full KV in input
    # In s_packed: input0 occupies first 4 positions of k_out
    # The s_packed KV should be the suffix K/V of input0 (which is the whole sequence)
    # k_out[0:4] should match the first 4 rows of the input key
    provider_k = k_out[:4]
    expected_provider_k = key[:4]  # layout splits by kept_lengths_q, first input is first 4
    assert torch.allclose(provider_k, expected_provider_k), \
        "Provider KV in s_packed should equal its own suffix KV"


# ------------------------------------------------------------------
# torch_ref.attention with s_packed
# ------------------------------------------------------------------

def test_attention_s_packed_mask_respects_visibility():
    """Attention with s_packed mask: Q only attends to its own KV ranges."""
    input_ids = [[1, 2, 3, 4], [1, 2, 5, 6]]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1))
    plan = planner.plan(input_ids)
    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

    backend = TorchReferenceBackend()

    num_heads, head_dim = 2, 8
    total_q = plan.s_packed_length  # = 6
    k_out = torch.randn(total_q, num_heads, head_dim)
    v_out = torch.randn(total_q, num_heads, head_dim)
    # Q: each input's suffix tokens
    q = torch.randn(total_q, num_heads, head_dim)

    out = backend.attention(q, k_out, v_out, plan, packed_batch_layout=layout)

    # Output shape: total_q tokens in s_packed Q order
    assert out.shape == (total_q, num_heads, head_dim)
    # No zeros (real attention)
    assert not torch.all(out == 0)


def test_attention_s_packed_output_per_input():
    """Attention output is correctly split per input."""
    input_ids = [[1, 2, 3, 4], [1, 2, 5, 6]]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1))
    plan = planner.plan(input_ids)
    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

    backend = TorchReferenceBackend()

    num_heads, head_dim = 2, 8
    total_q = plan.s_packed_length
    k_out = torch.randn(total_q, num_heads, head_dim)
    v_out = torch.randn(total_q, num_heads, head_dim)
    q = torch.randn(total_q, num_heads, head_dim)

    out = backend.attention(q, k_out, v_out, plan, packed_batch_layout=layout)

    # Output for input0: first 4 tokens
    assert out[:4].shape == (4, num_heads, head_dim)
    # Output for input1: last 2 tokens
    assert out[4:6].shape == (2, num_heads, head_dim)


def test_attention_s_packed_mask_shape_matches():
    """Global custom mask shape matches (total_q, s_packed_length)."""
    input_ids = [[1, 2, 3, 4], [1, 2, 5, 6], [1, 2, 3, 7]]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1))
    plan = planner.plan(input_ids)

    mask = plan.build_global_custom_mask(device="cpu")

    total_q = sum(plan.s_packed_q_lengths)
    assert mask.shape == (total_q, plan.s_packed_length)


# ------------------------------------------------------------------
# Reuser and provider distinction in s_packed
# ------------------------------------------------------------------

def test_reuser_q_only_has_suffix():
    """Reuser's Q in s_packed consists only of its suffix tokens."""
    input_ids = [[1, 2, 3, 4, 5], [1, 2, 3, 6, 7]]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2))
    plan = planner.plan(input_ids)

    # Provider (input0): Q len = 5 (full sequence)
    # Reuser (input1): prefix_len=3, Q len = 2 (suffix [6,7])
    assert plan.s_packed_q_lengths[0] == 5
    assert plan.s_packed_q_lengths[1] == 2
    # Total Q in s_packed = 7
    assert sum(plan.s_packed_q_lengths) == 7


def test_s_packed_length_matches_unique_tokens():
    """s_packed_length equals total unique KV tokens."""
    input_ids = [[1, 2, 3, 4], [1, 2, 5, 6], [1, 2, 3, 7]]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1))
    plan = planner.plan(input_ids)

    # Unique tokens: [1,2,3,4] + [5,6] + [7] = 7
    assert plan.s_packed_length == 7
    # KV storage is deduplicated
    assert plan.s_packed_length <= sum(plan.original_lengths)


# ------------------------------------------------------------------
# Mask cache (lazy evaluation)
# ------------------------------------------------------------------

def test_mask_cached_on_repeated_build():
    """build_global_custom_mask caches the result."""
    input_ids = [[1, 2, 3, 4], [1, 2, 5, 6]]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1))
    plan = planner.plan(input_ids)

    mask1 = plan.build_global_custom_mask(device="cpu")
    mask2 = plan.build_global_custom_mask(device="cpu")

    assert mask1 is mask2, "Mask should be cached (same object)"


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_single_input_no_sharing():
    """Single input: s_packed = that input's tokens, mask = standard causal."""
    input_ids = [[1, 2, 3, 4, 5]]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1))
    plan = planner.plan(input_ids)

    assert plan.s_packed_length == 5
    assert plan.s_packed_q_lengths[0] == 5
    mask = plan.build_global_custom_mask(device="cpu")
    # Standard causal: True = visible, lower triangular
    expected = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    assert torch.equal(mask, expected)


def test_empty_batch():
    """Empty batch: s_packed_length=0, no errors."""
    input_ids: list[list[int]] = []
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1))
    plan = planner.plan(input_ids)

    assert plan.s_packed_length == 0
    assert plan.batch_size == 0
    assert plan.s_packed_q_lengths == []
    mask = plan.build_global_custom_mask(device="cpu")
    assert mask.shape == (0, 0)
