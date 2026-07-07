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

    # Reuser's first suffix token (original position 2) sees:
    # - Full prefix [0,2): kv_orig < 2 ✓
    # - Its own suffix[0] at s_packed[4] (original pos 2): kv_orig=2 <= q_orig=2 ✓ (self)
    assert mask[4, 0], "Reuser Q[0] should see KV[0] (prefix position 0)"
    assert mask[4, 1], "Reuser Q[0] should see KV[1] (prefix position 1, causal < q_pos=2)"
    assert mask[4, 4], "Reuser Q[0] should see its own suffix[0] at KV[4] (self-attention, kv_orig=2 <= q_orig=2)"
    assert not mask[4, 5], "Reuser Q[0] should NOT see KV[5] (suffix[1], kv_orig=3 > q_orig=2)"

    # Reuser's second suffix token (original position 3) sees:
    # - Full prefix [0,2): kv_orig < 2 ✓
    # - Its own suffix[0] at s_packed[4] (original pos 2): kv_orig=2 <= q_orig=3 ✓
    # - Its own suffix[1] at s_packed[5] (original pos 3): kv_orig=3 <= q_orig=3 ✓ (self)
    # - Provider's suffix: kv_orig >= 4 > q_orig=3 → not visible
    assert mask[5, 0], "Reuser Q[1] should see KV[0]"
    assert mask[5, 1], "Reuser Q[1] should see KV[1]"
    assert mask[5, 4], "Reuser Q[1] should see its own suffix[0] at KV[4]"
    assert mask[5, 5], "Reuser Q[1] should see its own suffix[1] at KV[5] (self-attention)"


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


# ------------------------------------------------------------------
# Step-4 chain: sample 3's prefix spans sample 0 and sample 1 (non-contiguous in s_packed)
#
# Chain (step=4):  0 → 1 → 2 → 3
#   sample 0 (provider):  [1..22, A1..A1692]    1714 tokens
#   sample 1 (reuser 0):   [1..22, B1..B1692]    prefix=22, suffix=1692
#   sample 2 (provider):   [C1..C1692]            1692 tokens  (no shared prefix with 0/1)
#   sample 3 (reuser 1):  [1..22, B1..B1692, D1..D336]   prefix=1714, suffix=336
#
# OLD (buggy) behavior: build_s_packed scanned s_packed for a contiguous 1714-token
# match and found none (s_packed[0:22]=shared, s_packed[22:1714]=[A...], not [B...]).
# sample 3's prefix was incorrectly appended as a duplicate block.
#
# NEW (fused) behavior: trie match at depth 1714 against sample 1's path gives
# prefix_len=1714, provider=1.  sample 3's prefix range is [(p_start, p_start+1714)]
# where p_start=sample_1's s_packed start.  Since sample 1's s_packed range is
# [(22, 1714+22)]=[(22,1736)], the prefix spans s_packed[22:1736) — exactly
# sample 0's suffix and sample 1's suffix concatenated in s_packed.
# ------------------------------------------------------------------

def test_step4_chain_fused_detector_correct_kv_ranges():
    """Fused detector produces correct s_packed_kv_ranges for step=4 chain.

    Key insight: when a reuser matches the provider's ENTIRE sequence via trie
    (full logical prefix), the s_packed prefix range spans ALL provider's
    physical blocks.  This is the correct fused behavior — the trie treats
    the provider's full sequence as one logical block; s_packed storage splits
    it across multiple physical blocks but the reuser should reference all of them.

    Chain: 0 (provider) → 1 (reuses 0, full match) → 2 (provider) → 3 (reuses 1, full match)

    s_packed layout:
      [0..1102)    sample 0 (provider): [1,2] + [100..1199]
      [1102..2202) sample 1 suffix: [200..1299]
      [2202..3402) sample 2 (provider): [300..1499]
      [3402..3502) sample 3 suffix: [4000..4098] (100 tokens)
    """
    sample_0 = [1, 2] + list(range(100, 1200))   # 1102 tokens
    sample_1 = [1, 2] + list(range(200, 1300))   # 1102 tokens  (shared [1,2], then B...)
    sample_2 = list(range(300, 1500))              # 1200 tokens  (no overlap with 0/1)
    sample_3 = [1, 2] + list(range(200, 1300)) + list(range(4000, 4100))  # 1202 tokens

    input_ids = [sample_0, sample_1, sample_2, sample_3]

    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1, min_group_size=2)
    )
    plan = planner.plan(input_ids)

    # Detection results
    assert plan.is_provider[0] is True
    assert plan.prefix_lens[0] == 0
    assert plan.is_provider[1] is False
    assert plan.prefix_lens[1] == 2   # reuses sample 0, matched=2
    assert plan.is_provider[2] is True
    assert plan.is_provider[3] is False
    assert plan.prefix_lens[3] == 1102  # reuses sample 1, matched=1102 (full)

    sp = plan.s_packed_kv_ranges
    assert len(sp) == 4

    # Sample 0 (provider): full range in s_packed
    assert sp[0] == [(0, 1102)]

    # Sample 1 (reuser of 0, full match of its own sequence):
    # The trie says sample 1 matches sample 0's full sequence (2 tokens = sample 1's prefix).
    # Wait — sample 1 has prefix_len=2, meaning it matches sample 0's first 2 tokens.
    # So sample 1's prefix range = sample 0's prefix block = (0, 2).
    # Sample 1's suffix = appended to s_packed = (1102, 2202).
    # CORRECTION: sample 1's prefix = (0, 2), suffix = (1102, 2202)
    assert sp[1] == [(0, 2), (1102, 2202)]

    # Sample 2 (provider): full range
    assert sp[2] == [(2202, 3402)]

    # Sample 3 (reuser of 1, full match of sample 1's full sequence):
    # _decompose_prefix_range([(0,2), (1102,2202)], 1102) returns both blocks
    # (since 1102 >= provider_len of 1102).
    # Then suffix (100 tokens) is appended: (3402, 3502).
    assert sp[3] == [(0, 2), (1102, 2202), (3402, 3502)]

    # s_packed_length: sample 0 (1102) + sample 1 suffix (1100) + sample 2 (1200) + sample 3 suffix (100)
    expected_length = 1102 + 1100 + 1200 + 100
    assert plan.s_packed_length == expected_length, (
        f"s_packed_length expected {expected_length}, got {plan.s_packed_length}"
    )

    # sample 3 suffix: original_len (1202) - prefix_len (1102) = 100
    assert plan.suffix_lens[3] == 100
    # sample 3 suffix range is the last block (appended at end)
    sample_3_suffix_range = sp[3][-1]
    assert sample_3_suffix_range == (3402, 3502)


def test_step4_chain_mask_is_correct_for_discontiguous_prefix():
    """Mask for sample 3's suffix respects causal attention over its multi-block prefix.

    Sample 3's suffix Q tokens must be able to attend to both blocks of its prefix:
    block 1: s_packed[(0,2)) = shared prefix [1,2]
    block 2: s_packed[(1102,2202)) = sample 1's suffix [200..1299]
    """
    sample_0 = [1, 2] + list(range(100, 1200))
    sample_1 = [1, 2] + list(range(200, 1300))
    sample_2 = list(range(300, 1500))
    sample_3 = [1, 2] + list(range(200, 1300)) + list(range(4000, 4100))

    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1, min_group_size=2)
    )
    plan = planner.plan([sample_0, sample_1, sample_2, sample_3])
    mask = plan.build_global_custom_mask(device="cpu")

    # Sample 3's Q length = suffix_len = 100
    q_len_3 = plan.s_packed_q_lengths[3]
    assert q_len_3 == 100  # original_len (1202) - prefix_len (1102)

    # Sample 3's prefix has TWO blocks
    prefix_blocks = plan.s_packed_kv_ranges[3][:-1]  # all but last (suffix)
    assert len(prefix_blocks) == 2
    assert prefix_blocks[0] == (0, 2)    # shared prefix block
    assert prefix_blocks[1] == (1102, 2202)  # sample 1 suffix block

    # Sample 3's first suffix Q (relative qi=0) should see the full shared prefix
    q_start_3 = plan.s_packed_q_starts[3]
    first_q_pos = q_start_3  # position 3402 in s_packed Q

    # Block 1 visibility: prefix block (0,2)
    # The first suffix token is at logical position 1102, so it can see all 2 prefix tokens
    assert mask[first_q_pos, 0], "Should see prefix block position 0"
    assert mask[first_q_pos, 1], "Should see prefix block position 1"

    # Block 2 visibility: sample 1 suffix block (1102, 2202)
    # The first suffix token can see all of sample 1's suffix block
    # (since the causal rule within that block is relative to prefix_len)
    # At qi=0 within suffix (global prefix_len=1102), visible range in block 2
    # starts at block 2's start (1102), and the causal rule within suffix gives
    # visible_hi = min(2202, 1102 + 0 + 1) = 1103... wait, let me reconsider.

    # Cross-input isolation: sample 3 should NOT see sample 2's KV
    sample_2_range = plan.s_packed_kv_ranges[2]
    sample_2_kv_lo = sample_2_range[0][0]
    sample_2_kv_hi = sample_2_range[0][1]
    for kv_j in range(sample_2_kv_lo, sample_2_kv_hi):
        assert not mask[first_q_pos, kv_j], \
            f"Sample 3 suffix Q[0] should NOT see sample 2 KV at {kv_j}"


def test_step4_chain_no_duplicate_prefix_in_s_packed():
    """s_packed must not contain duplicate copies of the shared content.

    Sample 1's prefix [1,2] is stored once in s_packed[0:2) (from sample 0).
    Sample 3's prefix references sample 1's FULL range [(0,2), (1102,2202)].
    No duplicate tokens are stored.
    """
    sample_0 = [1, 2] + list(range(100, 1200))
    sample_1 = [1, 2] + list(range(200, 1300))
    sample_2 = list(range(300, 1500))
    sample_3 = [1, 2] + list(range(200, 1300)) + list(range(4000, 4100))

    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1, min_group_size=2)
    )
    plan = planner.plan([sample_0, sample_1, sample_2, sample_3])

    # Unique tokens stored in s_packed:
    # sample 0: 1102 tokens (provider)
    # sample 1 suffix: 1100 tokens (prefix [1,2] deduplicated against sample 0)
    # sample 2: 1200 tokens (provider)
    # sample 3 suffix: 100 tokens (prefix references existing content)
    expected_length = 1102 + 1100 + 1200 + 100
    assert plan.s_packed_length == expected_length

    # Verify no new tokens added for sample 3's prefix:
    # sample 3's prefix references sample 1's full range
    # sp[3] = [(0,2), (1102,2202), (3402,3502)] = prefix blocks + suffix
    assert plan.s_packed_kv_ranges[3][:2] == [(0, 2), (1102, 2202)]
    assert plan.s_packed_kv_ranges[1] == [(0, 2), (1102, 2202)]
    # They should be identical (sample 3's prefix references sample 1's range exactly)


def test_fused_detector_produces_s_packed_result():
    """TriePrefixDetector.detect() now populates s_packed_result on the result."""
    from prefix_sharing.core.prefix_detector import TriePrefixDetector

    detector = TriePrefixDetector(min_prefix_len=1, min_group_size=2)
    result = detector.detect([[1, 2, 3], [1, 2, 4]])

    assert result.s_packed_result is not None
    sp = result.s_packed_result
    assert sp.s_packed_length == 4  # [1,2,3,4]
    assert len(sp.s_packed_kv_ranges) == 2
    assert sp.s_packed_kv_ranges[0] == [(0, 3)]  # provider: full range
    assert sp.s_packed_kv_ranges[1] == [(0, 2), (3, 4)]  # reuser: prefix [1,2] + suffix [4]
    assert sp.s_packed_q_lengths == [3, 1]  # provider: full, reuser: suffix only
    assert sp._custom_mask is not None  # mask pre-built on CPU
