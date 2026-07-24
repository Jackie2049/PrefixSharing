"""Tests for prefix_block_mask: chain limit matrix + mask_mod semantics.

The mask_mod / cross_limit produced by ``prefix_sharing.backends.prefix_block_mask``
must reproduce, over the **unexpanded** packed kept-token stream, exactly the
visibility that ``build_kv`` + ``block_causal_mask`` express over the expanded
stream:

* same row: causal on absolute positions;
* cross row: reuser sees the provider chain's tokens whose absolute position
  falls inside the shared prefix window (chained reuse included).

All tests here are CPU-only and use an independent recursive oracle
(:func:`_logical_prefix_tokens`) — never the code under test — as reference.
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.prefix_block_mask import (
    build_chain_limit_matrix,
    build_token_index_tensors,
    get_or_create_block_mask,
    make_prefix_sharing_mask_mod,
)
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner


# ---------------------------------------------------------------------------
# Plan construction helpers
# ---------------------------------------------------------------------------

def _plan(sequences):
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1)
    planner = PrefixSharingPlanner(config)
    return planner.plan(sequences)


# ---------------------------------------------------------------------------
# Independent recursive oracle (NOT the code under test)
# ---------------------------------------------------------------------------

def _packed_tokens(plan):
    """(row, abs_pos) for every token of the packed kept-token stream."""
    tokens = []
    for row in range(plan.batch_size):
        offset = plan.q_position_offsets[row]
        for local_index in range(plan.kept_lengths_q[row]):
            tokens.append((row, offset + local_index))
    return tokens


def _logical_prefix_tokens(plan, row):
    """Set of (row, abs_pos) visible to *row* via its provider chain.

    A reuser's prefix window [0, prefix_len) of its logical sequence consists
    of its provider's own kept tokens below prefix_len plus (recursively) the
    provider's own prefix-chain tokens below prefix_len.
    """
    if not plan.is_reuser(row):
        return set()
    provider = plan.provider_index[row]
    limit = plan.prefix_lens[row]
    result = set()
    own_start = plan.q_position_offsets[provider]
    own_end = own_start + plan.kept_lengths_q[provider]
    for pos in range(own_start, min(own_end, limit)):
        result.add((provider, pos))
    for token_row, token_pos in _logical_prefix_tokens(plan, provider):
        if token_pos < limit:
            result.add((token_row, token_pos))
    return result


def _reference_dense_visibility(plan):
    """Dense (T, T) visibility oracle over the packed stream, True = visible."""
    tokens = _packed_tokens(plan)
    total = len(tokens)
    visible = torch.zeros(total, total, dtype=torch.bool)
    prefix_sets = [_logical_prefix_tokens(plan, row) for row in range(plan.batch_size)]
    for q_index, (row_q, pos_q) in enumerate(tokens):
        for kv_index, (row_kv, pos_kv) in enumerate(tokens):
            if row_q == row_kv:
                visible[q_index, kv_index] = pos_kv <= pos_q
            else:
                visible[q_index, kv_index] = (row_kv, pos_kv) in prefix_sets[row_q]
    return visible


def _mask_mod_dense(plan):
    """Evaluate the mask_mod under test over the full (T, T) grid."""
    token_row, token_pos = build_token_index_tensors(plan)
    cross_limit = build_chain_limit_matrix(plan)
    mask_mod = make_prefix_sharing_mask_mod(token_row, token_pos, cross_limit)
    total = token_row.shape[0]
    dummy = torch.zeros(total, total, dtype=torch.int64)
    q_index = torch.arange(total).unsqueeze(1).expand(total, total)
    kv_index = torch.arange(total).unsqueeze(0).expand(total, total)
    return mask_mod(dummy, dummy, q_index, kv_index)


# ---------------------------------------------------------------------------
# mask_mod semantics vs recursive oracle
# ---------------------------------------------------------------------------

def test_no_sharing_is_block_diagonal_causal():
    plan = _plan([[1, 2, 3, 4], [5, 6, 7]])
    reference = _reference_dense_visibility(plan)
    # Sanity: oracle itself is two independent causal blocks.
    assert reference[:4, :4].equal(torch.tril(torch.ones(4, 4, dtype=torch.bool)))
    assert reference[4:, 4:].equal(torch.tril(torch.ones(3, 3, dtype=torch.bool)))
    assert not reference[:4, 4:].any()
    assert not reference[4:, :4].any()
    assert _mask_mod_dense(plan).equal(reference)


def test_single_provider_with_reuser():
    provider = [1, 2, 3, 4, 5, 6]
    reuser = provider[:4] + [20, 21]
    plan = _plan([provider, reuser])
    assert plan.is_reuser(1)
    assert plan.prefix_lens[1] == 4
    reference = _reference_dense_visibility(plan)
    # Reuser Q rows (packed rows 6..7, abs pos 4..5) see provider tokens 0..3
    # plus their own causal suffix.
    assert reference[6, :4].all(), "reuser must see all provider prefix tokens"
    assert _mask_mod_dense(plan).equal(reference)


def test_chained_reuse():
    provider_a = [1, 2, 3, 4, 5, 6, 7, 8]
    reuser_b = provider_a[:5] + [20, 21, 22]
    reuser_c = reuser_b[:7] + [30, 31]
    plan = _plan([provider_a, reuser_b, reuser_c])
    assert plan.is_reuser(1) and plan.is_reuser(2)
    # Whatever provider the detector picked for C, semantics must match oracle.
    assert _mask_mod_dense(plan).equal(_reference_dense_visibility(plan))
    # And the reuse must actually span the chain: C's Q rows see tokens from
    # both A's region (abs pos < 5) and B's own region (abs pos 5..6).
    tokens = _packed_tokens(plan)
    c_first_q = next(i for i, (row, _) in enumerate(tokens) if row == 2)
    reference = _reference_dense_visibility(plan)
    a_cols = [i for i, (row, pos) in enumerate(tokens) if row == 0 and pos < 5]
    b_cols = [i for i, (row, pos) in enumerate(tokens) if row == 1 and pos < 7]
    assert reference[c_first_q, a_cols].all()
    assert reference[c_first_q, b_cols].all()


def test_reuser_of_reuser_with_shorter_prefix():
    provider_a = [1, 2, 3, 4, 5, 6, 7, 8]
    reuser_b = provider_a[:5] + [20, 21, 22]
    reuser_c = reuser_b[:3] + [30, 31]
    plan = _plan([provider_a, reuser_b, reuser_c])
    reference = _reference_dense_visibility(plan)
    assert _mask_mod_dense(plan).equal(reference)
    # C's window (3) is shorter than B's own prefix (5): C must NOT see any of
    # B's own tokens (abs pos >= 5).
    tokens = _packed_tokens(plan)
    c_first_q = next(i for i, (row, _) in enumerate(tokens) if row == 2)
    b_own_cols = [i for i, (row, pos) in enumerate(tokens) if row == 1 and pos >= 5]
    assert not reference[c_first_q, b_own_cols].any()


# ---------------------------------------------------------------------------
# cross_limit matrix vs brute force
# ---------------------------------------------------------------------------

def _brute_force_limit(plan, row, kv_row):
    positions = [pos for (r, pos) in _logical_prefix_tokens(plan, row) if r == kv_row]
    return max(positions) + 1 if positions else 0


@pytest.mark.parametrize(
    "sequences",
    [
        [[1, 2, 3, 4], [5, 6, 7]],
        [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 20, 21]],
        [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 20, 21, 22], [1, 2, 3, 4, 5, 20, 21, 30, 31]],
        [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 20, 21, 22], [1, 2, 3, 30, 31]],
    ],
)
def test_chain_limit_matrix_matches_brute_force(sequences):
    plan = _plan(sequences)
    cross_limit = build_chain_limit_matrix(plan)
    for row in range(plan.batch_size):
        for kv_row in range(plan.batch_size):
            assert int(cross_limit[row, kv_row]) == _brute_force_limit(plan, row, kv_row), (
                f"cross_limit[{row}, {kv_row}] mismatch"
            )


def test_provider_before_reuser_violation_raises():
    plan = _plan([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 20, 21]])
    # Forge an invalid plan: row 0 reuses the later row 1.
    bad_plan = dataclasses.replace(
        plan,
        provider_index=[1, 1],
        prefix_lens=[4, 4],
    )
    with pytest.raises(ValueError, match="provider-before-reuser"):
        build_chain_limit_matrix(bad_plan)


# ---------------------------------------------------------------------------
# BlockMask integration (torch flex_attention, CPU)
# ---------------------------------------------------------------------------

def test_block_mask_non_multiple_of_block_size():
    pytest.importorskip("torch.nn.attention.flex_attention")
    provider = list(range(1, 60))
    reuser = provider[:40] + list(range(100, 130))
    plan = _plan([provider, reuser])
    total = plan.cu_seqlens_q[-1]
    assert total % 128 != 0, "test requires a non-128-multiple packed length"

    block_mask = get_or_create_block_mask(plan, device="cpu")
    assert block_mask.seq_lengths == (total, total)
    # The internal dense materialization happens at create time and includes
    # padding to block-size multiples.  We only assert construction succeeds;
    # exact mask_mod semantics are already checked by _mask_mod_dense above.


def test_block_mask_cache_reuses_entry_for_same_plan():
    pytest.importorskip("torch.nn.attention.flex_attention")
    plan = _plan([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 20, 21]])
    cache = {}
    first = get_or_create_block_mask(plan, device="cpu", cache=cache)
    second = get_or_create_block_mask(plan, device="cpu", cache=cache)
    assert first is second
    assert len(cache) == 1
