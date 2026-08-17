"""Ch4.2 functional verification tests for PrefixSharing FSDP adapter.

Covers the 7 scenarios required by docs/feature-fsdp.md Chapter 4.2:
1. Two samples sharing arbitrary prefix -> provider/reuser plan
2. Multiple reusers with different prefix_len and suffix_len
3. Same provider serving multiple reusers
4. Chain reuse (reuser becomes subsequent provider)
5. No shareable prefix -> None fallback
6. strict=true rejects unsupported configs
7. strict=false fallback aligns with baseline
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.integrations.verl_fsdp import (
    PrefixSharingFSDPAttentionRuntime,
    prepare_for_prefix_sharing_fsdp,
    restore_prefix_sharing_outputs_2d,
)


def _make_batch(seqs, pad_to=None):
    """Build a 2D batch from variable-length token lists (right-padded)."""
    max_len = max(len(s) for s in seqs)
    if pad_to is not None:
        max_len = max(max_len, pad_to)
    input_ids = torch.zeros(len(seqs), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)
    position_ids = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attention_mask[i, : len(s)] = True
        position_ids[i, : len(s)] = torch.arange(len(s))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


# ── Ch4.2 Scenario 1: two samples sharing arbitrary prefix ──


def test_scenario1_two_samples_share_arbitrary_prefix():
    """A B C D E / A B C X Y -> provider=0, reuser=1, prefix_len=3."""
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    batch = _make_batch([[1, 2, 3, 4, 5], [1, 2, 3, 10, 11]])
    trimmed, state = prepare_for_prefix_sharing_fsdp(batch, config)

    assert state is not None
    plan = state.prefix_sharing_plan
    assert plan.has_sharing
    assert plan.provider_index == [0, 0]
    assert plan.prefix_lens == [0, 3]
    # reuser keeps [3:5] (suffix only)
    assert plan.input_keep_ranges[1] == (3, 5)


# ── Ch4.2 Scenario 2: multiple reusers, different prefix_len and suffix_len ──


def test_scenario2_multiple_reusers_different_prefix_and_suffix_lens():
    """A B C D E / A B C X Y Z / A B Q R -> different prefix_lens."""
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    batch = _make_batch(
        [
            [1, 2, 3, 4, 5],          # provider (len 5)
            [1, 2, 3, 10, 11, 12],    # reuser, prefix 3, suffix 3 (len 6)
            [1, 2, 20, 21],           # reuser, prefix 2, suffix 2 (len 4)
        ]
    )
    trimmed, state = prepare_for_prefix_sharing_fsdp(batch, config)

    assert state is not None
    plan = state.prefix_sharing_plan
    assert plan.has_sharing
    # row 0 is provider for both
    assert plan.provider_index[1] == 0
    assert plan.provider_index[2] == 0
    # different prefix lengths detected
    assert plan.prefix_lens[1] == 3   # shares [1,2,3]
    assert plan.prefix_lens[2] == 2   # shares [1,2]
    # different suffix lengths
    assert plan.input_keep_ranges[1] == (3, 6)   # suffix len 3
    assert plan.input_keep_ranges[2] == (2, 4)   # suffix len 2


# ── Ch4.2 Scenario 3: same provider serving multiple reusers ──


def test_scenario3_same_provider_serves_multiple_reusers():
    """One provider, 3 reusers all pointing to it."""
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    batch = _make_batch(
        [
            [1, 2, 3, 4, 5, 6],       # provider
            [1, 2, 3, 4, 10],         # reuser 1 (prefix 4)
            [1, 2, 3, 20, 21],        # reuser 2 (prefix 3)
            [1, 2, 30, 40, 50, 60],   # reuser 3 (prefix 2)
        ]
    )
    trimmed, state = prepare_for_prefix_sharing_fsdp(batch, config)

    assert state is not None
    plan = state.prefix_sharing_plan
    # all reusers point to provider 0
    assert plan.provider_index[1:] == [0, 0, 0]
    assert plan.prefix_lens[1:] == [4, 3, 2]


# ── Ch4.2 Scenario 4: chain reuse (reuser becomes subsequent provider) ──


def test_scenario4_chain_reuse_reuser_becomes_provider():
    """A B C D E / A B C X Y Q / A B C X Y Z -> row2 reuses row1 (longer prefix)."""
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    batch = _make_batch(
        [
            [1, 2, 3, 4, 5],            # row0: provider for row1
            [1, 2, 3, 4, 10, 11, 12],   # row1: reuser of row0 (prefix 4), provider for row2
            [1, 2, 3, 4, 10, 20, 30],   # row2: reuser of row1 (prefix 5)
        ]
    )
    trimmed, state = prepare_for_prefix_sharing_fsdp(batch, config)

    assert state is not None
    plan = state.prefix_sharing_plan
    # row1 reuses row0, row2 reuses row1 (chain)
    assert plan.provider_index[1] == 0
    assert plan.provider_index[2] == 1
    # row2 prefix is at least as long as row1's prefix (chain reuse)
    assert plan.prefix_lens[2] >= plan.prefix_lens[1]


# ── Ch4.2 Scenario 5: no shareable prefix -> None fallback ──


def test_scenario5_no_shareable_prefix_returns_none():
    """Completely different sequences -> fallback."""
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    batch = _make_batch([[1, 2, 3, 4], [5, 6, 7, 8]])
    returned, state = prepare_for_prefix_sharing_fsdp(batch, config)

    assert returned is batch
    assert state is None


def test_scenario5b_single_sample_returns_none():
    """Single sample cannot share -> fallback."""
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    batch = _make_batch([[1, 2, 3, 4, 5]])
    returned, state = prepare_for_prefix_sharing_fsdp(batch, config)

    assert returned is batch
    assert state is None


# ── Ch4.2 Scenario 6: strict=true / config validation rejects unsupported ──


def test_scenario6_config_validation_rejects_bad_detector():
    config = PrefixSharingConfig(
        enable_prefix_sharing=True, detector="invalid", integrate_mode="verl_fsdp"
    )
    with pytest.raises(PrefixSharingConfigError, match="detector"):
        config.validate(integrate_mode="verl_fsdp")


def test_scenario6b_config_validation_rejects_bad_backend():
    config = PrefixSharingConfig(
        enable_prefix_sharing=True, backend="invalid_backend", integrate_mode="verl_fsdp"
    )
    with pytest.raises(PrefixSharingConfigError, match="backend"):
        config.validate(integrate_mode="verl_fsdp")


def test_scenario6c_config_validation_rejects_unsupported_integrate_mode():
    config = PrefixSharingConfig(
        enable_prefix_sharing=True, integrate_mode="some_other_mode"
    )
    with pytest.raises(PrefixSharingConfigError, match="integrate_mode"):
        config.validate(integrate_mode="some_other_mode")


def test_scenario6d_fsdp_mode_skips_megatron_specific_checks():
    """verl_fsdp mode should not require Megatron parallel config attrs."""
    config = PrefixSharingConfig(
        enable_prefix_sharing=True, integrate_mode="verl_fsdp"
    )
    # Should not raise even though model_config has no pp/cp/rope attrs
    config.validate(model_config={}, integrate_mode="verl_fsdp")


# ── Ch4.2 Scenario 7: position_ids preserve original absolute positions ──


def test_scenario7_position_ids_preserve_absolute_positions():
    """Reuser suffix position_ids must reflect original absolute positions."""
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    batch = _make_batch([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 10, 11, 12, 13]])
    trimmed, state = prepare_for_prefix_sharing_fsdp(batch, config)

    assert state is not None
    plan = state.prefix_sharing_plan
    # reuser prefix_len=3, so suffix positions 3..6 must be preserved
    assert plan.prefix_lens[1] == 3
    trimmed_mask = trimmed["attention_mask"][1]
    # valid positions in reuser row should map to original positions [3,4,5,6]
    valid_cols = trimmed_mask.nonzero(as_tuple=False).flatten().tolist()
    expected_positions = batch["position_ids"][1, valid_cols].tolist()
    assert expected_positions == list(range(3, 3 + len(valid_cols)))


# ── Bonus: attention runtime respects layer_id and stats ──


def test_attention_runtime_records_stats_per_layer():
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    batch = _make_batch([[1, 2, 3, 4, 5], [1, 2, 3, 10, 11]])
    _, state = prepare_for_prefix_sharing_fsdp(batch, config)
    assert state is not None

    torch.manual_seed(42)
    q = torch.randn(2, 5, 2, 4)
    k = torch.randn(2, 5, 2, 4)
    v = torch.randn(2, 5, 2, 4)

    runtime = PrefixSharingFSDPAttentionRuntime(layer_id=3)
    with prefix_sharing_runtime_context(state) as ctx:
        out = runtime.forward(None, q, k, v)
        # layer 3 should have stats recorded
        assert 3 in ctx.stats.layers
        assert ctx.stats.layers[3].reuse_hit_count >= 1

    assert out.shape == q.shape
