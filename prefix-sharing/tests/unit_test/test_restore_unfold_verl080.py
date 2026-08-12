"""Unit tests for restore_via_2d_unfold_verl080.

Verifies the core logic of the v080 restore wrapper:
- Early return when no context / not a NestedTensor
- Round-trip symmetry of _unfold_trimmed_nested_to_2d + _fold_2d_to_nested
- Interior copies logp/entropy from provider
- Prefix-last recomputes logp using saved logits + label_value, entropy copied from provider
- After folding back, NestedTensor offsets are restored to full [prefix | suffix] length

Full PS accuracy comparison (vs baseline forward) belongs to integrated_test; it requires
a real verl/Megatron/GPU environment. This file only covers locally verifiable tensor logic.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo
from prefix_sharing.integrations.verl_mcore import (
    PrefixSharingRuntimeState,
    _fold_2d_to_nested,
    _unfold_trimmed_nested_to_2d,
    restore_via_2d_unfold_verl080,
)


def _make_state(sequences, min_prefix_len=3):
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=min_prefix_len)
    )
    plan = planner.plan(sequences, forward_id=10, micro_batch_id=20)
    return PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        attention_backend=None,
        packed_batch_layout=PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q),
        parallel_info=MegatronParallelInfo(),
    )


def _mock_vocab_log_probs_fn(logits, labels):
    """Simple log_softmax gather, simulating vocab_parallel_log_probs_from_logits.

    Labels are taken modulo to prevent out-of-bounds (small vocab dim in tests, large token ids).
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    labels = labels.long() % logits.size(-1)
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


# ═══════════════════════════════════════
# Early return paths
# ═══════════════════════════════════════


def test_no_context_returns_output_unchanged():
    """When contextvar has no context, return immediately without any processing."""
    output = {"log_probs": torch.tensor([1.0, 2.0])}
    result = restore_via_2d_unfold_verl080(output, _mock_vocab_log_probs_fn)
    assert result is output


def test_non_nested_log_probs_returns_unchanged():
    """When log_probs is not a NestedTensor, early return (possibly entered via v070 2D path)."""
    state = _make_state([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]])
    original = torch.zeros(2, 5)
    with prefix_sharing_runtime_context(state):
        output = {"log_probs": original.clone()}
        result = restore_via_2d_unfold_verl080(output, _mock_vocab_log_probs_fn)
    assert result is output
    assert torch.equal(result["log_probs"], original)


def test_no_sharing_returns_output_unchanged():
    """When plan has no reuser (has_sharing=False), early return; output unchanged."""
    # Two sequences with no common prefix -> no reuse -> has_sharing=False
    state = _make_state([[1, 2, 3, 10, 11], [4, 5, 6, 7, 8]])
    nested = torch.nested.nested_tensor(
        [torch.tensor([1.0, 2.0]), torch.tensor([3.0])], layout=torch.jagged
    )
    output = {"log_probs": nested}
    with prefix_sharing_runtime_context(state) as ctx:
        assert not ctx.prefix_sharing_plan.has_sharing
        result = restore_via_2d_unfold_verl080(output, _mock_vocab_log_probs_fn)
        assert result is output


# ═══════════════════════════════════════
# Round-trip symmetry
# ═══════════════════════════════════════


def test_unfold_fold_roundtrip_preserves_values():
    """Trimmed NestedTensor -> 2D left-pad -> fold back; values and shapes are correct.

    row0 provider: len=5 (full), right-padded to L_max=6
    row1 reuser:   suffix len=3, left-pad prefix_len=3 zeros
    """
    row0 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    row1 = torch.tensor([1.1, 1.2, 1.3])  # reuser suffix only
    nested = torch.nested.nested_tensor([row0, row1], layout=torch.jagged)

    original_lengths = [5, 6]
    input_keep_ranges = [(0, 5), (3, 6)]
    L_max = 6

    logp_2d, entropy_2d = _unfold_trimmed_nested_to_2d(
        nested, None, original_lengths, input_keep_ranges, L_max, 2,
    )
    assert entropy_2d is None
    assert logp_2d.shape == (2, 6)
    # provider row: original values + right-pad zeros
    assert torch.allclose(logp_2d[0], torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.0]))
    # reuser row: left-pad 3 zeros + suffix
    assert torch.allclose(logp_2d[1], torch.tensor([0.0, 0.0, 0.0, 1.1, 1.2, 1.3]))

    folded = _fold_2d_to_nested(logp_2d, original_lengths)
    assert folded.offsets().tolist() == [0, 5, 11]
    assert torch.allclose(folded.values()[0:5], row0)
    # After folding, reuser row has full length 6 (including left-padded prefix segment)
    assert torch.allclose(folded.values()[5:11], logp_2d[1, 0:6])


def test_unfold_handles_pure_reuser_zero_length_row():
    """Pure reuser (suffix_len=0) row has length 0 after trimming NestedTensor.

    Expands to an all-padding row; after restore, interior is filled, prefix-last may
    still be padding (consistent with baseline not computing next-token semantics).
    """
    # Row with suffix_len=0: suffix_data is empty
    row0 = torch.tensor([0.5, 0.6, 0.7])  # provider
    empty_row = torch.tensor([])  # reuser suffix is empty
    nested = torch.nested.nested_tensor([row0, empty_row], layout=torch.jagged)

    original_lengths = [3, 3]  # provider=3, reuser orig=3 (prefix3+suffix0)
    input_keep_ranges = [(0, 3), (3, 3)]  # reuser keep[3:3]=empty
    L_max = 3

    logp_2d, _ = _unfold_trimmed_nested_to_2d(
        nested, None, original_lengths, input_keep_ranges, L_max, 2,
    )
    assert logp_2d.shape == (2, 3)
    # reuser row is all zeros (left-pad 3 zeros, no suffix)
    assert torch.allclose(logp_2d[1], torch.zeros(3))


# ═══════════════════════════════════════
# Restore behavior (interior copy + prefix-last recompute)
# ═══════════════════════════════════════


def test_restore_copies_interior_and_recomputes_prefix_last():
    """Full restore: 2 interior copies + 1 prefix-last recompute.

    sequences: [[1,2,3,10,11], [1,2,3,20,21,22]]
      provider=row0 [1,2,3,10,11] len5
      reuser=row1  [1,2,3,20,21,22] len6, prefix_len=3, suffix=[20,21,22]
      restore: interior @ target_2d_pos=0,1 ; prefix-last @ target_2d_pos=2 (label=20)
    """
    state = _make_state([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]])
    plan = state.prefix_sharing_plan
    assert plan.original_lengths == [5, 6]
    assert plan.input_keep_ranges[1] == (3, 6)

    provider_logp = torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.5])
    reuser_suffix_logp = torch.tensor([-1.1, -1.2, -1.3])
    nested_logp = torch.nested.nested_tensor(
        [provider_logp, reuser_suffix_logp], layout=torch.jagged
    )
    output = {"log_probs": nested_logp}

    with prefix_sharing_runtime_context(state) as ctx:
        # Only 1 prefix-last entry (interior is handled by restore-side bulk slicing, no index built)
        assert len(ctx.prefix_last_restore_indices) == 1
        prefix_last_idx = ctx.prefix_last_restore_indices[0]
        assert prefix_last_idx.target_2d_pos == 2
        assert prefix_last_idx.label_value == 20  # input_ids[1][3]

        # Pre-fill saved logits (vocab dim=4), simulating results saved by vocab_logprobs patch
        saved_logits = torch.tensor([[0.5, 0.3, 0.1, 0.1]])  # [1, 4]
        ctx.prefix_last_logits_saved[
            (prefix_last_idx.reuse_idx_in_batch, prefix_last_idx.target_2d_pos)
        ] = saved_logits

        result = restore_via_2d_unfold_verl080(output, _mock_vocab_log_probs_fn)

    restored = result["log_probs"]
    offsets = restored.offsets()
    assert offsets.tolist() == [0, 5, 11]  # provider 5 + reuser full 6

    values = restored.values()
    # provider unchanged
    assert torch.allclose(values[0:5], provider_logp)

    reuser = values[5:11]
    # interior pos0,1 copied from provider
    assert torch.allclose(reuser[0], provider_logp[0])
    assert torch.allclose(reuser[1], provider_logp[1])
    # prefix-last pos2 recomputed: log_softmax(saved)[label=20%4=0]
    expected_plast = torch.log_softmax(saved_logits.float(), dim=-1)[0, 0]
    assert torch.allclose(reuser[2], expected_plast)
    # suffix preserved as-is
    assert torch.allclose(reuser[3:6], reuser_suffix_logp)


def test_restore_with_entropy_copies_both_logp_and_entropy():
    """Entropy sync restore: both interior and prefix-last are copied from provider (not recomputed)."""
    state = _make_state([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]])

    provider_logp = torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.5])
    reuser_logp = torch.tensor([-1.1, -1.2, -1.3])
    provider_ent = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    reuser_ent = torch.tensor([1.5, 1.6, 1.7])

    nested_logp = torch.nested.nested_tensor(
        [provider_logp, reuser_logp], layout=torch.jagged
    )
    nested_ent = torch.nested.nested_tensor(
        [provider_ent, reuser_ent], layout=torch.jagged
    )
    output = {"log_probs": nested_logp, "entropy": nested_ent}

    with prefix_sharing_runtime_context(state) as ctx:
        prefix_last_idx = ctx.prefix_last_restore_indices[0]
        ctx.prefix_last_logits_saved[
            (prefix_last_idx.reuse_idx_in_batch, prefix_last_idx.target_2d_pos)
        ] = torch.tensor([[0.5, 0.3, 0.1, 0.1]])

        result = restore_via_2d_unfold_verl080(output, _mock_vocab_log_probs_fn)

    ent_values = result["entropy"].values()
    ent_offsets = result["entropy"].offsets()
    assert ent_offsets.tolist() == [0, 5, 11]
    reuser_ent = ent_values[5:11]
    # interior + prefix-last entropy all copied from provider
    assert torch.allclose(reuser_ent[0], provider_ent[0])  # interior pos0
    assert torch.allclose(reuser_ent[1], provider_ent[1])  # interior pos1
    assert torch.allclose(reuser_ent[2], provider_ent[2])  # prefix-last pos2 (copied, not recomputed)
    # suffix entropy as-is
    assert torch.allclose(reuser_ent[3:6], torch.tensor([1.5, 1.6, 1.7]))


def test_restore_clears_saved_logits_is_callers_responsibility():
    """Wrapper does not clear prefix_last_logits_saved (caller is responsible for clearing after forward_step patch call).

    Verifies that saved dict is still non-empty after restore (cleanup is caller's responsibility).
    """
    state = _make_state([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]])
    nested_logp = torch.nested.nested_tensor(
        [torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.5]),
         torch.tensor([-1.1, -1.2, -1.3])],
        layout=torch.jagged,
    )
    output = {"log_probs": nested_logp}

    with prefix_sharing_runtime_context(state) as ctx:
        prefix_last_idx = ctx.prefix_last_restore_indices[0]
        key = (prefix_last_idx.reuse_idx_in_batch, prefix_last_idx.target_2d_pos)
        ctx.prefix_last_logits_saved[key] = torch.tensor([[0.5, 0.3, 0.1, 0.1]])
        restore_via_2d_unfold_verl080(output, _mock_vocab_log_probs_fn)
        # Wrapper does not clear; caller is responsible
        assert key in ctx.prefix_last_logits_saved
