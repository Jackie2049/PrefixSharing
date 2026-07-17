"""Tests for BSHD integration seam: format resolution, restore, vocab indexing, context."""

from __future__ import annotations

import math

import pytest
import torch

from prefix_sharing.backends.batched_layout import BatchedBatchLayout
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import (
    PackedPrefixLastRestoreIndex,
    PrefixSharingRuntimeContext,
    _build_prefix_last_restore_indices,
    prefix_sharing_runtime_context,
)
from prefix_sharing.integrations.megatron_runtime import _prefix_attention_bshd
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo
from prefix_sharing.integrations.verl_mcore import (
    PrefixSharingRuntimeState,
    _resolve_batch_format,
    restore_via_bshd,
)

# We do NOT import megatron attention modules here; the _prefix_attention_bshd
# integration is tested indirectly via the factory / context plumbing tests
# below.  The full forward pass (rotary_pos_emb, linear_proj) requires a
# Megatron runtime environment and is exercised in the NPU/GPU integrated
# test suite.


# ── _resolve_batch_format ──────────────────────────────────────────────


class TestResolveBatchFormat:
    def test_thd_config_returns_thd(self):
        cfg = PrefixSharingConfig(enable_prefix_sharing=True, batch_format="thd")
        assert _resolve_batch_format(cfg, True) == "thd"
        assert _resolve_batch_format(cfg, False) == "thd"  # explicit wins

    def test_bshd_config_returns_bshd(self):
        cfg = PrefixSharingConfig(enable_prefix_sharing=True, batch_format="bshd")
        assert _resolve_batch_format(cfg, True) == "bshd"
        assert _resolve_batch_format(cfg, False) == "bshd"

    def test_auto_with_remove_padding_true_returns_thd(self):
        cfg = PrefixSharingConfig(enable_prefix_sharing=True, batch_format="auto")
        assert _resolve_batch_format(cfg, True) == "thd"

    def test_auto_with_remove_padding_false_returns_bshd(self):
        cfg = PrefixSharingConfig(enable_prefix_sharing=True, batch_format="auto")
        assert _resolve_batch_format(cfg, False) == "bshd"


# ── context: _build_prefix_last_restore_indices ───────────────────────


def _shared_plan():
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    )
    return planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]],
        forward_id=1, micro_batch_id=1,
    )


class TestBuildPrefixLastRestoreIndicesBshd:
    def test_bshd_identity_mapping(self):
        """BSHD layout: provider_1d_pos == target_2d_pos (identity)."""
        plan = _shared_plan()
        layout = BatchedBatchLayout.from_valid_lengths(plan.original_lengths)
        indices = _build_prefix_last_restore_indices(plan, layout)
        assert len(indices) == 1
        spec = plan.prefix_last_restore[0]
        idx = indices[0]
        assert idx.provider_1d_pos == spec.target_2d_pos  # identity
        assert idx.target_2d_pos == spec.target_2d_pos
        assert idx.reuse_idx_in_batch == spec.reuse_idx_in_batch

    def test_thd_packed_index_unchanged(self):
        """THD layout: provider_1d_pos computed from packed_index (regression)."""
        plan = _shared_plan()
        kept_rows = [
            torch.tensor(list(range(r[0], r[1])), dtype=torch.long)
            for r in plan.input_keep_ranges
        ]
        layout = PackedBatchLayout.from_kept_position_rows(kept_rows, align_size=1)
        indices = _build_prefix_last_restore_indices(plan, layout)
        assert len(indices) == 1
        # provider row0, offset=2 within kept range → packed_index(0,2) = 0+2 = 2
        assert indices[0].provider_1d_pos == 2


# ── restore_via_bshd ──────────────────────────────────────────────────


class TestRestoreViaBshd:
    def test_no_ctx_early_return(self):
        output = {"log_probs": torch.randn(3)}
        result = restore_via_bshd(output, None)
        assert result is output

    def test_no_sharing_early_return(self):
        planner = PrefixSharingPlanner(
            PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=100)
        )
        plan = planner.plan(
            [[1, 2, 3], [4, 5, 6]],
            forward_id=1, micro_batch_id=1,
        )
        assert not plan.has_sharing
        state = PrefixSharingRuntimeState(
            prefix_sharing_plan=plan,
            attention_backend=None,
            packed_batch_layout=BatchedBatchLayout.from_valid_lengths(plan.original_lengths),
            parallel_info=MegatronParallelInfo(),
        )
        from prefix_sharing.core.prefix_store import PrefixAttentionStore
        with prefix_sharing_runtime_context(state) as ctx:
            output = {"log_probs": torch.nested.as_nested_tensor(
                [torch.randn(3), torch.randn(3)], layout=torch.jagged,
            )}
            result = restore_via_bshd(output, lambda x, y: torch.tensor(0.0))
            assert result is output  # early return, unchanged

    def test_nested_restore_round_trip(self):
        """NestedTensor → unfold → restore (identity caller) → fold → nested."""
        planner = PrefixSharingPlanner(
            PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
        )
        # Two rows, share 2-token prefix, lengths 4 and 5.
        plan = planner.plan(
            [[1, 2, 10, 11], [1, 2, 20, 21, 22]],
            forward_id=1, micro_batch_id=1,
        )
        assert plan.has_sharing

        layout = BatchedBatchLayout.from_valid_lengths(plan.original_lengths)
        state = PrefixSharingRuntimeState(
            prefix_sharing_plan=plan,
            attention_backend=None,
            packed_batch_layout=layout,
            parallel_info=MegatronParallelInfo(),
        )
        from prefix_sharing.core.prefix_store import PrefixAttentionStore

        # Provider has log_probs [a0,a1,a2,a3]; reuser has [z0,z1,z2,z3,z4].
        lp_provider = torch.tensor([1., 2., 3., 4.])
        lp_reuser = torch.tensor([9., 9., 8., 7., 6.])
        lp_nested = torch.nested.as_nested_tensor([lp_provider, lp_reuser], layout=torch.jagged)
        output = {"log_probs": lp_nested}

        # Fake prefix-last saved logits: reuser row=1, col=1 (prefix_len-1=1)
        # provider logits value for label=20 would be computed by vocab_logprobs patch.
        # Simulate a simple identity log_prob function.
        with prefix_sharing_runtime_context(state) as ctx:
            # Saved logit [1, 1] (1 token, vocab=1).  Log_prob function returns a scalar.
            ctx.prefix_last_logits_saved[(1, 1)] = torch.tensor([[1.0]])  # [1, 1]
            result = restore_via_bshd(
                output,
                vocab_parallel_log_probs_fn=lambda logits, label: logits.reshape(()),
            )

        restored = result["log_probs"]
        # Unfold the result for checking.
        vals = restored.values()
        offs = restored.offsets()
        row0 = vals[offs[0]:offs[1]]
        row1 = vals[offs[1]:offs[2]]

        assert row0.tolist() == [1., 2., 3., 4.]  # provider unchanged
        # Interior [0, 0] (prefix_len-2) bulk-copied from provider → row1[0] = row0[0] = 1.
        assert row1[0].item() == 1.0
        # Prefix-last col 1 recomputed from saved logit [1.0] → logp=1.0.
        assert row1[1].item() == 1.0
        # Suffix [2,3,4] unchanged (original reuser values).
        assert row1[2:].tolist() == [8., 7., 6.]


# ── runtime state compatibility ───────────────────────────────────────


class TestRuntimeStateBshdCompat:
    def test_bshd_layout_works_with_runtime_state(self):
        plan = _shared_plan()
        layout = BatchedBatchLayout.from_valid_lengths(plan.original_lengths)
        state = PrefixSharingRuntimeState(
            prefix_sharing_plan=plan,
            attention_backend=None,
            packed_batch_layout=layout,
            parallel_info=MegatronParallelInfo(),
        )
        assert state.prefix_sharing_plan.has_sharing
        assert state.packed_batch_layout.is_bshd()
        assert state.packed_batch_layout.batch_size == 2

    def test_thd_layout_works_with_runtime_state(self):
        plan = _shared_plan()
        layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
        state = PrefixSharingRuntimeState(
            prefix_sharing_plan=plan,
            attention_backend=None,
            packed_batch_layout=layout,
            parallel_info=MegatronParallelInfo(),
        )
        assert not state.packed_batch_layout.is_bshd()


# ── Forward step probe diagnostic compatibility ────────────────────────


class TestForwardStepProbeBshd:
    def test_probe_compatible_with_bshd_layout(self):
        """The diagnostics probe must not crash on BSHD layout (no padded_lengths/cu_seqlens)."""
        plan = _shared_plan()
        layout = BatchedBatchLayout.from_valid_lengths(plan.original_lengths)
        state = PrefixSharingRuntimeState(
            prefix_sharing_plan=plan,
            attention_backend=None,
            packed_batch_layout=layout,
            parallel_info=MegatronParallelInfo(),
        )
        # Simulate what forward_step's probe does.
        layout_str = (
            f"valid={state.packed_batch_layout.valid_lengths},"
            f"type={'BSHD' if getattr(state.packed_batch_layout, 'is_bshd', lambda: False)() else 'THD'}"
        )
        assert "BSHD" in layout_str
        assert "valid" in layout_str
