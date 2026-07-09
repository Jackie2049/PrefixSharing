import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import (
    current_prefix_sharing_context,
    gather_prefix_last_logits_across_context_parallel,
    prefix_sharing_runtime_context,
)
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo
from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState


def _prefix_sharing_runtime_state():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    prefix_sharing_plan = planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]],
        forward_id=10,
        micro_batch_id=20,
    )
    return PrefixSharingRuntimeState(
        prefix_sharing_plan=prefix_sharing_plan,
        attention_backend=None,
        packed_batch_layout=PackedBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q),
        parallel_info=MegatronParallelInfo(pp_rank=1, pp_size=2, is_pipeline_first_stage=False),
    )


def test_prefix_sharing_runtime_context_sets_and_clears_current_context():
    prefix_sharing_runtime_state = _prefix_sharing_runtime_state()
    assert current_prefix_sharing_context() is None
    with prefix_sharing_runtime_context(prefix_sharing_runtime_state) as ctx:
        assert current_prefix_sharing_context() is ctx
        assert ctx.prefix_sharing_plan is prefix_sharing_runtime_state.prefix_sharing_plan
        # 1 prefix-last spec (interior is bulk-sliced in the restore, not indexed).
        assert len(ctx.prefix_last_restore_indices) == 1
        assert ctx.prefix_last_restore_indices[0].provider_1d_pos == 2  # prefix-last, direct provider seq0 offset 2
        assert ctx.prefix_last_restore_indices[0].target_2d_pos == 2
        assert ctx.prefix_last_restore_indices[0].label_value == 20  # seq1[3]
        assert ctx.parallel_info is prefix_sharing_runtime_state.parallel_info
        assert ctx.parallel_info.pp_rank == 1
        assert ctx.parallel_info.pp_size == 2
        assert ctx.stats.original_tokens == 11
        assert ctx.stats.kept_valid_tokens == 8
        assert ctx.stats.expected_reused_counts_per_layer == 1
        assert ctx.stats.expected_reused_prefix_tokens_per_layer == 3
        assert not ctx.store.closed
    assert current_prefix_sharing_context() is None
    assert ctx.store.closed


def test_prefix_sharing_runtime_context_uses_padded_layout_for_restore_indices():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    prefix_sharing_plan = planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]],
        forward_id=10,
        micro_batch_id=20,
    )
    runtime_state = PrefixSharingRuntimeState(
        prefix_sharing_plan=prefix_sharing_plan,
        attention_backend=None,
        packed_batch_layout=PackedBatchLayout(
            valid_lengths=[5, 2],
            padded_lengths=[6, 2],
            cu_seqlens=[0, 6, 8],
            max_seqlen=6,
        ),
        parallel_info=MegatronParallelInfo(),
    )

    with prefix_sharing_runtime_context(runtime_state) as ctx:
        # 1 prefix-last spec (interior bulk-sliced, not indexed).
        assert len(ctx.prefix_last_restore_indices) == 1
        assert ctx.prefix_last_restore_indices[0].provider_1d_pos == 2  # prefix-last, direct provider seq0 offset 2
        assert ctx.stats.kept_padded_tokens == 8


def test_prefix_sharing_runtime_context_builds_cp_local_logits_save_indices():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    prefix_sharing_plan = planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]],
        forward_id=10,
        micro_batch_id=20,
    )
    layout = PackedBatchLayout.from_kept_position_rows(
        [torch.arange(5), torch.arange(3, 5)],
        align_size=8,
        cp_rank=1,
        cp_size=2,
    )
    runtime_state = PrefixSharingRuntimeState(
        prefix_sharing_plan=prefix_sharing_plan,
        attention_backend=None,
        packed_batch_layout=layout,
        parallel_info=MegatronParallelInfo(cp_rank=1, cp_size=2),
    )

    with prefix_sharing_runtime_context(runtime_state) as ctx:
        assert len(ctx.prefix_last_restore_indices) == 1
        assert ctx.prefix_last_restore_indices[0].provider_1d_pos == 2
        assert len(ctx.prefix_last_logits_save_indices) == 1
        save_index = ctx.prefix_last_logits_save_indices[0]
        assert save_index.provider_global_1d_pos == 2
        assert save_index.provider_local_1d_pos == 0
        assert save_index.owner_cp_rank == 1
        assert save_index.target_2d_pos == 2


def test_prefix_sharing_runtime_context_skips_non_owner_cp_local_logits_save_index():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    prefix_sharing_plan = planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]],
        forward_id=10,
        micro_batch_id=20,
    )
    layout = PackedBatchLayout.from_kept_position_rows(
        [torch.arange(5), torch.arange(3, 5)],
        align_size=8,
        cp_rank=0,
        cp_size=2,
    )
    runtime_state = PrefixSharingRuntimeState(
        prefix_sharing_plan=prefix_sharing_plan,
        attention_backend=None,
        packed_batch_layout=layout,
        parallel_info=MegatronParallelInfo(cp_rank=0, cp_size=2),
    )

    with prefix_sharing_runtime_context(runtime_state) as ctx:
        save_index = ctx.prefix_last_logits_save_indices[0]
        assert save_index.provider_global_1d_pos == 2
        assert save_index.provider_local_1d_pos is None
        assert save_index.owner_cp_rank == 1


def test_gather_prefix_last_logits_across_context_parallel_fills_non_owner_rank(monkeypatch):
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    prefix_sharing_plan = planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]],
        forward_id=10,
        micro_batch_id=20,
    )
    layout = PackedBatchLayout.from_kept_position_rows(
        [torch.arange(5), torch.arange(3, 5)],
        align_size=8,
        cp_rank=0,
        cp_size=2,
    )
    runtime_state = PrefixSharingRuntimeState(
        prefix_sharing_plan=prefix_sharing_plan,
        attention_backend=None,
        packed_batch_layout=layout,
        parallel_info=MegatronParallelInfo(cp_rank=0, cp_size=2),
    )
    owner_logits = torch.randn(1, 7, requires_grad=True)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    import sys
    from types import ModuleType

    parallel_state = ModuleType("megatron.core.parallel_state")
    parallel_state.get_context_parallel_group = lambda: object()
    core = ModuleType("megatron.core")
    core.parallel_state = parallel_state
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)

    def fake_all_gather(local_logits, group=None):
        return (local_logits, owner_logits)

    monkeypatch.setattr("torch.distributed.nn.functional.all_gather", fake_all_gather)

    with prefix_sharing_runtime_context(runtime_state) as ctx:
        save_index = ctx.prefix_last_logits_save_indices[0]
        key = (save_index.reuse_idx_in_batch, save_index.target_2d_pos)
        assert key not in ctx.prefix_last_logits_saved

        gather_prefix_last_logits_across_context_parallel(ctx, torch.zeros(1, 7))

        assert ctx.prefix_last_logits_saved[key] is owner_logits


def test_chain_reuse_prefix_last_resolves_to_direct_provider():
    """Chain reuse: a reuser's prefix-last index points at its **direct**
    provider (not the root) and resolves to a real packed slot there.

    Trie detection produces a natural chain:
        seq0: [1, 2, 3, 4, 5]   provider (root)
        seq1: [1, 2, 3, 7, 8]   reuse(seq0), shared prefix [1,2,3] (len 3)
        seq2: [1, 2, 3, 7, 9]   reuse(seq1), shared prefix [1,2,3,7] (len 4)

    seq2's prefix-last is position 3 (predicts token 9). Chain reuse forces
    prefix_len_reuser (4) > prefix_len_provider (3), so position 3 lands at
    seq1's keep_start — inside seq1's computed suffix region. The prefix-last
    logits therefore live on the direct provider seq1 and need no walk up to
    seq0. Interior positions are no longer indexed: the restore bulk-slices
    them from the provider's already-restored 2D row.
    """
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2))
    plan = planner.plan(
        [[1, 2, 3, 4, 5], [1, 2, 3, 7, 8], [1, 2, 3, 7, 9]],
        forward_id=10,
        micro_batch_id=20,
    )
    # Chain: seq2 -> seq1 -> seq0; seq2 shares the longer prefix [1,2,3,7].
    assert plan.provider_index == [0, 0, 1]
    assert plan.prefix_lens == [0, 3, 4]

    runtime_state = PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        attention_backend=None,
        packed_batch_layout=PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q),
        parallel_info=MegatronParallelInfo(),
    )
    with prefix_sharing_runtime_context(runtime_state) as ctx:
        # One prefix-last index per reuser-with-suffix (seq1, seq2); interior
        # is no longer indexed.
        by_row = {
            idx.reuse_idx_in_batch: idx for idx in ctx.prefix_last_restore_indices
        }
        assert set(by_row) == {1, 2}

        # seq1 prefix-last: position 2 (predicts token 7), direct provider seq0.
        assert by_row[1].provider_idx_in_batch == 0
        assert by_row[1].target_2d_pos == 2
        assert by_row[1].label_value == 7
        assert by_row[1].provider_1d_pos != -1

        # seq2 prefix-last: position 3 (predicts token 9), direct provider seq1
        # (NOT the root seq0) — the key chain-reuse property.
        plast = by_row[2]
        assert plast.provider_idx_in_batch == 1
        assert plast.target_2d_pos == 3
        assert plast.label_value == 9
        assert plast.provider_1d_pos != -1, (
            "seq2 prefix-last must resolve to a real packed slot on its direct "
            "provider seq1, not walk up to the root seq0"
        )
