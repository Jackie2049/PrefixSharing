from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner


def test_planner_builds_phase_one_prefix_sharing_plan_and_restore_specs():
    input_ids = [
        [1, 2, 3, 4, 5, 10, 11],
        [1, 2, 3, 20, 21, 22],
        [1, 2, 3, 4, 5, 30, 31],
        [9, 9, 9],
    ]
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    prefix_sharing_plan = planner.plan(input_ids, forward_id=7, micro_batch_id=3)

    assert prefix_sharing_plan.forward_id == 7
    assert prefix_sharing_plan.micro_batch_id == 3
    assert prefix_sharing_plan.original_lengths == [7, 6, 7, 3]
    assert [(s.reuse_idx_in_batch, s.provider_idx_in_batch, s.prefix_len) for s in prefix_sharing_plan.reuse_specs] == [
        (1, 0, 3),
        (2, 0, 5),
    ]
    assert prefix_sharing_plan.is_provider == [True, False, False, True]
    assert prefix_sharing_plan.provider_index == [0, 0, 0, 3]
    assert prefix_sharing_plan.prefix_lens == [0, 3, 5, 0]
    assert prefix_sharing_plan.suffix_lens == [7, 3, 2, 3]
    assert prefix_sharing_plan.input_keep_ranges == [(0, 7), (3, 6), (5, 7), (0, 3)]
    assert prefix_sharing_plan.kept_lengths_q == [7, 3, 2, 3]
    assert prefix_sharing_plan.expanded_lengths_kv == [7, 6, 7, 3]
    # cu_seqlens_q: cumsum of kept_lengths_q (reuser Q packs suffix only) -> total 15.
    # cu_seqlens_kv: cumsum of original_lengths (full logical KV: injected prefix + suffix) -> 23.
    assert prefix_sharing_plan.cu_seqlens_q == [0, 7, 10, 12, 15]
    assert prefix_sharing_plan.cu_seqlens_kv == [0, 7, 13, 20, 23]
    # q_position_offsets: logical position of the first packed Q token (reuser == prefix_len).
    # kv_position_offsets: KV always numbered from logical position 0.
    assert prefix_sharing_plan.q_position_offsets == [0, 3, 5, 0]
    assert prefix_sharing_plan.kv_position_offsets == [0, 0, 0, 0]

    # prefix_last_restore: reuser Q skips prefix; only the prefix-last token
    # needs a spec (interior columns are bulk-sliced by the 2D restore).
    # Row 1 (prefix_len=3): 1 prefix-last spec.
    # Row 2 (prefix_len=5): 1 prefix-last spec.
    # Total: 2 specs.
    all_specs = prefix_sharing_plan.prefix_last_restore
    assert len(all_specs) == 2

    # Row 1 prefix-last spec
    spec1 = all_specs[0]
    assert spec1.reuse_idx_in_batch == 1
    assert spec1.provider_idx_in_batch == 0
    assert spec1.target_2d_pos == 2  # prefix_len - 1
    assert spec1.label_value == 20  # input_ids[1][3], the first suffix token

    # Row 2 prefix-last spec
    spec2 = all_specs[1]
    assert spec2.reuse_idx_in_batch == 2
    assert spec2.provider_idx_in_batch == 0
    assert spec2.target_2d_pos == 4  # prefix_len - 1
    assert spec2.label_value == 30  # input_ids[2][5], the first suffix token


def test_planner_no_shared_prefix_keeps_original_shapes():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2))
    prefix_sharing_plan = planner.plan([[1, 2], [1, 3], [4, 5]], forward_id=1, micro_batch_id=1)

    assert not prefix_sharing_plan.has_sharing
    assert prefix_sharing_plan.group_ids == [-1, -1, -1]
    assert prefix_sharing_plan.input_keep_ranges == [(0, 2), (0, 2), (0, 2)]
    assert prefix_sharing_plan.reuse_specs == []
    assert prefix_sharing_plan.prefix_last_restore == []


def test_planner_emits_only_prefix_last_not_interior():
    # seq1 (provider): [1,2,3 | A,B,C]   prompt=[1,2,3], response=[A,B,C]
    # seq2 (reuser):   [1,2,3 | A,D,E]   prompt=[1,2,3], response=[A,D,E]
    # Shared prefix: [1,2,3,A] (len 4). A is a response token in both, so the
    # prefix spans response positions — but interior columns are now bulk-sliced
    # by the 2D restore; the planner only emits the single prefix-last spec.
    input_ids = [
        [1, 2, 3, 4, 5, 6],     # 1,2,3,A,B,C
        [1, 2, 3, 4, 7, 8],     # 1,2,3,A,D,E
    ]
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    )
    plan = planner.plan(input_ids, forward_id=1, micro_batch_id=1)

    # Only the prefix-last spec is emitted (interior is handled by the restore).
    # Total: 1 spec.
    assert len(plan.prefix_last_restore) == 1

    prefix_last_spec = plan.prefix_last_restore[0]
    assert prefix_last_spec.reuse_idx_in_batch == 1
    assert prefix_last_spec.provider_idx_in_batch == 0
    assert prefix_last_spec.target_2d_pos == 3  # prefix_len - 1
    assert prefix_last_spec.label_value == 7  # input_ids[1][4] = first suffix token D

    # Trimming still at prefix_len
    assert plan.input_keep_ranges == [(0, 6), (4, 6)]
    assert plan.kept_lengths_q == [6, 2]


def test_planner_emits_prefix_last_with_minimal_args():
    """Only the prefix-last spec is emitted; interior needs no spec."""
    input_ids = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 6],
    ]
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    )
    plan = planner.plan(input_ids, forward_id=1, micro_batch_id=1)

    # Only the prefix-last spec is emitted (interior handled by the restore).
    assert len(plan.prefix_last_restore) == 1
    spec = plan.prefix_last_restore[0]
    assert spec.reuse_idx_in_batch == 1
