"""Framework-independent sparse attention layout tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from prefix_sharing.core.attention_layout import (
    PrefixAttentionMaskType,
    build_prefix_tree_attention_layout,
)
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner


def _planner() -> PrefixSharingPlanner:
    return PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1)
    )


def test_star_layout_encodes_causal_self_and_only_visible_provider_prefix() -> None:
    plan = _planner().plan(
        [
            [1, 2, 3, 4, 5, 10, 11],
            [1, 2, 3, 20, 21, 22],
            [1, 2, 3, 4, 5, 30, 31],
            [9, 9, 9],
        ]
    )

    layout = build_prefix_tree_attention_layout(plan)

    assert layout.total_tokens == 15
    assert layout.node_ranges == ((0, 7), (7, 10), (10, 12), (12, 15))
    assert layout.node_position_offsets == (0, 3, 5, 0)
    assert layout.parent_indices == (0, 0, 0, 3)

    assert layout.slices_for_node(1) == (
        (PrefixAttentionMaskType.CAUSAL, 7, 10, 7, 10),
        (PrefixAttentionMaskType.FULL, 7, 10, 0, 3),
    )
    assert layout.slices_for_node(2) == (
        (PrefixAttentionMaskType.CAUSAL, 10, 12, 10, 12),
        (PrefixAttentionMaskType.FULL, 10, 12, 0, 5),
    )

    # Row 1 can see its provider's shared prefix and its own causal history.
    assert layout.is_visible(7, 0)
    assert layout.is_visible(7, 2)
    assert not layout.is_visible(7, 3)  # provider private suffix is not visible.
    assert layout.is_visible(7, 7)
    assert not layout.is_visible(7, 8)
    assert layout.is_visible(9, 8)
    # Siblings and independent trees remain invisible.
    assert not layout.is_visible(7, 10)
    assert not layout.is_visible(10, 7)
    assert not layout.is_visible(7, 12)


def test_chain_layout_reaches_strict_ancestors_without_leaking_private_suffixes() -> None:
    plan = _planner().plan(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 100, 101, 102, 103],
            [1, 2, 3, 100, 101, 200],
        ]
    )
    assert plan.provider_index == [0, 0, 1]
    assert plan.prefix_lens == [0, 3, 5]

    layout = build_prefix_tree_attention_layout(plan)

    # The deepest leaf sees root tokens [0, 3) and row 1 tokens that make up
    # positions [3, 5), but not row 1's private continuation.
    deepest_start, _ = layout.node_ranges[2]
    assert layout.is_visible(deepest_start, 0)
    assert layout.is_visible(deepest_start, 2)
    assert layout.is_visible(deepest_start, 8)
    assert layout.is_visible(deepest_start, 9)
    assert not layout.is_visible(deepest_start, 10)

    full_slices = [
        slice_
        for slice_ in layout.attention_slices
        if slice_.query_start == deepest_start
        and slice_.mask_type is PrefixAttentionMaskType.FULL
    ]
    assert [(slice_.key_start, slice_.key_end) for slice_ in full_slices] == [(8, 10), (0, 3)]


def test_no_sharing_layout_contains_only_per_row_causal_slices() -> None:
    plan = _planner().plan([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    assert not plan.has_sharing

    layout = build_prefix_tree_attention_layout(plan)

    assert len(layout.attention_slices) == plan.batch_size
    assert all(slice_.mask_type is PrefixAttentionMaskType.CAUSAL for slice_ in layout.attention_slices)
    assert layout.logical_attention_elements == 6 + 3 + 10
    assert not layout.is_visible(0, 3)
    assert not layout.is_visible(3, 0)


def test_signature_captures_tree_visibility_not_only_total_token_count() -> None:
    planner = _planner()
    star = planner.plan([[1, 2, 3, 10], [1, 2, 3, 20]])
    no_sharing = planner.plan([[1, 2, 3, 10], [9]])

    star_layout = build_prefix_tree_attention_layout(star)
    no_sharing_layout = build_prefix_tree_attention_layout(no_sharing)

    assert star_layout.total_tokens == no_sharing_layout.total_tokens
    assert star_layout.signature != no_sharing_layout.signature


def test_layout_rejects_non_topological_or_cyclic_provider_relation() -> None:
    plan = _planner().plan([[1, 2, 3, 4], [1, 2, 3, 5]])
    invalid_plan = replace(plan, provider_index=[1, 0])

    with pytest.raises(ValueError, match="provider-before-reuser"):
        build_prefix_tree_attention_layout(invalid_plan)
