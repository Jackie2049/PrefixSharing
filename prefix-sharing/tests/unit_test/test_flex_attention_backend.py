"""CPU-safe unit tests for FlexAttention backend metadata and guards."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.flex_attention import (
    FlexAttentionBackend,
    FlexAttentionValidationError,
    _build_device_metadata,
)
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.attention_layout import build_prefix_tree_attention_layout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner


def _chain_layout():
    plan = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1)
    ).plan(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 100, 101, 102, 103],
            [1, 2, 3, 100, 101, 200],
        ]
    )
    return plan, build_prefix_tree_attention_layout(plan)


def test_device_metadata_preserves_chain_specific_ancestor_boundaries() -> None:
    _, layout = _chain_layout()

    token_nodes, token_positions, visible_ends = _build_device_metadata(
        layout,
        torch=torch,
        device=torch.device("cpu"),
    )

    assert token_nodes.tolist() == [0] * 8 + [1] * 4 + [2]
    assert token_positions.tolist() == list(range(8)) + [3, 4, 5, 6] + [5]
    # Deepest node (2) sees root only through original offset 3 and direct
    # provider through original offset 5.
    assert visible_ends[0, 2].item() == 3
    assert visible_ends[1, 2].item() == 5
    assert visible_ends[2, 2].item() == -1


def test_prepare_runtime_fails_fast_without_cuda() -> None:
    plan, layout = _chain_layout()
    packed_layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

    with pytest.raises(FlexAttentionValidationError, match="requires CUDA"):
        FlexAttentionBackend().prepare_runtime(
            prefix_tree_attention_layout=layout,
            packed_batch_layout=packed_layout,
            device=torch.device("cpu"),
        )
