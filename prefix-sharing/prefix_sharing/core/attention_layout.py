"""Backend-neutral sparse attention semantics for a prefix-sharing plan.

``PrefixSharingPlan`` describes trimming, provider relations, and restore
semantics.  This module derives the separate question required by sparse
attention backends: which deduplicated query tokens may attend to which
deduplicated key tokens.  It deliberately contains only immutable Python
values, so it can be shared by FlexAttention, future device backends, and CPU
reference tests without importing torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from prefix_sharing.core.planner import PrefixSharingPlan


class PrefixAttentionMaskType(str, Enum):
    """Visibility rule for one rectangular query/key slice."""

    CAUSAL = "causal"
    FULL = "full"


@dataclass(frozen=True)
class PrefixTreeAttentionSlice:
    """One non-overlapping visibility region of a prefix tree node.

    A node always owns one ``CAUSAL`` slice over its retained Q-path tokens.
    Every strict ancestor that contributes tokens to the node's original
    prefix contributes one ``FULL`` slice.  All offsets are deduplicated
    packed-token coordinates, with an exclusive end.
    """

    query_start: int
    query_end: int
    key_start: int
    key_end: int
    mask_type: PrefixAttentionMaskType

    def __post_init__(self) -> None:
        if self.query_start < 0 or self.key_start < 0:
            raise ValueError("attention slice bounds must be non-negative")
        if self.query_end <= self.query_start:
            raise ValueError("attention slice query range must be non-empty")
        if self.key_end <= self.key_start:
            raise ValueError("attention slice key range must be non-empty")


@dataclass(frozen=True)
class PrefixTreeAttentionLayout:
    """Sparse visibility layout over the plan's deduplicated Q/K/V tokens."""

    total_tokens: int
    node_ranges: tuple[tuple[int, int], ...]
    node_position_offsets: tuple[int, ...]
    parent_indices: tuple[int, ...]
    prefix_lengths: tuple[int, ...]
    attention_slices: tuple[PrefixTreeAttentionSlice, ...]
    logical_attention_elements: int
    max_depth: int
    signature: tuple[object, ...]

    def slices_for_node(
        self, node_index: int
    ) -> tuple[tuple[PrefixAttentionMaskType, int, int, int, int], ...]:
        """Return slices for one node in their deterministic construction order."""

        if not 0 <= node_index < len(self.node_ranges):
            raise IndexError(f"node_index={node_index} is outside the layout")
        query_start, query_end = self.node_ranges[node_index]
        return tuple(
            (
                slice_.mask_type,
                slice_.query_start,
                slice_.query_end,
                slice_.key_start,
                slice_.key_end,
            )
            for slice_ in self.attention_slices
            if slice_.query_start == query_start and slice_.query_end == query_end
        )

    def is_visible(self, query_index: int, key_index: int) -> bool:
        """Return token-level visibility for reference tests and mask builders."""

        if not 0 <= query_index < self.total_tokens:
            raise IndexError(f"query_index={query_index} is outside the layout")
        if not 0 <= key_index < self.total_tokens:
            raise IndexError(f"key_index={key_index} is outside the layout")

        for slice_ in self.attention_slices:
            if not (slice_.query_start <= query_index < slice_.query_end):
                continue
            if not (slice_.key_start <= key_index < slice_.key_end):
                continue
            if slice_.mask_type is PrefixAttentionMaskType.FULL:
                return True
            return key_index - slice_.key_start <= query_index - slice_.query_start
        return False

    def dense_visibility(self) -> tuple[tuple[bool, ...], ...]:
        """Materialize a small CPU reference mask; never use this in a backend."""

        return tuple(
            tuple(self.is_visible(query_index, key_index) for key_index in range(self.total_tokens))
            for query_index in range(self.total_tokens)
        )


def build_prefix_tree_attention_layout(
    prefix_sharing_plan: PrefixSharingPlan,
) -> PrefixTreeAttentionLayout:
    """Derive sparse tree visibility from a framework-independent plan.

    A reuser's retained node contains only its suffix.  Its original prefix is
    represented by the visible portions of its direct provider and all strict
    ancestors.  For an ancestor whose retained node begins at original offset
    ``p``, only the interval ``[p, descendant_prefix_len)`` is visible.  This
    prevents a provider's private suffix from leaking to a sibling reuser.
    """

    batch_size = prefix_sharing_plan.batch_size
    if batch_size == 0:
        return PrefixTreeAttentionLayout(
            total_tokens=0,
            node_ranges=(),
            node_position_offsets=(),
            parent_indices=(),
            prefix_lengths=(),
            attention_slices=(),
            logical_attention_elements=0,
            max_depth=0,
            signature=((), (), (), (), ()),
        )

    node_ranges = tuple(prefix_sharing_plan.q_range_for_batch(index) for index in range(batch_size))
    node_offsets = tuple(int(offset) for offset in prefix_sharing_plan.q_position_offsets)
    parent_indices = tuple(int(parent) for parent in prefix_sharing_plan.provider_index)
    prefix_lengths = tuple(int(prefix_len) for prefix_len in prefix_sharing_plan.prefix_lens)
    total_tokens = node_ranges[-1][1]

    _validate_plan_coordinates(
        prefix_sharing_plan,
        node_ranges=node_ranges,
        node_offsets=node_offsets,
        parent_indices=parent_indices,
        prefix_lengths=prefix_lengths,
    )

    slices: list[PrefixTreeAttentionSlice] = []
    logical_attention_elements = 0
    max_depth = 0

    for node_index, (node_start, node_end) in enumerate(node_ranges):
        node_length = node_end - node_start
        slices.append(
            PrefixTreeAttentionSlice(
                query_start=node_start,
                query_end=node_end,
                key_start=node_start,
                key_end=node_end,
                mask_type=PrefixAttentionMaskType.CAUSAL,
            )
        )
        logical_attention_elements += node_length * (node_length + 1) // 2

        lineage = _lineage_to_root(node_index, parent_indices)
        ancestors = lineage[1:]
        max_depth = max(max_depth, len(ancestors))
        descendant_prefix_len = prefix_lengths[node_index]
        for lineage_index, ancestor_index in enumerate(ancestors):
            ancestor_start, ancestor_end = node_ranges[ancestor_index]
            ancestor_offset = node_offsets[ancestor_index]
            child_toward_descendant = lineage[lineage_index]
            child_offset = node_offsets[child_toward_descendant]
            visible_end_in_original = min(
                descendant_prefix_len,
                child_offset,
                ancestor_offset + (ancestor_end - ancestor_start),
            )
            visible_length = visible_end_in_original - ancestor_offset
            if visible_length <= 0:
                continue
            key_end = ancestor_start + visible_length
            slices.append(
                PrefixTreeAttentionSlice(
                    query_start=node_start,
                    query_end=node_end,
                    key_start=ancestor_start,
                    key_end=key_end,
                    mask_type=PrefixAttentionMaskType.FULL,
                )
            )
            logical_attention_elements += node_length * visible_length

    immutable_slices = tuple(slices)
    signature = (
        node_ranges,
        node_offsets,
        parent_indices,
        prefix_lengths,
        tuple(
            (
                slice_.query_start,
                slice_.query_end,
                slice_.key_start,
                slice_.key_end,
                slice_.mask_type.value,
            )
            for slice_ in immutable_slices
        ),
    )
    return PrefixTreeAttentionLayout(
        total_tokens=total_tokens,
        node_ranges=node_ranges,
        node_position_offsets=node_offsets,
        parent_indices=parent_indices,
        prefix_lengths=prefix_lengths,
        attention_slices=immutable_slices,
        logical_attention_elements=logical_attention_elements,
        max_depth=max_depth,
        signature=signature,
    )


def _validate_plan_coordinates(
    prefix_sharing_plan: PrefixSharingPlan,
    *,
    node_ranges: tuple[tuple[int, int], ...],
    node_offsets: tuple[int, ...],
    parent_indices: tuple[int, ...],
    prefix_lengths: tuple[int, ...],
) -> None:
    """Validate only invariants required to turn a plan into a tree layout."""

    expected_start = 0
    for index, (node_start, node_end) in enumerate(node_ranges):
        if node_start != expected_start or node_end <= node_start:
            raise ValueError("node ranges must be contiguous, non-empty Q-path ranges")
        expected_start = node_end
        if node_offsets[index] < 0:
            raise ValueError("node position offsets must be non-negative")
        if prefix_lengths[index] < 0:
            raise ValueError("prefix lengths must be non-negative")

        parent_index = parent_indices[index]
        is_reuser = prefix_sharing_plan.is_reuser(index)
        if is_reuser:
            if not 0 <= parent_index < index:
                raise ValueError("reuser relations must satisfy provider-before-reuser ordering")
            if prefix_lengths[index] <= 0:
                raise ValueError("reuser relations require a non-zero prefix length")
        elif parent_index != index:
            raise ValueError("provider relations must satisfy provider-before-reuser ordering")


def _lineage_to_root(node_index: int, parent_indices: tuple[int, ...]) -> tuple[int, ...]:
    """Return node, direct provider, then outward ancestors to the root."""

    lineage = [node_index]
    current = node_index
    while parent_indices[current] != current:
        current = parent_indices[current]
        lineage.append(current)
    return tuple(lineage)


__all__ = [
    "PrefixAttentionMaskType",
    "PrefixTreeAttentionLayout",
    "PrefixTreeAttentionSlice",
    "build_prefix_tree_attention_layout",
]
