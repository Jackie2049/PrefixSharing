"""Prefix detection for shared token sequence identification.

This module provides algorithms to detect shared prefixes among token sequences
in a **batch** (multiple sequences processed together in one detection pass).
For each reuser sequence, the detector records one **reuse relation**: which
previous provider sequence can supply a reusable prefix slice and how long that
slice is. A provider may serve different reusers with different prefix lengths.

Core Responsibilities:
    1. Identify common prefixes across multiple token sequences.
    2. Emit per-sample reuse relations using configurable thresholds.
    3. Preserve compatibility group fields for diagnostics, while keeping
       relation data as the semantic source of truth.

Key Concepts:
    - Provider: The earlier sequence in a reuse relation whose logical KV can
      supply a prefix slice to a later sequence.
    - Reuser: A sequence that reuses the provider's prefix KV.
      Reusers skip prefix computation and attend to the provider's KV cache.

Key Components:
    - PrefixReuseSpec: Represents one relation
      ``(reuse_idx_in_batch, provider_idx_in_batch, prefix_len)``.
    - PrefixGroup: Compatibility/debug view grouping relations with identical
      ``(provider_index, prefix_len)``.
    - PrefixDetectionResult: Container for detection output with per-sequence
      metadata including group membership, provider assignment, and reuse flags.
    - PrefixDetector: Abstract base class defining the detector interface.
    - TriePrefixDetector: Concrete implementation using a trie data structure.
    - common_prefix_len: Utility to compute common prefix length across sequences.

Design Principles:
    - Per-sample relation first: ``provider_index[i]`` and ``prefix_lens[i]``
      are the authoritative plan for each row. Groups are secondary.
    - Online provider selection: Phase 1 follows PrefixTrain_dev's practical
      approach--a sequence may reuse the longest prefix found in earlier
      sequences, then becomes available as a provider for later sequences.
    - Multi-length provider reuse: The same provider can serve prefix ``0..5``
      to one reuser and ``0..10`` to another.
    - Configurable thresholds: Minimum prefix length and group size allow
      tuning the detection behavior for different workloads.

Example:
    >>> detector = TriePrefixDetector(min_prefix_len=3, min_group_size=2)
    >>> sequences = [[1, 2, 3, 4, 5], [1, 2, 3, 6, 7], [8, 9, 10]]
    >>> result = detector.detect(sequences)
    >>> # sequences[1] reuses sequences[0] prefix [1, 2, 3] with length 3
    >>> # sequences[0] is the provider, sequences[1] is a reuser
    >>> result.prefix_lens
    (3, 3, 0)
    >>> result.is_provider
    (True, False, True)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence


TokenSequence = Sequence[int]


class PrefixDetector(ABC):
    """Abstract base class for prefix detectors.

    Subclasses implement ``detect()`` to analyze a batch of token sequences
    and identify groups that can share prefix KV caches.
    """

    @abstractmethod
    def detect(self, input_ids: Sequence[TokenSequence]) -> PrefixDetectionResult:
        """Detect prefix groups in the input batch.

        Args:
            input_ids: A sequence of token sequences to analyze.

        Returns:
            PrefixDetectionResult containing group assignments and metadata.
        """
        ...


@dataclass(frozen=True)
class PrefixGroup:
    group_id: int
    member_indices: tuple[int, ...]
    prefix_len: int
    provider_index: int


@dataclass(frozen=True)
class PrefixReuseSpec:
    """One reuse edge: reuser row ``reuse_idx_in_batch`` borrows KV from ``provider_idx_in_batch``."""

    reuse_idx_in_batch: int
    provider_idx_in_batch: int
    prefix_len: int


@dataclass(frozen=True)
class PrefixDetectionResult:
    """Per-batch detection output with per-sample reuse relations.

    ``reuse_specs`` is the semantic source of truth. The tuple fields
    ``group_ids``, ``provider_index``, ``prefix_lens``, and ``is_provider`` are
    indexed by batch position ``i`` for convenient planner use.

    Example (``TriePrefixDetector(min_prefix_len=2, min_group_size=2)``)::

        sequences = [
            [1, 2, 3, 4, 5, 10],  # index 0
            [1, 2, 3, 20],        # index 1, reuses index 0 length 3
            [1, 2, 3, 4, 5, 30],  # index 2, reuses index 0 length 5
        ]
        result = TriePrefixDetector(min_prefix_len=2, min_group_size=2).detect(sequences)

    Index 0 computes fully. Index 1 reuses index 0 length 3. Index 2 reuses
    index 0 length 5. The same provider therefore serves multiple prefix
    lengths.

    Corresponding field values::

        batch_size      == 3
        reuse_specs     == (
            PrefixReuseSpec(reuse_idx_in_batch=1, provider_idx_in_batch=0, prefix_len=3),
            PrefixReuseSpec(reuse_idx_in_batch=2, provider_idx_in_batch=0, prefix_len=5),
        )
        provider_index  == (0, 0, 0)
        prefix_lens     == (0, 3, 5)
        is_provider     == (True, False, False)
    """

    batch_size: int
    reuse_specs: tuple[PrefixReuseSpec, ...]
    groups: tuple[PrefixGroup, ...]
    group_ids: tuple[int, ...]
    provider_index: tuple[int, ...]
    prefix_lens: tuple[int, ...]
    is_provider: tuple[bool, ...]


class _TrieNode:
    __slots__ = ("children", "indices", "depth", "provider_index")

    def __init__(self, depth: int = 0) -> None:
        self.children: dict[int, _TrieNode] = {}
        self.indices: list[int] = []
        self.depth = depth
        self.provider_index = -1


class TriePrefixDetector(PrefixDetector):
    """Detect per-sample reuse relations with an online token trie.

    Each sequence is matched against previously inserted sequences. If the
    longest match satisfies the configured thresholds, the current sequence
    becomes a reuser of the provider recorded at the matched trie node. The
    current sequence is then inserted, allowing it to provide longer prefixes to
    later samples.
    """

    def __init__(self, min_prefix_len: int = 1, min_group_size: int = 2) -> None:
        if min_prefix_len < 1:
            raise ValueError("min_prefix_len must be >= 1")
        if min_group_size < 2:
            raise ValueError("min_group_size must be >= 2")
        self.min_prefix_len = min_prefix_len
        self.min_group_size = min_group_size

    def detect(self, input_ids: Sequence[TokenSequence]) -> PrefixDetectionResult:
        batch_size = len(input_ids)
        root = _TrieNode()
        group_ids = [-1] * batch_size
        provider_index = list(range(batch_size))
        prefix_lens = [0] * batch_size
        is_provider = [True] * batch_size
        reuse_specs: list[PrefixReuseSpec] = []
        group_key_to_id: dict[tuple[int, int], int] = {}
        group_members: dict[int, list[int]] = {}

        # 诊断开关：跳过 trie 检测，直接返回 0-prefix（所有序列当 provider，无复用）。
        # 用法见 PREFIX_SHARING_FORCE_ZERO_PREFIX 说明（配合 build 层的 has_sharing 旁路）。
        import os as _os
        if _os.environ.get("PREFIX_SHARING_FORCE_ZERO_PREFIX"):
            return PrefixDetectionResult(
                batch_size=batch_size,
                reuse_specs=(),
                groups=(),
                group_ids=tuple(group_ids),
                provider_index=tuple(provider_index),
                prefix_lens=tuple(prefix_lens),
                is_provider=tuple(is_provider),
            )

        for index, seq in enumerate(input_ids):
            node = root
            matched = 0
            matched_provider = -1
            matched_group_size = 0
            for token in seq:
                child = node.children.get(int(token))
                if child is None:
                    break
                node = child
                matched += 1
                if node.provider_index >= 0:
                    matched_provider = node.provider_index
                    matched_group_size = len(node.indices) + 1

            if (
                matched >= self.min_prefix_len
                and matched_provider >= 0
                and matched_group_size >= self.min_group_size
            ):
                spec = PrefixReuseSpec(
                    reuse_idx_in_batch=index,
                    provider_idx_in_batch=matched_provider,
                    prefix_len=matched,
                )
                reuse_specs.append(spec)
                provider_index[index] = matched_provider
                prefix_lens[index] = matched
                is_provider[index] = False

                group_key = (matched_provider, matched)
                group_id = group_key_to_id.setdefault(group_key, len(group_key_to_id))
                group_ids[index] = group_id
                if group_id not in group_members:
                    group_members[group_id] = [matched_provider]
                group_members[group_id].append(index)

            node = root
            node.indices.append(index)
            for token in seq:
                token = int(token)
                child = node.children.get(token)
                if child is None:
                    child = _TrieNode(node.depth + 1)
                    child.provider_index = index
                    node.children[token] = child
                node = child
                node.indices.append(index)

        groups = [
            PrefixGroup(
                group_id=group_id,
                member_indices=tuple(members),
                prefix_len=prefix_len,
                provider_index=provider,
            )
            for (provider, prefix_len), group_id in sorted(
                group_key_to_id.items(), key=lambda item: item[1]
            )
            for members in (group_members[group_id],)
        ]

        return PrefixDetectionResult(
            batch_size=batch_size,
            reuse_specs=tuple(reuse_specs),
            groups=tuple(groups),
            group_ids=tuple(group_ids),
            provider_index=tuple(provider_index),
            prefix_lens=tuple(prefix_lens),
            is_provider=tuple(is_provider),
        )


def common_prefix_len(sequences: Iterable[TokenSequence]) -> int:
    iterator = iter(sequences)
    try:
        first = list(next(iterator))
    except StopIteration:
        return 0
    prefix_len = len(first)
    for seq in iterator:
        limit = min(prefix_len, len(seq))
        index = 0
        while index < limit and first[index] == seq[index]:
            index += 1
        prefix_len = index
        if prefix_len == 0:
            break
    return prefix_len
