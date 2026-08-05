"""Prefix detection for shared token sequence identification.

This module provides algorithms to detect shared prefixes among token sequences
in a **batch** (multiple sequences processed together in one detection pass).
For each reuser sequence, the detector records one **reuse relation**: which
previous provider sequence can supply a reusable prefix slice and how long that
slice is. A provider may serve different reusers with different prefix lengths.

Core Responsibilities:
    1. Identify common prefixes across multiple token sequences.
    2. Emit per-sample reuse relations using configurable thresholds.

Key Concepts:
    - Provider: The earlier sequence in a reuse relation whose logical KV can
      supply a prefix slice to a later sequence.
    - Reuser: A sequence that reuses the provider's prefix KV.
      Reusers skip prefix computation and attend to the provider's KV cache.

Key Components:
    - PrefixReuseSpec: Represents one relation
      ``(reuse_idx_in_batch, provider_idx_in_batch, prefix_len)``.
    - PrefixDetectionResult: Container for detection output with per-sequence
      metadata including provider assignment and reuse flags.
    - PrefixDetector: Abstract base class defining the detector interface.
    - TriePrefixDetector: Concrete implementation using a trie data structure.
    - common_prefix_len: Utility to compute common prefix length across sequences.

Design Principles:
    - Per-sample relation first: ``provider_index[i]`` and ``prefix_lens[i]``
      are the authoritative plan for each row.
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
class PrefixReuseSpec:
    """One reuse edge: reuser row ``reuse_idx_in_batch`` borrows KV from ``provider_idx_in_batch``."""

    reuse_idx_in_batch: int
    provider_idx_in_batch: int
    prefix_len: int


@dataclass(frozen=True)
class PrefixDetectionResult:
    """Per-batch detection output with per-sample reuse relations.

    ``reuse_specs`` is the semantic source of truth. The tuple fields
    ``provider_index``, ``prefix_lens``, and ``is_provider`` are indexed by
    batch position ``i`` for convenient planner use.

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

    def reset(self, depth: int = 0) -> None:
        """Reset node for pool reuse."""
        self.children.clear()
        self.indices.clear()
        self.depth = depth
        self.provider_index = -1


class _TrieNodePool:
    """Pre-allocated pool of _TrieNode to avoid repeated memory allocation.

    Each detect() creates ~10k+ TrieNode objects. When pymalloc's free arenas
    are exhausted, the next allocation triggers a slow OS-level memory request,
    causing periodic multi-hundred-ms spikes.

    The pool pre-allocates nodes and reuses them across detect() calls,
    eliminating the allocation spike. On exhaustion it grows by 50%
    (minimum 4096); the extra capacity persists across reset(), so growth
    happens at most until the largest micro-batch has been seen once.

    IMPORTANT: the pool must be shared across detect() calls (module-level
    singleton), otherwise the pre-allocation cost itself is paid per-call
    and the spike just moves from detect() to pool construction.
    """

    __slots__ = ("_nodes", "_index")

    def __init__(self, initial_size: int = 16384) -> None:
        self._nodes: list[_TrieNode] = [_TrieNode() for _ in range(initial_size)]
        self._index = 0

    def get(self, depth: int = 0) -> _TrieNode:
        if self._index >= len(self._nodes):
            # Pool exhausted: extend by 50%
            extend = max(4096, len(self._nodes) // 2)
            self._nodes.extend([_TrieNode() for _ in range(extend)])
        node = self._nodes[self._index]
        self._index += 1
        node.reset(depth)
        return node

    def reset(self) -> None:
        """Reset pool for next detect() call."""
        self._index = 0


# Module-level singleton: allocated once per process, reused across all
# detect() calls (all micro-batches, all steps). Training is single-threaded
# per worker process, so no locking is needed.
_SHARED_NODE_POOL = _TrieNodePool(initial_size=16384)


class TriePrefixDetector(PrefixDetector):
    """Detect per-sample reuse relations with an online token trie.

    Each sequence is matched against previously inserted sequences. If the
    longest match satisfies the configured thresholds, the current sequence
    becomes a reuser of the provider recorded at the matched trie node. The
    current sequence is then inserted, allowing it to provide longer prefixes to
    later samples.

    Uses the module-level shared node pool to avoid periodic memory allocation
    spikes (see _TrieNodePool docstring).
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
        _SHARED_NODE_POOL.reset()
        root = _SHARED_NODE_POOL.get(depth=0)
        provider_index = list(range(batch_size))
        prefix_lens = [0] * batch_size
        is_provider = [True] * batch_size
        reuse_specs: list[PrefixReuseSpec] = []

        for index, seq in enumerate(input_ids):
            node = root
            root.indices.append(index)

            matched = 0
            matched_provider = -1
            matched_group_size = 0
            still_matching = True

            for token in seq:
                token = int(token)
                child = node.children.get(token)

                if still_matching and child is not None:
                    # Still on a matching path: reuse existing node
                    node = child
                    matched += 1
                    if node.provider_index >= 0:
                        matched_provider = node.provider_index
                        # +1 because current index hasn't been appended yet
                        matched_group_size = len(node.indices) + 1
                else:
                    # Match ended (or never started): switch to insert mode
                    still_matching = False
                    if child is None:
                        child = _SHARED_NODE_POOL.get(depth=node.depth + 1)
                        child.provider_index = index
                        node.children[token] = child
                    node = child

                node.indices.append(index)

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

        return PrefixDetectionResult(
            batch_size=batch_size,
            reuse_specs=tuple(reuse_specs),
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
