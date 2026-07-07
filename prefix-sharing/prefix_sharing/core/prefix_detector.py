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

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence


TokenSequence = Sequence[int]


@dataclass(frozen=True)
class SPackedBuildResult:
    """s_packed build result produced during fused trie detection.

    Produced by :meth:`TriePrefixDetector.detect` as part of the fused
    detect + s_packed allocation pass.  These fields replace the separate
    ``build_s_packed`` call that used independent token-scanning and could
    disagree with the detector's trie-based prefix matching.
    """

    s_packed_kv_ranges: list[list[tuple[int, int]]]
    """Per-input KV intervals in s_packed (inclusive [lo, hi)).  A reuser's
    interval may span multiple discontiguous blocks when the shared prefix
    was built from a provider whose sequence occupied multiple s_packed
    segments."""
    s_packed_length: int
    s_packed_q_lengths: list[int]
    s_packed_q_starts: list[int]
    s_packed_prefix_ends: list[int]
    """Per-input: position past the last prefix token in s_packed.
    For a provider: same as the range start (provider has no prefix)."""
    s_packed_suffix_starts: list[int]
    """Per-input: s_packed position where the suffix starts (for mask construction).
    For a provider: start of their range (suffix = full sequence).
    For a reuser: position where suffix was appended (or equal to prefix_end if no suffix)."""
    _custom_mask: torch.Tensor | None


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
    s_packed_result: SPackedBuildResult | None = None


class _TrieNode:
    __slots__ = ("children", "indices", "depth", "provider_index", "s_packed_pos")

    def __init__(self, depth: int = 0) -> None:
        self.children: dict[int, _TrieNode] = {}
        self.indices: list[int] = []
        self.depth = depth
        self.provider_index = -1
        self.s_packed_pos = -1


def _decompose_prefix_range(
    provider_blocks: list[tuple[int, int]],
    prefix_len: int,
) -> list[tuple[int, int]]:
    """Decompose a prefix of a provider's s_packed blocks into physical intervals.

    The provider may have multiple non-contiguous s_packed blocks
    (e.g. prefix from a shared block + suffix appended).  We need to collect
    exactly ``prefix_len`` tokens from those blocks, respecting block boundaries.

    Args:
        provider_blocks: List of (block_start, block_end) intervals for the provider.
        prefix_len: How many tokens of the provider's content the reuser needs.

    Returns:
        A list of (block_start, block_end) intervals covering exactly
        ``prefix_len`` tokens from the provider's blocks.

    Example:
        provider blocks = [(0, 2), (1102, 2202)], total tokens = 2202
        prefix_len = 1102  (reuser needs full provider content)
        Returns [(0, 2), (1102, 2202)] — all blocks

        prefix_len = 2  (reuser needs only first 2 tokens)
        Returns [(0, 2)] — first block only
    """
    result: list[tuple[int, int]] = []
    remaining = prefix_len
    for block_start, block_end in provider_blocks:
        block_len = block_end - block_start
        if remaining <= 0:
            break
        if block_len <= remaining:
            # Take the whole block
            result.append((block_start, block_end))
            remaining -= block_len
        else:
            # Take a partial block
            result.append((block_start, block_start + remaining))
            remaining = 0
    return result


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

        # --- s_packed build state (fused with trie traversal) ---
        s_packed: list[int] = []
        s_packed_kv_ranges: list[list[tuple[int, int]]] = []
        s_packed_q_lengths: list[int] = []
        s_packed_q_starts: list[int] = []
        s_packed_prefix_ends: list[int] = []

        # Per-sample context recorded during trie traversal.
        # sample_original_lengths[i] = original length of sample i.
        sample_original_lengths: list[int] = []

        # sample_s_packed_blocks[i] = list of (start, end) blocks for sample i.
        # For a provider: one block [start, end).  For a reuser: equals the
        # provider's blocks (transitive chain — later samples can reuse the reuser).
        sample_s_packed_blocks: list[list[tuple[int, int]]] = []

        # --- Phase A: trie match + s_packed allocation (single pass per sample) ---
        for index, seq in enumerate(input_ids):
            seq_list = list(seq)
            original_len = len(seq_list)

            # Phase A1: trie walk — find longest match against previously
            #           inserted sequences.
            node = root
            matched = 0
            matched_provider = -1
            matched_group_size = 0
            for token in seq_list:
                child = node.children.get(int(token))
                if child is None:
                    break
                node = child
                matched += 1
                if node.provider_index >= 0:
                    matched_provider = node.provider_index
                    matched_group_size = len(node.indices) + 1

            # Phase A2: record whether this sample qualifies as a reuser.
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

            # Phase A3: allocate s_packed intervals consistent with trie match.
            # A reuser's prefix references the provider's s_packed blocks,
            # truncated to prefix_len tokens.  A provider appends its entire
            # sequence to s_packed and records the range.
            if is_provider[index]:
                # Provider: append entire sequence, record s_packed range.
                start_pos = len(s_packed)
                s_packed.extend(seq_list)
                end_pos = len(s_packed)
                blocks = [(start_pos, end_pos)]
                s_packed_kv_ranges.append(blocks)
                sample_s_packed_blocks.append(blocks)
                sample_original_lengths.append(original_len)
            else:
                # Reuser: prefix references the provider's physical s_packed blocks,
                # decomposed to match the provider's storage layout.
                p = provider_index[index]
                p_blocks = sample_s_packed_blocks[p]
                prefix_len = prefix_lens[index]
                suffix_len = original_len - prefix_len

                # Decompose the reuser's prefix into provider's physical blocks.
                prefix_blocks = _decompose_prefix_range(
                    provider_blocks=p_blocks,
                    prefix_len=prefix_len,
                )
                ranges = list(prefix_blocks)
                if suffix_len > 0:
                    suffix_start = len(s_packed)
                    s_packed.extend(seq_list[prefix_len:])
                    ranges.append((suffix_start, len(s_packed)))
                s_packed_kv_ranges.append(ranges)
                # Reuser's blocks for downstream transitive reuse: prefix blocks + suffix.
                if suffix_len > 0:
                    reuser_blocks = list(prefix_blocks) + [(suffix_start, len(s_packed))]
                else:
                    reuser_blocks = list(prefix_blocks)
                sample_s_packed_blocks.append(reuser_blocks)
                sample_original_lengths.append(original_len)

            # Phase A4: insert this sequence into the trie, recording s_packed
            #           start position on every node visited / created.
            # For a provider: newly created nodes get the current s_packed start.
            # For a reuser: all nodes already exist (from the provider), so we
            # only append the index — s_packed_pos is already correctly set from
            # when the provider was inserted.
            start_pos = sample_s_packed_blocks[index][0][0]
            node = root
            node.indices.append(index)
            for depth_val, token in enumerate(seq_list):
                token = int(token)
                child = node.children.get(token)
                if child is None:
                    child = _TrieNode(node.depth + 1)
                    child.provider_index = index
                    # New node: record where this token lives in s_packed.
                    child.s_packed_pos = start_pos + depth_val
                    node.children[token] = child
                # Existing node: do NOT overwrite s_packed_pos — it was set
                # when the provider was inserted and is the correct position.
                node = child
                node.indices.append(index)

        # --- Phase B: derive Q-side layout from s_packed_kv_ranges ---
        q_cumsum = 0
        s_packed_suffix_starts: list[int] = []
        for i in range(batch_size):
            original_len = sample_original_lengths[i]
            prefix_len = prefix_lens[i]
            if is_provider[i]:
                q_len = original_len
            else:
                q_len = original_len - prefix_len
            s_packed_q_lengths.append(q_len)
            s_packed_q_starts.append(q_cumsum)
            q_cumsum += q_len
            prefix_start = s_packed_kv_ranges[i][0][0] if s_packed_kv_ranges[i] else 0
            prefix_end = prefix_start + prefix_len if not is_provider[i] else prefix_start
            s_packed_prefix_ends.append(prefix_end)
            # suffix_s_packed_start: s_packed position where this sample's suffix starts.
            # For provider: suffix = full sequence, starts at their range start.
            # For reuser: suffix appended at current end of s_packed.
            if is_provider[i]:
                suffix_start = prefix_start
            else:
                suffix_start = len(s_packed) if original_len > prefix_len else prefix_end
            s_packed_suffix_starts.append(suffix_start)

        # --- Phase C: build global custom causal mask ---
        custom_mask: torch.Tensor | None = None
        total_q = q_cumsum
        T = len(s_packed)
        if T > 0 and total_q > 0:
            try:
                device = torch.device("cpu")
                mask = torch.zeros(total_q, T, dtype=torch.bool, device=device)
                for i in range(batch_size):
                    q_start = s_packed_q_starts[i]
                    q_len = s_packed_q_lengths[i]
                    prefix_len = prefix_lens[i]
                    prefix_end = s_packed_prefix_ends[i]
                    suffix_s_start = s_packed_suffix_starts[i]

                    for qi in range(q_len):
                        q_pos = q_start + qi
                        q_original_pos = prefix_len + qi

                        for (kv_lo, kv_hi) in s_packed_kv_ranges[i]:
                            if kv_hi <= prefix_end:
                                # Prefix block: positions are contiguous starting from 0.
                                visible_hi = min(kv_hi, q_original_pos)
                                if kv_lo < visible_hi:
                                    mask[q_pos, kv_lo:visible_hi] = True
                            else:
                                # Suffix block: s_packed position kv_lo maps to original position kv_lo - prefix_len
                                # (suffix in s_packed occupies the same positions as in original space).
                                # Causal: kv_original_pos <= q_original_pos
                                # -> kv - prefix_len <= q_original_pos
                                # -> kv <= q_original_pos + prefix_len
                                visible_hi = min(kv_hi, q_original_pos + prefix_len + 1)
                                if kv_lo < visible_hi:
                                    mask[q_pos, kv_lo:visible_hi] = True
                custom_mask = mask
            except Exception:
                custom_mask = None

        s_packed_result = SPackedBuildResult(
            s_packed_kv_ranges=s_packed_kv_ranges,
            s_packed_length=len(s_packed),
            s_packed_q_lengths=s_packed_q_lengths,
            s_packed_q_starts=s_packed_q_starts,
            s_packed_prefix_ends=s_packed_prefix_ends,
            s_packed_suffix_starts=s_packed_suffix_starts,
            _custom_mask=custom_mask,
        )

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
            s_packed_result=s_packed_result,
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
