"""Prefix-sharing planning: turn detection output into batch metadata.

This module sits between **prefix detection** (:mod:`prefix_sharing.core.prefix_detector`)
and **downstream execution** (mapping, attention backends, RoPE). It does not
discover shared prefixes; it consumes :class:`~prefix_sharing.core.prefix_detector.PrefixDetectionResult`
and materializes per-row layout: what to keep on the Q path, cumulative sequence
lengths for packed tensors, position offsets, and optional prefix-last restore
hints for RL / logprob alignment.

Core Responsibilities:
    1. **Per-row layout after detection**: for each batch index, decide how that
       row participates in packed execution—whether the Q path uses the full
       sequence or only the post-prefix suffix, how long the row occupies on
       the KV path, and which contiguous token spans to keep for inputs, labels,
       and loss masks. Inputs are only ``provider_index``, ``prefix_lens``, and
       per-row original length; the result is encoded as lengths, cumulative
       seqlens, and keep ranges inside ``PrefixSharingPlan``.
    2. Set ``q_position_offsets`` so reuser rows preserve correct absolute positions
       when the Q path only packs the suffix after a shared prefix.
    3. Emit :class:`PrefixLastRestoreSpec` entries when a reuser has both a
       non-zero prefix and a non-empty suffix.

Key Concepts:
    - Reuser (planner view): ``provider_index[i] != i`` and ``prefix_lens[i] > 0``.
      The Q path keeps ``input_ids[i][prefix_len:original_len)``; KV expanded
      length stays the full original length for backend conventions.
    - Provider or standalone row: full sequence on the Q path; offsets zero.

Key Components:
    - :class:`PrefixSharingPlanner`: Optional embedded :class:`~prefix_sharing.core.prefix_detector.TriePrefixDetector`;
      :meth:`~PrefixSharingPlanner.plan` runs detect-then-plan;
      :meth:`~PrefixSharingPlanner.plan_from_detection` accepts precomputed detection
      (tests or alternate detectors).
    - :func:`_cumsum`: Computes cumulative lengths for ``cu_seqlens_q`` and ``cu_seqlens_kv``.

Design Principles:
    - Single layout source: all ranges and lengths derive from detector fields
      plus original lengths; planner does not reinterpret group structure.
    - Detector-agnostic planning: only ``PrefixDetectionResult`` + ``input_ids``
      lengths matter in :meth:`~PrefixSharingPlanner.plan_from_detection`.
    - Correlation hooks: optional ``forward_id`` and ``micro_batch_id`` for logging,
      cache keys, and pipeline tracing (auto-assigned when omitted).

Example:
    >>> from prefix_sharing.core.config import PrefixSharingConfig
    >>> planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    >>> prefix_sharing_plan = planner.plan([[1, 2, 3, 4], [1, 2, 3, 5]], forward_id=1, micro_batch_id=1)
    >>> prefix_sharing_plan.has_sharing
    True
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.prefix_detector import PrefixDetectionResult, PrefixReuseSpec, TriePrefixDetector


@dataclass(frozen=True)
class PrefixLastRestoreSpec:
    """Plan for one reuse row's prefix-last logprob restore.

    A reuser's first suffix token is predicted from the shared prefix-last
    position (``logits[prefix_len - 1]`` predicting ``input_ids[prefix_len]``).
    Since different reusers may have different first suffix labels, the
    provider's full logits at ``prefix_len - 1`` must be stored and the logprob
    recomputed per reuser using that reuser's label.

    Interior prefix positions (``[0, prefix_len - 2]``) are **not** represented
    here: their logprob is identical across the shared prefix (same logits +
    same labels), so the 2D restore bulk-slices the whole interval off the
    direct provider's already-restored row — no per-position spec is needed.
    """

    reuse_idx_in_batch: int
    provider_idx_in_batch: int
    group_id: int
    target_2d_pos: int = -1
    """The prefix-last position (``prefix_len - 1``): the last token of the
    shared prefix. By the log_probs layout invariant
    (``log_probs[i] = log_softmax(logits[i])[label[i+1]]``) this single index
    is **both** the provider logits row that predicts the reuser's first
    suffix token (read side) and the output column the recomputed logprob is
    written to (write side) — so one field covers both roles.
    """
    label_value: int = -1
    """The reuser's first suffix token ID (``input_ids[prefix_len]``), used as
    the label for logprob recompute. Needed because the trimmed packed labels
    don't contain this position.
    """


Range = tuple[int, int]


@dataclass(frozen=True)
class PrefixSharingPlan:
    """Complete framework-independent plan for a single micro-batch."""

    # 基础标识信息
    forward_id: int                              # 前向传播唯一标识
    micro_batch_id: int                          # micro-batch序号
    batch_size: int                              # 批次内序列数量
    original_lengths: list[int]                   # 各序列原始长度

    # 前缀复用关系
    reuse_specs: list[PrefixReuseSpec]           # 序列间复用关系规范
    group_ids: list[int]                        # 各序列所属前缀组ID
    is_provider: list[bool]                      # 各序列是否为provider（被复用方）
    provider_index: list[int]                    # 各序列的provider在batch中的索引
    prefix_lens: list[int]                      # 各序列前缀长度（可共享部分）
    suffix_lens: list[int]                      # 各序列后缀长度（需独立计算部分）

    # THD格式下的序列长度管理
    kept_lengths_q: list[int]                   # 裁剪后各序列的Q长度
    expanded_lengths_kv: list[int]             # 扩展后各序列的KV长度（包含共享前缀）
    cu_seqlens_q: list[int]                     # Q的累积序列长度（用于THD索引）
    cu_seqlens_kv: list[int]                   # KV的累积序列长度（用于THD索引）
    max_seqlen_q: int                          # Q的最大序列长度
    max_seqlen_kv: int                         # KV的最大序列长度

    # 位置偏移（用于恢复原始位置信息）
    q_position_offsets: list[int]               # Q相对于原始序列的位置偏移
    kv_position_offsets: list[int]             # KV相对于原始序列的位置偏移

    # 裁剪保留范围（start, end）
    input_keep_ranges: list[Range]              # input_ids保留范围
    label_keep_ranges: list[Range]             # labels保留范围
    loss_mask_keep_ranges: list[Range]         # loss_mask保留范围

    # 恢复点信息（用于logprob恢复）
    prefix_last_restore: list[PrefixLastRestoreSpec] = field(default_factory=list)  # reuser的suffix-first位置恢复规范

    def __post_init__(self) -> None:
        expected = self.batch_size
        fields: Sequence[tuple[str, list[object]]] = (
            ("original_lengths", self.original_lengths),
            ("group_ids", self.group_ids),
            ("is_provider", self.is_provider),
            ("provider_index", self.provider_index),
            ("prefix_lens", self.prefix_lens),
            ("suffix_lens", self.suffix_lens),
            ("kept_lengths_q", self.kept_lengths_q),
            ("expanded_lengths_kv", self.expanded_lengths_kv),
            ("q_position_offsets", self.q_position_offsets),
            ("kv_position_offsets", self.kv_position_offsets),
            ("input_keep_ranges", self.input_keep_ranges),
            ("label_keep_ranges", self.label_keep_ranges),
            ("loss_mask_keep_ranges", self.loss_mask_keep_ranges),
        )
        for name, value in fields:
            if len(value) != expected:
                raise ValueError(f"{name} length must equal batch_size={expected}")
        if len(self.cu_seqlens_q) != expected + 1:
            raise ValueError("cu_seqlens_q length must equal batch_size + 1")
        if len(self.cu_seqlens_kv) != expected + 1:
            raise ValueError("cu_seqlens_kv length must equal batch_size + 1")

    @property
    def has_sharing(self) -> bool:
        return bool(self.reuse_specs)

    def is_reuser(self, idx_in_batch: int) -> bool:
        return self.provider_index[idx_in_batch] != idx_in_batch and self.prefix_lens[idx_in_batch] > 0

    def q_range_for_batch(self, idx_in_batch: int) -> Range:
        return self.cu_seqlens_q[idx_in_batch], self.cu_seqlens_q[idx_in_batch + 1]

    def kv_range_for_batch(self, idx_in_batch: int) -> Range:
        return self.cu_seqlens_kv[idx_in_batch], self.cu_seqlens_kv[idx_in_batch + 1]

    def restore_for_reuse(self, idx_in_batch: int) -> PrefixLastRestoreSpec | None:
        for spec in self.prefix_last_restore:
            if spec.reuse_idx_in_batch == idx_in_batch:
                return spec
        return None


_forward_ids = itertools.count(1)


def _cumsum(lengths: Sequence[int]) -> list[int]:
    values = [0]
    total = 0
    for length in lengths:
        total += int(length)
        values.append(total)
    return values


@dataclass
class PrefixSharingPlanner:
    config: PrefixSharingConfig
    detector: TriePrefixDetector | None = None
    _micro_batch_counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.detector is None:
            self.detector = TriePrefixDetector(
                min_prefix_len=self.config.min_prefix_len,
                min_group_size=self.config.min_group_size,
            )

    def plan(
        self,
        input_ids: Sequence[Sequence[int]],
        *,
        forward_id: int | None = None,
        micro_batch_id: int | None = None,
    ) -> PrefixSharingPlan:
        detection = self.detector.detect(input_ids)
        return self.plan_from_detection(
            input_ids,
            detection,
            forward_id=forward_id,
            micro_batch_id=micro_batch_id,
        )

    def plan_from_detection(
        self,
        input_ids: Sequence[Sequence[int]],
        detection: PrefixDetectionResult,
        *,
        forward_id: int | None = None,
        micro_batch_id: int | None = None,
    ) -> PrefixSharingPlan:
        if len(input_ids) != detection.batch_size:
            raise ValueError("input_ids batch size does not match detection result")
        if forward_id is None:
            forward_id = next(_forward_ids)
        if micro_batch_id is None:
            self._micro_batch_counter += 1
            micro_batch_id = self._micro_batch_counter

        batch_size = len(input_ids)
        original_lengths = [len(seq) for seq in input_ids]
        group_ids = list(detection.group_ids)
        is_provider = list(detection.is_provider)
        provider_index = list(detection.provider_index)
        prefix_lens = list(detection.prefix_lens)
        reuse_specs = list(detection.reuse_specs)

        suffix_lens: list[int] = []
        kept_lengths_q: list[int] = []
        expanded_lengths_kv: list[int] = []
        q_position_offsets: list[int] = []
        kv_position_offsets: list[int] = []
        input_keep_ranges: list[tuple[int, int]] = []
        label_keep_ranges: list[tuple[int, int]] = []
        loss_mask_keep_ranges: list[tuple[int, int]] = []
        restore_specs: list[PrefixLastRestoreSpec] = []

        for index, original_len in enumerate(original_lengths):
            prefix_len = prefix_lens[index]
            if prefix_len > original_len:
                raise ValueError(f"prefix_len exceeds sequence length for batch index {index}")
            is_reuser = provider_index[index] != index and prefix_len > 0
            suffix_len = original_len - prefix_len if is_reuser else original_len
            suffix_lens.append(suffix_len)
            expanded_lengths_kv.append(original_len)

            if is_reuser:
                keep_start = prefix_len
                keep_end = original_len
                kept_len = suffix_len
                q_offset = prefix_len

                # --- Prefix-last restore (first suffix token) ---
                # Interior prefix positions [0, prefix_len-2] are restored by
                # the 2D restore, which bulk-slices them off the direct
                # provider's already-restored row (identical across the shared
                # prefix), so the planner only emits the prefix-last token.
                # The logits at position prefix_len-1 predict the first suffix
                # token whose label is input_ids[prefix_len] (differs per
                # reuser); the restored logprob is written to 2D position
                # prefix_len - 1 (the label slot for first-suffix prediction).
                if prefix_len > 0 and suffix_len > 0:
                    restore_specs.append(
                        PrefixLastRestoreSpec(
                            reuse_idx_in_batch=index,
                            provider_idx_in_batch=provider_index[index],
                            group_id=group_ids[index],
                            target_2d_pos=prefix_len - 1,
                            label_value=input_ids[index][prefix_len],
                        )
                    )
            else:
                keep_start = 0
                keep_end = original_len
                kept_len = original_len
                q_offset = 0

            kept_lengths_q.append(kept_len)
            q_position_offsets.append(q_offset)
            kv_position_offsets.append(0)
            keep_range = (keep_start, keep_end)
            input_keep_ranges.append(keep_range)
            label_keep_ranges.append(keep_range)
            loss_mask_keep_ranges.append(keep_range)

        cu_seqlens_q = _cumsum(kept_lengths_q)
        cu_seqlens_kv = _cumsum(expanded_lengths_kv)

        return PrefixSharingPlan(
            forward_id=forward_id,
            micro_batch_id=micro_batch_id,
            batch_size=batch_size,
            original_lengths=original_lengths,
            reuse_specs=reuse_specs,
            group_ids=group_ids,
            is_provider=is_provider,
            provider_index=provider_index,
            prefix_lens=prefix_lens,
            suffix_lens=suffix_lens,
            kept_lengths_q=kept_lengths_q,
            expanded_lengths_kv=expanded_lengths_kv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max(kept_lengths_q, default=0),
            max_seqlen_kv=max(expanded_lengths_kv, default=0),
            q_position_offsets=q_position_offsets,
            kv_position_offsets=kv_position_offsets,
            input_keep_ranges=input_keep_ranges,
            label_keep_ranges=label_keep_ranges,
            loss_mask_keep_ranges=loss_mask_keep_ranges,
            prefix_last_restore=restore_specs,
        )
