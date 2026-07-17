"""BSHD batch layout — the padded Batch-Sequence coordinate system.

This layout is the BSHD counterpart of :class:`PackedBatchLayout`.  Instead of a
packed 1D token sequence indexed by ``cu_seqlens``, BSHD describes a standard
``[B, S]`` padded tensor where each row's valid tokens are left-aligned
(``preprocess_bshd_engine`` convention, see ``verl/models/mcore/util.py``).

Coordinate convention (absolute coordinates, used by every BSHD component):

* row ``i``'s valid tokens occupy columns ``[0, valid_lengths[i])``
* position id of column ``c`` is ``c`` itself (verl's BSHD preprocessing
  assigns ``arange(S)`` per row)
* a reuser row's suffix occupies columns ``[prefix_len, valid_len)`` — the
  prefix columns ``[0, prefix_len)`` still physically exist and are hidden
  exclusively by the backend's 4-D attention mask, never by data trimming.

``BatchedBatchLayout`` and ``PackedBatchLayout`` share a common subset of
properties (``valid_lengths``, ``batch_size``, ``total_valid_length``,
``total_padded_length``) and the ``is_bshd()`` discriminator so caller code can
dispatch without caring about the concrete type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BatchedBatchLayout:
    """BSHD batch layout — standard ``[B, S]`` padded format (absolute coords).

    ``valid_lengths`` holds each row's *full* valid length (identical to
    ``plan.original_lengths``).  BSHD never trims data, so there is no
    "kept lengths" concept here; a reuser's suffix range is expressed by
    ``plan.prefix_lens`` + ``valid_lengths`` jointly.
    """

    valid_lengths: list[int]
    """Per-row number of valid (non-padding) tokens — full original lengths."""

    seq_length: int
    """Lower bound of the padded sequence length S (= ``max(valid_lengths)``).

    The real padded S produced inside the engine may be larger (TP alignment,
    see ``preprocess_bshd_engine``).  Backends must size their masks from the
    actual query/key tensor shapes, not from this field.
    """

    def __post_init__(self) -> None:
        if len(self.valid_lengths) == 0:
            # Empty batch — allow any seq_length.
            return
        if self.seq_length <= 0:
            raise ValueError("seq_length must be > 0 for a non-empty batch")
        if any(length < 0 for length in self.valid_lengths):
            raise ValueError("valid_lengths must be non-negative")
        if any(length > self.seq_length for length in self.valid_lengths):
            raise ValueError("valid_lengths cannot exceed seq_length")

    # ── Format identification ────────────────────────────────────────────

    @staticmethod
    def is_bshd() -> bool:
        """Returns ``True`` — this is the BSHD layout.

        Together with :meth:`PackedBatchLayout.is_bshd` this lets downstream
        code write ``layout.is_bshd()`` uniformly on either type.
        """
        return True

    # ── Derived properties ───────────────────────────────────────────────

    @property
    def batch_size(self) -> int:
        return len(self.valid_lengths)

    @property
    def total_valid_length(self) -> int:
        return sum(self.valid_lengths)

    @property
    def total_padded_length(self) -> int:
        """Lower-bound padded token count (``B * seq_length``).

        Provided for ``PrefixSharingStats.from_plan`` compatibility, which
        reads this attribute on either layout type.  The engine-side padded S
        may be larger due to TP alignment.
        """
        return self.batch_size * self.seq_length

    def to_cu_seqlens(self) -> list[int]:
        """Cumulative valid lengths (for varlen boundary conversion)."""
        return _cumsum(self.valid_lengths)

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_valid_lengths(cls, valid_lengths: Sequence[int]) -> "BatchedBatchLayout":
        """Build layout from per-row full valid lengths."""
        lengths = [int(length) for length in valid_lengths]
        return cls(
            valid_lengths=lengths,
            seq_length=max(lengths, default=0),
        )


def _cumsum(lengths: Sequence[int]) -> list[int]:
    values = [0]
    running = 0
    for length in lengths:
        running += int(length)
        values.append(running)
    return values
