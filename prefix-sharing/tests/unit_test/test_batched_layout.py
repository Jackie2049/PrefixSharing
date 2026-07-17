"""Tests for BatchedBatchLayout (BSHD layout)."""

from __future__ import annotations

import pytest

from prefix_sharing.backends.batched_layout import BatchedBatchLayout
from prefix_sharing.backends.packed_layout import PackedBatchLayout


class TestConstruction:
    def test_basic(self):
        layout = BatchedBatchLayout(valid_lengths=[5, 6, 3], seq_length=6)
        assert layout.batch_size == 3
        assert layout.seq_length == 6
        assert layout.valid_lengths == [5, 6, 3]

    def test_from_valid_lengths(self):
        layout = BatchedBatchLayout.from_valid_lengths([4, 7, 2])
        assert layout.seq_length == 7
        assert layout.batch_size == 3

    def test_empty_batch_allowed(self):
        layout = BatchedBatchLayout(valid_lengths=[], seq_length=0)
        assert layout.batch_size == 0
        assert layout.total_valid_length == 0

    def test_seq_length_must_cover_valid(self):
        with pytest.raises(ValueError, match="cannot exceed seq_length"):
            BatchedBatchLayout(valid_lengths=[5], seq_length=4)

    def test_negative_valid_length_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            BatchedBatchLayout(valid_lengths=[-1], seq_length=4)

    def test_zero_seq_length_rejected_for_non_empty(self):
        with pytest.raises(ValueError, match="seq_length must be > 0"):
            BatchedBatchLayout(valid_lengths=[1], seq_length=0)


class TestFormatIdentification:
    def test_is_bshd_true(self):
        layout = BatchedBatchLayout(valid_lengths=[3], seq_length=3)
        assert layout.is_bshd() is True

    def test_packed_layout_is_bshd_false(self):
        packed = PackedBatchLayout.from_valid_lengths([3, 4])
        assert packed.is_bshd() is False

    def test_dispatch_uniform(self):
        """Caller code can dispatch on is_bshd() for either layout type."""
        bshd = BatchedBatchLayout(valid_lengths=[3], seq_length=3)
        thd = PackedBatchLayout.from_valid_lengths([3])
        formats = {layout.is_bshd() for layout in (bshd, thd)}
        assert formats == {True, False}


class TestDerivedProperties:
    def test_total_valid_length(self):
        layout = BatchedBatchLayout(valid_lengths=[5, 6, 3], seq_length=6)
        assert layout.total_valid_length == 14

    def test_total_padded_length(self):
        layout = BatchedBatchLayout(valid_lengths=[5, 6, 3], seq_length=6)
        assert layout.total_padded_length == 18

    def test_to_cu_seqlens(self):
        layout = BatchedBatchLayout(valid_lengths=[5, 6, 3], seq_length=6)
        assert layout.to_cu_seqlens() == [0, 5, 11, 14]

    def test_to_cu_seqlens_empty(self):
        layout = BatchedBatchLayout(valid_lengths=[], seq_length=0)
        assert layout.to_cu_seqlens() == [0]
