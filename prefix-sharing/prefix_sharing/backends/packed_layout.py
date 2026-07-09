"""Packed batch layout shared by backend implementations.

The layout describes the packed-token coordinate system seen by attention
backends. It is not prefix-sharing semantic metadata; integrations construct it
from framework inputs, and backends consume it to split tensors safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class ContextParallelPackedView:
    cp_rank: int
    cp_size: int
    local_padded_lengths: list[int]
    local_cu_seqlens: list[int]
    local_total_padded_length: int
    global_indices_by_local: Any
    local_indices_by_global: Any
    local_position_ids: Any | None = None
    local_valid_token_mask: Any | None = None

    def global_to_local(self, global_index: int) -> int | None:
        local_index = self.local_indices_by_global[int(global_index)]
        if hasattr(local_index, "item"):
            local_index = int(local_index.item())
        else:
            local_index = int(local_index)
        return None if local_index < 0 else local_index

    def local_to_global(self, local_index: int) -> int:
        global_index = self.global_indices_by_local[int(local_index)]
        if hasattr(global_index, "item"):
            return int(global_index.item())
        return int(global_index)

    def row_start(self, row: int) -> int:
        return self.local_cu_seqlens[row]

    def row_local_slice(self, row: int) -> slice:
        return slice(self.local_cu_seqlens[row], self.local_cu_seqlens[row + 1])

    def row_local_valid_mask(self, row: int) -> Any:
        if self.local_valid_token_mask is None:
            return None
        row_slice = self.row_local_slice(row)
        return self.local_valid_token_mask[row_slice]


@dataclass(frozen=True)
class PackedBatchLayout:
    valid_lengths: list[int]
    padded_lengths: list[int]
    cu_seqlens: list[int]
    max_seqlen: int
    packed_position_ids: Any | None = None
    valid_token_mask: Any | None = None
    context_parallel: ContextParallelPackedView | None = None

    def __post_init__(self) -> None:
        batch_size = len(self.valid_lengths)
        if len(self.padded_lengths) != batch_size:
            raise ValueError("padded_lengths must match valid_lengths")
        if len(self.cu_seqlens) != batch_size + 1:
            raise ValueError("cu_seqlens length must equal batch_size + 1")
        if self.cu_seqlens[0] != 0:
            raise ValueError("cu_seqlens must start with 0")
        if any(length < 0 for length in self.valid_lengths):
            raise ValueError("valid_lengths must be non-negative")
        if any(length < 0 for length in self.padded_lengths):
            raise ValueError("padded_lengths must be non-negative")
        for valid_length, padded_length in zip(self.valid_lengths, self.padded_lengths):
            if padded_length < valid_length:
                raise ValueError("padded_lengths cannot be smaller than valid_lengths")
        if self.cu_seqlens != _cumsum(self.padded_lengths):
            raise ValueError("cu_seqlens must be the cumulative padded lengths")
        if self.context_parallel is not None:
            if self.context_parallel.cp_size < 1:
                raise ValueError("context parallel size must be >= 1")
            if not 0 <= self.context_parallel.cp_rank < self.context_parallel.cp_size:
                raise ValueError("context parallel rank must be in [0, cp_size)")
            if len(self.context_parallel.local_padded_lengths) != batch_size:
                raise ValueError("local_padded_lengths must match valid_lengths")
            if self.context_parallel.local_cu_seqlens != _cumsum(self.context_parallel.local_padded_lengths):
                raise ValueError("local_cu_seqlens must be the cumulative local padded lengths")
            if self.context_parallel.local_total_padded_length != self.context_parallel.local_cu_seqlens[-1]:
                raise ValueError("local_total_padded_length must match local_cu_seqlens[-1]")

    @classmethod
    def from_valid_lengths(cls, valid_lengths: Sequence[int]) -> "PackedBatchLayout":
        lengths = [int(length) for length in valid_lengths]
        return cls(
            valid_lengths=lengths,
            padded_lengths=list(lengths),
            cu_seqlens=_cumsum(lengths),
            max_seqlen=max(lengths, default=0),
            packed_position_ids=None,
            valid_token_mask=None,
        )

    @classmethod
    def from_kept_position_rows(
        cls,
        kept_position_rows: Sequence[Any],
        *,
        align_size: int,
        cp_rank: int = 0,
        cp_size: int = 1,
    ) -> "PackedBatchLayout":
        if align_size < 1:
            raise ValueError("align_size must be >= 1")
        if cp_size < 1:
            raise ValueError("cp_size must be >= 1")
        if not 0 <= cp_rank < cp_size:
            raise ValueError("cp_rank must be in [0, cp_size)")
        if not kept_position_rows:
            return cls.from_valid_lengths([])

        first = kept_position_rows[0]
        valid_lengths = [int(row.shape[0]) for row in kept_position_rows]
        padded_lengths = [_pad_to_multiple(length, align_size) for length in valid_lengths]
        packed_position_rows = []
        valid_mask_rows = []
        for row, valid_length, padded_length in zip(kept_position_rows, valid_lengths, padded_lengths):
            row = row.to(first.device)
            pad_length = padded_length - valid_length
            valid_mask_rows.append(
                torch.cat(
                    [
                        torch.ones(valid_length, dtype=torch.bool, device=first.device),
                        torch.zeros(pad_length, dtype=torch.bool, device=first.device),
                    ],
                    dim=0,
                )
            )
            if pad_length == 0:
                packed_position_rows.append(row)
                continue
            padding = torch.zeros(pad_length, dtype=row.dtype, device=first.device)
            packed_position_rows.append(torch.cat([row, padding], dim=0))

        packed_position_ids = torch.cat(packed_position_rows, dim=0)
        valid_token_mask = torch.cat(valid_mask_rows, dim=0)
        cu_seqlens = _cumsum(padded_lengths)
        context_parallel = None
        if cp_size > 1:
            context_parallel = _build_context_parallel_view(
                padded_lengths=padded_lengths,
                valid_lengths=valid_lengths,
                cu_seqlens=cu_seqlens,
                packed_position_ids=packed_position_ids,
                cp_rank=cp_rank,
                cp_size=cp_size,
                device=first.device,
            )

        return cls(
            valid_lengths=valid_lengths,
            padded_lengths=padded_lengths,
            cu_seqlens=cu_seqlens,
            max_seqlen=max(padded_lengths, default=0),
            packed_position_ids=packed_position_ids,
            valid_token_mask=valid_token_mask,
            context_parallel=context_parallel,
        )

    @property
    def batch_size(self) -> int:
        return len(self.valid_lengths)

    @property
    def has_padding(self) -> bool:
        """True when at least one row has a pad token (``padded > valid``)."""
        return self.valid_lengths != self.padded_lengths

    @property
    def total_padded_length(self) -> int:
        return self.cu_seqlens[-1]

    @property
    def total_valid_length(self) -> int:
        return sum(self.valid_lengths)

    def row_start(self, row: int) -> int:
        return self.cu_seqlens[row]

    def packed_index(self, row: int, valid_offset: int) -> int:
        if valid_offset < 0 or valid_offset >= self.valid_lengths[row]:
            raise IndexError("valid_offset is outside the row valid token range")
        return self.row_start(row) + valid_offset

    def valid_slice(self, row: int, tensor: Any) -> Any:
        start = self.row_start(row)
        end = start + self.valid_lengths[row]
        return tensor[start:end]

    def unpad(self, tensor: Any) -> Any:
        """Strip TP padding from a packed tensor, keeping only valid tokens.

        When the layout has no padding this is a no-op that returns *tensor*
        unchanged.
        """
        if not self.has_padding:
            return tensor
        rows = list(torch.split(tensor, self.padded_lengths, dim=0))
        return torch.cat(
            [row[:vl] for row, vl in zip(rows, self.valid_lengths)], dim=0,
        )

    def repad(self, tensor: Any) -> Any:
        """Re-apply TP padding to a valid-only tensor.

        *tensor* must be packed along dim 0 following :attr:`valid_lengths`;
        the returned tensor follows :attr:`padded_lengths`.  When the layout
        has no padding this is a no-op.
        """
        if not self.has_padding:
            return tensor
        rows = list(torch.split(tensor, self.valid_lengths, dim=0))
        repadded: list[Any] = []
        for row, padded_len, valid_len in zip(rows, self.padded_lengths, self.valid_lengths):
            if valid_len == padded_len:
                repadded.append(row)
                continue
            pad_len = padded_len - valid_len
            pad_tensor = torch.zeros(
                pad_len, *row.shape[1:], dtype=row.dtype, device=row.device,
            )
            repadded.append(torch.cat([row, pad_tensor], dim=0))
        return torch.cat(repadded, dim=0)


def _pad_to_multiple(length: int, align_size: int) -> int:
    return int(length + (align_size - length % align_size) % align_size)


def _cumsum(lengths: Sequence[int]) -> list[int]:
    values = [0]
    running = 0
    for length in lengths:
        running += int(length)
        values.append(running)
    return values


def _build_context_parallel_view(
    *,
    padded_lengths: Sequence[int],
    valid_lengths: Sequence[int],
    cu_seqlens: Sequence[int],
    packed_position_ids: Any,
    cp_rank: int,
    cp_size: int,
    device: Any,
) -> ContextParallelPackedView:
    local_padded_lengths: list[int] = []
    global_indices: list[int] = []
    valid_flags: list[bool] = []
    for row, (padded_length, valid_length) in enumerate(zip(padded_lengths, valid_lengths)):
        if padded_length % cp_size != 0:
            raise ValueError("CP padded length must be divisible by cp_size")
        chunk_len = int(padded_length) // cp_size
        if chunk_len % 2 != 0:
            raise ValueError("CP local chunk length must be divisible by 2")
        half = chunk_len // 2
        row_start = int(cu_seqlens[row])
        front_start = row_start + half * cp_rank
        front_end = row_start + half * (cp_rank + 1)
        back_start = row_start + int(padded_length) - half * (cp_rank + 1)
        back_end = row_start + int(padded_length) - half * cp_rank
        row_indices = list(range(front_start, front_end)) + list(range(back_start, back_end))
        local_padded_lengths.append(len(row_indices))
        global_indices.extend(row_indices)
        valid_end = row_start + int(valid_length)
        valid_flags.extend(index < valid_end for index in row_indices)

    local_cu_seqlens = _cumsum(local_padded_lengths)
    global_index_tensor = torch.tensor(global_indices, dtype=torch.long, device=device)
    local_index_tensor = torch.full((int(cu_seqlens[-1]),), -1, dtype=torch.long, device=device)
    if global_indices:
        local_index_tensor[global_index_tensor] = torch.arange(
            len(global_indices), dtype=torch.long, device=device,
        )
    local_position_ids = packed_position_ids.index_select(0, global_index_tensor) if global_indices else packed_position_ids.new_empty((0,))
    local_valid_token_mask = torch.tensor(valid_flags, dtype=torch.bool, device=device)
    return ContextParallelPackedView(
        cp_rank=cp_rank,
        cp_size=cp_size,
        local_padded_lengths=local_padded_lengths,
        local_cu_seqlens=local_cu_seqlens,
        local_total_padded_length=local_cu_seqlens[-1],
        global_indices_by_local=global_index_tensor,
        local_indices_by_global=local_index_tensor,
        local_position_ids=local_position_ids,
        local_valid_token_mask=local_valid_token_mask,
    )
