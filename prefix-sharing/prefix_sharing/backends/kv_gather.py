"""Gather-based KV expansion for Flash Attention backends.

This module replaces the per-row ``copy_`` assembly loop of
:meth:`TorchReferenceBackend.build_kv` with a single ``index_select``
gather per K/V tensor.

Motivation (measured on the verl FSDP path, Qwen2.5-0.5B, 4-row
micro-batches): the ``new_empty`` + ``copy_`` chain leaves one ``CopySlices``
autograd node per copy — 2 per provider row, 4 per reuser row, per layer.
Each node's backward clones the *entire* expanded-KV gradient buffer and
zeroes its own slice, so the backward of one micro-batch executes ~1.5k
small strictly-serial kernels (24 layers × ~14 nodes × ~4-5 kernels),
costing a token-count-independent ~30-60 ms per micro-batch.
``index_select`` records a single autograd node per tensor whose backward
is one ``index_add`` kernel.

Semantics are identical to the copy-based reference:

* provider row: expanded KV = its own valid packed tokens;
* reuser row: expanded KV = provider-chain prefix followed by its own
  valid suffix tokens.  Transitive reuse chains are resolved directly
  against the packed (unexpanded) K/V coordinates, so no store load/store
  round-trip is needed for the KV content itself;
* the store is **not** touched: its only readers were the copy loop
  itself and the DeltaNet path, so publishing per-row expanded views
  would be write-only dead work;
* gradients flow to the packed K/V rows that produced each token: the
  provider prefix rows receive the sum of all reusers' contributions via
  the gather's ``index_add`` backward (CUDA uses atomicAdd — values are
  exact up to floating-point summation order, same as any scatter-add).

The gather index tensor depends only on ``(plan, packed_batch_layout)`` and
is cached on the plan object (per-micro-batch lifetime), so all layers and
both K and V of one micro-batch share a single index construction + H2D
copy.
"""

from __future__ import annotations

from typing import Any

import torch

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.observability import PrefixSharingStats
from prefix_sharing.core.planner import PrefixSharingPlan

_CACHE_ATTR = "_kv_gather_index_cache"


def _truncate_segments(
    segments: list[tuple[int, int]],
    length: int,
) -> list[tuple[int, int]]:
    """Take the first ``length`` tokens of a segment list (expanded order)."""
    out: list[tuple[int, int]] = []
    remaining = length
    for start, seg_len in segments:
        if remaining <= 0:
            break
        take = min(seg_len, remaining)
        out.append((start, take))
        remaining -= take
    return out


def _build_gather_index(
    plan: PrefixSharingPlan,
    layout: PackedBatchLayout,
    device: Any,
) -> torch.Tensor:
    """Build the packed→expanded gather index tensor for one micro-batch.

    Single forward pass over rows: the online-detector invariant (a provider
    always precedes its reusers, the same invariant the copy-based build_kv
    relies on for its store lookups) means each row's expanded-row segment
    decomposition is already computed when a later row reuses it.  A
    reuser's prefix segments are therefore just its provider's expanded
    segments truncated to ``prefix_len`` — no recursion, no re-resolution
    of ancestor chains.
    """
    row_starts = list(layout.cu_seqlens[:-1])
    segments: list[tuple[int, int]] = []
    expanded_segments_by_row: list[list[tuple[int, int]]] = []
    for row in range(plan.batch_size):
        valid_length = layout.valid_lengths[row]
        if plan.is_reuser(row):
            provider = plan.provider_index[row]
            if provider >= row:
                raise RuntimeError(
                    f"provider index {provider} must precede reuser row {row} "
                    "(online-detector invariant violated)"
                )
            prefix_len = plan.prefix_lens[row]
            prefix_segments = _truncate_segments(expanded_segments_by_row[provider], prefix_len)
            own_segments = [(row_starts[row], valid_length)] if valid_length > 0 else []
            row_segments = prefix_segments + own_segments
        else:
            row_segments = [(row_starts[row], valid_length)] if valid_length > 0 else []
        expanded_segments_by_row.append(row_segments)
        segments.extend(row_segments)

    expanded_total = sum(length for _, length in segments)
    if expanded_total != sum(plan.expanded_lengths_kv):
        raise RuntimeError(
            f"gather index total ({expanded_total}) does not match "
            f"sum(plan.expanded_lengths_kv) ({sum(plan.expanded_lengths_kv)})"
        )

    index = torch.cat(
        [torch.arange(start, start + length, dtype=torch.long) for start, length in segments]
    ) if segments else torch.empty(0, dtype=torch.long)
    return index.to(device)


def get_kv_gather_index(
    plan: PrefixSharingPlan,
    layout: PackedBatchLayout,
    device: Any,
) -> torch.Tensor:
    """Return the cached packed→expanded gather index for ``(plan, layout, device)``.

    The index is identical for every layer and for K and V of one
    micro-batch, so it is built once and cached on the plan object (whose
    lifetime is exactly one micro-batch).  The layout is checked by
    identity to guard against ``id()`` reuse after GC.
    """
    cache = getattr(plan, _CACHE_ATTR, None)
    if cache is None:
        cache = {}
        object.__setattr__(plan, _CACHE_ATTR, cache)
    cache_key = str(device)
    entry = cache.get(cache_key)
    if entry is not None and entry[0] is layout:
        return entry[1]
    index = _build_gather_index(plan, layout, device)
    cache[cache_key] = (layout, index)
    return index


def build_kv_via_gather(
    key: Any,
    value: Any,
    prefix_sharing_plan: PrefixSharingPlan,
    *,
    packed_batch_layout: Any | None = None,
    layer_id: int,
    stats: PrefixSharingStats | None = None,
) -> tuple[Any, Any]:
    """Expand packed K/V to the plan's expanded layout via one gather.

    Drop-in replacement for the copy-loop assembly in
    :meth:`TorchReferenceBackend.build_kv`, with identical output values
    and stats accounting.  See the module docstring for the autograd
    motivation.

    Unlike the copy path, this function has no ``store`` parameter at all:
    transitive reuse is resolved at index-build time directly in packed
    coordinates, and the store's only readers were the copy loop itself
    and the DeltaNet path.  ``layer_id`` is likewise retained only for
    stats accounting (the gather itself is layer-independent — the index
    is shared by all layers).
    """
    plan = prefix_sharing_plan
    layout = packed_batch_layout or PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

    index = get_kv_gather_index(plan, layout, key.device)
    expanded_key = key.index_select(0, index)
    expanded_value = value.index_select(0, index)

    reuse_count = 0
    reused_prefix_tokens = 0
    for batch_index in range(plan.batch_size):
        if plan.is_reuser(batch_index):
            reuse_count += 1
            reused_prefix_tokens += int(plan.prefix_lens[batch_index])

    if stats is not None:
        stats.record_attention_kv_build(
            layer_id=layer_id,
            store_count=plan.batch_size,
            reuse_count=reuse_count,
            reuse_hit_count=reuse_count,
            reuse_miss_count=0,
            stored_tokens=int(expanded_key.shape[0]),
            reused_prefix_tokens=reused_prefix_tokens,
            expanded_kv_tokens=int(expanded_key.shape[0]),
            valid_q_tokens=layout.total_valid_length,
            padded_q_tokens=layout.total_padded_length,
        )
    return expanded_key, expanded_value
