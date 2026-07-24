"""Prefix-sharing BlockMask construction for FlexAttention backends.

This module encodes prefix-sharing visibility as a FlexAttention ``mask_mod``
over the **unexpanded** packed kept-token stream (Q_LEN == KV_LEN == total
kept tokens), so mask-based backends can skip ``build_kv`` entirely: a reuser's
queries attend directly to the provider chain's KV columns in the packed
stream instead of to a physically concatenated copy.

Visibility semantics (identical to ``block_causal_mask`` over the expanded
layout):

* same row: causal on absolute positions (``kv_pos <= q_pos``);
* cross row: row ``i`` sees row ``j``'s tokens whose absolute position is
  below ``cross_limit[i, j]`` — the shared prefix window, propagated along
  the reuse chain.

The chain limit matrix is built **top-down in a single pass**: the plan's row
order guarantees every provider precedes its reusers (the same invariant
``TorchReferenceBackend.build_kv`` relies on), so a reuser's visibility is its
provider's already-computed row, clamped to its own prefix length::

    L[i, :] = min(prefix_len[i], L[provider, :])
    L[i, provider] = prefix_len[i]

No recursion over ancestors is needed: the provider's row already encodes the
whole chain, and the clamp is exactly "the provider's mask row at the last
shared-prefix token".
"""

from __future__ import annotations

from typing import Any, Callable

from prefix_sharing.core.planner import PrefixSharingPlan


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("prefix_block_mask requires PyTorch") from exc
    return torch


def build_chain_limit_matrix(prefix_sharing_plan: PrefixSharingPlan) -> Any:
    """Build the ``cross_limit`` matrix, shape (batch_size, batch_size), int64.

    ``cross_limit[i, j]`` is the exclusive upper bound on absolute positions of
    row-``j`` KV tokens visible to row-``i`` queries via the prefix chain;
    ``0`` means row ``j`` is invisible to row ``i`` (including ``i == j`` —
    same-row visibility is handled by the causal branch of the mask_mod).

    Built top-down in one O(batch_size^2) pass; requires the
    provider-before-reuser ordering invariant.
    """
    torch = _torch()
    plan = prefix_sharing_plan
    batch_size = plan.batch_size
    cross_limit = torch.zeros(batch_size, batch_size, dtype=torch.int64)
    for row in range(batch_size):
        if not plan.is_reuser(row):
            continue
        provider = plan.provider_index[row]
        if provider >= row:
            raise ValueError(
                "prefix-sharing plan violates the provider-before-reuser "
                f"ordering invariant: row {row} reuses later row {provider}"
            )
        prefix_len = int(plan.prefix_lens[row])
        inherited = torch.clamp(cross_limit[provider], max=prefix_len)
        inherited[provider] = prefix_len
        inherited[row] = 0
        cross_limit[row] = inherited
    return cross_limit


def build_token_index_tensors(
    prefix_sharing_plan: PrefixSharingPlan,
    *,
    device: Any | None = None,
) -> tuple[Any, Any]:
    """Build ``token_row`` / ``token_pos`` over the packed kept-token stream.

    Both are int64 tensors of shape ``(total_kept_tokens,)``:
    ``token_row[t]`` is the batch row of packed token ``t`` and
    ``token_pos[t]`` its absolute position (reuser suffix tokens start at
    ``q_position_offsets[row] == prefix_len``).
    """
    torch = _torch()
    plan = prefix_sharing_plan
    row_chunks = []
    pos_chunks = []
    for row in range(plan.batch_size):
        kept_length = int(plan.kept_lengths_q[row])
        offset = int(plan.q_position_offsets[row])
        row_chunks.append(torch.full((kept_length,), row, dtype=torch.int64))
        pos_chunks.append(torch.arange(offset, offset + kept_length, dtype=torch.int64))
    if not row_chunks:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty
    token_row = torch.cat(row_chunks).to(device)
    token_pos = torch.cat(pos_chunks).to(device)
    return token_row, token_pos


def make_prefix_sharing_mask_mod(
    token_row: Any,
    token_pos: Any,
    cross_limit: Any,
) -> Callable[[Any, Any, Any, Any], Any]:
    """Create the FlexAttention ``mask_mod`` closure.

    The three index tensors are captured by the closure; their *values* never
    participate in torch.compile guards, so one compiled kernel serves every
    micro-batch (only shapes matter, handled by ``dynamic=True``).  All
    position/prefix information must flow in as tensors — do not introduce
    Python scalar branches on plan values inside the closure, or recompiles
    will occur per micro-batch.
    """

    def mask_mod(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        row_q = token_row[q_idx]
        row_kv = token_row[kv_idx]
        same_row_causal = (row_q == row_kv) & (token_pos[kv_idx] <= token_pos[q_idx])
        cross_prefix = cross_limit[row_q, row_kv] > token_pos[kv_idx]
        return same_row_causal | cross_prefix

    return mask_mod


def get_or_create_block_mask(
    prefix_sharing_plan: PrefixSharingPlan,
    *,
    device: Any,
    cache: dict | None = None,
    block_size: int | tuple[int, int] | None = None,
) -> Any:
    """Return the ``BlockMask`` for this plan, building and caching it if needed.

    The mask depends only on the plan's sequence structure, which is shared by
    every layer of one micro-batch forward, so it is built **once per
    micro-batch** and reused across layers.  ``cache`` (owned by the backend
    instance) maps ``(id(plan), total_tokens, device)`` to
    ``(plan, block_mask)`` — the plan reference is kept alive so its ``id``
    cannot be recycled while cached; entries are evicted FIFO beyond a small
    bound (a micro-batch's plan dies with its runtime context anyway).

    Sequence lengths need not be multiples of BLOCK_SIZE: modern PyTorch pads
    the mask internally during block conversion (padding evaluates to masked).
    """
    torch = _torch()
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "prefix_block_mask requires torch.nn.attention.flex_attention "
            "(PyTorch >= 2.5)"
        ) from exc

    total = int(prefix_sharing_plan.cu_seqlens_q[-1])
    cache_key = (id(prefix_sharing_plan), total, str(device))
    if cache is not None and cache_key in cache:
        return cache[cache_key][1]

    token_row, token_pos = build_token_index_tensors(prefix_sharing_plan, device=device)
    cross_limit = build_chain_limit_matrix(prefix_sharing_plan).to(device)
    mask_mod = make_prefix_sharing_mask_mod(token_row, token_pos, cross_limit)
    extra_kwargs: dict[str, Any] = {}
    if block_size is not None:
        extra_kwargs["BLOCK_SIZE"] = block_size
    block_mask = create_block_mask(
        mask_mod, 1, 1, total, total, device=device, **extra_kwargs
    )

    if cache is not None:
        max_entries = 8
        cache[cache_key] = (prefix_sharing_plan, block_mask)
        while len(cache) > max_entries:
            cache.pop(next(iter(cache)))
    return block_mask
