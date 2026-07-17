"""Pure PyTorch BSHD reference backend.

This backend is the BSHD counterpart of :class:`TorchReferenceBackend`: it
receives and produces ``[B, S, H, D]`` padded tensors instead of THD packed
``[total_tokens, H, D]``.

Coordinate convention (absolute coordinates — see ``batched_layout.py``):

* row ``i``'s valid tokens occupy columns ``[0, valid_lengths[i])``
* a reuser row's suffix Q occupies rows ``[prefix_len, valid_len)``; its prefix
  rows ``[0, prefix_len)`` physically exist but are fully masked in attention
  and zeroed in the output (they are never read downstream: every layer's
  build_kv replaces the reuser's prefix KV with the provider's stored KV, and
  the final prefix-column logprobs are restored from the provider row).
* the expanded KV of a reuser is ``cat(provider_prefix, own_suffix)`` whose
  column indices coincide with the padded column space.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.batched_layout import BatchedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.observability import PrefixSharingStats
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_ATTENTION_KV,
    PrefixActivationSlotId,
    PrefixAttentionStore,
)


class TorchReferenceBackendBshd:
    """BSHD reference backend — pure PyTorch, ``[B, S, H, D]`` padded tensors."""

    capabilities = BackendCapabilities(
        name="torch_ref_bshd",
        supports_cpu=True,
        supports_cuda=True,
        supports_cann=True,
        supports_different_q_kv_lengths=True,
        supports_prefix_last_restore=True,
        supports_gated_attention=False,
        supports_deltanet_state_reuse=False,
        supports_bshd=True,
    )

    def validate(self, config: PrefixSharingConfig, model_config: Any | None = None) -> None:
        config.validate(model_config=model_config)

    # ── build_kv ──────────────────────────────────────────────────────────

    def build_kv(
        self,
        key: Any,
        value: Any,
        store: PrefixAttentionStore,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        layer_id: int,
        tp_rank: int = 0,
        stats: PrefixSharingStats | None = None,
    ) -> tuple[Any, Any]:
        """Build prefix-expanded K/V in ``[B, max_kv, H, D]`` BSHD format.

        Each provider row's valid KV ``[0, expanded_len)`` is stored in the
        prefix store.  Each reuser row loads the provider's prefix KV and
        concatenates its own suffix KV ``[prefix_len, expanded_len)``.  All
        rows are right-padded to ``max_kv`` and stacked.

        Slicing uses absolute coordinates on the untrimmed padded input:
        ``expanded_lengths_kv[i] == original_lengths[i]`` for every row.
        """
        B = key.shape[0]
        expanded_keys: list[torch.Tensor] = []
        expanded_values: list[torch.Tensor] = []
        store_count = 0
        reuse_count = 0
        reuse_hit_count = 0
        reuse_miss_count = 0
        stored_tokens = 0
        reused_prefix_tokens = 0

        # Same online-detector invariant as the THD reference backend: a
        # provider appears before every reuser that loads from it.  Do not
        # reorder or parallelize this loop.
        for batch_index in range(B):
            expanded_len = prefix_sharing_plan.expanded_lengths_kv[batch_index]
            if not prefix_sharing_plan.is_reuser(batch_index):
                # Provider: store ALL valid tokens as the reusable KV.
                valid_key = key[batch_index, :expanded_len]
                valid_value = value[batch_index, :expanded_len]
                slot_id = PrefixActivationSlotId(
                    prefix_sharing_plan.forward_id,
                    prefix_sharing_plan.micro_batch_id,
                    layer_id,
                    batch_index,
                    PREFIX_STATE_TYPE_ATTENTION_KV,
                    tp_rank,
                )
                store.store(
                    slot_id,
                    key_tensor=valid_key,
                    value_tensor=valid_value,
                    prefix_len=valid_key.shape[0],
                    overwrite=True,
                )
                store_count += 1
                stored_tokens += int(valid_key.shape[0])
                expanded_keys.append(valid_key)
                expanded_values.append(valid_value)
            else:
                provider = prefix_sharing_plan.provider_index[batch_index]
                provider_slot_id = PrefixActivationSlotId(
                    prefix_sharing_plan.forward_id,
                    prefix_sharing_plan.micro_batch_id,
                    layer_id,
                    provider,
                    PREFIX_STATE_TYPE_ATTENTION_KV,
                    tp_rank,
                )
                reuse_count += 1
                try:
                    entry = store.load(provider_slot_id)
                except KeyError:
                    reuse_miss_count += 1
                    if stats is not None:
                        stats.record_attention_kv_build(
                            layer_id=layer_id,
                            store_count=store_count,
                            reuse_count=reuse_count,
                            reuse_hit_count=reuse_hit_count,
                            reuse_miss_count=reuse_miss_count,
                            stored_tokens=stored_tokens,
                            reused_prefix_tokens=reused_prefix_tokens,
                            expanded_kv_tokens=sum(int(r.shape[0]) for r in expanded_keys),
                            valid_q_tokens=sum(prefix_sharing_plan.original_lengths),
                            padded_q_tokens=B * key.shape[1],
                        )
                    raise
                reuse_hit_count += 1
                prefix_len = prefix_sharing_plan.prefix_lens[batch_index]
                reused_prefix_tokens += int(prefix_len)
                # Reuser's own suffix KV: absolute columns [prefix_len, expanded_len).
                suffix_key = key[batch_index, prefix_len:expanded_len]
                suffix_value = value[batch_index, prefix_len:expanded_len]
                expanded_key = torch.cat([entry.key_tensor[:prefix_len], suffix_key], dim=0)
                expanded_value = torch.cat([entry.value_tensor[:prefix_len], suffix_value], dim=0)
                # Publish expanded KV for transitive reuse (a later row may
                # reuse this longer prefix).
                own_slot_id = PrefixActivationSlotId(
                    prefix_sharing_plan.forward_id,
                    prefix_sharing_plan.micro_batch_id,
                    layer_id,
                    batch_index,
                    PREFIX_STATE_TYPE_ATTENTION_KV,
                    tp_rank,
                )
                store.store(
                    own_slot_id,
                    key_tensor=expanded_key,
                    value_tensor=expanded_value,
                    prefix_len=expanded_key.shape[0],
                    overwrite=True,
                )
                store_count += 1
                stored_tokens += int(expanded_key.shape[0])
                expanded_keys.append(expanded_key)
                expanded_values.append(expanded_value)

        if stats is not None:
            stats.record_attention_kv_build(
                layer_id=layer_id,
                store_count=store_count,
                reuse_count=reuse_count,
                reuse_hit_count=reuse_hit_count,
                reuse_miss_count=reuse_miss_count,
                stored_tokens=stored_tokens,
                reused_prefix_tokens=reused_prefix_tokens,
                expanded_kv_tokens=sum(int(r.shape[0]) for r in expanded_keys),
                valid_q_tokens=sum(prefix_sharing_plan.original_lengths),
                padded_q_tokens=B * key.shape[1],
            )

        # Right-pad to max KV length, stack → [B, max_kv, H, D].
        if not expanded_keys:
            return key, value  # empty batch
        max_kv_len = max(k.shape[0] for k in expanded_keys)
        _, H, D = key.shape[1:]
        padded_k = torch.zeros(B, max_kv_len, H, D, dtype=key.dtype, device=key.device)
        padded_v = torch.zeros(B, max_kv_len, H, D, dtype=value.dtype, device=value.device)
        for i, (k_row, v_row) in enumerate(zip(expanded_keys, expanded_values)):
            length = k_row.shape[0]
            padded_k[i, :length] = k_row
            padded_v[i, :length] = v_row

        return padded_k, padded_v

    # ── attention ─────────────────────────────────────────────────────────

    def attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """BSHD batched attention with per-sample prefix-aware causal mask.

        Input shapes:
            query: ``[B, S_q, H_q, D]`` (untrimmed padded)
            key:   ``[B, S_kv, H_kv, D]`` (build_kv output, right-padded)
            value: ``[B, S_kv, H_kv, D]``

        Returns ``[B, S_q, H_q, D]``; non-kept Q rows (provider padding and
        reuser prefix rows ``[0, prefix_len)``) are exactly zero.
        """
        layout: BatchedBatchLayout | None = (
            packed_batch_layout if isinstance(packed_batch_layout, BatchedBatchLayout) else None
        )
        valid_lengths = (
            layout.valid_lengths if layout is not None else list(prefix_sharing_plan.original_lengths)
        )
        expanded_kv_lengths = prefix_sharing_plan.expanded_lengths_kv

        B, S_q, H_q, D = query.shape
        device = query.device
        dtype = query.dtype

        # GQA/MQA head expansion (repeat_interleave, same as TorchReferenceBackend).
        kv_heads = key.shape[2]
        if H_q != kv_heads:
            if H_q % kv_heads != 0:
                raise ValueError("query heads must be a multiple of kv heads for grouped-query attention")
            repeat = H_q // kv_heads
            key = key.repeat_interleave(repeat, dim=2)
            value = value.repeat_interleave(repeat, dim=2)

        # 4-D prefix-aware causal mask [B, 1, S_q, S_kv] in absolute coordinates.
        mask = _build_bshd_attention_mask(
            plan=prefix_sharing_plan,
            valid_lengths=valid_lengths,
            expanded_kv_lengths=expanded_kv_lengths,
            max_q=S_q,
            max_kv=key.shape[1],
            batch_size=B,
            device=device,
        )

        scale = 1.0 / math.sqrt(D)
        scores = torch.einsum("bqhd,bkhd->bhqk", query, key) * scale
        scores = scores.masked_fill(mask, torch.finfo(dtype).min)
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", probs, value)

        # Zero non-kept Q rows (provider padding + reuser prefix rows).  Fully
        # masked rows above produce finite garbage (finfo.min → uniform softmax);
        # those columns are never read downstream, but zeroing keeps the output
        # deterministic and prevents any accidental leakage through FFN/residual.
        keep = _kept_q_row_mask(prefix_sharing_plan, valid_lengths, S_q, device)
        return out * keep.unsqueeze(-1).unsqueeze(-1).to(dtype)


# ── BSHD mask construction ──────────────────────────────────────────────


def _build_bshd_attention_mask(
    plan: PrefixSharingPlan,
    valid_lengths: list[int],
    expanded_kv_lengths: list[int],
    max_q: int,
    max_kv: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a ``[B, 1, max_q, max_kv]`` prefix-aware causal mask.

    * ``True`` = masked (not attended to).
    * ``False`` = visible (attended to).

    Absolute coordinates (row/col indices == padded column space):

    * provider row ``i``: standard causal within ``[0, valid) × [0, kv)``.
    * reuser row ``i``: Q rows are the suffix rows ``[prefix_len, valid)``;
      prefix KV columns ``[0, prefix_len)`` are all-visible; suffix KV columns
      ``[prefix_len, kv)`` are causal within the suffix.
    * everything else (padding Q rows, reuser prefix Q rows, padding KV
      columns) stays ``True``.
    """
    mask = torch.ones(batch_size, 1, max_q, max_kv, dtype=torch.bool, device=device)

    for i in range(batch_size):
        q_val = valid_lengths[i]
        kv_val = expanded_kv_lengths[i]
        if q_val == 0 or kv_val == 0:
            continue

        if plan.is_reuser(i):
            prefix_len = int(plan.prefix_lens[i])
            suffix_len = kv_val - prefix_len
            # Reuser Q rows [prefix_len, q_val): prefix KV columns all visible.
            if prefix_len > 0:
                mask[i, 0, prefix_len:q_val, :prefix_len] = False
            # Suffix KV columns: causal within the suffix (j sees k <= j).
            if suffix_len > 0:
                causal = torch.ones(suffix_len, suffix_len, dtype=torch.bool, device=device).tril(diagonal=0)
                mask[i, 0, prefix_len:q_val, prefix_len:prefix_len + suffix_len] = ~causal
        else:
            # Provider: standard causal within [q_val, kv_val].
            block = torch.ones(q_val, kv_val, dtype=torch.bool, device=device)
            mask[i, 0, :q_val, :kv_val] = torch.triu(block, diagonal=1)

    return mask


def _kept_q_row_mask(
    plan: PrefixSharingPlan,
    valid_lengths: list[int],
    max_q: int,
    device: torch.device,
) -> torch.Tensor:
    """``[B, max_q]`` bool — ``True`` for Q rows whose output is meaningful.

    Provider: ``[0, valid)``; reuser: suffix rows ``[prefix_len, valid)``.
    """
    keep = torch.zeros(plan.batch_size, max_q, dtype=torch.bool, device=device)
    for i in range(plan.batch_size):
        start = int(plan.prefix_lens[i]) if plan.is_reuser(i) else 0
        keep[i, start:valid_lengths[i]] = True
    return keep
