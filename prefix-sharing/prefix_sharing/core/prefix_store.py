"""Prefix activation stores for one forward/backward lifecycle.

This module manages **logical** tensors produced during prefix sharing
within a single forward/backward pass. It sits downstream of planning
(:mod:`prefix_sharing.core.planner`) and is consumed by attention backends
(e.g. :mod:`prefix_sharing.backends.torch_ref`) and integration context
(:mod:`prefix_sharing.integrations.context`). Stores do not decide reuse
relations or prefix lengths; they only store and retrieve framework runtime
state by composite keys.

Core Responsibilities:
    1. **Isolate prefix activations by scope**: index entries with ``forward_id``,
       ``micro_batch_id``, ``layer_id``, ``sample_idx_in_batch``, and
       ``tp_rank`` so micro-batches, layers, and tensor-parallel ranks do not
       collide.
    2. **Preserve attention KV history**: store logical attention K/V tensors
       while sharing lifecycle and isolation mechanics.
    3. **Lifecycle management**: ``clear`` / ``close`` bound the cache to one
       runtime context; after ``close``, ``store`` and ``load`` raise.

Key Concepts:
    - Attention entry: K/V written when a row is not a reuser, keyed by its
      batch index; reuser entries concatenate provider prefix K/V with suffix
      K/V for transitive reuse in deeper layers.
    - ``prefix_len`` on :class:`StoredAttentionKV`: metadata for how much of the
      stored tensor counts as prefix when slicing; backends use it with per-row
      ``prefix_lens`` from batch metadata.

Key Components:
    - :class:`PrefixActivationSlotId`: Immutable identifier for one reusable prefix activation.
    - :class:`PrefixActivationStore`: Shared lifecycle and isolation rules for prefix activation stores.
    - :class:`StoredAttentionKV`: Attention K/V tensors plus logical prefix length.
    - :class:`PrefixAttentionStore`: Typed store for attention K/V entries.

Design Principles:
    - **Never detach**: cached tensors must retain the autograd graph so gradients
      flow correctly through shared prefix K/V.
    - **No overwrite by default**: duplicate slots raise unless ``overwrite=True``
      (backends set this when updating expanded reuser entries).
    - **Framework-agnostic**: stored tensors/states are ``Any``; no PyTorch
      import in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PREFIX_STATE_TYPE_ATTENTION_KV = "attention_kv"


@dataclass(frozen=True)
class PrefixActivationSlotId:
    """Unique slot id for stored prefix activation within one runtime context."""

    forward_id: int
    micro_batch_id: int
    layer_id: int
    sample_idx_in_batch: int
    prefix_state_type: str
    tp_rank: int = 0


@dataclass(frozen=True)
class StoredAttentionKV:
    """Stored logical prefix K/V tensors plus the prefix slice length."""

    key_tensor: Any
    value_tensor: Any
    prefix_len: int


class PrefixActivationStore:
    """Shared lifecycle-bound store for reusable prefix activation entries."""

    def __init__(self) -> None:
        self._entries: dict[PrefixActivationSlotId, Any] = {}
        self._closed = False

    def store_entry(
        self,
        slot_id: PrefixActivationSlotId,
        *,
        entry: Any,
        overwrite: bool = False,
    ) -> None:
        self._ensure_open()
        if slot_id in self._entries and not overwrite:
            raise KeyError(f"stored prefix state already exists for {slot_id}")
        self._entries[slot_id] = entry

    def load_entry(self, slot_id: PrefixActivationSlotId) -> Any:
        self._ensure_open()
        try:
            return self._entries[slot_id]
        except KeyError as exc:
            raise KeyError(f"missing stored prefix state for {slot_id}") from exc

    def contains(self, slot_id: PrefixActivationSlotId) -> bool:
        return slot_id in self._entries

    def clear(self) -> None:
        self._entries.clear()

    def close(self) -> None:
        self.clear()
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def size(self) -> int:
        return len(self._entries)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self.__class__.__name__} is closed")


class PrefixAttentionStore(PrefixActivationStore):
    """Typed store for logical attention K/V tensors."""

    def store(
        self,
        slot_id: PrefixActivationSlotId,
        *,
        key_tensor: Any,
        value_tensor: Any,
        prefix_len: int,
        overwrite: bool = False,
    ) -> None:
        if slot_id.prefix_state_type != PREFIX_STATE_TYPE_ATTENTION_KV:
            raise ValueError("PrefixAttentionStore requires prefix_state_type='attention_kv'")
        if prefix_len < 0:
            raise ValueError("prefix_len must be >= 0")
        self.store_entry(
            slot_id,
            entry=StoredAttentionKV(key_tensor=key_tensor, value_tensor=value_tensor, prefix_len=prefix_len),
            overwrite=overwrite,
        )

    def load(self, slot_id: PrefixActivationSlotId) -> StoredAttentionKV:
        entry = self.load_entry(slot_id)
        if not isinstance(entry, StoredAttentionKV):
            raise TypeError(f"stored prefix state is not attention KV for {slot_id}")
        return entry
