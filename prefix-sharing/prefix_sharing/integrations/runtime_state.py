"""Shared runtime state carried by verl integration paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.attention_layout import PrefixTreeAttentionLayout
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo


@dataclass(frozen=True)
class PrefixSharingRuntimeState:
    prefix_sharing_plan: PrefixSharingPlan
    attention_backend: Any
    packed_batch_layout: PackedBatchLayout
    parallel_info: MegatronParallelInfo
    kept_position_ids: Any | None = None
    prefix_tree_attention_layout: PrefixTreeAttentionLayout | None = None
