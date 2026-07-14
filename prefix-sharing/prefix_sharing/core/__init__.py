"""Framework-independent prefix sharing semantics."""

from prefix_sharing.core.batch_trim import TrimmedBatch, trim_batch, trim_inputs, trim_labels, trim_loss_masks
from prefix_sharing.core.attention_layout import (
    PrefixAttentionMaskType,
    PrefixTreeAttentionLayout,
    PrefixTreeAttentionSlice,
    build_prefix_tree_attention_layout,
)
from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError
from prefix_sharing.core.observability import PrefixSharingLayerStats, PrefixSharingStats
from prefix_sharing.core.prefix_detector import PrefixDetectionResult, PrefixReuseSpec, TriePrefixDetector
from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_ATTENTION_KV,
    PrefixActivationSlotId,
    PrefixActivationStore,
    PrefixAttentionStore,
    StoredAttentionKV,
)
from prefix_sharing.core.planner import PrefixLastRestoreSpec, PrefixSharingPlan, PrefixSharingPlanner

__all__ = [
    "PrefixDetectionResult",
    "PrefixAttentionMaskType",
    "PrefixReuseSpec",
    "PREFIX_STATE_TYPE_ATTENTION_KV",
    "PrefixActivationSlotId",
    "PrefixActivationStore",
    "PrefixAttentionStore",
    "PrefixSharingLayerStats",
    "PrefixTreeAttentionLayout",
    "PrefixTreeAttentionSlice",
    "PrefixSharingPlan",
    "PrefixSharingStats",
    "PrefixSharingConfig",
    "PrefixSharingConfigError",
    "PrefixLastRestoreSpec",
    "PrefixSharingPlanner",
    "StoredAttentionKV",
    "TrimmedBatch",
    "TriePrefixDetector",
    "build_prefix_tree_attention_layout",
    "trim_batch",
    "trim_inputs",
    "trim_labels",
    "trim_loss_masks",
]
