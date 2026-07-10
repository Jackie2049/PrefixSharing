"""Framework-independent prefix sharing semantics."""

from prefix_sharing.core.batch_trim import TrimmedBatch, trim_batch, trim_inputs, trim_labels, trim_loss_masks
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
    "PrefixReuseSpec",
    "PREFIX_STATE_TYPE_ATTENTION_KV",
    "PrefixActivationSlotId",
    "PrefixActivationStore",
    "PrefixAttentionStore",
    "PrefixSharingLayerStats",
    "PrefixSharingPlan",
    "PrefixSharingStats",
    "PrefixSharingConfig",
    "PrefixSharingConfigError",
    "PrefixLastRestoreSpec",
    "PrefixSharingPlanner",
    "StoredAttentionKV",
    "TrimmedBatch",
    "TriePrefixDetector",
    "trim_batch",
    "trim_inputs",
    "trim_labels",
    "trim_loss_masks",
]
