"""Backend interface consumed by integrations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan


class PrefixAttentionExecutionMode(str, Enum):
    """Physical Q/K/V layout consumed by an attention backend.

    ``EXPANDED_KV`` is the established PrefixSharing execution path: reuser
    prefixes are materialized into per-row K/V before attention.  Sparse
    backends consume the original deduplicated Q/K/V together with a separate
    visibility layout, so they must not be forced through ``build_kv()``.
    """

    EXPANDED_KV = "expanded_kv"
    DEDUPLICATED_QKV = "deduplicated_qkv"


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    supports_cpu: bool
    supports_cuda: bool
    supports_cann: bool
    supports_different_q_kv_lengths: bool
    supports_prefix_last_restore: bool
    supports_fused_rope: bool = False
    supports_context_parallel: bool = False
    supports_pipeline_parallel: bool = False
    supports_flash_attention: bool = False
    execution_mode: PrefixAttentionExecutionMode = PrefixAttentionExecutionMode.EXPANDED_KV


class PrefixAttentionBackend(Protocol):
    capabilities: BackendCapabilities

    def validate(self, config: PrefixSharingConfig, model_config: Any | None = None) -> None:
        ...

    def apply_rope(self, query: Any, key: Any, prefix_sharing_plan: PrefixSharingPlan, **kwargs: Any) -> tuple[Any, Any]:
        ...

    def attention(self, query: Any, key: Any, value: Any, prefix_sharing_plan: PrefixSharingPlan, **kwargs: Any) -> Any:
        ...


class ExpandedKVPrefixAttentionBackend(PrefixAttentionBackend, Protocol):
    """Backend contract for paths that require materialized expanded K/V."""

    def build_kv(
        self,
        key: Any,
        value: Any,
        store: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        layer_id: int,
        tp_rank: int = 0,
        stats: Any | None = None,
    ) -> tuple[Any, Any]:
        ...

class DeduplicatedPrefixAttentionBackend(PrefixAttentionBackend, Protocol):
    """Backend contract for sparse paths consuming deduplicated Q/K/V directly.

    The execution layout is supplied through backend-specific runtime metadata
    (for example a FlexAttention BlockMask).  This protocol intentionally has
    no ``build_kv`` method: creating a no-op implementation would hide the
    physical-input distinction that integration code must respect.
    """

    def prepare_runtime(self, **kwargs: Any) -> Any:
        ...


def requires_expanded_kv(backend: PrefixAttentionBackend) -> bool:
    """Return whether integration must materialize prefix-expanded K/V."""

    return backend.capabilities.execution_mode is PrefixAttentionExecutionMode.EXPANDED_KV
