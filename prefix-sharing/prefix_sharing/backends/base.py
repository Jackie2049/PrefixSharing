"""Backend interface consumed by integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan


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


class PrefixAttentionBackend(Protocol):
    capabilities: BackendCapabilities

    def validate(self, config: PrefixSharingConfig, model_config: Any | None = None) -> None:
        ...

    def apply_rope(self, query: Any, key: Any, prefix_sharing_plan: PrefixSharingPlan, **kwargs: Any) -> tuple[Any, Any]:
        ...

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

    def attention(self, query: Any, key: Any, value: Any, prefix_sharing_plan: PrefixSharingPlan, **kwargs: Any) -> Any:
        ...
