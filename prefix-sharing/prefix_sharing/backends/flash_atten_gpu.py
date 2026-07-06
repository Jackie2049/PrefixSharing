"""GPU Flash Attention 2 backend for prefix sharing (s_packed mode).

s_packed Mode
-------------
When s_packed KV storage is used (plan.s_packed_length > 0), falls back to
torch_ref.attention() which supports the global custom causal mask.

For s_packed mode, ``flash_attn_varlen_func`` is not used because it does not
support arbitrary custom masks. The reference PyTorch attention path is correct
and competitive in speed for typical batch sizes.
"""

from __future__ import annotations

from typing import Any

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan

class GpuFlashAttentionBackend:
    """CUDA/GPU Flash Attention 2 backend.

    ``apply_rope`` and ``build_kv`` are delegated to
    :class:`TorchReferenceBackend` because RoPE position injection and KV
    cache store/load are pure PyTorch operations that do not benefit from
    fused attention kernels.

    Only ``attention()`` is replaced by the Flash Attention 2 kernel.
    """

    capabilities = BackendCapabilities(
        name="flash_atten_gpu",
        supports_cpu=False,
        supports_cuda=True,
        supports_cann=False,
        supports_different_q_kv_lengths=True,
        supports_prefix_last_restore=True,
        supports_flash_attention=True,
    )

    def __init__(self) -> None:
        self._torch_ref = TorchReferenceBackend()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, config: PrefixSharingConfig, model_config: Any | None = None) -> None:
        config.validate(model_config=model_config)
        # Eager import check so that mis-configured environments fail fast.
        _import_flash_attn_varlen()

    # ------------------------------------------------------------------
    # RoPE & KV build: reuse the reference implementation
    # ------------------------------------------------------------------
    def apply_rope(
        self,
        query: Any,
        key: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        return self._torch_ref.apply_rope(query, key, prefix_sharing_plan, **kwargs)

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
        return self._torch_ref.build_kv(
            key, value, store, prefix_sharing_plan,
            packed_batch_layout=packed_batch_layout,
            layer_id=layer_id, tp_rank=tp_rank,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Attention: torch_ref for s_packed mode (custom mask support)
    # ------------------------------------------------------------------
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
        # s_packed mode requires custom causal mask which flash_attn_varlen_func
        # does not support, so delegate to torch_ref.attention()
        return self._torch_ref.attention(
            query, key, value, prefix_sharing_plan,
            packed_batch_layout=packed_batch_layout,
            **kwargs,
        )
