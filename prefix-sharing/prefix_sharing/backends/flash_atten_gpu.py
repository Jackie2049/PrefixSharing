"""GPU Flash Attention 2 backend for prefix sharing.

This backend replaces the reference PyTorch attention with
``flash_attn.flash_attn_interface.flash_attn_varlen_func``, which natively
supports different Q and KV sequence lengths via ``cu_seqlens_q`` and
``cu_seqlens_kv``.  This is exactly what prefix sharing needs: reusers have
shorter Q (suffix only) but full-length KV (prefix + suffix).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.flash_atten_base import FlashAttentionMixin, FlashBackendValidationError
from prefix_sharing.backends.kv_builder import apply_rope_with_plan, build_prefix_expanded_kv
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan

@lru_cache(maxsize=None)
def _import_flash_attn_varlen() -> Any:
    """Lazy-import ``flash_attn_varlen_func`` once and cache the result.

    Using ``@lru_cache`` guarantees the import (and the underlying library
    initialisation) happens at most once per process, while still keeping the
    import lazy so that CPU-only environments can import this module without
    crashing.
    """
    try:
        from flash_attn import flash_attn_varlen_func
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GpuFlashAttentionBackend requires flash-attn. "
            "Please Install flash-attention first"
        ) from exc
    return flash_attn_varlen_func


class GpuFlashAttentionBackend(FlashAttentionMixin):
    """CUDA/GPU Flash Attention 2 backend.

    ``apply_rope`` and ``build_kv`` use shared PyTorch helpers because RoPE
    position injection and KV cache store/load do not benefit from fused
    attention kernels.

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
        return apply_rope_with_plan(query, key, prefix_sharing_plan, **kwargs)

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
        return build_prefix_expanded_kv(
            key, value, store, prefix_sharing_plan,
            packed_batch_layout=packed_batch_layout,
            layer_id=layer_id, tp_rank=tp_rank,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Attention: Flash Attention 2 kernel
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
        q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, pad_layout = (
            self._prepare_flash_inputs(
                query, key, value, prefix_sharing_plan,
                attention_mask=kwargs.get("attention_mask"),
                packed_batch_layout=packed_batch_layout,
            )
        )

        flash_attn_varlen_func = _import_flash_attn_varlen()

        try:
            out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                dropout_p=kwargs.get("dropout_p", 0.0),
                softmax_scale=kwargs.get("softmax_scale", None),  # defaults to 1/sqrt(head_dim)
                causal=kwargs.get("causal", True),
                window_size=kwargs.get("window_size", (-1, -1)),
                softcap=kwargs.get("softcap", 0.0),
                alibi_slopes=kwargs.get("alibi_slopes", None),
                deterministic=kwargs.get("deterministic", False),
            )
        except Exception as exc:
            raise FlashBackendValidationError(
                f"flash_attn_varlen_func failed on device={q.device}, "
                f"q_shape={tuple(q.shape)}, k_shape={tuple(k.shape)}, "
                f"cu_seqlens_q={cu_seqlens_q.tolist()}, cu_seqlens_kv={cu_seqlens_kv.tolist()}, "
                f"max_seqlen_q={max_seqlen_q}, max_seqlen_kv={max_seqlen_kv}"
            ) from exc

        # Re-apply TP padding that was stripped in _prepare_flash_inputs so
        # the output shape matches the original (padded) query tensor.
        if pad_layout is not None:
            out = self._repad_output(out, pad_layout)

        return out
