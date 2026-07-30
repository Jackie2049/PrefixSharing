"""GPU Flash Attention 2 backend for prefix sharing.

This backend replaces the reference PyTorch attention with
``flash_attn.flash_attn_interface.flash_attn_varlen_func``, which natively
supports different Q and KV sequence lengths via ``cu_seqlens_q`` and
``cu_seqlens_kv``.  This is exactly what prefix sharing needs: reusers have
shorter Q (suffix only) but full-length KV (prefix + suffix).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.flash_atten_base import FlashAttentionMixin, FlashBackendValidationError
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
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

    ``apply_rope`` is delegated to :class:`TorchReferenceBackend` because
    RoPE position injection is a pure PyTorch operation that does not
    benefit from fused attention kernels.

    ``build_kv`` uses the gather-based expansion in
    :mod:`prefix_sharing.backends.kv_gather` (single ``index_select`` per
    K/V, one ``index_add`` in backward) instead of the reference
    ``new_empty`` + ``copy_`` loop, whose per-copy ``CopySlices`` autograd
    nodes made the backward launch ~1.5k serial small kernels per
    micro-batch.  Set ``PREFIX_SHARING_FA_BUILD_KV=copy`` to fall back to
    the reference copy loop (A/B debugging).

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
    # RoPE: reuse the reference implementation
    # KV build: gather-based expansion (see kv_gather module docstring);
    # PREFIX_SHARING_FA_BUILD_KV=copy falls back to the reference copy loop.
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
        if os.environ.get("PREFIX_SHARING_FA_BUILD_KV", "gather") == "copy":
            return self._torch_ref.build_kv(
                key, value, store, prefix_sharing_plan,
                packed_batch_layout=packed_batch_layout,
                layer_id=layer_id, tp_rank=tp_rank,
                stats=stats,
            )
        from prefix_sharing.backends.kv_gather import build_kv_via_gather

        return build_kv_via_gather(
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
