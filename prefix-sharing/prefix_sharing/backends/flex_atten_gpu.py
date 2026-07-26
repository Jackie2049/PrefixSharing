"""GPU FlexAttention backend for prefix sharing.

This backend uses PyTorch's ``torch.nn.attention.flex_attention`` with a
per-micro-batch ``BlockMask`` to encode prefix-sharing visibility.  Because the
mask itself expresses which provider KV columns each reuser query can attend to,
there is no need for the physical KV expansion performed by ``build_kv``.  For
backends that do this, ``BackendCapabilities.requires_kv_expansion`` is
``False`` and integrations skip the build_kv copy loop entirely.

Layout convention (same as the rest of the backend layer):

* ``query``: ``(total_q_padded, Hq, D)`` packed THD;
* ``key`` / ``value``: ``(total_kv_padded, Hkv, D)`` packed THD;
* attention output: ``(total_q_padded, Hq, D)``.

In the FSDP packed remove-padding path ``total_q_padded == total_kv_padded`` and
there is no TP padding (``align_size=1``).  The tensors are therefore reshaped
and permuted to ``(1, H, T, D)`` for FlexAttention (single batch dim).
"""

from __future__ import annotations

import inspect
import os
from functools import lru_cache
from typing import Any

import torch

from prefix_sharing.backends.base import BackendCapabilities, PrefixAttentionBackend
from prefix_sharing.backends.prefix_block_mask import get_or_create_block_mask
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan


def _flex_attn_env_options() -> tuple[dict[str, Any] | None, tuple[int, int] | None]:
    """Parse optional FlexAttention kernel / BlockMask overrides from env.

    When ``PREFIX_SHARING_FLEX_ATTN_BLOCK_SIZE`` is set, we keep the flex_attention
    kernel tile size and the BlockMask block size in sync.  A mismatch between
    these two is a common cause of illegal memory accesses.

    Returns:
        ``(kernel_options, block_mask_block_size)``.  If the env var is unset or
        invalid, both entries are ``None`` and the backend uses PyTorch defaults.
    """
    block_size_env = os.environ.get("PREFIX_SHARING_FLEX_ATTN_BLOCK_SIZE", "")
    if not block_size_env:
        return None, None

    try:
        block_m, block_n = block_size_env.split(",")
        block_m_i = int(block_m)
        block_n_i = int(block_n)
    except Exception:
        print(
            f"[flex_atten_gpu] ignoring invalid PREFIX_SHARING_FLEX_ATTN_BLOCK_SIZE="
            f"{block_size_env!r}; expected M,N",
            flush=True,
        )
        return None, None

    num_stages = 2
    try:
        num_stages = int(os.environ.get("PREFIX_SHARING_FLEX_ATTN_NUM_STAGES", "2"))
    except Exception:
        num_stages = 2

    kernel_options = {
        "BLOCK_M": block_m_i,
        "BLOCK_N": block_n_i,
        "BLOCK_M1": block_m_i,
        "BLOCK_N1": block_n_i,
        "BLOCK_M2": block_m_i,
        "BLOCK_N2": block_n_i,
        "num_stages": num_stages,
    }
    print(
        f"[flex_atten_gpu] override kernel_options/block_size: "
        f"BLOCK_M={block_m_i}, BLOCK_N={block_n_i}, num_stages={num_stages}",
        flush=True,
    )
    return kernel_options, (block_m_i, block_n_i)


@lru_cache(maxsize=None)
def _import_flex_attention() -> Any:
    """Lazy-import flex_attention once per process and cache the result.

    This keeps the module importable in CPU-only environments (tests) while
    still failing fast at runtime validation when flex_attention is missing.
    """
    try:
        from torch.nn.attention import flex_attention
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GpuFlexAttentionBackend requires torch.nn.attention.flex_attention. "
            "Upgrade to PyTorch >= 2.5."
        ) from exc
    return flex_attention


class GpuFlexAttentionBackend(PrefixAttentionBackend):
    """CUDA/GPU FlexAttention backend with per-micro-batch BlockMask.

    ``apply_rope`` is delegated to :class:`TorchReferenceBackend` because RoPE
    position injection is handled by the integration before the backend is
    called.

    ``build_kv`` is **not used** by integrations: the backend declares
    ``requires_kv_expansion=False`` and ``attention`` consumes the unexpanded
    packed K/V directly.  ``build_kv`` still exists on the class to satisfy the
    protocol, but it raises if called.
    """

    capabilities = BackendCapabilities(
        name="flex_atten_gpu",
        supports_cpu=False,
        supports_cuda=True,
        supports_cann=False,
        supports_different_q_kv_lengths=True,
        supports_prefix_last_restore=True,
        supports_flash_attention=True,
        requires_kv_expansion=False,
    )

    def __init__(self) -> None:
        self._torch_ref = TorchReferenceBackend()
        self._block_mask_cache: dict[Any, Any] = {}
        self._compiled_flex_attention: Any | None = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, config: PrefixSharingConfig, model_config: Any | None = None) -> None:
        config.validate(model_config=model_config)
        flex_module = _import_flex_attention()
        # Fail fast if enable_gqa is not supported by this torch version.
        if "enable_gqa" not in inspect.signature(flex_module.flex_attention).parameters:
            raise RuntimeError(
                "GpuFlexAttentionBackend requires flex_attention(enable_gqa=...). "
                "Upgrade to PyTorch >= 2.6."
            )

    # ------------------------------------------------------------------
    # RoPE & KV build
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
        # Integrations must skip build_kv for this backend because of
        # requires_kv_expansion=False.  Keeping this method on the class only
        # satisfies the PrefixAttentionBackend Protocol; reaching it is a bug.
        raise RuntimeError(
            "GpuFlexAttentionBackend does not use build_kv; "
            "integrations must pass unexpanded packed K/V directly when "
            "requires_kv_expansion=False."
        )

    # ------------------------------------------------------------------
    # Attention: FlexAttention kernel
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
        q: torch.Tensor = query
        k: torch.Tensor = key
        v: torch.Tensor = value

        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            raise ValueError(
                "GpuFlexAttentionBackend expects packed 3-D tensors "
                "(total_tokens, num_heads, head_dim), got shapes "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
            )

        # Unpad if TP padding was ever present (defensive: FSDP path has none).
        if packed_batch_layout is not None and getattr(
            packed_batch_layout, "has_padding", False
        ):
            q = packed_batch_layout.unpad(q)
            k = packed_batch_layout.unpad(k)
            v = packed_batch_layout.unpad(v)
            repad_layout = packed_batch_layout
        else:
            repad_layout = None

        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError(
                "GpuFlexAttentionBackend expects equal packed lengths for Q/K/V "
                f"(got q_len={q.shape[0]}, k_len={k.shape[0]}, v_len={v.shape[0]})."
            )

        total, Hq, D = q.shape
        Hkv = k.shape[1]

        # FlexAttention uses (B, H, S, D); packed THD becomes (1, H, T, D).
        q = q.permute(1, 0, 2).unsqueeze(0).contiguous()          # (1, Hq, T, D)
        k = k.permute(1, 0, 2).unsqueeze(0).contiguous()          # (1, Hkv, T, D)
        v = v.permute(1, 0, 2).unsqueeze(0).contiguous()          # (1, Hkv, T, D)

        flex_module = _import_flex_attention()
        kernel_options, block_size = _flex_attn_env_options()
        block_mask = get_or_create_block_mask(
            prefix_sharing_plan,
            device=q.device,
            cache=self._block_mask_cache,
            block_size=block_size,
        )

        if self._compiled_flex_attention is None:
            self._compiled_flex_attention = torch.compile(
                flex_module.flex_attention, dynamic=True
            )

        out = self._compiled_flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            enable_gqa=True,
            scale=kwargs.get("softmax_scale", None),
            kernel_options=kernel_options,
        )

        # Back to packed THD: (1, H, T, D) -> (T, H, D).
        out = out.squeeze(0).permute(1, 0, 2).contiguous()

        if repad_layout is not None:
            out = repad_layout.repad(out)

        return out

    def gated_attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        gate: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError(
            "GpuFlexAttentionBackend does not support gated_attention."
        )
