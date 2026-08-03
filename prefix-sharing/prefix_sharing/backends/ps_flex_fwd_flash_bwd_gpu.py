"""Hybrid PrefixSharing attention backend: FlexAttention forward + FlashAttention backward.

This backend combines the best of two worlds:

* **Forward**: PyTorch ``flex_attention`` with a per-micro-batch ``BlockMask``.
  No physical KV expansion is needed; visibility is encoded in the mask.
* **Backward**: The Flash Attention 2 varlen backward kernel, which is much
  faster than the Inductor-generated FlexAttention backward for our workload.

The crucial memory trick is wrapped in a custom ``torch.autograd.Function``:
autograd only saves the **packed** K/V tensors.  During ``backward`` the
expanded KV layout is recomputed via ``index_select`` and immediately consumed
by the Flash backward kernel, so the expanded layout is never retained between
layers.  Compared with the FA path that saves expanded K/V, this saves
``(G - 1) * prefix_len * 2 * Dkv`` per layer.

Layout convention (same as the rest of the backend layer):

* ``query``: ``(total_q_padded, Hq, D)`` packed THD;
* ``key`` / ``value``: ``(total_kv_padded, Hkv, D)`` packed THD;
* attention output: ``(total_q_padded, Hq, D)``.

In the FSDP packed remove-padding path ``total_q_padded == total_kv_padded``.
"""

from __future__ import annotations

import inspect
import os
from functools import lru_cache
from typing import Any

import torch

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.flash_atten_base import FlashAttentionMixin, FlashBackendValidationError
from prefix_sharing.backends.kv_gather import get_kv_gather_index
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.backends.prefix_block_mask import get_or_create_block_mask
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan


# ------------------------------------------------------------------------------
# FlexAttention helpers (mirrored from flex_atten_gpu.py on open-source-FL)
# ------------------------------------------------------------------------------

def _flex_attn_env_options() -> dict[str, Any] | None:
    """Parse optional FlexAttention kernel tile overrides from env.

    ``PREFIX_SHARING_FLEX_ATTN_BLOCK_SIZE`` only controls the attention kernel
    tile size via ``kernel_options`` (``BLOCK_M/BLOCK_N`` and the backward
    ``BLOCK_M1/N1/M2/N2``).  The ``BlockMask`` is left at PyTorch's default
    ``BLOCK_SIZE=128`` because Inductor requires the mask block size to be
    divisible by the autotune config's tile sizes.
    """
    block_size_env = os.environ.get("PREFIX_SHARING_FLEX_ATTN_BLOCK_SIZE", "")
    if not block_size_env:
        return None

    try:
        block_m, block_n = block_size_env.split(",")
        block_m_i = int(block_m)
        block_n_i = int(block_n)
    except Exception:
        print(
            f"[ps_flex_fwd_flash_bwd_gpu] ignoring invalid PREFIX_SHARING_FLEX_ATTN_BLOCK_SIZE="
            f"{block_size_env!r}; expected M,N",
            flush=True,
        )
        return None

    num_stages = 2
    try:
        num_stages = int(os.environ.get("PREFIX_SHARING_FLEX_ATTN_NUM_STAGES", "2"))
    except Exception:
        num_stages = 2

    return {
        "BLOCK_M": block_m_i,
        "BLOCK_N": block_n_i,
        "BLOCK_M1": block_m_i,
        "BLOCK_N1": block_n_i,
        "BLOCK_M2": block_m_i,
        "BLOCK_N2": block_m_i,
        "num_stages": num_stages,
    }


@lru_cache(maxsize=None)
def _import_flex_attention() -> Any:
    """Lazy-import flex_attention once per process."""
    try:
        from torch.nn.attention import flex_attention
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PSFlexFwdFlashBwdBackend requires torch.nn.attention.flex_attention. "
            "Upgrade to PyTorch >= 2.5."
        ) from exc
    return flex_attention


@lru_cache(maxsize=None)
def _import_flash_attn_varlen_backward() -> Any:
    """Lazy-import Flash Attention varlen backward internal API."""
    try:
        from flash_attn.flash_attn_interface import _flash_attn_varlen_backward
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PSFlexFwdFlashBwdBackend requires flash-attn with _flash_attn_varlen_backward. "
            "Install flash-attention >= 2.5."
        ) from exc
    return _flash_attn_varlen_backward


_COMPILED_FLEX_ATTENTION: Any | None = None


def _get_compiled_flex_attention(flex_module: Any) -> Any:
    """Return a process-wide compiled flex_attention callable."""
    global _COMPILED_FLEX_ATTENTION
    if _COMPILED_FLEX_ATTENTION is None:
        _COMPILED_FLEX_ATTENTION = torch.compile(
            flex_module.flex_attention, dynamic=True
        )
    return _COMPILED_FLEX_ATTENTION


# ------------------------------------------------------------------------------
# LSE layout normalization
# ------------------------------------------------------------------------------

def _normalize_lse_for_flash(lse: torch.Tensor, total_q: int) -> torch.Tensor:
    """Convert flex_attention ``return_lse`` layout to flash varlen backward layout.

    flex_attention returns lse with shape ``[B, H, M]`` where ``M`` may be
    rounded up to the kernel block size.  Flash varlen backward expects
    ``[H, total_q]``.
    """
    if lse.dim() == 3:
        if lse.shape[0] != 1:
            raise ValueError(
                f"PSFlexFwdFlashBwdFunction expects batch dim 1 for packed input, "
                f"got lse shape {tuple(lse.shape)}"
            )
        lse = lse.squeeze(0)
    if lse.dim() != 2:
        raise ValueError(
            f"PSFlexFwdFlashBwdFunction expects lse of rank 2 or 3, "
            f"got shape {tuple(lse.shape)}"
        )
    if lse.shape[-1] < total_q:
        raise ValueError(
            f"flex_attention lse last dim ({lse.shape[-1]}) is smaller than total_q ({total_q})"
        )
    if lse.shape[-1] > total_q:
        lse = lse[..., :total_q]
    return lse


# ------------------------------------------------------------------------------
# Custom autograd Function: packed KV retention + Flash backward
# ------------------------------------------------------------------------------

class PSFlexFwdFlashBwdFunction(torch.autograd.Function):
    """FlexAttention forward, FlashAttention backward, packed K/V retention."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        packed_k: torch.Tensor,
        packed_v: torch.Tensor,
        index: torch.Tensor,
        block_mask: Any,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_q: int,
        max_kv: int,
        fa_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        # Forward is performed under no_grad so that flex_attention does not
        # create its own autograd subgraph; this Function is the only node.
        with torch.no_grad():
            q4 = q.permute(1, 0, 2).unsqueeze(0)
            k4 = packed_k.permute(1, 0, 2).unsqueeze(0)
            v4 = packed_v.permute(1, 0, 2).unsqueeze(0)

            flex_module = _import_flex_attention()
            compiled_flex_attention = _get_compiled_flex_attention(flex_module)

            out4, lse = compiled_flex_attention(
                q4,
                k4,
                v4,
                block_mask=block_mask,
                enable_gqa=True,
                scale=fa_kwargs.get("softmax_scale"),
                kernel_options=_flex_attn_env_options(),
                return_lse=True,
            )
            out = out4.squeeze(0).permute(1, 0, 2)

        # Save ONLY packed tensors.  expanded K/V are not retained.
        ctx.save_for_backward(
            q, packed_k, packed_v, out, lse, index, cu_seqlens_q, cu_seqlens_kv
        )
        ctx.fa_meta = (max_q, max_kv, fa_kwargs)
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Any, ...]:
        (
            q,
            packed_k,
            packed_v,
            out,
            lse,
            index,
            cu_seqlens_q,
            cu_seqlens_kv,
        ) = ctx.saved_tensors
        max_q, max_kv, fa_kwargs = ctx.fa_meta

        # 1) Recompute the expanded KV layout from the packed tensors.
        expanded_k = packed_k.index_select(0, index)
        expanded_v = packed_v.index_select(0, index)

        # 2) Normalize LSE layout for Flash varlen backward.
        total_q = q.shape[0]
        lse = _normalize_lse_for_flash(lse, total_q)

        # Flash kernels like contiguous inputs.
        grad_out = grad_out.contiguous()
        q = q.contiguous()
        expanded_k = expanded_k.contiguous()
        expanded_v = expanded_v.contiguous()
        out = out.contiguous()

        _flash_attn_varlen_backward = _import_flash_attn_varlen_backward()

        # Inspect signature and pass optional kwargs by name so we stay
        # compatible across flash-attn 2.5.x/2.6.x/2.7.x.
        bwd_sig = inspect.signature(_flash_attn_varlen_backward)
        bwd_params = set(bwd_sig.parameters.keys())

        bwd_kwargs: dict[str, Any] = {
            "dropout_p": fa_kwargs.get("dropout_p", 0.0),
            "softmax_scale": fa_kwargs.get("softmax_scale", None),
            "causal": fa_kwargs.get("causal", True),
            "window_size": fa_kwargs.get("window_size", (-1, -1)),
            "softcap": fa_kwargs.get("softcap", 0.0),
            "alibi_slopes": fa_kwargs.get("alibi_slopes", None),
            "deterministic": fa_kwargs.get("deterministic", False),
        }
        if "rng_state" in bwd_params:
            bwd_kwargs["rng_state"] = None
        if "gen_bias_batch_group" in bwd_params:
            bwd_kwargs["gen_bias_batch_group"] = None

        dq, dk_exp, dv_exp = _flash_attn_varlen_backward(
            grad_out,
            q,
            expanded_k,
            expanded_v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_q,
            max_kv,
            **bwd_kwargs,
        )

        # 3) Scatter expanded gradients back to the packed layout.
        dk = torch.zeros_like(packed_k).index_add_(0, index, dk_exp)
        dv = torch.zeros_like(packed_v).index_add_(0, index, dv_exp)

        return (
            dq,
            dk,
            dv,
            None,  # index
            None,  # block_mask
            None,  # cu_seqlens_q
            None,  # cu_seqlens_kv
            None,  # max_q
            None,  # max_kv
            None,  # fa_kwargs
        )


# ------------------------------------------------------------------------------
# Backend class
# ------------------------------------------------------------------------------

class PSFlexFwdFlashBwdBackend(FlashAttentionMixin):
    """CUDA/GPU hybrid backend: FlexAttention forward + FlashAttention backward.

    ``apply_rope`` is delegated to :class:`TorchReferenceBackend`.

    ``build_kv`` is not used because ``requires_kv_expansion=False``: the
    backend consumes the unexpanded packed K/V directly.
    """

    capabilities = BackendCapabilities(
        name="ps_flex_fwd_flash_bwd_gpu",
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

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, config: PrefixSharingConfig, model_config: Any | None = None) -> None:
        config.validate(model_config=model_config)

        flex_module = _import_flex_attention()
        flex_sig = inspect.signature(flex_module.flex_attention)
        for required_param in ("enable_gqa", "return_lse"):
            if required_param not in flex_sig.parameters:
                raise RuntimeError(
                    f"PSFlexFwdFlashBwdBackend requires flex_attention({required_param}=...). "
                    "Upgrade to PyTorch >= 2.6."
                )

        # Fail fast if flash-attn backward is missing or has an unexpected shape.
        _import_flash_attn_varlen_backward()

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
        raise RuntimeError(
            "PSFlexFwdFlashBwdBackend does not use build_kv; "
            "integrations must pass unexpanded packed K/V directly when "
            "requires_kv_expansion=False."
        )

    # ------------------------------------------------------------------
    # Attention: hybrid Function
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
        q = self._ensure_3d_thd(query, "query")
        packed_k = self._ensure_3d_thd(key, "key")
        packed_v = self._ensure_3d_thd(value, "value")

        # Strip any TP padding defensively.  In the FSDP packed path there is
        # none, but the helper keeps the output shape aligned with cu_seqlens.
        q, pad_layout = self._strip_tp_padding(q, packed_batch_layout)
        packed_k, _ = self._strip_tp_padding(packed_k, packed_batch_layout)
        packed_v, _ = self._strip_tp_padding(packed_v, packed_batch_layout)

        if q.shape[0] != packed_k.shape[0] or q.shape[0] != packed_v.shape[0]:
            raise FlashBackendValidationError(
                "PSFlexFwdFlashBwdBackend expects equal packed lengths for Q/K/V "
                f"(got q_len={q.shape[0]}, k_len={packed_k.shape[0]}, v_len={packed_v.shape[0]})."
            )

        device = q.device
        self._validate_plan_for_flash(prefix_sharing_plan)

        cu_seqlens_q = self._build_cu_seqlens_tensor(
            prefix_sharing_plan.cu_seqlens_q, device=device, dtype=torch.int32
        )
        cu_seqlens_kv = self._build_cu_seqlens_tensor(
            prefix_sharing_plan.cu_seqlens_kv, device=device, dtype=torch.int32
        )

        layout = packed_batch_layout or PackedBatchLayout.from_valid_lengths(
            prefix_sharing_plan.kept_lengths_q
        )
        index = get_kv_gather_index(prefix_sharing_plan, layout, device)

        block_mask = get_or_create_block_mask(
            prefix_sharing_plan,
            device=device,
            cache=self._block_mask_cache,
        )

        fa_kwargs = {
            "dropout_p": kwargs.get("dropout_p", 0.0),
            "softmax_scale": kwargs.get("softmax_scale", None),
            "causal": kwargs.get("causal", True),
            "window_size": kwargs.get("window_size", (-1, -1)),
            "softcap": kwargs.get("softcap", 0.0),
            "alibi_slopes": kwargs.get("alibi_slopes", None),
            "deterministic": kwargs.get("deterministic", False),
        }

        out = PSFlexFwdFlashBwdFunction.apply(
            q,
            packed_k,
            packed_v,
            index,
            block_mask,
            cu_seqlens_q,
            cu_seqlens_kv,
            prefix_sharing_plan.max_seqlen_q,
            prefix_sharing_plan.max_seqlen_kv,
            fa_kwargs,
        )

        if pad_layout is not None:
            out = self._repad_output(out, pad_layout)

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
            "PSFlexFwdFlashBwdBackend does not support gated_attention."
        )
