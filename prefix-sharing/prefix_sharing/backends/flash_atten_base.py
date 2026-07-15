"""Flash Attention common abstraction layer.

This module provides shared input-normalization logic for GPU and NPU Flash
Attention backends.  It converts the packed THD tensors produced by the
Megatron hook into the shape/dtype/device layout expected by Flash Attention
kernels, while keeping hardware-specific dispatch in the concrete backends.
"""

from __future__ import annotations

from typing import Any

import torch

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.planner import PrefixSharingPlan


class FlashBackendValidationError(RuntimeError):
    """Raised when a flash-attention backend cannot run with the given plan."""


class FlashAttentionMixin:
    """Shared helpers for GPU / NPU Flash Attention backends.

    Concrete subclasses must still define ``capabilities``, ``validate()``,
    ``apply_rope()``, ``build_kv()`` and ``attention()``; this mixin only
    normalises the tensor shapes for the ``attention()`` step.
    """

    capabilities: BackendCapabilities

    # ------------------------------------------------------------------
    # Shared validation
    # ------------------------------------------------------------------
    def _validate_plan_for_flash(self, prefix_sharing_plan: PrefixSharingPlan) -> None:
        """Common checks before calling any Flash Attention kernel."""
        if prefix_sharing_plan.cu_seqlens_q is None or prefix_sharing_plan.cu_seqlens_kv is None:
            raise FlashBackendValidationError(
                "Flash Attention requires cu_seqlens_q and cu_seqlens_kv; "
                "got None. Make sure PrefixSharingPlan was built with packed THD metadata."
            )
        if len(prefix_sharing_plan.cu_seqlens_q) != prefix_sharing_plan.batch_size + 1:
            raise FlashBackendValidationError(
                f"cu_seqlens_q length ({len(prefix_sharing_plan.cu_seqlens_q)}) must be "
                f"batch_size + 1 ({prefix_sharing_plan.batch_size + 1})"
            )
        if len(prefix_sharing_plan.cu_seqlens_kv) != prefix_sharing_plan.batch_size + 1:
            raise FlashBackendValidationError(
                f"cu_seqlens_kv length ({len(prefix_sharing_plan.cu_seqlens_kv)}) must be "
                f"batch_size + 1 ({prefix_sharing_plan.batch_size + 1})"
            )

    # ------------------------------------------------------------------
    # Tensor normalisation helpers
    # ------------------------------------------------------------------
    def _ensure_3d_thd(self, tensor: Any, name: str) -> Any:
        """Ensure packed tensor is (total_tokens, num_heads, head_dim).

        The Megatron hook squeezes the dummy batch dimension before handing
        tensors to the backend, so we expect 3-D inputs.  If 2-D is ever
        received we raise loudily rather than guessing.
        """
        if tensor.dim() == 3:
            return tensor
        if tensor.dim() == 2:
            raise FlashBackendValidationError(
                f"{name} has 2 dims {tuple(tensor.shape)}; Flash Attention expects "
                "(total_tokens, num_heads, head_dim)."
            )
        raise FlashBackendValidationError(
            f"{name} has unexpected rank {tensor.dim()} with shape {tuple(tensor.shape)}"
        )

    def _build_cu_seqlens_tensor(self, lengths: list[int], device: Any, dtype: Any = None) -> Any:
        """Convert a Python list of cumulative lengths to a CUDA/NPU tensor."""
        t = torch.tensor(lengths, device=device)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return t

    def _strip_tp_padding(
        self, q: Any, packed_batch_layout: Any | None
    ) -> tuple[Any, PackedBatchLayout | None]:
        """Strip TP padding from Q and validate the tensor shape.

        Returns ``(q_unpadded, layout)`` where *layout* is the
        :class:`PackedBatchLayout` when padding was stripped, or ``None``
        when no stripping was needed.  The caller must pass *layout* to
        :meth:`_repad_output` after the FA kernel.
        """
        if not isinstance(packed_batch_layout, PackedBatchLayout):
            return q, None
        if not packed_batch_layout.has_padding:
            return q, None

        # Defensive: ensure the padded Q tensor shape matches the layout.
        expected_len = packed_batch_layout.total_padded_length
        if q.shape[0] != expected_len:
            raise FlashBackendValidationError(
                f"Query total_tokens={q.shape[0]} does not match "
                f"packed_batch_layout total_padded_length={expected_len} "
                f"(padded_lengths={packed_batch_layout.padded_lengths})"
            )

        q_unpadded = packed_batch_layout.unpad(q)
        return q_unpadded, packed_batch_layout

    def _prepare_flash_inputs(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        attention_mask: Any | None = None,
        packed_batch_layout: Any | None = None,
    ) -> tuple[Any, Any, Any, Any, Any, int, int, PackedBatchLayout | None]:
        """Normalise inputs for ``flash_attn_varlen_func``.

        When *packed_batch_layout* is provided and contains TP-induced
        padding (``has_padding`` is True, i.e. TP > 1), the padding
        tokens are **stripped from Q** before the FA call so that the
        kernel uses the semantic (unpadded) ``cu_seqlens`` where the
        causal alignment is correct.  The padding is re-applied to the
        output by the caller via :meth:`_repad_output`.

        K/V tensors are already de-padded from :meth:`build_kv` and
        follow ``prefix_sharing_plan.expanded_lengths_kv``.

        Returns
        -------
        q, k, v : (total_tokens, num_heads, head_dim)  — Q is de-padded when layout has padding
        cu_seqlens_q : (batch_size + 1,)
        cu_seqlens_kv : (batch_size + 1,)
        max_seqlen_q : int
        max_seqlen_kv : int
        pad_layout : PackedBatchLayout | None
            When not None the caller must re-pad the FA output via
            :meth:`_repad_output`.
        """
        self._validate_plan_for_flash(prefix_sharing_plan)

        q = self._ensure_3d_thd(query, "query")
        k = self._ensure_3d_thd(key, "key")
        v = self._ensure_3d_thd(value, "value")

        # ---- Detect & strip TP padding ----
        q, pad_layout = self._strip_tp_padding(q, packed_batch_layout)

        device = q.device

        # ---- Q cu_seqlens & max_seqlen ----
        # After de-padding, Q follows the plan's semantic kept_lengths_q.
        cu_seqlens_q = self._build_cu_seqlens_tensor(
            prefix_sharing_plan.cu_seqlens_q, device=device, dtype=torch.int32
        )
        max_seqlen_q = prefix_sharing_plan.max_seqlen_q

        # ---- KV cu_seqlens & max_seqlen ----
        # K/V have already been de-padded and prefix-expanded by build_kv();
        # they follow the plan's semantic expanded_lengths_kv without any TP
        # alignment padding.
        cu_seqlens_kv = self._build_cu_seqlens_tensor(
            prefix_sharing_plan.cu_seqlens_kv, device=device, dtype=torch.int32
        )
        max_seqlen_kv = prefix_sharing_plan.max_seqlen_kv

        # Flash Attention ignores dense attention_mask; if one is passed we
        # simply drop it because cu_seqlens already encodes the per-sample
        # boundaries.  This is consistent with Megatron's THD path.
        _ = attention_mask

        return q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, pad_layout

    def _repad_output(
        self,
        output: Any,
        pad_layout: PackedBatchLayout,
    ) -> Any:
        """Re-apply TP padding to the FA output so it matches the original Q shape.

        Delegates to :meth:`PackedBatchLayout.repad`.
        """
        return pad_layout.repad(output)
