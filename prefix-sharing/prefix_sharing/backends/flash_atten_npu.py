"""CANN/NPU Flash Attention backend for prefix sharing.

Uses MindSpeed's ``npu_fusion_attention`` fused kernel in **BSH layout**
with **per-sample padded tensors** and a batched 4-D mask.

Why BSH instead of TND (varlen)
-------------------------------
The TND / varlen forward kernel ``aclnnFlashAttentionVarLenScoreV2``
accepts attention masks with irregular dimensions, but the corresponding
gradient kernel ``aclnnFlashAttentionUnpaddingScoreGradV2`` requires the
mask dimensions to be multiples of the tile-block size (128).  There is
no ``aclnnFlashAttentionVarLenScoreGrad`` in CANN 8.5.0, and the
UnpaddingScoreGrad V2/V3/V4/V5 all route through the same
``s1s2_bn2gs1s2_sab`` tiling path which enforces this constraint.

By converting the per-sample THD tokens into padded BSHD tensors and
omitting ``actual_seq_qlen`` / ``actual_seq_kvlen``, the kernel dispatches
to the non-varlen pair:

  Forward:  ``aclnnFlashAttentionScoreV2``  (no 128 constraint)
  Backward: ``aclnnFlashAttentionScoreGradV2`` (no 128 constraint)

The small cost of split / pad / stack is negligible compared to the NPU
fused attention kernel time.

Mask semantics
--------------
``atten_mask``: True = masked (not participate), False = visible.
Shape = ``(batch_size, 1, max_q, max_kv)`` — per-sample, prefix-aware:

  - **Provider** → standard causal (upper-tri True).
  - **Reuser**   → prefix KV columns all-visible, suffix KV columns causal.
  - Padding rows / cols (past valid lengths) are left ``True`` so the
    kernel sees them as invisible.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, List
import importlib

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.flash_atten_base import (
    FlashAttentionMixin,
    FlashBackendValidationError,
)
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan


_CANDIDATES = [
    ("mindspeed.ops.fusion_attention_v2", "npu_fusion_attention"),
    ("mindspeed.ops", "npu_fusion_attention"),
]

@lru_cache(maxsize=None)
def _import_npu_fusion_attention():
    last_err = None
    for module_name, attr in _CANDIDATES:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr)
        except ImportError as e:
            last_err = e
    raise RuntimeError(
        "NpuFlashAttentionBackend requires MindSpeed (mindspeed.ops). "
        "Install MindSpeed matching your CANN version."
    ) from last_err


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("NpuFlashAttentionBackend requires PyTorch") from exc
    return torch


# ---------------------------------------------------------------------------
# Per-sample pad-mask builder
# ---------------------------------------------------------------------------

def _build_per_sample_mask(
    plan: PrefixSharingPlan,
    valid_lens: List[int],
    expanded_kv_lens: List[int],
    max_q: int,
    max_kv: int,
    device: Any,
) -> Any:
    """Build ``(batch_size, 1, max_q, max_kv)`` mask for the full batch.

    Each sample *i* occupies rows ``[0, valid_lens[i])`` and columns
    ``[0, expanded_kv_lens[i])`` within its own ``[max_q, max_kv]``
    per-sample sub-tensor.  Padding rows/cols outside valid ranges stay
    ``True`` (hidden).
    """
    torch = _torch()
    batch_size = plan.batch_size
    mask = torch.ones(batch_size, 1, max_q, max_kv, dtype=torch.bool, device=device)

    for i in range(batch_size):
        q_val = valid_lens[i]
        kv_val = expanded_kv_lens[i]
        if q_val == 0 or kv_val == 0:
            continue

        if plan.is_reuser(i):
            prefix_len = int(plan.prefix_lens[i])
            # Prefix KV columns: all Q tokens see all prefix KV tokens.
            if prefix_len > 0:
                mask[i, 0, :q_val, :prefix_len] = False

            # Suffix KV columns: causal.
            suffix_len = kv_val - prefix_len
            if suffix_len > 0:
                suffix_block = torch.ones(q_val, suffix_len, dtype=torch.bool, device=device)
                # tril(0) → lower-tri visible → ~ → upper masked
                mask[i, 0, :q_val, prefix_len:prefix_len + suffix_len] = \
                    ~suffix_block.tril(diagonal=0)
        else:
            # Provider: standard causal within [q_val, kv_val].
            block = torch.ones(q_val, kv_val, dtype=torch.bool, device=device)
            if q_val <= kv_val:
                mask[i, 0, :q_val, :kv_val] = torch.triu(block, diagonal=1)
            else:
                # Tall block: shift causal diagonal by (q_val - kv_val) rows.
                mask[i, 0, :q_val, :kv_val] = torch.triu(
                    block, diagonal=q_val - kv_val + 1,
                )

    return mask


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class NpuFlashAttentionBackend(FlashAttentionMixin):
    """Ascend NPU backend via ``npu_fusion_attention`` (BSH, single batched call)."""

    capabilities = BackendCapabilities(
        name="flash_atten_npu",
        supports_cpu=False,
        supports_cuda=False,
        supports_cann=True,
        supports_different_q_kv_lengths=True,
        supports_prefix_last_restore=True,
    )

    def __init__(self) -> None:
        self._torch_ref = TorchReferenceBackend()

    def validate(self, config: PrefixSharingConfig, model_config: Any | None = None) -> None:
        config.validate(model_config=model_config)
        _import_npu_fusion_attention()

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
        """Delegate KV expansion to the torch reference backend.

        Returns per-sample expanded K/V concatenated in THD order.  The
        caller (megatron_runtime) still passes the THD-concatenated result
        to ``attention()``, where we split it back into per-sample rows for
        the BSHD conversion.
        """
        return self._torch_ref.build_kv(
            key,
            value,
            store,
            prefix_sharing_plan,
            packed_batch_layout=packed_batch_layout,
            layer_id=layer_id,
            tp_rank=tp_rank,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # attention — THD → BSHD → npu_fusion_attention → THD
    # ------------------------------------------------------------------
    def attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        **kwargs: Any,
    ) -> Any:
        """Run prefix-sharing attention via BSH-mode ``npu_fusion_attention``.

        1. Split incoming THD Q/K/V into per-sample rows.
        2. Pad each row to ``(max_q, ...)`` / ``(max_kv, ...)`` and stack
           into BSHD layout.
        3. Build per-sample ``(batch_size, 1, max_q, max_kv)`` prefix-aware causal mask.
        4. Invoke ``npu_fusion_attention`` once with ``input_layout="BSH"``
           and **no** ``actual_seq_qlen`` / ``actual_seq_kvlen`` so that
           both forward and backward route through the non-varlen CANN APIs.
        5. Unpack the BSHD output back to THD.
        """
        layer_id = kwargs.get('layer_id', '?')
        print(
            f"[PS][backend] flash_atten_npu attention: "
            f"layer={layer_id}, "
            f"q_shape={tuple(query.shape)}, k_shape={tuple(key.shape)}, "
            f"v_shape={tuple(value.shape)}"
        )

        torch = _torch()
        npu_fusion_attention = _import_npu_fusion_attention()

        q = self._ensure_3d_thd(query, "query")        # [T_q, n_heads, d]
        k = self._ensure_3d_thd(key, "key")              # [T_kv, n_kv_heads, d]
        v = self._ensure_3d_thd(value, "value")          # [T_kv, n_kv_heads, d]

        packed_layout: PackedBatchLayout = kwargs.get("packed_batch_layout")
        if packed_layout is None:
            raise FlashBackendValidationError(
                "flash_atten_npu.attention requires packed_batch_layout kwarg."
            )

        plan = prefix_sharing_plan
        batch_size = plan.batch_size

        # --- metadata ---
        q_cus = packed_layout.cu_seqlens                      # cumulative padded
        kv_cus = plan.cu_seqlens_kv                            # cumulative expanded
        valid_lens = packed_layout.valid_lengths               # per-sample valid Q
        kv_lens = plan.expanded_lengths_kv                     # per-sample expanded KV

        total_q = q_cus[-1]
        total_kv = kv_cus[-1]

        if q.shape[0] != total_q:
            raise FlashBackendValidationError(
                f"q.shape[0]={q.shape[0]} != total_q={total_q}"
            )
        if k.shape[0] != total_kv:
            raise FlashBackendValidationError(
                f"k.shape[0]={k.shape[0]} != total_kv={total_kv}"
            )

        if total_q == 0 or total_kv == 0:
            return torch.zeros_like(q)

        num_q_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        head_dim = q.shape[-1]
        hidden_q = num_q_heads * head_dim
        hidden_kv = num_kv_heads * head_dim

        # --- Step 1: split THD → per-sample rows ---
        q_rows = _split_packed(q, packed_layout.padded_lengths)
        k_rows = _split_packed(k, kv_lens)
        v_rows = _split_packed(v, kv_lens)

        max_q = max(valid_lens)
        max_kv = max(kv_lens)

        # --- Step 2: pad & stack → BSH ---
        # Q: (T_q, nq, d) → split → pad → (batch_size, max_q, nq*d)
        # K: (T_kv, nkv, d) → split → pad → (batch_size, max_kv, nkv*d)
        q_bsh = torch.zeros(batch_size, max_q, hidden_q, dtype=q.dtype, device=q.device)
        k_bsh = torch.zeros(batch_size, max_kv, hidden_kv, dtype=k.dtype, device=k.device)
        v_bsh = torch.zeros(batch_size, max_kv, hidden_kv, dtype=v.dtype, device=v.device)

        for i in range(batch_size):
            if valid_lens[i] > 0:
                q_bsh[i, :valid_lens[i], :] = \
                    q_rows[i][:valid_lens[i]].reshape(valid_lens[i], hidden_q)
            if kv_lens[i] > 0:
                k_bsh[i, :kv_lens[i], :] = \
                    k_rows[i].reshape(kv_lens[i], hidden_kv)
                v_bsh[i, :kv_lens[i], :] = \
                    v_rows[i].reshape(kv_lens[i], hidden_kv)

        # --- Step 3: build per-sample mask ---
        atten_mask = _build_per_sample_mask(
            plan, valid_lens, kv_lens, max_q, max_kv, q.device,
        )

        # --- Step 4: invoke npu_fusion_attention (BSH, non-varlen) ---
        scale = kwargs.get("softmax_scale") or (1.0 / math.sqrt(head_dim))
        dropout_p = kwargs.get("dropout_p", 0.0)
        keep_prob = kwargs.get("keep_prob", 1.0 - dropout_p)

        try:
            result = npu_fusion_attention(
                q_bsh, k_bsh, v_bsh,
                num_q_heads,
                "BSH",
                atten_mask=atten_mask,
                scale=scale,
                keep_prob=keep_prob,
                sparse_mode=1,
            )
        except Exception as exc:
            raise FlashBackendValidationError(
                f"npu_fusion_attention (BSH) failed: q={tuple(q_bsh.shape)}, "
                f"k={tuple(k_bsh.shape)}, v={tuple(v_bsh.shape)}, "
                f"mask={tuple(atten_mask.shape)}, batch_size={batch_size}, "
                f"max_q={max_q}, max_kv={max_kv}"
            ) from exc

        output_bsh = result[0] if isinstance(result, (tuple, list)) else result
        # output_bsh: (batch_size, max_q, nq*d)

        # --- Step 5: unpack BSHD → THD ---
        output_thd = torch.zeros(total_q, num_q_heads, head_dim,
                                 dtype=q.dtype, device=q.device)
        for i in range(batch_size):
            vlen = valid_lens[i]
            if vlen == 0:
                continue
            q_lo = q_cus[i]
            q_hi = q_lo + vlen
            output_thd[q_lo:q_hi] = output_bsh[i, :vlen, :].reshape(vlen, num_q_heads, head_dim)

        return output_thd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_packed(tensor: Any, lengths: List[int]) -> List[Any]:
    """Split a packed tensor along dim 0 by the given *lengths*."""
    rows: List[Any] = []
    offset = 0
    for length in lengths:
        rows.append(tensor[offset:offset + length])
        offset += length
    return rows
