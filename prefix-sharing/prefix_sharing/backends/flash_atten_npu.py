"""CANN/NPU Flash Attention backend for prefix sharing (s_packed mode).

Uses MindSpeed's ``npu_fusion_attention`` fused kernel with a **s_packed custom
atten_mask** (sparse_mode=1).

Two layout paths are supported:

TND single-sample path (default, recommended)
---------------------------------------------
All samples are concatenated into a single "super-sample" in TND format.
The planner's global custom mask ``(total_q, s_packed_length)`` in SS format
is used directly as ``atten_mask``.  This eliminates Q padding, K/V batch
expansion, and the quadratic ``(B, max_q, s_packed_length)`` mask.

Memory reduction vs BSH: ~4-8x (eliminates batch dim and padding overhead).

BSH fallback path
-----------------
Each sample is padded to ``max_q`` and stacked into ``(B, max_q, H)`` with
K/V expanded along the batch dim.  The BSHD mask
``(batch_size, 1, max_q, s_packed_length)`` is built per-sample.  Use this
path if TND + sparse_mode=1 is not supported on a particular CANN version.

Mask semantics
--------------
- TND path: ``atten_mask`` shape ``(total_q, s_packed_length)``,
  True = masked (not participate), False = visible.
- BSH path: ``atten_mask`` shape ``(batch_size, 1, max_q, s_packed_length)``,
  True = masked (not participate), False = visible.
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
# s_packed mask builder (BSHD)
# ---------------------------------------------------------------------------

def _build_s_packed_mask(
    plan: PrefixSharingPlan,
    valid_lens: list[int],
    max_q: int,
    s_packed_length: int,
    device: Any,
) -> Any:
    """Build BSHD mask from s_packed plan with causal constraints.

    Returns mask of shape (batch_size, 1, max_q, s_packed_length) where:
    - True = masked (invisible), False = visible  (matches npu_fusion_attention atten_mask convention)
    - Each batch's visible KV positions come from plan.s_packed_kv_ranges,
      with causal masking applied within each block.
    """
    torch = _torch()
    batch_size = plan.batch_size
    # True = masked (invisible) by default, set False for visible positions.
    mask = torch.ones(batch_size, 1, max_q, s_packed_length, dtype=torch.bool, device=device)

    for i in range(batch_size):
        q_len = valid_lens[i]
        if q_len == 0:
            continue

        prefix_len = plan.prefix_lens[i]
        prefix_end = plan.s_packed_prefix_end[i]

        for qi in range(q_len):
            q_s_packed = plan.s_packed_q_starts[i] + qi
            q_original_pos = prefix_len + qi

            for kv_lo, kv_hi in plan.s_packed_kv_ranges[i]:
                if kv_hi <= prefix_end:
                    # Prefix block: causal boundary is q_original_pos.
                    visible_hi = min(kv_hi, q_original_pos)
                else:
                    # Suffix block: causal boundary is q_original_pos + prefix_len.
                    visible_hi = min(kv_hi, q_original_pos + prefix_len + 1)

                if kv_lo < visible_hi:
                    # Set visible positions to False (unmasked).
                    mask[i, 0, qi, kv_lo:visible_hi] = False

    return mask


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class NpuFlashAttentionBackend(FlashAttentionMixin):
    """Ascend NPU backend via ``npu_fusion_attention`` (s_packed mode).

    Supports two layout paths:
    - **TND** (default): single super-sample, SS mask, no padding/expansion.
    - **BSH** (fallback): per-sample padded batch, BSHD mask.
    """

    capabilities = BackendCapabilities(
        name="flash_atten_npu",
        supports_cpu=False,
        supports_cuda=False,
        supports_cann=True,
        supports_different_q_kv_lengths=True,
        supports_prefix_last_restore=True,
        supports_gated_attention=False,
        supports_deltanet_state_reuse=False,
    )

    def __init__(self, *, use_tnd: bool = True) -> None:
        """
        Args:
            use_tnd: If True (default), use the TND single-sample path which
                eliminates Q padding, K/V batch expansion, and the quadratic
                BSHD mask.  Falls back to BSH if set to False.
        """
        self._torch_ref = TorchReferenceBackend()
        self._use_tnd = use_tnd

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
    # attention — dispatch TND or BSH path
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
        """Run prefix-sharing attention via ``npu_fusion_attention`` in s_packed mode.

        Dispatches to TND (default) or BSH path based on ``self._use_tnd``.

        Q: (total_q, n_heads, d) — THD format
        K/V: (s_packed_length, n_kv_heads, d) — THD format, deduplicated
        """
        if self._use_tnd:
            return self._attention_tnd(query, key, value, prefix_sharing_plan, **kwargs)
        else:
            return self._attention_bsh(
                query, key, value, prefix_sharing_plan,
                packed_batch_layout=packed_batch_layout, **kwargs,
            )

    # ------------------------------------------------------------------
    # TND path — single super-sample, SS mask (recommended)
    # ------------------------------------------------------------------
    def _attention_tnd(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        **kwargs: Any,
    ) -> Any:
        """TND single-sample path: all Q tokens as one sequence, SS custom mask.

        Eliminates:
        - Q padding to max_q (saves memory + compute)
        - K/V batch expansion (avoids potential materialization)
        - Quadratic (B, max_q, s_packed_length) mask

        The planner's global custom mask ``(total_q, s_packed_length)`` with
        ``True = visible`` is inverted to ``True = masked`` and passed directly
        as the SS-format ``atten_mask`` to the TND kernel.
        """
        layer_id = kwargs.get('layer_id', '?')
        print(
            f"[PS][backend][s_packed] flash_atten_npu TND attention: "
            f"layer={layer_id}, "
            f"q_shape={tuple(query.shape)}, k_shape={tuple(key.shape)}, "
            f"v_shape={tuple(value.shape)}"
        )

        torch = _torch()
        npu_fusion_attention = _import_npu_fusion_attention()

        q = self._ensure_3d_thd(query, "query")
        k = self._ensure_3d_thd(key, "key")
        v = self._ensure_3d_thd(value, "value")

        plan = prefix_sharing_plan
        total_q = sum(plan.s_packed_q_lengths)
        s_packed_length = plan.s_packed_length

        if q.shape[0] != total_q:
            raise FlashBackendValidationError(
                f"q.shape[0]={q.shape[0]} != total_q={total_q}"
            )
        if k.shape[0] != s_packed_length:
            raise FlashBackendValidationError(
                f"k.shape[0]={k.shape[0]} != s_packed_length={s_packed_length}"
            )

        if total_q == 0 or s_packed_length == 0:
            return torch.zeros_like(q)

        num_q_heads = q.shape[1]
        head_dim = q.shape[-1]

        # --- Mask: reuse planner's cached global custom mask, invert polarity ---
        # planner: (total_q, s_packed_length), True = visible
        # NPU:     (total_q, s_packed_length), True = masked
        global_mask = plan.build_global_custom_mask(q.device)
        atten_mask = ~global_mask

        # --- Invoke TND npu_fusion_attention ---
        scale = kwargs.get("softmax_scale") or (1.0 / math.sqrt(head_dim))
        dropout_p = kwargs.get("dropout_p", 0.0)
        keep_prob = kwargs.get("keep_prob", 1.0 - dropout_p)

        try:
            result = npu_fusion_attention(
                q, k, v,
                num_q_heads,
                "TND",
                atten_mask=atten_mask,
                scale=scale,
                keep_prob=keep_prob,
                sparse_mode=1,
                actual_seq_qlen=[total_q],
                actual_seq_kvlen=[s_packed_length],
            )
        except Exception as exc:
            raise FlashBackendValidationError(
                f"npu_fusion_attention (TND s_packed) failed: "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}, "
                f"mask={tuple(atten_mask.shape)}, "
                f"total_q={total_q}, s_packed_length={s_packed_length}, "
                f"num_q_heads={num_q_heads}"
            ) from exc

        output = result[0] if isinstance(result, (tuple, list)) else result
        return output  # Already (total_q, N_q, D) — no unpacking needed

    # ------------------------------------------------------------------
    # BSH fallback path — per-sample padded batch, BSHD mask
    # ------------------------------------------------------------------
    def _attention_bsh(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """BSH fallback: per-sample padded batch with BSHD mask.

        Use this path if TND + sparse_mode=1 is not supported on a particular
        CANN version.

        Q: (total_q, n_heads, d) → padded to (B, max_q, H)
        K/V: (s_packed_length, n_kv_heads, d) → expanded to (B, s_packed, H)
        mask: (B, 1, max_q, s_packed_length) built from scratch
        """
        layer_id = kwargs.get('layer_id', '?')
        print(
            f"[PS][backend][s_packed] flash_atten_npu BSH attention: "
            f"layer={layer_id}, "
            f"q_shape={tuple(query.shape)}, k_shape={tuple(key.shape)}, "
            f"v_shape={tuple(value.shape)}"
        )

        torch = _torch()
        npu_fusion_attention = _import_npu_fusion_attention()

        q = self._ensure_3d_thd(query, "query")
        k = self._ensure_3d_thd(key, "key")
        v = self._ensure_3d_thd(value, "value")

        packed_layout: PackedBatchLayout = packed_batch_layout
        if packed_layout is None:
            raise FlashBackendValidationError(
                "flash_atten_npu BSH path requires packed_batch_layout kwarg."
            )

        plan = prefix_sharing_plan
        batch_size = plan.batch_size
        s_packed_length = plan.s_packed_length
        valid_lens = packed_layout.valid_lengths
        max_q = max(valid_lens)

        total_q = sum(plan.s_packed_q_lengths)
        if q.shape[0] != total_q:
            raise FlashBackendValidationError(
                f"q.shape[0]={q.shape[0]} != total_q={total_q}"
            )
        if k.shape[0] != s_packed_length:
            raise FlashBackendValidationError(
                f"k.shape[0]={k.shape[0]} != s_packed_length={s_packed_length}"
            )

        if total_q == 0 or s_packed_length == 0:
            return torch.zeros_like(q)

        num_q_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        head_dim = q.shape[-1]
        hidden_q = num_q_heads * head_dim
        hidden_kv = num_kv_heads * head_dim

        # --- Step 1: split THD → per-sample rows ---
        q_rows = _split_packed(q, packed_layout.padded_lengths)

        # --- Step 2: pad & stack → BSH ---
        k_bsh = k.reshape(s_packed_length, hidden_kv).unsqueeze(0).expand(batch_size, -1, -1)
        v_bsh = v.reshape(s_packed_length, hidden_kv).unsqueeze(0).expand(batch_size, -1, -1)

        q_bsh = torch.zeros(batch_size, max_q, hidden_q, dtype=q.dtype, device=q.device)

        for i in range(batch_size):
            if valid_lens[i] > 0:
                q_bsh[i, :valid_lens[i], :] = \
                    q_rows[i][:valid_lens[i]].reshape(valid_lens[i], hidden_q)

        # --- Step 3: build s_packed custom mask ---
        atten_mask = _build_s_packed_mask(
            plan, valid_lens, max_q, s_packed_length, q.device,
        )

        # --- Step 4: invoke npu_fusion_attention (BSH) ---
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
                f"npu_fusion_attention (BSH s_packed) failed: q={tuple(q_bsh.shape)}, "
                f"k={tuple(k_bsh.shape)}, v={tuple(v_bsh.shape)}, "
                f"mask={tuple(atten_mask.shape)}, batch_size={batch_size}, "
                f"max_q={max_q}, s_packed_length={s_packed_length}"
            ) from exc

        output_bsh = result[0] if isinstance(result, (tuple, list)) else result

        # --- Step 5: unpack BSHD → THD ---
        output_thd = torch.zeros(total_q, num_q_heads, head_dim,
                                 dtype=q.dtype, device=q.device)
        q_cus = packed_layout.cu_seqlens
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
