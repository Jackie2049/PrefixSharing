"""CANN/NPU Flash Attention backend for prefix sharing (s_packed mode).

Uses MindSpeed's ``npu_fusion_attention`` fused kernel in **BSH layout** with a
**s_packed custom causal mask**.

s_packed Mode
-------------
All inputs are packed into a single s_packed sequence with deduplicated KV
storage (shared prefixes are stored once). A global custom causal mask
(built from plan.s_packed_kv_ranges) controls visibility.

Mask semantics: ``atten_mask``: True = masked (not participate), False = visible.
Shape = ``(batch_size, 1, max_q, s_packed_length)``:
  - Each batch's visible KV positions come from plan.s_packed_kv_ranges.
  - Padding rows/cols are left ``True`` so the kernel ignores them.
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
# s_packed mask builder
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
    """Ascend NPU backend via ``npu_fusion_attention`` (BSH, s_packed mode)."""

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
    # attention — s_packed mode via BSHD npu_fusion_attention
    # ------------------------------------------------------------------
    def attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        **kwargs: Any,
    ) -> Any:
        """Run prefix-sharing attention via BSHD ``npu_fusion_attention`` in s_packed mode.

        Q: (total_q, n_heads, d) — per-input suffix tokens, padded per input
        K/V: (s_packed_length, n_kv_heads, d) — s_packed unique KV (去重)
        mask: (batch_size, 1, max_q, s_packed_length) — global custom causal
        """
        layer_id = kwargs.get('layer_id', '?')
        print(
            f"[PS][backend][s_packed] flash_atten_npu attention: "
            f"layer={layer_id}, "
            f"q_shape={tuple(query.shape)}, k_shape={tuple(key.shape)}, "
            f"v_shape={tuple(value.shape)}"
        )

        torch = _torch()
        npu_fusion_attention = _import_npu_fusion_attention()

        q = self._ensure_3d_thd(query, "query")
        k = self._ensure_3d_thd(key, "key")
        v = self._ensure_3d_thd(value, "value")

        packed_layout: PackedBatchLayout = kwargs.get("packed_batch_layout")
        if packed_layout is None:
            raise FlashBackendValidationError(
                "flash_atten_npu.attention requires packed_batch_layout kwarg."
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

        # --- Step 1: split THD → per-sample rows (only Q needs split; K/V are
        #            already the complete s_packed sequence and must NOT be re-split) ---
        q_rows = _split_packed(q, packed_layout.padded_lengths)

        # --- Step 2: pad & stack → BSH ---
        # K/V: each batch row receives the FULL s_packed K/V (no split needed).
        # Reshape THD → (s_packed_length, hidden_kv) first, then expand to BSH.
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
