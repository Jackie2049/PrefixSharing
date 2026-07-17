"""CANN/NPU Flash Attention BSHD backend for prefix sharing.

This is the BSHD counterpart of :class:`NpuFlashAttentionBackend`: it
receives and produces ``[B, S, H, D]`` padded tensors instead of converting
between THD and BSHD internally.  The layout is already ``[B, S, H, D]`` when
it reaches this backend (via ``_prefix_attention_bshd`` in megatron_runtime),
so the THD → split → pad → stack step is unnecessary.

The NPU kernel is called with ``input_layout="BSH"`` (flattened hidden dim)
and **no** ``actual_seq_qlen`` / ``actual_seq_kvlen``, routing through the
non-varlen CANN APIs (``aclnnFlashAttentionScoreV2`` /
``aclnnFlashAttentionScoreGradV2``) which do not have the 128-tile constraint
of the varlen gradient kernels.

Mask semantics (same as ``_build_bshd_attention_mask`` in torch_ref_bshd):
``atten_mask`` is ``[B, 1, S_q, S_kv]`` with ``True`` = masked.

.. note::
   This backend cannot be tested without NPU hardware.  The build_kv
   delegation and factory registration are verified via mock tests on CPU.
   Full on-device verification is required before production use.
   TND (varlen) migration is future work — see design doc §11.2.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any
import importlib

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.batched_layout import BatchedBatchLayout
from prefix_sharing.backends.torch_ref_bshd import (
    TorchReferenceBackendBshd,
    _build_bshd_attention_mask,
    _kept_q_row_mask,
)
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.observability import PrefixSharingStats
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.core.prefix_store import PrefixAttentionStore


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
        "NpuFlashAttentionBackendBshd requires MindSpeed (mindspeed.ops). "
        "Install MindSpeed matching your CANN version."
    ) from last_err


def _torch() -> Any:
    try:
        import torch  # noqa: F401 — lazy import for environments without torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("NpuFlashAttentionBackendBshd requires PyTorch") from exc
    return __import__("torch")


class NpuFlashAttentionBackendBshd:
    """Ascend NPU BSHD backend via ``npu_fusion_attention`` (BSH, single batched call).

    Receives ``[B, S, H, D]`` padded tensors directly (BSHD format from
    ``_prefix_attention_bshd``).

    * build_kv → delegates to :class:`TorchReferenceBackendBshd`
    * attention → ``npu_fusion_attention`` with ``input_layout="BSH"`` and
      a per-sample absolute-coordinate prefix-aware causal mask.
    """

    capabilities = BackendCapabilities(
        name="flash_atten_npu_bshd",
        supports_cpu=False,
        supports_cuda=False,
        supports_cann=True,
        supports_different_q_kv_lengths=True,
        supports_prefix_last_restore=True,
        supports_gated_attention=False,
        supports_deltanet_state_reuse=False,
        supports_bshd=True,
    )

    def __init__(self) -> None:
        self._torch_ref = TorchReferenceBackendBshd()

    def validate(self, config: PrefixSharingConfig, model_config: Any | None = None) -> None:
        config.validate(model_config=model_config)
        _import_npu_fusion_attention()

    # ── RoPE: not supported on NPU BSHD path — handled in megatron_runtime ──

    def apply_rope(
        self,
        query: Any,
        key: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """RoPE is applied in ``_prefix_attention_bshd`` before reaching this backend.
        Raising an error here catches incorrect callers."""
        raise NotImplementedError(
            "NpuFlashAttentionBackendBshd.apply_rope is not used — "
            "RoPE is applied in megatron_runtime._prefix_attention_bshd "
            "via Megatron's native SBHD broadcast path.  If you see this "
            "error, the caller is applying RoPE too late."
        )

    # ── build_kv: delegate directly to TorchReferenceBackendBshd ────────────

    def build_kv(
        self,
        key: Any,
        value: Any,
        store: PrefixAttentionStore,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        layer_id: int,
        tp_rank: int = 0,
        stats: PrefixSharingStats | None = None,
    ) -> tuple[Any, Any]:
        """Delegate KV expansion to :class:`TorchReferenceBackendBshd`.

        Returns expanded K/V in ``[B, max_kv, H, D]`` BSHD format, right-padded
        to the longest expanded KV row.
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

    # ── attention: BSHD → BSH → npu_fusion_attention → BSHD ────────────────

    def attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        attention_mask: Any = None,  # noqa: ARG002 — unused; mask built from plan
        layer_id: int = 0,
        **kwargs: Any,
    ) -> Any:
        """Run prefix-sharing attention via BSH-mode ``npu_fusion_attention``.

        Input shapes (BSHD padded):
            query: ``[B, S_q, H_q, D]``
            key:   ``[B, S_kv, H_kv, D]``  (build_kv output, right-padded)
            value: ``[B, S_kv, H_kv, D]``

        Returns ``[B, S_q, H_q, D]``; non-kept Q rows are exactly zero.
        """
        torch = _torch()
        npu_fusion_attention = _import_npu_fusion_attention()

        layout: BatchedBatchLayout | None = (
            packed_batch_layout if isinstance(packed_batch_layout, BatchedBatchLayout) else None
        )
        valid_lengths = (
            layout.valid_lengths if layout is not None
            else list(prefix_sharing_plan.original_lengths)
        )
        expanded_kv_lengths = prefix_sharing_plan.expanded_lengths_kv

        B, S_q, H_q, D = query.shape
        _, S_kv, H_kv, _ = key.shape
        device = query.device
        dtype = query.dtype

        # ── build per-sample prefix-aware mask ──
        atten_mask = _build_bshd_attention_mask(
            plan=prefix_sharing_plan,
            valid_lengths=valid_lengths,
            expanded_kv_lengths=expanded_kv_lengths,
            max_q=S_q,
            max_kv=S_kv,
            batch_size=B,
            device=device,
        )  # [B, 1, S_q, S_kv], True=masked

        # ── reshape BSHD → BSH (flatten heads*dim) ──
        q_bsh = query.reshape(B, S_q, H_q * D)
        k_bsh = key.reshape(B, S_kv, H_kv * D)
        v_bsh = value.reshape(B, S_kv, H_kv * D)

        scale = kwargs.get("softmax_scale") or (1.0 / math.sqrt(D))
        dropout_p = kwargs.get("dropout_p", 0.0)
        keep_prob = kwargs.get("keep_prob", 1.0 - dropout_p)

        try:
            result = npu_fusion_attention(
                q_bsh, k_bsh, v_bsh,
                H_q,
                "BSH",
                atten_mask=atten_mask,
                scale=scale,
                keep_prob=keep_prob,
                sparse_mode=1,
                num_key_value_heads=H_kv,
            )
        except Exception as exc:
            raise RuntimeError(
                f"NpuFlashAttentionBackendBshd: npu_fusion_attention failed: "
                f"q={tuple(q_bsh.shape)}, k={tuple(k_bsh.shape)}, "
                f"v={tuple(v_bsh.shape)}, mask={tuple(atten_mask.shape)}, "
                f"layer={layer_id}"
            ) from exc

        output_bsh = result[0] if isinstance(result, (tuple, list)) else result
        # output_bsh: [B, S_q, H_q * D]

        # ── reshape back to BSHD ──
        out = output_bsh.reshape(B, S_q, H_q, D)

        # ── zero non-kept Q rows (consistent with TorchReferenceBackendBshd) ──
        keep = _kept_q_row_mask(prefix_sharing_plan, valid_lengths, S_q, device)
        return out * keep.unsqueeze(-1).unsqueeze(-1).to(dtype)
