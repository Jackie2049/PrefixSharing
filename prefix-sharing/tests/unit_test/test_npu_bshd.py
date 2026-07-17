"""Tests for NpuFlashAttentionBackendBshd — factory, delegation, mock integration."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from prefix_sharing.backends.batched_layout import BatchedBatchLayout
from prefix_sharing.backends.factory import get_bshd_backend_instance
from prefix_sharing.backends.torch_ref_bshd import TorchReferenceBackendBshd
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner


# ── helpers ────────────────────────────────────────────────────────────────


def _plan():
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    )
    return planner.plan(
        [[1, 2, 10, 11], [1, 2, 20, 21, 22]],
        forward_id=1, micro_batch_id=1,
    )


# ── factory ────────────────────────────────────────────────────────────────


class TestFactory:
    def test_get_bshd_backend_instance_npu(self):
        """Factory creates NpuFlashAttentionBackendBshd when asked."""
        backend = get_bshd_backend_instance(
            PrefixSharingConfig(enable_prefix_sharing=True, batch_format="bshd"),
            backend="flash_atten_npu_bshd",
        )
        from prefix_sharing.backends.flash_atten_npu_bshd import NpuFlashAttentionBackendBshd
        assert isinstance(backend, NpuFlashAttentionBackendBshd)

    def test_bshd_backend_capabilities(self):
        backend = get_bshd_backend_instance(
            PrefixSharingConfig(enable_prefix_sharing=True, batch_format="bshd"),
            backend="flash_atten_npu_bshd",
        )
        caps = backend.capabilities
        assert caps.supports_bshd is True
        assert caps.supports_cann is True
        assert caps.supports_cpu is False
        assert caps.supports_cuda is False

    def test_torch_ref_bshd_via_explicit_name(self):
        """Factory creates TorchReferenceBackendBshd via torch_ref_bshd."""
        backend = get_bshd_backend_instance(
            PrefixSharingConfig(enable_prefix_sharing=True, backend="torch_ref_bshd", batch_format="bshd"),
        )
        assert isinstance(backend, TorchReferenceBackendBshd)


# ── build_kv delegation ────────────────────────────────────────────────────


class TestBuildKvDelegation:
    def test_build_kv_delegates_to_torch_ref(self):
        """build_kv on NPU BSHD backend delegates to TorchReferenceBackendBshd."""
        from prefix_sharing.backends.flash_atten_npu_bshd import NpuFlashAttentionBackendBshd
        from prefix_sharing.core.prefix_store import PrefixAttentionStore

        plan = _plan()
        layout = BatchedBatchLayout.from_valid_lengths(plan.original_lengths)
        B, H, D = 2, 4, 64
        S = layout.seq_length

        backend = NpuFlashAttentionBackendBshd()
        # Patch the internal torch_ref so we can verify delegation.
        mock_ref = MagicMock(wraps=backend._torch_ref)
        backend._torch_ref = mock_ref

        store = PrefixAttentionStore()
        key = torch.randn(B, S, 1, D)  # MQA: 1 kv head
        value = torch.randn(B, S, 1, D)

        expanded_k, expanded_v = backend.build_kv(
            key, value, store, plan,
            packed_batch_layout=layout,
            layer_id=1, tp_rank=0,
        )

        mock_ref.build_kv.assert_called_once()
        assert isinstance(expanded_k, torch.Tensor)
        assert isinstance(expanded_v, torch.Tensor)


# ── attention (mock npu_fusion_attention) ──────────────────────────────────


class TestAttentionMock:
    """Mock-level tests for attention mask construction and kernel invocation.

    These tests do *not* require NPU hardware — they mock the
    ``npu_fusion_attention`` call and verify shapes and arguments.
    """

    @staticmethod
    def _make_backend():
        from prefix_sharing.backends.flash_atten_npu_bshd import NpuFlashAttentionBackendBshd
        return NpuFlashAttentionBackendBshd()

    def _run_attention(self, backend, plan, patch_fn=None):
        """Helper: creates BSHD tensors and calls attention with a mock NPU kernel."""
        layout = BatchedBatchLayout.from_valid_lengths(plan.original_lengths)
        B = len(plan.original_lengths)
        H_q, H_kv, D = 4, 1, 64  # MQA: 4 query heads, 1 kv head
        S_q = max(plan.original_lengths)
        S_kv = max(plan.expanded_lengths_kv)

        query = torch.randn(B, S_q, H_q, D)
        key = torch.randn(B, S_kv, H_kv, D)
        value = torch.randn(B, S_kv, H_kv, D)

        if patch_fn:
            patcher = patch(
                "prefix_sharing.backends.flash_atten_npu_bshd._import_npu_fusion_attention",
                return_value=patch_fn,
            )
        else:
            # Default mock: return a simple identity-like result.
            def default_mock(q, k, v, num_heads, input_layout, **kw):
                # Return q reshaped → [B, S, H*D] as output (identity passthrough)
                return (q,)
            patcher = patch(
                "prefix_sharing.backends.flash_atten_npu_bshd._import_npu_fusion_attention",
                return_value=default_mock,
            )

        with patcher:
            result = backend.attention(
                query, key, value, plan,
                packed_batch_layout=layout,
                layer_id=0,
            )
        return result, layout, query, key, value

    def test_output_shape(self):
        """Output shape matches query [B, S_q, H_q, D]."""
        plan = _plan()
        backend = self._make_backend()
        result, layout, q, *_ = self._run_attention(backend, plan)
        assert result.shape == q.shape, f"{result.shape} != {q.shape}"

    def test_non_kept_rows_zeroed(self):
        """Reuser prefix Q rows and padding rows are exactly zero."""
        plan = _plan()
        backend = self._make_backend()
        result, layout, *_ = self._run_attention(backend, plan)

        # Row 0 is a provider → all [0, valid) should be non-zero.
        v0 = layout.valid_lengths[0]
        assert result[0, :v0].abs().sum().item() > 0, "provider valid rows should be non-zero"

        # Row 1 is a reuser with prefix_len=2 → rows [0, 2) should be zero.
        prefix_len = int(plan.prefix_lens[1])
        assert prefix_len == 2
        assert result[1, :prefix_len].abs().sum().item() == 0.0, \
            "reuser prefix Q rows should be zero"

    def test_mask_absolute_coordinates(self):
        """The mask passed to npu_fusion_attention uses absolute coordinates."""
        plan = _plan()
        backend = self._make_backend()

        captured_mask = {}

        def mock_npu(q, k, v, num_heads, input_layout, **kw):
            captured_mask["mask"] = kw.get("atten_mask")
            return (q,)

        self._run_attention(backend, plan, patch_fn=mock_npu)
        mask = captured_mask["mask"]
        assert mask is not None
        B, _, M_q, M_kv = mask.shape
        assert M_q == max(plan.original_lengths)
        assert M_kv == max(plan.expanded_lengths_kv)

        # Provider row 0: causal lower-tri (upper-tri True=masked).
        v0 = plan.original_lengths[0]
        assert mask[0, 0, 0, 0].item() is False, "provider diagonal should be visible"
        assert mask[0, 0, 0, 1].item() is True, "provider above-diag should be masked"

        # Reuser row 1: prefix columns [0,2) all visible for suffix Q rows.
        prefix_len = int(plan.prefix_lens[1])
        assert prefix_len == 2
        assert mask[1, 0, prefix_len, 0].item() is False, \
            "reuser suffix Q should see prefix KV col 0"

    def test_gqa_kv_heads_inferred_from_shape(self):
        """num_key_value_heads is NOT passed to npu_fusion_attention; GQA is
        handled by the kernel inferring kv heads from the K tensor shape."""
        plan = _plan()
        backend = self._make_backend()

        captured = {}

        def mock_npu(q, k, v, num_heads, input_layout, **kw):
            captured["num_kv_heads"] = kw.get("num_key_value_heads")
            return (q,)

        self._run_attention(backend, plan, patch_fn=mock_npu)
        assert captured["num_kv_heads"] is None, \
            "num_key_value_heads must not be passed (kernel infers from K shape)"

    def test_sparse_mode_1(self):
        """sparse_mode=1 is passed to npu_fusion_attention."""
        plan = _plan()
        backend = self._make_backend()

        captured = {}

        def mock_npu(q, k, v, num_heads, input_layout, **kw):
            captured["sparse_mode"] = kw.get("sparse_mode")
            return (q,)

        self._run_attention(backend, plan, patch_fn=mock_npu)
        assert captured["sparse_mode"] == 1

    def test_scale_default(self):
        """Default scale = 1/sqrt(D)."""
        plan = _plan()
        backend = self._make_backend()

        captured = {}

        def mock_npu(q, k, v, num_heads, input_layout, **kw):
            captured["scale"] = kw.get("scale")
            return (q,)

        self._run_attention(backend, plan, patch_fn=mock_npu)
        expected = 1.0 / math.sqrt(64)
        assert abs(captured["scale"] - expected) < 1e-6


# ── validate (without NPU, expect ImportError) ─────────────────────────────


class TestValidate:
    def test_validate_requires_mindspeed(self):
        """validate raises because there is no MindSpeed on CPU."""
        from prefix_sharing.backends.flash_atten_npu_bshd import NpuFlashAttentionBackendBshd
        backend = NpuFlashAttentionBackendBshd()
        with pytest.raises(RuntimeError, match="MindSpeed"):
            backend.validate(PrefixSharingConfig(enable_prefix_sharing=True))


# ── apply_rope raises ──────────────────────────────────────────────────────


class TestApplyRope:
    def test_apply_rope_not_implemented(self):
        from prefix_sharing.backends.flash_atten_npu_bshd import NpuFlashAttentionBackendBshd
        backend = NpuFlashAttentionBackendBshd()
        with pytest.raises(NotImplementedError, match="RoPE"):
            backend.apply_rope(None, None, _plan())
