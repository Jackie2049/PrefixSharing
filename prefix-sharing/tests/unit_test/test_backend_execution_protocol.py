"""Execution-mode contract tests for prefix attention backends."""

from __future__ import annotations

from prefix_sharing.backends.base import (
    BackendCapabilities,
    PrefixAttentionExecutionMode,
    requires_expanded_kv,
)
from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend
from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend
from prefix_sharing.backends.torch_ref import TorchReferenceBackend


def test_existing_backends_explicitly_require_expanded_kv() -> None:
    for backend in (
        TorchReferenceBackend(),
        GpuFlashAttentionBackend(),
        NpuFlashAttentionBackend(),
    ):
        assert backend.capabilities.execution_mode is PrefixAttentionExecutionMode.EXPANDED_KV
        assert requires_expanded_kv(backend)


def test_deduplicated_capability_does_not_require_build_kv() -> None:
    capabilities = BackendCapabilities(
        name="deduplicated_test",
        supports_cpu=False,
        supports_cuda=True,
        supports_cann=False,
        supports_different_q_kv_lengths=False,
        supports_prefix_last_restore=True,
        execution_mode=PrefixAttentionExecutionMode.DEDUPLICATED_QKV,
    )

    class _DeduplicatedBackend:
        pass

    _DeduplicatedBackend.capabilities = capabilities

    assert not requires_expanded_kv(_DeduplicatedBackend())
