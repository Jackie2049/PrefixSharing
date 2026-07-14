"""FSDP dispatch tests for backends that consume deduplicated Q/K/V."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.base import (
    BackendCapabilities,
    PrefixAttentionExecutionMode,
)
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.integrations.verl_fsdp import (
    PrefixSharingFSDPAttentionRuntime,
    build_prefix_sharing_micro_batch_fsdp,
)


class _RecordingDeduplicatedBackend:
    """Sparse backend fake: deliberately has no build_kv method."""

    capabilities = BackendCapabilities(
        name="recording_deduplicated",
        supports_cpu=True,
        supports_cuda=False,
        supports_cann=False,
        supports_different_q_kv_lengths=False,
        supports_prefix_last_restore=True,
        execution_mode=PrefixAttentionExecutionMode.DEDUPLICATED_QKV,
    )

    def __init__(self) -> None:
        self.prepare_calls = []
        self.attention_calls = []

    def prepare_runtime(self, *, prefix_tree_attention_layout, packed_batch_layout, device):
        marker = object()
        self.prepare_calls.append((prefix_tree_attention_layout, packed_batch_layout, device, marker))
        return marker

    def attention(
        self,
        query,
        key,
        value,
        prefix_sharing_plan,
        *,
        packed_batch_layout,
        prefix_tree_attention_layout,
        runtime,
    ):
        self.attention_calls.append(
            (query, key, value, prefix_sharing_plan, packed_batch_layout, prefix_tree_attention_layout, runtime)
        )
        return query + value


def _shared_batch():
    return {
        "input_ids": torch.tensor(
            [[1, 2, 3, 4, 5], [1, 2, 3, 6, 7]],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.arange(5).expand(2, -1),
    }


def test_fsdp_runtime_dispatches_deduplicated_backend_without_build_kv_and_reuses_runtime():
    backend = _RecordingDeduplicatedBackend()
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    _, state = build_prefix_sharing_micro_batch_fsdp(_shared_batch(), config, backend=backend)

    assert state is not None
    assert state.prefix_tree_attention_layout is not None
    assert not hasattr(backend, "build_kv")

    q = torch.randn(2, 5, 2, 4)
    k = torch.randn(2, 5, 2, 4)
    v = torch.randn(2, 5, 2, 4)
    with prefix_sharing_runtime_context(state) as ctx:
        first = PrefixSharingFSDPAttentionRuntime(layer_id=0).forward(None, q, k, v)
        second = PrefixSharingFSDPAttentionRuntime(layer_id=1).forward(None, q, k, v)

        assert len(backend.prepare_calls) == 1
        assert len(backend.attention_calls) == 2
        assert backend.attention_calls[0][-1] is backend.attention_calls[1][-1]
        assert backend.attention_calls[0][-2] is state.prefix_tree_attention_layout
        assert ctx.attention_backend_runtime is backend.attention_calls[0][-1]

    assert first.shape == q.shape
    assert second.shape == q.shape
