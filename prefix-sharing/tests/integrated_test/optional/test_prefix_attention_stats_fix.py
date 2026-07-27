"""Regression test: prefix_attention must not NameError on 'ctx' or 'prefix_log'."""
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.integrations.megatron_runtime import prefix_attention
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.integrations.runtime_state import PrefixSharingRuntimeState
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo


def _install_megatron_parallel_state(monkeypatch, tp_size=1):
    parallel_state = types.ModuleType("megatron.core.parallel_state")
    parallel_state.get_tensor_model_parallel_world_size = lambda: tp_size
    parallel_state.get_tensor_model_parallel_rank = lambda: 0
    parallel_state.get_context_parallel_world_size = lambda: 1
    parallel_state.get_context_parallel_rank = lambda: 0
    parallel_state.get_pipeline_model_parallel_world_size = lambda: 1
    parallel_state.get_pipeline_model_parallel_rank = lambda: 0
    parallel_state.is_pipeline_first_stage = lambda ignore_virtual=True: True
    parallel_state.is_pipeline_last_stage = lambda ignore_virtual=True: True
    parallel_state.get_virtual_pipeline_model_parallel_world_size = lambda: None
    core = types.ModuleType("megatron.core")
    core.parallel_state = parallel_state
    megatron = types.ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)


def _install_rope_passthrough(monkeypatch):
    rope_utils = types.ModuleType("megatron.core.models.common.embeddings.rope_utils")
    rope_utils.apply_rotary_pos_emb = lambda t, freqs, **kw: t
    embeddings = types.ModuleType("megatron.core.models.common.embeddings")
    embeddings.rope_utils = rope_utils
    common = types.ModuleType("megatron.core.models.common")
    common.embeddings = embeddings
    models = types.ModuleType("megatron.core.models")
    models.common = common
    core = sys.modules["megatron.core"]
    core.models = models
    monkeypatch.setitem(sys.modules, "megatron.core.models", models)
    monkeypatch.setitem(sys.modules, "megatron.core.models.common", common)
    monkeypatch.setitem(sys.modules, "megatron.core.models.common.embeddings", embeddings)
    monkeypatch.setitem(sys.modules, "megatron.core.models.common.embeddings.rope_utils", rope_utils)


def test_prefix_attention_runs_without_nameerror(monkeypatch):
    _install_megatron_parallel_state(monkeypatch, tp_size=1)
    _install_rope_passthrough(monkeypatch)

    # Build state inline (the 070 batch constructor was removed)
    config = PrefixSharingConfig.from_raw({"enable_prefix_sharing": True, "min_prefix_len": 3})
    sequences = [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]
    plan = PrefixSharingPlanner(config).plan(sequences)
    kept_position_rows = [torch.arange(5) for _ in range(2)]
    layout = PackedBatchLayout.from_kept_position_rows(kept_position_rows, align_size=1)
    state = PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        attention_backend=None,
        packed_batch_layout=layout,
        parallel_info=MegatronParallelInfo(
            global_rank=0, tp_rank=0, tp_size=1,
            cp_rank=0, cp_size=1,
            pp_rank=0, pp_size=1,
            is_pipeline_first_stage=True, is_pipeline_last_stage=True,
        ),
    )

    layout = state.packed_batch_layout
    total = layout.total_padded_length  # 7

    hidden = 4
    query = torch.randn(total, 1, hidden)
    key = torch.randn(total, 1, hidden)
    value = torch.randn(total, 1, hidden)

    # pos_emb shape: [max_pos, 1, 1, hidden*2]
    pos_emb = torch.zeros(total, 1, 1, hidden * 2)

    packed_seq_params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
    )

    linear_proj = MagicMock(return_value=torch.randn(total, 1, hidden))
    attention_module = SimpleNamespace(
        config=SimpleNamespace(sequence_parallel=False, num_layers=2),
        layer_number=1,
        linear_proj=linear_proj,
    )

    with prefix_sharing_runtime_context(state) as ctx:
        result = prefix_attention(
            attention_module,
            query,
            key,
            value,
            attention_mask=None,
            rotary_pos_emb=(pos_emb, pos_emb),
            packed_seq_params=packed_seq_params,
        )

    assert linear_proj.called, "prefix_attention did not reach linear_proj"
