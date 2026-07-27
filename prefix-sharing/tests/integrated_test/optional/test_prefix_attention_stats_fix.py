"""Regression test: prefix_attention must not NameError on 'ctx' or 'prefix_log'."""
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.integrations.megatron_runtime import prefix_attention
from prefix_sharing.integrations.verl_mcore import build_prefix_sharing_micro_batch_verl070


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

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.arange(5).repeat(2, 1),
        "responses": torch.tensor([[10, 11], [20, 21]]),
    }
    actor_config = {
        "prefix_sharing_config": {"enable_prefix_sharing": True, "min_prefix_len": 3},
        "megatron": {"use_remove_padding": True},
    }
    model_config = SimpleNamespace(
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
        apply_rope_fusion=False,
        fused_single_qkv_rope=False,
        model_type="text_only_causal_lm",
    )
    _, state = build_prefix_sharing_micro_batch_verl070(batch, actor_config, model_config)

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
