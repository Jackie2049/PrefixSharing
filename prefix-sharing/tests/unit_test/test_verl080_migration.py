"""verl080_mcore0161_ms0160 patch set 和配套接口的单元测试。

本文件测试本地可验证的部分：
- PrefixSharingConfig.validate_for_engine()
- PrefixSharingRuntimeState.kept_position_ids 字段
- PrefixSharingRuntimeContext.kept_position_ids 传递
- compat_matrix 新条目版本匹配
- integrations.verl_mcore.read_ps_config_from_engine_config 读取逻辑
- integrations.megatron_runtime v0.16.1 API helpers

需要 verl/Megatron 运行环境的 patch 联动测试（forward_step → attention → vocab_logprobs）
属于 integrated_test，本地不可运行，在 NPU/GPU 环境中单独验证。
"""

from dataclasses import dataclass

import pytest

from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import (
    current_prefix_sharing_context,
    prefix_sharing_runtime_context,
)
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo
from prefix_sharing.integrations.runtime_state import PrefixSharingRuntimeState


# ═══════════════════════════════════════
# PrefixSharingConfig.validate_for_engine()
# ═══════════════════════════════════════


def test_validate_for_engine_disabled_does_not_raise():
    config = PrefixSharingConfig(enable_prefix_sharing=False)
    config.validate_for_engine(use_remove_padding=True)
    config.validate_for_engine(use_remove_padding=False)


def test_validate_for_engine_accepts_remove_padding_true():
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    config.validate_for_engine(use_remove_padding=True)


def test_validate_for_engine_rejects_remove_padding_false():
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    with pytest.raises(PrefixSharingConfigError, match="use_remove_padding"):
        config.validate_for_engine(use_remove_padding=False)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("detector", "prompt", "detector"),
        ("backend", "unknown_backend", "backend"),
        ("boundary_strategy", "restore_last_prefix_token", "boundary_strategy"),
        ("min_prefix_len", 0, "min_prefix_len"),
        ("min_group_size", 1, "min_group_size"),
    ],
)
def test_validate_for_engine_rejects_unsupported_options(field, value, message):
    config = PrefixSharingConfig(enable_prefix_sharing=True, **{field: value})
    with pytest.raises(PrefixSharingConfigError, match=message):
        config.validate_for_engine(use_remove_padding=True)


# ═══════════════════════════════════════
# PrefixSharingRuntimeState.kept_position_ids
# ═══════════════════════════════════════


def _make_runtime_state(kept_position_ids=None):
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    )
    plan = planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]],
        forward_id=1,
        micro_batch_id=1,
    )
    return PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        attention_backend=None,
        packed_batch_layout=PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q),
        parallel_info=MegatronParallelInfo(),
        kept_position_ids=kept_position_ids,
    )


def test_runtime_state_accepts_kept_position_ids():
    state = _make_runtime_state(kept_position_ids=[0, 1, 2, 3, 4, 0, 1, 2])
    assert state.kept_position_ids == [0, 1, 2, 3, 4, 0, 1, 2]


def test_runtime_state_defaults_kept_position_ids_to_none():
    state = _make_runtime_state()
    assert state.kept_position_ids is None


def test_runtime_state_is_frozen():
    state = _make_runtime_state(kept_position_ids=[0, 1, 2])
    with pytest.raises(AttributeError):
        state.kept_position_ids = [5, 6]


# ═══════════════════════════════════════
# PrefixSharingRuntimeContext.kept_position_ids 传递
# ═══════════════════════════════════════


def test_context_propagates_kept_position_ids_from_state():
    state = _make_runtime_state(kept_position_ids=[10, 11, 12])
    with prefix_sharing_runtime_context(state) as ctx:
        assert ctx.kept_position_ids == [10, 11, 12]


def test_context_kept_position_ids_none_when_state_has_none():
    state = _make_runtime_state(kept_position_ids=None)
    with prefix_sharing_runtime_context(state) as ctx:
        assert ctx.kept_position_ids is None


def test_context_kept_position_ids_none_when_state_has_no_attr():
    """旧配套的 PrefixSharingRuntimeState 没有 kept_position_ids 字段时，
    context 应通过 getattr 回退到 None，保持向后兼容。"""
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    )
    plan = planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]],
        forward_id=1,
        micro_batch_id=1,
    )
    # 模拟旧版 state（没有 kept_position_ids）
    # 用 Namespace 模拟，因为 frozen dataclass 不能动态删除字段
    import types

    old_state = types.SimpleNamespace(
        prefix_sharing_plan=plan,
        attention_backend=None,
        packed_batch_layout=PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q),
        parallel_info=MegatronParallelInfo(),
    )
    with prefix_sharing_runtime_context(old_state) as ctx:
        assert ctx.kept_position_ids is None


# ═══════════════════════════════════════
# compat_matrix FSDP-first 匹配
# ═══════════════════════════════════════


def test_compat_matrix_prefers_verl080_fsdp_even_when_mcore_is_installed():
    from prefix_sharing.setup.compat_matrix import COMPAT_MATRIX
    from prefix_sharing.setup.version_guard import DetectedVersions

    versions = DetectedVersions(
        verl="0.8.0.dev",
        megatron_core="0.16.1",
        mindspeed="0.16.0",
    )
    matching = [e for e in COMPAT_MATRIX if e.match(versions)]
    assert matching
    assert matching[0].patch_set_id == "verl080_fsdp"
    assert any(e.patch_set_id == "verl080_mcore0161_ms0160" for e in matching)


def test_compat_matrix_matches_verl080_fsdp_without_mcore_or_mindspeed():
    from prefix_sharing.setup.compat_matrix import COMPAT_MATRIX
    from prefix_sharing.setup.version_guard import DetectedVersions

    versions = DetectedVersions(
        verl="0.8.0.dev",
        megatron_core=None,
        mindspeed=None,
    )
    matching = [e for e in COMPAT_MATRIX if e.match(versions)]
    assert matching
    assert matching[0].patch_set_id == "verl080_fsdp"


def test_compat_matrix_no_match_raises_incompatible():
    from prefix_sharing.setup.compat_matrix import COMPAT_MATRIX
    from prefix_sharing.setup.version_guard import DetectedVersions

    versions = DetectedVersions(
        verl="0.7.0",  # 不匹配任何条目
        megatron_core="0.12.0",
        mindspeed="0.12.0",
    )
    matching = [e for e in COMPAT_MATRIX if e.match(versions)]
    assert len(matching) == 0


# ═══════════════════════════════════════
# integrations.verl_mcore.read_ps_config_from_engine_config 读取逻辑
# ═══════════════════════════════════════


def test_read_ps_config_from_override_dict():
    from prefix_sharing.integrations.verl_mcore import (
        read_ps_config_from_engine_config,
    )

    engine_config = type("EngineConfig", (), {
        "override_transformer_config": {
            "prefix_sharing_config": {"enable_prefix_sharing": True},
        },
    })()
    result = read_ps_config_from_engine_config(engine_config)
    assert result == {"enable_prefix_sharing": True}


def test_read_ps_config_returns_none_when_missing():
    from prefix_sharing.integrations.verl_mcore import (
        read_ps_config_from_engine_config,
    )

    engine_config = type("EngineConfig", (), {
        "override_transformer_config": {},
    })()
    result = read_ps_config_from_engine_config(engine_config)
    assert result is None


def test_read_ps_config_from_direct_attr():
    from prefix_sharing.integrations.verl_mcore import (
        read_ps_config_from_engine_config,
    )

    engine_config = type("EngineConfig", (), {
        "override_transformer_config": type("Override", (), {
            "prefix_sharing_config": {"enable_prefix_sharing": True},
        })(),
    })()
    result = read_ps_config_from_engine_config(engine_config)
    assert result == {"enable_prefix_sharing": True}


def test_read_ps_config_returns_none_for_empty_config():
    from prefix_sharing.integrations.verl_mcore import (
        read_ps_config_from_engine_config,
    )

    engine_config = type("EngineConfig", (), {})()
    result = read_ps_config_from_engine_config(engine_config)
    assert result is None


# ═══════════════════════════════════════
# integrations.megatron_runtime v0.16.1 helpers
# ═══════════════════════════════════════


def test_extract_cu_seqlens_returns_none_for_none_params():
    from prefix_sharing.integrations.megatron_runtime import _extract_cu_seqlens
    assert _extract_cu_seqlens(None, "cu_seqlens_q_padded", "cu_seqlens_q") is None


def test_extract_cu_seqlens_prefers_padded_attr():
    from prefix_sharing.integrations.megatron_runtime import _extract_cu_seqlens
    params = type("Params", (), {
        "cu_seqlens_q_padded": [0, 10],
        "cu_seqlens_q": [0, 8],
    })()
    result = _extract_cu_seqlens(params, "cu_seqlens_q_padded", "cu_seqlens_q")
    assert result == [0, 10]


def test_extract_cu_seqlens_falls_back_to_primary():
    from prefix_sharing.integrations.megatron_runtime import _extract_cu_seqlens
    params = type("Params", (), {
        "cu_seqlens_q": [0, 8],
    })()
    result = _extract_cu_seqlens(params, "cu_seqlens_q_padded", "cu_seqlens_q")
    assert result == [0, 8]


def test_get_yarn_mscale_fallback_without_megatron():
    from prefix_sharing.integrations.megatron_runtime import _get_yarn_mscale
    # Fake attention_module without real Megatron — should fallback to 1.0
    fake_module = type("Attn", (), {"config": None})()
    assert _get_yarn_mscale(fake_module) == 1.0


def test_get_cp_group_returns_none_without_pg_collection():
    from prefix_sharing.integrations.megatron_runtime import _get_cp_group
    fake_module = type("Attn", (), {})()
    assert _get_cp_group(fake_module) is None


# ═══════════════════════════════════════
# __init__.py auto-activation 逻辑
# ═══════════════════════════════════════


def test_auto_activation_always_attempts_and_handles_missing_env(monkeypatch):
    """patch 始终尝试安装，环境兼容时（verl/Megatron 存在）应成功安装。"""
    monkeypatch.delenv("ENABLE_PREFIX_SHARING", raising=False)
    import importlib
    import prefix_sharing
    importlib.reload(prefix_sharing)
    # 服务器上 verl+Megatron 已安装，patch 安装应成功
    assert prefix_sharing._patch_handle is not None
    assert prefix_sharing._patch_handle.patch_set_sizes[0] == 7


def test_auto_activation_handles_env_var_false(monkeypatch):
    """ENABLE_PREFIX_SHARING=0 时仍尝试安装 patch，兼容性错误安全回退。"""
    monkeypatch.setenv("ENABLE_PREFIX_SHARING", "0")
    import importlib
    import prefix_sharing
    importlib.reload(prefix_sharing)
    # 服务器上 verl+Megatron 已安装，patch 安装应成功
    assert prefix_sharing._patch_handle is not None
    assert prefix_sharing._patch_handle.patch_set_sizes[0] == 7


def test_auto_activation_handles_env_var_true(monkeypatch):
    """ENABLE_PREFIX_SHARING=1 时尝试安装 patch，兼容性错误安全回退。"""
    monkeypatch.setenv("ENABLE_PREFIX_SHARING", "1")
    import importlib
    import prefix_sharing
    importlib.reload(prefix_sharing)
    assert prefix_sharing._patch_handle is not None
    assert prefix_sharing._patch_handle.patch_set_sizes[0] == 7
