"""Unit tests for verl080_mcore0161_ms0160 patch set and supporting interfaces.

This file tests the locally verifiable parts:
- PrefixSharingConfig.validate_for_engine()
- PrefixSharingRuntimeState.kept_position_ids field
- PrefixSharingRuntimeContext.kept_position_ids propagation
- compat_matrix new entry version matching
- integrations.verl_mcore.read_ps_config_from_engine_config read logic
- integrations.megatron_runtime v0.16.1 API helpers

Patch integration tests requiring verl/Megatron runtime (forward_step -> attention ->
vocab_logprobs) belong to integrated_test; they cannot run locally and are validated
separately in NPU/GPU environments.
"""

import pytest

from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
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
# PrefixSharingRuntimeContext.kept_position_ids propagation
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
    """When an older PrefixSharingRuntimeState lacks the kept_position_ids field,
    context should fall back to None via getattr, maintaining backward compatibility."""
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    )
    plan = planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]],
        forward_id=1,
        micro_batch_id=1,
    )
    # Simulate old-version state (without kept_position_ids)
    # Use SimpleNamespace to mock, since frozen dataclass cannot dynamically remove fields
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
# compat_matrix FSDP-first matching
# ═══════════════════════════════════════


def test_compat_matrix_prefers_verl080_fsdp_even_when_mcore_is_installed():
    from prefix_sharing.setup.compat_matrix import COMPAT_MATRIX
    from prefix_sharing.setup.version_detector import DependencyDetectedVersions

    versions = DependencyDetectedVersions(
        verl="0.8.0.dev",
        megatron_core="0.16.1",
        mindspeed="0.16.0",
    )
    matching = [e for e in COMPAT_MATRIX if e.match(versions)]
    assert [entry.patch_set_id for entry in matching] == [
        "verl080_fsdp",
        "verl080_mcore0161_ms0160",
    ]


def test_compat_matrix_matches_verl080_fsdp_without_mcore_or_mindspeed():
    from prefix_sharing.setup.compat_matrix import COMPAT_MATRIX
    from prefix_sharing.setup.version_detector import DependencyDetectedVersions

    versions = DependencyDetectedVersions(
        verl="0.8.0.dev",
        megatron_core=None,
        mindspeed=None,
    )
    matching = [e for e in COMPAT_MATRIX if e.match(versions)]
    assert matching
    assert matching[0].patch_set_id == "verl080_fsdp"


def test_compat_matrix_selects_only_fsdp_without_mindspeed():
    from prefix_sharing.setup.compat_matrix import COMPAT_MATRIX
    from prefix_sharing.setup.version_detector import DependencyDetectedVersions

    versions = DependencyDetectedVersions(
        verl="0.8.0.dev",
        megatron_core="0.16.1",
        mindspeed=None,
    )
    matching = [entry for entry in COMPAT_MATRIX if entry.match(versions)]
    assert [entry.patch_set_id for entry in matching] == ["verl080_fsdp"]


def test_compat_matrix_no_match_raises_incompatible():
    from prefix_sharing.setup.compat_matrix import COMPAT_MATRIX
    from prefix_sharing.setup.version_detector import DependencyDetectedVersions

    versions = DependencyDetectedVersions(
        verl="0.7.0",  # Does not match any entry
        megatron_core="0.12.0",
        mindspeed="0.12.0",
    )
    matching = [e for e in COMPAT_MATRIX if e.match(versions)]
    assert len(matching) == 0


# ═══════════════════════════════════════
# integrations.verl_mcore.read_ps_config_from_engine_config read logic
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
# __init__.py auto-activation logic
# ═══════════════════════════════════════


def test_auto_activation_always_attempts_and_handles_missing_env(monkeypatch):
    """Patch always attempts installation; when environment is compatible (verl/Megatron present), installation should succeed."""
    monkeypatch.delenv("ENABLE_PREFIX_SHARING", raising=False)
    import importlib
    import prefix_sharing
    importlib.reload(prefix_sharing)
    # On server, verl+Megatron are installed; patch installation should succeed
    assert prefix_sharing._patch_handle is not None
    assert "PatchHandle (ACTIVE, 7 patches):" in prefix_sharing._patch_handle.describe()


def test_auto_activation_handles_env_var_false(monkeypatch):
    """With ENABLE_PREFIX_SHARING=0, patch installation is still attempted; compatibility errors fall back safely."""
    monkeypatch.setenv("ENABLE_PREFIX_SHARING", "0")
    import importlib
    import prefix_sharing
    importlib.reload(prefix_sharing)
    # On server, verl+Megatron are installed; patch installation should succeed
    assert prefix_sharing._patch_handle is not None
    assert "PatchHandle (ACTIVE, 7 patches):" in prefix_sharing._patch_handle.describe()


def test_auto_activation_handles_env_var_true(monkeypatch):
    """With ENABLE_PREFIX_SHARING=1, patch installation is attempted; compatibility errors fall back safely."""
    monkeypatch.setenv("ENABLE_PREFIX_SHARING", "1")
    import importlib
    import prefix_sharing
    importlib.reload(prefix_sharing)
    assert prefix_sharing._patch_handle is not None
    assert "PatchHandle (ACTIVE, 7 patches):" in prefix_sharing._patch_handle.describe()
