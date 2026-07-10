import pytest

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.integrations.verl_utils import read_ps_config_from_engine_config
from prefix_sharing.setup.logged_patch import LoggedPatchManager


class Target:
    def method(self):
        return "original"


def test_logged_patch_manager_installs_and_disables_patch():
    target = Target()
    manager = LoggedPatchManager()

    def replacement(instance):
        return "patched"

    manager.patch_attr(Target, "method", replacement)
    handle = manager.handle()
    assert target.method() == "patched"
    assert handle.active

    handle.disable()
    assert target.method() == "original"
    assert not handle.active


def test_logged_patch_manager_context_manager_restores_original():
    target = Target()
    manager = LoggedPatchManager()
    manager.patch_attr(Target, "method", lambda instance: "patched")

    with manager.handle():
        assert target.method() == "patched"
    assert target.method() == "original"


def test_setup_can_load_explicit_verl080_fsdp_patch_set():
    from prefix_sharing.setup import _load_patch_set

    patch_set = _load_patch_set("verl080_fsdp")

    assert len(patch_set) == 2
    assert patch_set[0].module_name == "verl.workers.engine.fsdp.transformer_impl"
    assert "FSDPEngineWithLMHead.forward_step" in patch_set[0].description
    assert patch_set[1].module_name == "transformers.modeling_utils"
    assert "ALL_ATTENTION_FUNCTIONS" in patch_set[1].description


def test_prefix_sharing_config_from_raw_accepts_nested_config():
    config = PrefixSharingConfig.from_raw(
        {
            "enable_prefix_sharing": True,
            "min_prefix_len": 4,
            "min_group_size": 3,
            "boundary_strategy": "prefix_last_restore",
        }
    )

    assert config.enable_prefix_sharing is True
    assert config.min_prefix_len == 4
    assert config.min_group_size == 3


def test_read_ps_config_from_engine_config_prefers_explicit_prefix_sharing_config():
    class EngineConfig:
        prefix_sharing_config = {"enable_prefix_sharing": False}
        use_prefix_grouper = True
        prefix_grouper = {"mode": "arbitrary_prefix", "min_prefix_len": 8}

    raw = read_ps_config_from_engine_config(EngineConfig())

    assert raw == {"enable_prefix_sharing": False}


def test_read_ps_config_from_prefix_grouper_prompt_only_disables_prefix_sharing():
    class EngineConfig:
        use_prefix_grouper = True
        prefix_grouper = {"mode": "prompt_only"}

    raw = read_ps_config_from_engine_config(EngineConfig())
    config = PrefixSharingConfig.from_raw(raw)

    assert config.enable_prefix_sharing is False


def test_read_ps_config_from_prefix_grouper_arbitrary_prefix_enables_prefix_sharing():
    class EngineConfig:
        use_prefix_grouper = True
        prefix_grouper = {
            "mode": "arbitrary_prefix",
            "min_prefix_len": 8,
            "min_group_size": 3,
            "strict": True,
        }

    raw = read_ps_config_from_engine_config(EngineConfig())
    config = PrefixSharingConfig.from_raw(raw)

    assert config.enable_prefix_sharing is True
    assert config.min_prefix_len == 8
    assert config.min_group_size == 3
    assert "strict" not in raw


def test_read_ps_config_from_prefix_grouper_rejects_unknown_mode():
    class EngineConfig:
        use_prefix_grouper = True
        prefix_grouper = {"mode": "unknown"}

    with pytest.raises(ValueError, match="prefix_grouper.mode"):
        read_ps_config_from_engine_config(EngineConfig())


def test_prefix_sharing_config_from_raw_rejects_legacy_enabled_key():
    with pytest.raises(TypeError, match="enabled"):
        PrefixSharingConfig.from_raw({"enabled": True})


def test_prefix_sharing_config_from_raw_accepts_env_enable(monkeypatch):
    monkeypatch.setenv("ENABLE_PREFIX_SHARING", "1")

    config = PrefixSharingConfig.from_raw(None)

    assert config.enable_prefix_sharing is True
