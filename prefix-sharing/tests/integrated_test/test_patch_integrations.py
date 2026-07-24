import pytest

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.integrations.megatron_attention import (
    IntegrationUnavailable,
    MegatronAttentionIntegration,
)
from prefix_sharing.integrations.patch_manager import PatchManager
from prefix_sharing.integrations.verl_mcore import (
    VerlMCoreIntegration,
    prefix_sharing_enabled,
    read_ps_config_from_engine_config,
)
class Target:
    def method(self):
        return "original"


def test_patch_manager_installs_and_disables_patch():
    target = Target()
    manager = PatchManager()

    def replacement(instance):
        return "patched"

    manager.patch_attr(Target, "method", replacement)
    handle = manager.handle()
    assert target.method() == "patched"
    assert handle.active

    handle.disable()
    assert target.method() == "original"
    assert not handle.active


def test_patch_manager_context_manager_restores_original():
    target = Target()
    manager = PatchManager()
    manager.patch_attr(Target, "method", lambda instance: "patched")

    with manager.handle():
        assert target.method() == "patched"
    assert target.method() == "original"


def test_patch_manager_installs_and_disables_dict_item_patch():
    registry = {"eager": lambda: "original"}
    original = registry["eager"]
    manager = PatchManager()

    def replacement():
        return "patched"

    manager.patch_item(registry, "eager", replacement)
    handle = manager.handle()
    assert registry["eager"]() == "patched"

    handle.disable()
    assert registry["eager"] is original


def test_megatron_integration_reports_missing_dependency_cleanly(monkeypatch):
    import importlib

    _original_import = importlib.import_module

    def _mock_import(name, package=None):
        if name == "megatron.core.transformer.attention":
            raise ModuleNotFoundError("No module named 'megatron'")
        return _original_import(name, package=package)

    monkeypatch.setattr(importlib, "import_module", _mock_import)

    config = PrefixSharingConfig(enable_prefix_sharing=True)
    integration = MegatronAttentionIntegration(config=config, backend=TorchReferenceBackend())
    with pytest.raises(IntegrationUnavailable, match="Megatron"):
        integration.install(model_config={})


def test_verl_integration_reports_missing_dependency_cleanly(monkeypatch):
    import importlib

    _original_import = importlib.import_module

    def _mock_import(name, package=None):
        if name == "verl":
            raise ModuleNotFoundError("No module named 'verl'")
        return _original_import(name, package=package)

    monkeypatch.setattr(importlib, "import_module", _mock_import)

    config = PrefixSharingConfig(enable_prefix_sharing=True)
    integration = VerlMCoreIntegration(config=config)
    with pytest.raises(IntegrationUnavailable, match="verl"):
        integration.install(model_config={})


def test_setup_can_load_explicit_verl080_fsdp_patch_set():
    from prefix_sharing.setup import _load_patch_set

    patch_set = _load_patch_set("verl080_fsdp")

    assert len(patch_set) == 7
    assert patch_set[0].module_name == "verl.workers.engine.fsdp.transformer_impl"
    assert "FSDPEngineWithLMHead.forward_step" in patch_set[0].description
    assert patch_set[1].module_name == "verl.trainer.ppo.ray_trainer"
    assert "RayPPOTrainer.fit" in patch_set[1].description
    # Diag-dump wrapper is installed on top of the profiler wrapper.
    assert patch_set[4].module_name == "verl.workers.engine.fsdp.transformer_impl"
    assert "dump weight gradients" in patch_set[4].description


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


def test_prefix_sharing_enabled_propagates_install_failure(monkeypatch):
    class FakeIntegration:
        def __init__(self, config, backend=None):
            pass

        def install(self, model_config=None):
            raise RuntimeError("install failed")

    monkeypatch.setattr("prefix_sharing.integrations.verl_mcore.VerlMCoreIntegration", FakeIntegration)
    with pytest.raises(RuntimeError, match="install failed"):
        with prefix_sharing_enabled(PrefixSharingConfig(enable_prefix_sharing=True)):
            pass
