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
)
from prefix_sharing.integrations.verl_fsdp import VerlFSDPIntegration


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


def test_verl_fsdp_integration_reports_missing_dependency_cleanly(monkeypatch):
    import importlib

    _original_import = importlib.import_module

    def _mock_import(name, package=None):
        if name == "verl":
            raise ModuleNotFoundError("No module named 'verl'")
        return _original_import(name, package=package)

    monkeypatch.setattr(importlib, "import_module", _mock_import)

    config = PrefixSharingConfig(enable_prefix_sharing=True)
    integration = VerlFSDPIntegration(config=config)
    with pytest.raises(IntegrationUnavailable, match="verl"):
        integration.install(model_config={})


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
