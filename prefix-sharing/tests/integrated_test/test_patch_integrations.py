import pytest

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.integrations.verl_utils import read_ps_config_from_engine_config
from prefix_sharing.setup.patch_installer import LoggedPatchManager
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


def test_logged_patch_manager_restores_mapping_item():
    manager = LoggedPatchManager()
    mapping = {"flash_attention_2": "original"}

    manager.patch_item(mapping, "flash_attention_2", "patched")

    assert mapping["flash_attention_2"] == "patched"
    manager.handle().disable()
    assert mapping["flash_attention_2"] == "original"


def test_setup_can_load_explicit_verl080_fsdp_patch_set():
    import importlib

    mod = importlib.import_module("prefix_sharing.setup.patches.verl080_fsdp")
    patch_set = mod.PATCH_SET

    assert len(patch_set) == 7
    assert patch_set[0].module_name == "verl.workers.engine.fsdp.transformer_impl"
    assert "FSDPEngineWithLMHead.forward_step" in patch_set[0].description
    assert patch_set[1].module_name == "verl.trainer.ppo.ray_trainer"
    assert "RayPPOTrainer.fit" in patch_set[1].description
    # Diag-dump wrapper is installed on top of the profiler wrapper.
    assert patch_set[4].module_name == "verl.workers.engine.fsdp.transformer_impl"
    assert "dump weight gradients" in patch_set[4].description



def test_default_install_selects_all_matching_patch_sets(monkeypatch):
    from prefix_sharing.setup import _resolve_patch_set_ids
    from prefix_sharing.setup.compat_matrix import CompatEntry
    from prefix_sharing.setup.version_detector import DependencyDetectedVersions

    versions = DependencyDetectedVersions("0.8.0.dev", "0.16.1", "0.16.0")
    entries = [
        CompatEntry("0.8.0.dev", "*", "*", "verl080_fsdp"),
        CompatEntry("0.8.0.dev", "0.16.1", "0.16.0", "verl080_mcore0161_ms0160"),
    ]
    monkeypatch.setattr(
        "prefix_sharing.setup.detect_dependency_versions",
        lambda: versions,
    )
    monkeypatch.setattr(
        "prefix_sharing.setup._match_compat_entries",
        lambda _versions: entries,
    )

    assert _resolve_patch_set_ids(None) == [
        "verl080_fsdp",
        "verl080_mcore0161_ms0160",
    ]


def test_default_install_selects_fsdp_only_when_mcore_dependencies_absent(monkeypatch):
    from prefix_sharing.setup import _resolve_patch_set_ids
    from prefix_sharing.setup.compat_matrix import CompatEntry
    from prefix_sharing.setup.version_detector import DependencyDetectedVersions

    versions = DependencyDetectedVersions("0.8.0.dev", None, None)
    entries = [CompatEntry("0.8.0.dev", "*", "*", "verl080_fsdp")]
    monkeypatch.setattr(
        "prefix_sharing.setup.detect_dependency_versions",
        lambda: versions,
    )
    monkeypatch.setattr(
        "prefix_sharing.setup._match_compat_entries",
        lambda _versions: entries,
    )

    assert _resolve_patch_set_ids(None) == ["verl080_fsdp"]


def test_explicit_patch_set_accepts_comma_separated_list():
    from prefix_sharing.setup import _resolve_patch_set_ids

    patch_set_ids = _resolve_patch_set_ids(
        " verl080_fsdp, verl080_mcore0161_ms0160 ",
    )

    assert patch_set_ids == ["verl080_fsdp", "verl080_mcore0161_ms0160"]


def test_explicit_patch_set_deduplicates_preserving_order():
    from prefix_sharing.setup import _resolve_patch_set_ids

    patch_set_ids = _resolve_patch_set_ids(
        "verl080_fsdp,verl080_fsdp,verl080_mcore0161_ms0160",
    )

    assert patch_set_ids == ["verl080_fsdp", "verl080_mcore0161_ms0160"]



def test_install_loads_all_resolved_patch_sets(monkeypatch):
    import sys
    from types import SimpleNamespace

    from prefix_sharing.setup import install
    from prefix_sharing.setup.patch_installer import PatchHandle, PatchSpec

    installed_specs = []

    patch_set_ids = ["verl080_fsdp", "verl080_mcore0161_ms0160"]
    for patch_set_id in patch_set_ids:
        monkeypatch.setitem(
            sys.modules,
            f"prefix_sharing.setup.patches.{patch_set_id}",
            SimpleNamespace(
                PATCH_SET=[
                    PatchSpec(
                        module_name=f"fake.{patch_set_id}",
                        target_getter=lambda module: (module, "target"),
                        patch_factory=lambda original: original,
                        description=f"patch {patch_set_id}",
                    )
                ],
            ),
        )

    def fake_install_specs(specs):
        installed_specs.extend(specs)
        return PatchHandle([], specs=list(specs))

    monkeypatch.setattr(
        "prefix_sharing.setup._resolve_patch_set_ids",
        lambda _patch_set_id: patch_set_ids,
    )
    monkeypatch.setattr(
        "prefix_sharing.setup.PatchRegistry.install_specs",
        fake_install_specs,
    )

    handle = install()

    assert [spec.module_name for spec in installed_specs] == [
        "fake.verl080_fsdp",
        "fake.verl080_mcore0161_ms0160",
    ]
    assert handle.describe().startswith("PatchHandle")


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
