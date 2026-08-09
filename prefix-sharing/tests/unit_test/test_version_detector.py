"""Unit tests for setup dependency version detection."""

from __future__ import annotations

from types import SimpleNamespace

from prefix_sharing.setup import version_detector


def test_detect_from_module_prefers_an_already_loaded_module(monkeypatch):
    loaded_module = SimpleNamespace(__version__="1.2.3")
    monkeypatch.setitem(version_detector.sys.modules, "fake_dependency", loaded_module)

    def unexpected_import(_module_name):
        raise AssertionError("loaded modules must not be imported again")

    monkeypatch.setattr(version_detector.importlib, "import_module", unexpected_import)

    assert version_detector._detect_from_module("fake_dependency") == "1.2.3"


def test_detect_from_module_returns_none_when_dependency_is_missing(monkeypatch):
    monkeypatch.delitem(version_detector.sys.modules, "missing_dependency", raising=False)

    def missing_import(_module_name):
        raise ModuleNotFoundError

    monkeypatch.setattr(version_detector.importlib, "import_module", missing_import)

    assert version_detector._detect_from_module("missing_dependency") is None


def test_detect_from_metadata_uses_metadata_for_an_already_loaded_module(monkeypatch):
    monkeypatch.setitem(version_detector.sys.modules, "mindspeed", SimpleNamespace())
    monkeypatch.setattr(version_detector, "_metadata_version", lambda package: f"{package}-version")

    assert version_detector._detect_from_metadata("mindspeed") == "mindspeed-version"


def test_detect_versions_collects_each_supported_dependency(monkeypatch):
    values = iter(["0.8.0.dev", "0.16.1", "0.16.0"])
    monkeypatch.setattr(version_detector, "_detect_from_module", lambda _module: next(values))
    monkeypatch.setattr(version_detector, "_detect_from_metadata", lambda _module: next(values))

    assert version_detector.detect_versions() == version_detector.DetectedVersions(
        verl="0.8.0.dev",
        megatron_core="0.16.1",
        mindspeed="0.16.0",
    )
