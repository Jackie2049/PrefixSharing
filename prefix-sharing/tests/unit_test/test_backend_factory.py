"""Unit tests for the backend factory."""

from __future__ import annotations

import pytest

from prefix_sharing.backends.factory import get_backend_instance
from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend
from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig


def test_factory_torch_ref() -> None:
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="torch_ref")
    backend = get_backend_instance(config)
    assert isinstance(backend, TorchReferenceBackend)
    assert backend.capabilities.name == "torch_ref"


def test_factory_flash_atten_gpu() -> None:
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="flash_atten_gpu")
    backend = get_backend_instance(config)
    assert isinstance(backend, GpuFlashAttentionBackend)
    assert backend.capabilities.name == "flash_atten_gpu"
    assert backend.capabilities.supports_flash_attention


def test_factory_flash_atten_npu():
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="flash_atten_npu")
    backend = get_backend_instance(config)
    assert isinstance(backend, NpuFlashAttentionBackend)
    assert backend.capabilities.name == "flash_atten_npu"


def test_factory_unknown_backend() -> None:
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="unknown")
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend_instance(config)


def test_resolve_backend_explicit() -> None:
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="torch_ref")
    explicit = GpuFlashAttentionBackend()
    resolved = get_backend_instance(config, explicit)
    assert resolved is explicit


def test_resolve_backend_from_config() -> None:
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="flash_atten_gpu")
    resolved = get_backend_instance(config)
    assert isinstance(resolved, GpuFlashAttentionBackend)


def test_config_validates_backends() -> None:
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="flash_atten_gpu")
    config.validate()

    bad_config = PrefixSharingConfig(enable_prefix_sharing=True, backend="not_real")
    with pytest.raises(Exception):
        bad_config.validate()


def test_config_accepts_supported_backends():
    for name in ("torch_ref", "flash_atten_gpu", "flash_atten_npu"):
        cfg = PrefixSharingConfig(enable_prefix_sharing=True, backend=name)
        cfg.validate()  # should not raise
        