"""Backend factory: instantiate a concrete backend from config string."""

from __future__ import annotations

from typing import Any

from prefix_sharing.backends.base import PrefixAttentionBackend
from prefix_sharing.core.config import PrefixSharingConfig


def get_backend_instance(
    config: PrefixSharingConfig, backend: Any | None = None
) -> PrefixAttentionBackend:
    """Return an explicit ``backend`` if given, otherwise build from ``config``.

    Supported values for ``config.backend``:
    * ``"torch_ref"``      -> :class:`~prefix_sharing.backends.torch_ref.TorchReferenceBackend`
    * ``"flash_atten_gpu"`` -> :class:`~prefix_sharing.backends.flash_atten_gpu.GpuFlashAttentionBackend`
    * ``"flash_atten_npu"`` -> :class:`~prefix_sharing.backends.flash_atten_npu.NpuFlashAttentionBackend`
    """
    if backend is not None:
        return backend

    if config.backend == "torch_ref":
        from prefix_sharing.backends.torch_ref import TorchReferenceBackend
        return TorchReferenceBackend()

    if config.backend == "flash_atten_gpu":
        from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend
        return GpuFlashAttentionBackend()

    if config.backend == "flash_atten_npu":
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend
        return NpuFlashAttentionBackend()

    raise ValueError(
        f"Unknown backend '{config.backend}'. "
        f"Supported: torch_ref, flash_atten_gpu, flash_atten_npu"
    )


def get_bshd_backend_instance(
    config: PrefixSharingConfig, backend: Any | None = None
) -> Any:
    """Return a BSHD-compatible backend instance.

    Uses the same backend names as the THD factory and maps them to the
    BSHD implementations:

    * ``"torch_ref"``       -> TorchReferenceBackendBshd
    * ``"flash_atten_npu"`` -> NpuFlashAttentionBackendBshd
    * ``"flash_atten_gpu"`` -> not supported yet: flash-attn has no 4-D
      ``attn_mask`` support; a future GPU BSHD backend must convert to
      varlen at the boundary (``flash_attn_varlen_func`` + cu_seqlens).
    """
    backend_name = backend if backend is not None else config.backend

    if backend_name == "torch_ref":
        from prefix_sharing.backends.torch_ref_bshd import TorchReferenceBackendBshd
        return TorchReferenceBackendBshd()

    if backend_name == "flash_atten_npu":
        from prefix_sharing.backends.flash_atten_npu_bshd import NpuFlashAttentionBackendBshd
        return NpuFlashAttentionBackendBshd()

    if backend_name == "flash_atten_gpu":
        raise ValueError(
            "BSHD is not supported for backend 'flash_atten_gpu' yet "
            "(flash_attn_func has no attn_mask parameter). "
            "Use backend='torch_ref' for BSHD."
        )

    raise ValueError(
        f"Unknown BSHD backend '{backend_name}'. "
        f"Supported: torch_ref, flash_atten_npu"
    )
