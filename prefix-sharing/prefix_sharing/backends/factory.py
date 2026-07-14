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
    * ``"flex_attention"`` -> :class:`~prefix_sharing.backends.flex_attention.FlexAttentionBackend`
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

    if config.backend == "flex_attention":
        from prefix_sharing.backends.flex_attention import FlexAttentionBackend
        return FlexAttentionBackend(
            block_size=config.flex_attention_block_size,
            compile_attention=config.flex_attention_compile,
        )
    
    raise ValueError(
        f"Unknown backend '{config.backend}'. "
        f"Supported: torch_ref, flash_atten_gpu, flash_atten_npu, flex_attention"
    )
