"""Attention backend adapters."""

from prefix_sharing.backends.base import (
    BackendCapabilities,
    DeduplicatedPrefixAttentionBackend,
    ExpandedKVPrefixAttentionBackend,
    PrefixAttentionBackend,
    PrefixAttentionExecutionMode,
    requires_expanded_kv,
)
from prefix_sharing.backends.block_causal_mask import build_block_causal_mask
from prefix_sharing.backends.factory import get_backend_instance
from prefix_sharing.backends.flash_atten_base import FlashAttentionMixin, FlashBackendValidationError
from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend
from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend
from prefix_sharing.backends.flex_attention import FlexAttentionBackend, FlexAttentionRuntime
from prefix_sharing.backends.torch_ref import TorchReferenceBackend

__all__ = [
    "BackendCapabilities",
    "DeduplicatedPrefixAttentionBackend",
    "ExpandedKVPrefixAttentionBackend",
    "FlashAttentionMixin",
    "FlashBackendValidationError",
    "FlexAttentionBackend",
    "FlexAttentionRuntime",
    "GpuFlashAttentionBackend",
    "NpuFlashAttentionBackend",
    "PrefixAttentionBackend",
    "PrefixAttentionExecutionMode",
    "TorchReferenceBackend",
    "build_block_causal_mask",
    "get_backend_instance",
    "requires_expanded_kv",
]
