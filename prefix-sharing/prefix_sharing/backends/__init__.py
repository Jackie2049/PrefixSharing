"""Attention backend adapters."""

from prefix_sharing.backends.base import BackendCapabilities, PrefixAttentionBackend, PrefixDeltanetBackend
from prefix_sharing.backends.block_causal_mask import build_block_causal_mask
from prefix_sharing.backends.factory import get_backend_instance
from prefix_sharing.backends.flash_atten_base import FlashAttentionMixin, FlashBackendValidationError
from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend
from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend
from prefix_sharing.backends.torch_ref import TorchReferenceBackend

from prefix_sharing.backends.flex_atten_gpu import GpuFlexAttentionBackend
from prefix_sharing.backends.prefix_block_mask import (
    build_chain_limit_matrix,
    build_token_index_tensors,
    get_or_create_block_mask,
    make_prefix_sharing_mask_mod,
)

__all__ = [
    "BackendCapabilities",
    "FlashAttentionMixin",
    "FlashBackendValidationError",
    "GpuFlexAttentionBackend",
    "GpuFlashAttentionBackend",
    "NpuFlashAttentionBackend",
    "PrefixAttentionBackend",
    "PrefixDeltanetBackend",
    "TorchReferenceBackend",
    "build_block_causal_mask",
    "build_chain_limit_matrix",
    "build_token_index_tensors",
    "get_backend_instance",
    "get_or_create_block_mask",
    "make_prefix_sharing_mask_mod",
]
