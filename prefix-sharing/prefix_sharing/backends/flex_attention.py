"""PyTorch FlexAttention backend for deduplicated prefix-sharing Q/K/V.

This backend deliberately does not implement ``build_kv``.  Its input K/V are
the original deduplicated packed tensors and a ``PrefixTreeAttentionLayout``
provides sparse visibility.  The Flex API is imported lazily so existing
CPU/NPU and expanded-KV users do not acquire a CUDA import dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from prefix_sharing.backends.base import BackendCapabilities, PrefixAttentionExecutionMode
from prefix_sharing.backends.kv_builder import apply_rope_with_plan
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.attention_layout import PrefixAttentionMaskType, PrefixTreeAttentionLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan


class FlexAttentionValidationError(RuntimeError):
    """Raised when the deduplicated Flex execution contract is not satisfied."""


@dataclass(frozen=True)
class FlexAttentionRuntime:
    """Per-forward immutable Flex metadata, intended for cross-layer reuse."""

    layout_signature: tuple[object, ...]
    device: str
    block_size: int
    block_mask: Any


@lru_cache(maxsize=1)
def _import_flex_attention_api() -> tuple[Any, Any]:
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    except (ImportError, ModuleNotFoundError) as exc:
        raise FlexAttentionValidationError(
            "backend='flex_attention' requires a PyTorch build that provides "
            "torch.nn.attention.flex_attention"
        ) from exc
    return create_block_mask, flex_attention


class FlexAttentionBackend:
    """CUDA FlexAttention backend consuming deduplicated packed Q/K/V."""

    capabilities = BackendCapabilities(
        name="flex_attention",
        supports_cpu=False,
        supports_cuda=True,
        supports_cann=False,
        supports_different_q_kv_lengths=False,
        supports_prefix_last_restore=True,
        execution_mode=PrefixAttentionExecutionMode.DEDUPLICATED_QKV,
    )

    def __init__(self, *, block_size: int = 128, compile_attention: bool = True) -> None:
        self.block_size = int(block_size)
        self.compile_attention = bool(compile_attention)
        self._compiled_attention: Any | None = None

    def validate(self, config: PrefixSharingConfig, model_config: Any | None = None) -> None:
        config.validate(model_config=model_config, integrate_mode="verl_fsdp")
        if config.backend != "flex_attention":
            raise FlexAttentionValidationError(
                "FlexAttentionBackend requires config.backend='flex_attention'"
            )
        if self.block_size not in {64, 128, 256}:
            raise FlexAttentionValidationError("FlexAttention block_size must be 64, 128, or 256")
        _import_flex_attention_api()

    def apply_rope(
        self,
        query: Any,
        key: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        return apply_rope_with_plan(query, key, prefix_sharing_plan, **kwargs)

    def prepare_runtime(
        self,
        *,
        prefix_tree_attention_layout: PrefixTreeAttentionLayout,
        packed_batch_layout: PackedBatchLayout,
        device: Any,
    ) -> FlexAttentionRuntime:
        """Build one BlockMask for all layers of the current model forward."""

        torch = _import_torch()
        if getattr(device, "type", None) != "cuda":
            raise FlexAttentionValidationError(
                "backend='flex_attention' currently requires CUDA tensors"
            )
        if packed_batch_layout.has_padding:
            raise FlexAttentionValidationError(
                "backend='flex_attention' does not support packed TP padding; "
                "expected padded_lengths == valid_lengths"
            )
        if packed_batch_layout.total_padded_length != prefix_tree_attention_layout.total_tokens:
            raise FlexAttentionValidationError(
                "packed token length does not match PrefixTreeAttentionLayout total_tokens"
            )

        token_nodes, token_positions, visible_ends = _build_device_metadata(
            prefix_tree_attention_layout,
            torch=torch,
            device=device,
        )

        def mask_mod(batch: Any, head: Any, query_index: Any, key_index: Any) -> Any:
            del batch, head
            query_node = token_nodes[query_index]
            key_node = token_nodes[key_index]
            query_position = token_positions[query_index]
            key_position = token_positions[key_index]
            same_node = query_node == key_node
            causal_visible = same_node & (key_position <= query_position)
            ancestor_visible = (~same_node) & (
                key_position < visible_ends[key_node, query_node]
            )
            return causal_visible | ancestor_visible

        create_block_mask, _ = _import_flex_attention_api()
        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=prefix_tree_attention_layout.total_tokens,
            KV_LEN=prefix_tree_attention_layout.total_tokens,
            device=device,
            BLOCK_SIZE=self.block_size,
        )
        return FlexAttentionRuntime(
            layout_signature=prefix_tree_attention_layout.signature,
            device=str(device),
            block_size=self.block_size,
            block_mask=block_mask,
        )

    def attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: PackedBatchLayout | None = None,
        prefix_tree_attention_layout: PrefixTreeAttentionLayout | None = None,
        runtime: FlexAttentionRuntime | None = None,
        **_: Any,
    ) -> Any:
        del prefix_sharing_plan
        if packed_batch_layout is None:
            raise FlexAttentionValidationError("FlexAttention requires PackedBatchLayout")
        if runtime is None:
            if prefix_tree_attention_layout is None:
                raise FlexAttentionValidationError(
                    "FlexAttention requires PrefixTreeAttentionLayout or prepared runtime"
                )
            runtime = self.prepare_runtime(
                prefix_tree_attention_layout=prefix_tree_attention_layout,
                packed_batch_layout=packed_batch_layout,
                device=query.device,
            )
        if query.shape[0 if query.dim() == 3 else 1] != packed_batch_layout.total_padded_length:
            raise FlexAttentionValidationError(
                "FlexAttention Q token length does not match PackedBatchLayout"
            )
        if key.shape != value.shape:
            raise FlexAttentionValidationError("FlexAttention key and value shapes must match")

        query_4d, query_shape = _to_flex_layout(query, "query")
        key_4d, _ = _to_flex_layout(key, "key")
        value_4d, _ = _to_flex_layout(value, "value")
        if query_4d.shape[1] % key_4d.shape[1] != 0:
            raise FlexAttentionValidationError(
                "FlexAttention requires query heads to be divisible by KV heads"
            )

        attention_fn = self._attention_function()
        try:
            output = attention_fn(
                query_4d,
                key_4d,
                value_4d,
                block_mask=runtime.block_mask,
                enable_gqa=query_4d.shape[1] != key_4d.shape[1],
            )
        except Exception as exc:
            raise FlexAttentionValidationError(
                "flex_attention execution failed for "
                f"q_shape={tuple(query.shape)}, k_shape={tuple(key.shape)}, device={query.device}"
            ) from exc
        return _from_flex_layout(output, query_shape)

    def _attention_function(self) -> Any:
        _, flex_attention = _import_flex_attention_api()
        if not self.compile_attention:
            return flex_attention
        if self._compiled_attention is None:
            torch = _import_torch()
            if not hasattr(torch, "compile"):
                raise FlexAttentionValidationError(
                    "backend='flex_attention' with flex_attention_compile=True requires torch.compile"
                )
            self._compiled_attention = torch.compile(flex_attention)
        return self._compiled_attention


def _build_device_metadata(
    layout: PrefixTreeAttentionLayout,
    *,
    torch: Any,
    device: Any,
) -> tuple[Any, Any, Any]:
    """Encode tree slices as O(T + nodes^2) device metadata for ``mask_mod``."""

    node_count = len(layout.node_ranges)
    token_nodes = torch.empty(layout.total_tokens, dtype=torch.long, device=device)
    token_positions = torch.empty(layout.total_tokens, dtype=torch.long, device=device)
    for node_index, ((start, end), position_offset) in enumerate(
        zip(layout.node_ranges, layout.node_position_offsets)
    ):
        token_nodes[start:end] = node_index
        token_positions[start:end] = torch.arange(
            position_offset,
            position_offset + (end - start),
            dtype=torch.long,
            device=device,
        )

    visible_ends = torch.full((node_count, node_count), -1, dtype=torch.long, device=device)
    range_to_node = {node_range: index for index, node_range in enumerate(layout.node_ranges)}
    node_start_to_index = {start: index for index, (start, _) in enumerate(layout.node_ranges)}
    for slice_ in layout.attention_slices:
        if slice_.mask_type is not PrefixAttentionMaskType.FULL:
            continue
        query_node = range_to_node[(slice_.query_start, slice_.query_end)]
        key_node = node_start_to_index[slice_.key_start]
        key_offset = layout.node_position_offsets[key_node]
        visible_ends[key_node, query_node] = key_offset + (slice_.key_end - slice_.key_start)
    return token_nodes, token_positions, visible_ends


def _to_flex_layout(tensor: Any, name: str) -> tuple[Any, tuple[int, ...]]:
    if tensor.dim() == 3:
        return tensor.transpose(0, 1).unsqueeze(0).contiguous(), tuple(tensor.shape)
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        return tensor.transpose(1, 2).contiguous(), tuple(tensor.shape)
    raise FlexAttentionValidationError(
        f"FlexAttention {name} must be [T, H, D] or [1, T, H, D], got {tuple(tensor.shape)}"
    )


def _from_flex_layout(tensor: Any, original_shape: tuple[int, ...]) -> Any:
    if len(original_shape) == 3:
        return tensor.squeeze(0).transpose(0, 1).contiguous()
    return tensor.transpose(1, 2).contiguous()


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise FlexAttentionValidationError("backend='flex_attention' requires PyTorch") from exc
    return torch


__all__ = [
    "FlexAttentionBackend",
    "FlexAttentionRuntime",
    "FlexAttentionValidationError",
]
