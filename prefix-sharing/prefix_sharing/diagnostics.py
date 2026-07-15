"""Shared diagnostic-dump helpers for PrefixSharing.

Environment variables that affect runtime behavior:
- PREFIX_SHARING_DIAG_DUMP=/path/to/dump_dir: enables tensor dumps
- PREFIX_SHARING_AUDIT=1: enables per-micro-batch audit summary
"""

from __future__ import annotations

import os
from typing import Any

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FSDP_ATTN_BUFFER: dict[int, Any] = {}
_FSDP_ATTN_INPUT_BUFFER: dict[int, dict[str, Any]] = {}
_FSDP_EXPANDED_KV_BUFFER: dict[int, dict[str, Any]] = {}


def env_truthy(name: str) -> bool:
    """Check whether env var ``name`` is set to a truthy value."""
    return os.getenv(name, "").strip().lower() in _TRUE_VALUES


def audit_enabled() -> bool:
    return env_truthy("PREFIX_SHARING_AUDIT")


def diagnostic_dump_enabled() -> bool:
    return os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None


def dump_fsdp_attn_output(output: Any, module: Any) -> None:
    """Dump FSDP attention outputs when ``PREFIX_SHARING_DIAG_DUMP`` is set.

    The wrapper call sites should stay small and side-effect-free when dump is
    disabled.  This helper owns the per-forward layer buffer and rank-0 file
    write policy.

    Buffers accumulate all 24 layers during the first forward and are NOT
    cleared by recompute (checkpoint).  Subsequent forward calls append to
    ``_FSDP_ATTN_BUFFER_FULL`` if the first full sequence already exists,
    so that checkpoint recompute does not overwrite the diagnostic data.
    """

    if not diagnostic_dump_enabled():
        return

    import torch

    from prefix_sharing.tools.diagnostic_dump import _get_dump_dir, _rank0_only

    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    if isinstance(output, tuple):
        output = output[0]
    if not hasattr(output, "dim") or output.dim() < 3:
        return

    layer_number = int(getattr(module, "layer_idx", 0) or 0) + 1
    num_layers = int(getattr(getattr(module, "config", None), "num_hidden_layers", 0) or 0)
    if num_layers == 0:
        return

    hidden = output.shape[-1] * output.shape[-2]
    out_2d = output.reshape(-1, hidden).detach().cpu().contiguous()
    if layer_number == 1:
        _FSDP_ATTN_BUFFER.clear()
    _FSDP_ATTN_BUFFER[layer_number] = out_2d
    if layer_number == num_layers:
        # Save ONLY on the FIRST full forward; recompute / repeated forwards
        # skip saving so that the initial diagnostic data is not overwritten.
        # Use a module-level flag tracked by the global buffer status:
        if not getattr(_FSDP_ATTN_BUFFER, "_saved_once", False):
            if _rank0_only():
                torch.save(_FSDP_ATTN_BUFFER, os.path.join(dump_dir, "attn_outputs.pt"))
            _FSDP_ATTN_BUFFER._saved_once = True
        _FSDP_ATTN_BUFFER.clear()


def dump_fsdp_attention_inputs(query: Any, key: Any, value: Any, module: Any) -> None:
    """Dump post-RoPE attention inputs per layer for first-divergence analysis.

    HF calls its attention interface after rotary embedding, so these tensors
    isolate the Q/K/V values actually consumed by prefix sharing.  The helper
    is diagnostic-only: it is a no-op unless ``PREFIX_SHARING_DIAG_DUMP`` is
    set and writes once at the last layer of a forward.

    Buffers accumulate all 24 layers during the first forward and are NOT
    cleared by recompute (checkpoint).  Subsequent forward calls append to
    ``_FSDP_ATTN_INPUT_BUFFER_FULL`` if the first full sequence already exists,
    so that checkpoint recompute does not overwrite the diagnostic data.
    """
    if not diagnostic_dump_enabled():
        return

    import torch

    from prefix_sharing.tools.diagnostic_dump import _get_dump_dir, _rank0_only

    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    layer_number = int(getattr(module, "layer_idx", 0) or 0) + 1
    num_layers = int(getattr(getattr(module, "config", None), "num_hidden_layers", 0) or 0)
    if num_layers == 0:
        return
    if layer_number == 1:
        _FSDP_ATTN_INPUT_BUFFER.clear()
    _FSDP_ATTN_INPUT_BUFFER[layer_number] = {
        "query": query.detach().cpu().contiguous(),
        "key": key.detach().cpu().contiguous(),
        "value": value.detach().cpu().contiguous(),
    }
    if layer_number == num_layers:
        if not getattr(_FSDP_ATTN_INPUT_BUFFER, "_saved_once", False):
            if _rank0_only():
                torch.save(_FSDP_ATTN_INPUT_BUFFER, os.path.join(dump_dir, "attn_inputs.pt"))
            _FSDP_ATTN_INPUT_BUFFER._saved_once = True
        _FSDP_ATTN_INPUT_BUFFER.clear()


def dump_fsdp_expanded_kv(
    key: Any,
    value: Any,
    *,
    layer_id: int,
    num_layers: int,
) -> None:
    """Dump ON expanded K/V after store/load, before attention consumes them."""
    if not diagnostic_dump_enabled() or num_layers <= 0:
        return

    import torch

    from prefix_sharing.tools.diagnostic_dump import _get_dump_dir, _rank0_only

    layer_number = layer_id + 1
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    if layer_number == 1:
        _FSDP_EXPANDED_KV_BUFFER.clear()
    _FSDP_EXPANDED_KV_BUFFER[layer_number] = {
        "key": key.detach().cpu().contiguous(),
        "value": value.detach().cpu().contiguous(),
    }
    if layer_number == num_layers:
        if not getattr(_FSDP_EXPANDED_KV_BUFFER, "_saved_once", False):
            if _rank0_only():
                torch.save(_FSDP_EXPANDED_KV_BUFFER, os.path.join(dump_dir, "expanded_kv.pt"))
            _FSDP_EXPANDED_KV_BUFFER._saved_once = True
        _FSDP_EXPANDED_KV_BUFFER.clear()
