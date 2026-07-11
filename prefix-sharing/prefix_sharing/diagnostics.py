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
        if _rank0_only():
            torch.save(_FSDP_ATTN_BUFFER, os.path.join(dump_dir, "attn_outputs.pt"))
        _FSDP_ATTN_BUFFER.clear()
