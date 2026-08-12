"""Patch: HF attention functions — intercept for PrefixSharing + diagnostic dump.

Each attention function in ``ALL_ATTENTION_FUNCTIONS`` is wrapped so that when
a PrefixSharing runtime context is active, the attention is routed to
``PrefixSharingFSDPAttentionRuntime``.  When inactive, the original function is
called transparently (zero overhead beyond one ``ContextVar.get()``).

Activation via ``PREFIX_SHARING_DIAG_DUMP`` env var enables per-layer attention
output capture to ``attn_outputs.pt`` for both ON and OFF paths.

Exported:
    create_attention_wrapper — wrap one HF attention function
    install_attention_patch    — iterate ALL_ATTENTION_FUNCTIONS and apply
"""

from __future__ import annotations

import os
from typing import Any

from prefix_sharing.tools.perf_profiler import PerfProfiler

_SUPPORTED_ATTENTIONS = {
    "flash_attention_2",
    "flash_attention_3",
    "sdpa",
    "flex_attention",
    # "eager",  # replaced by eager_paged in this transformers version
}

# ##### [PS-diag] dump helpers ######

def _resolve_num_layers(module: Any) -> int:
    """Infer the total number of model layers from the attention module, with
    a fallback to ``module.model.config``.

    ``_dump_attn_output`` uses the same fallback logic; extracted here as a
    shared utility function.
    """
    num_layers = int(getattr(getattr(module, "config", None), "num_hidden_layers", 0) or 0)
    if num_layers == 0:
        root_config = getattr(getattr(module, "model", None), "config", None)
        num_layers = int(getattr(root_config, "num_hidden_layers", 0) or 0)
    return num_layers


def _pack_off_dense_for_dump(tensor: Any) -> Any:
    """OFF path: reshape dense [B,H,L,D] → [T,H,D] to match the packed input
    convention of the dump functions.
    """
    import torch as _torch
    B, H, L, D = tensor.shape
    return tensor.transpose(1, 2).reshape(_torch.Size([B * L, H, D]))


def _dump_off_rope_and_kv(module: Any, query: Any, key: Any, value: Any) -> None:
    layer_number = int(getattr(module, "layer_idx", 0) or 0) + 1
    num_layers = _resolve_num_layers(module)
    if num_layers == 0:
        return
    from prefix_sharing.tools.diagnostic_dump import dump_build_kv_input_v_on, dump_rope_postqk_verl080
    dump_rope_postqk_verl080(layer_number, _pack_off_dense_for_dump(query), _pack_off_dense_for_dump(key), num_layers)
    dump_build_kv_input_v_on(layer_number, _pack_off_dense_for_dump(value), num_layers)


def _dump_attn_output(output: Any, module: Any) -> None:
    """Thin wrapper: extract layer_number / num_layers from *module*, delegate to
    ``diagnostic_dump.dump_fsdp_attn_output`` for accumulation and flush.
    """
    layer_number = int(getattr(module, "layer_idx", 0) or 0) + 1  # 1-based
    num_layers = _resolve_num_layers(module)
    if num_layers == 0:
        return

    from prefix_sharing.tools.diagnostic_dump import dump_fsdp_attn_output
    dump_fsdp_attn_output(output, layer_number, num_layers)

    # ── backward gradient hook is registered in forward_step.py
    # (once per forward, survives AC recompute) ──

# ##### [PS-diag] end #####


def create_attention_wrapper(original_fn: Any) -> Any:
    """Wrap a single HF attention function with PrefixSharing support.

    *original_fn* has signature ``(module, query, key, value, attention_mask, ...)``
    and returns ``(attn_output, attn_weights_or_None)``.
    """

    def patched_attention(module: Any, query: Any, key: Any, value: Any,
                          attention_mask: Any, *args: Any, **kwargs: Any) -> Any:
        # Prefer module attribute (AC recompute compatible), fall back to ContextVar
        # (Megatron and other paths)
        ctx = getattr(module, '_ps_ctx', None)
        if ctx is None:
            from prefix_sharing.integrations.context import current_prefix_sharing_context
            ctx = current_prefix_sharing_context()

        # ── OFF path: no prefix sharing context → transparent passthrough ──
        if ctx is None:
            # [PS-perf] start — OFF attention timing (cross-layer + per-layer) —
            profiler = PerfProfiler.current()
            _per_layer_ok = profiler is not None and getattr(profiler, "per_layer_enabled", False)
            _off_layer_id = int(getattr(module, "layer_idx", 0) or 0)
            if profiler is not None:
                profiler.start_phase(PerfProfiler.PHASE_ATTN_OFF)
            if _per_layer_ok:
                profiler.start_phase(f"attn.off.l{_off_layer_id}")
            try:
                result = original_fn(module, query, key, value, attention_mask, *args, **kwargs)
            finally:
                if _per_layer_ok:
                    _off_elapsed = profiler.stop_phase(f"attn.off.l{_off_layer_id}")
                    profiler.record_per_layer(_off_layer_id, PerfProfiler.PHASE_ATTN_OFF, _off_elapsed)
                if profiler is not None:
                    profiler.stop_phase(PerfProfiler.PHASE_ATTN_OFF)
            # [PS-perf] end ——————————————————————————————————————

            # ##### [PS-diag] OFF per-layer dump (baseline / context inactive) #####
            if os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
                _dump_attn_output(result, module)
                _dump_off_rope_and_kv(module, query, key, value)
            # ##### [PS-diag] end #####
            return result

        # ── ON path: route through PrefixSharing attention runtime ──
        from prefix_sharing.integrations.verl_fsdp import PrefixSharingFSDPAttentionRuntime
        from prefix_sharing.integrations.context import _current_context

        layer_id = int(getattr(module, "layer_idx", 0) or 0)
        _num_layers = _resolve_num_layers(module)
        runtime = PrefixSharingFSDPAttentionRuntime(layer_id=layer_id, num_layers=_num_layers)

        # HF attention interface expects [B, H, L, D]; runtime works in [B, L, H, D]
        query_ld = query.transpose(1, 2)
        key_ld = key.transpose(1, 2)
        value_ld = value.transpose(1, 2)

        # runtime.forward() reads ContextVar internally; during AC recompute the
        # ContextVar may have expired, but ctx from module._ps_ctx is still valid.
        # Temporarily inject the ContextVar.
        _ctxvar_token = _current_context.set(ctx)
        # [PS-perf] start — ON attention timing (attn.on = pack+kv+comp+unpack) —
        profiler = PerfProfiler.current()
        _per_layer_ok = profiler is not None and getattr(profiler, "per_layer_enabled", False)
        if profiler is not None:
            profiler.start_phase(PerfProfiler.PHASE_ATTN_ON)
        if _per_layer_ok:
            profiler.start_phase(f"attn.on.l{layer_id}")
        try:
            output_ld = runtime.forward(None, query_ld, key_ld, value_ld)
        finally:
            if _per_layer_ok:
                _on_elapsed = profiler.stop_phase(f"attn.on.l{layer_id}")
                profiler.record_per_layer(layer_id, PerfProfiler.PHASE_ATTN_ON, _on_elapsed)
            if profiler is not None:
                profiler.stop_phase(PerfProfiler.PHASE_ATTN_ON)
            _current_context.reset(_ctxvar_token)
        # [PS-perf] end ————————————————————————————————————————

        # ##### [PS-diag] ON attn output dump (context active = PS path) #####
        if os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
            _dump_attn_output(output_ld, module)
        return output_ld, None

    return patched_attention


def install_attention_patch() -> None:
    """Directly modify ALL_ATTENTION_FUNCTIONS dict — same approach as verl's PrefixGrouper."""
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:
        print("[PS] transformers not available, skipping attention patch")
        return

    patched = []
    for name in list(ALL_ATTENTION_FUNCTIONS.keys()):
        if name in _SUPPORTED_ATTENTIONS:
            ALL_ATTENTION_FUNCTIONS[name] = create_attention_wrapper(
                ALL_ATTENTION_FUNCTIONS[name]
            )
            patched.append(name)

    if patched:
        print(f"[PS] Attention patch installed on: {patched}")
