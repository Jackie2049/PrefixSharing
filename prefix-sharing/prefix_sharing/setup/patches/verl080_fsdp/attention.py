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

_SUPPORTED_ATTENTIONS = {
    "flash_attention_2",
    "flash_attention_3",
    "sdpa",
    "flex_attention",
    # "eager",  # replaced by eager_paged in this transformers version
}


# ##### [PS-diag] dump helpers ######

def _resolve_num_layers(module: Any) -> int:
    """从 attention module 推导模型总层数，有 ``module.model.config`` 回退。

    ``_dump_attn_output`` 也用了同样的回退逻辑，此处抽取为公共函数。
    """
    num_layers = int(getattr(getattr(module, "config", None), "num_hidden_layers", 0) or 0)
    if num_layers == 0:
        root_config = getattr(getattr(module, "model", None), "config", None)
        num_layers = int(getattr(root_config, "num_hidden_layers", 0) or 0)
    return num_layers


def _pack_off_dense_for_dump(tensor: Any) -> Any:
    """OFF 路径：将 dense [B,H,L,D] → [T,H,D] 以匹配 dump 函数的 packed 入参约定。"""
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
        # 优先 module 属性（AC recompute 兼容），回退 ContextVar（Megatron 等路径）
        ctx = getattr(module, '_ps_ctx', None)
        if ctx is None:
            from prefix_sharing.integrations.context import current_prefix_sharing_context
            ctx = current_prefix_sharing_context()

        # ── OFF path: no prefix sharing context → transparent passthrough ──
        if ctx is None:
            result = original_fn(module, query, key, value, attention_mask, *args, **kwargs)
            # ##### [PS-diag] OFF per-layer dump (baseline / context 不激活) #####
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

        # runtime.forward() 内部读 ContextVar；AC recompute 时 ContextVar
        # 可能已过期，但 ctx 来自 module._ps_ctx 仍然有效。临时注入 ContextVar。
        _ctxvar_token = _current_context.set(ctx)
        try:
            output_ld = runtime.forward(None, query_ld, key_ld, value_ld)
        finally:
            _current_context.reset(_ctxvar_token)

        # ##### [PS-diag] ON attn output dump（context 激活 = PS 路径） #####
        if os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
            _dump_attn_output(output_ld, module)
        # ##### [PS-diag] end #####
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
