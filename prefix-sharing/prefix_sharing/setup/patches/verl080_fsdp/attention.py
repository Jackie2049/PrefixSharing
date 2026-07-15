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


def _dump_attn_output(output: Any, module: Any) -> None:
    """Thin wrapper: extract layer_number / num_layers from *module*, delegate to
    ``diagnostic_dump.dump_fsdp_attn_output`` for accumulation and flush.
    """
    layer_number = int(getattr(module, "layer_idx", 0) or 0) + 1  # 1-based
    num_layers = int(getattr(getattr(module, "config", None), "num_hidden_layers", 0) or 0)
    if num_layers == 0:
        root_config = getattr(getattr(module, "model", None), "config", None)
        num_layers = int(getattr(root_config, "num_hidden_layers", 0) or 0)
    if num_layers == 0:
        return

    from prefix_sharing.tools.diagnostic_dump import dump_fsdp_attn_output
    dump_fsdp_attn_output(output, layer_number, num_layers)


def create_attention_wrapper(original_fn: Any) -> Any:
    """Wrap a single HF attention function with PrefixSharing support.

    *original_fn* has signature ``(module, query, key, value, attention_mask, ...)``
    and returns ``(attn_output, attn_weights_or_None)``.
    """

    def patched_attention(module: Any, query: Any, key: Any, value: Any,
                          attention_mask: Any, *args: Any, **kwargs: Any) -> Any:
        from prefix_sharing.integrations.context import current_prefix_sharing_context

        ctx = current_prefix_sharing_context()

        # ── OFF path: no prefix sharing context → transparent passthrough ──
        if ctx is None:
            result = original_fn(module, query, key, value, attention_mask, *args, **kwargs)
            # ##### [PS-diag] OFF attn output dump（context 不激活 = baseline） #####
            if os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
                _dump_attn_output(result, module)
            # ##### [PS-diag] end #####
            return result

        # ── ON path: route through PrefixSharing attention runtime ──
        from prefix_sharing.integrations.verl_fsdp import PrefixSharingFSDPAttentionRuntime

        layer_id = int(getattr(module, "layer_idx", 0) or 0)
        runtime = PrefixSharingFSDPAttentionRuntime(layer_id=layer_id)

        # HF attention interface expects [B, H, L, D]; runtime works in [B, L, H, D]
        query_ld = query.transpose(1, 2)
        key_ld = key.transpose(1, 2)
        value_ld = value.transpose(1, 2)
        output_ld = runtime.forward(None, query_ld, key_ld, value_ld)

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
