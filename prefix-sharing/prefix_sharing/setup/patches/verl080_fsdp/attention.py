"""Patch: transformers ``ALL_ATTENTION_FUNCTIONS.get_interface`` — HF attention 拦截。

当 PrefixSharing runtime context 激活时，把 HF attention 的 Q/K/V 路由到
``PrefixSharingFSDPAttentionRuntime``（执行 KV store/load + expanded KV + attention）。
context 不激活时透传原 attention，零开销（仅一次 ContextVar 查询）。

兼容 GQA（Q 头数 != KV 头数）：runtime 按 [B,L] 对齐，head 维度可不同；
HF 调用 attention_interface 时 Q/K/V 形态为 [B,H,L,D]，这里转置为 [B,L,H,D]
喂给 runtime。注意：HF 的 attention_interface 返回值是 [B,L,H,D]（Qwen2Attention
随后用 ``attn_output.reshape(*input_shape, -1)`` 直接 reshape，不再 transpose），
而 runtime 恰好在 [B,L,H,D] 空间工作，因此输出无需再转置，直接返回即可。
"""

from __future__ import annotations

import os
from typing import Any


def _dump_attn_output(output: Any, module: Any) -> None:
    """Thin wrapper: extract layer_number / num_layers from *module*, delegate to
    ``diagnostic_dump.dump_fsdp_attn_output`` for accumulation and flush.
    """
    layer_number = int(getattr(module, "layer_idx", 0) or 0) + 1  # 1-based

    # Try module.config first; under FSDP wrapping, fall back to the root model config
    num_layers = int(getattr(getattr(module, "config", None), "num_hidden_layers", 0) or 0)
    if num_layers == 0:
        # FSDP may wrap the HF module — try to reach config via the root model
        root_config = getattr(getattr(module, "model", None), "config", None)
        num_layers = int(getattr(root_config, "num_hidden_layers", 0) or 0)
    if num_layers == 0:
        return

    from prefix_sharing.tools.diagnostic_dump import dump_fsdp_attn_output
    dump_fsdp_attn_output(output, layer_number, num_layers)


def patch_transformers_attention(original_get_interface: Any) -> Any:
    """创建 ``ALL_ATTENTION_FUNCTIONS.get_interface`` 的 PS-aware wrapper。"""

    def ps_aware_get_interface(attn_implementation: str, default: Any = None) -> Any:
        original_fn = original_get_interface(attn_implementation, default)

        def patched_attention(module: Any, query: Any, key: Any, value: Any,
                              attention_mask: Any, *args: Any, **kwargs: Any) -> Any:
            from prefix_sharing.integrations.context import current_prefix_sharing_context

            ctx = current_prefix_sharing_context()
            if ctx is None:
                result = original_fn(module, query, key, value, attention_mask, *args, **kwargs)
                # ##### [PS-diag] OFF attn output dump（context 不激活 = baseline） #####
                if os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
                    _dump_attn_output(result, module)
                # ##### [PS-diag] end #####
                return result

            from prefix_sharing.integrations.verl_fsdp import PrefixSharingFSDPAttentionRuntime

            layer_id = int(getattr(module, "layer_idx", 0) or 0)
            runtime = PrefixSharingFSDPAttentionRuntime(layer_id=layer_id)
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

    return ps_aware_get_interface
