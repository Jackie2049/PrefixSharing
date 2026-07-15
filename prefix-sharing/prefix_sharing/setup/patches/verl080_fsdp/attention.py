"""Patch the concrete HF attention registry entries used by verl Qwen2.

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

from prefix_sharing.diagnostics import dump_fsdp_attn_output

# per-forward 累积每层 attention 输出，最后一层 flush 成 attn_outputs.pt。
# layer_number == 1 时清空（新 forward 起点），== num_layers 时存盘。
# 与 cmp_diag_verl080.cmp_attn_layer 约定一致：dict {layer_1based: tensor[N, hidden]}。

def patch_transformers_attention(original_fn: Any) -> Any:
    """Wrap one concrete attention implementation from the HF registry."""

    def patched_attention(module: Any, query: Any, key: Any, value: Any,
                          attention_mask: Any, *args: Any, **kwargs: Any) -> Any:
        from prefix_sharing.integrations.context import current_prefix_sharing_context

        ctx = current_prefix_sharing_context()
        if os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
            from prefix_sharing.diagnostics import dump_fsdp_attention_inputs

            dump_fsdp_attention_inputs(query, key, value, module)
        if ctx is None:
            result = original_fn(module, query, key, value, attention_mask, *args, **kwargs)
            if os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
                dump_fsdp_attn_output(result, module)
            return result

        from prefix_sharing.integrations.verl_fsdp import PrefixSharingFSDPAttentionRuntime

        layer_id = int(getattr(module, "layer_idx", 0) or 0)
        num_layers = int(getattr(getattr(module, "config", None), "num_hidden_layers", 0) or 0)
        runtime = PrefixSharingFSDPAttentionRuntime(layer_id=layer_id, num_layers=num_layers)
        output_ld = runtime.forward(
            None,
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        if os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
            dump_fsdp_attn_output(output_ld, module)
        return output_ld, None

    return patched_attention


def install_prefix_sharing_attention_wrappers(attention_functions: Any, manager: Any) -> None:
    """Wrap registry values directly, matching verl PrefixGrouper's strategy.

    Qwen2 reads ``ALL_ATTENTION_FUNCTIONS[name]`` during every forward.  We
    therefore replace the actual mapping entries instead of patching mapping
    lookup mechanics.  ``LoggedPatchManager`` records every replacement so a
    patch handle can restore the registry exactly.
    """
    for name in list(attention_functions.keys()):
        original_fn = attention_functions[name]
        if getattr(original_fn, "_prefix_sharing_attention_wrapper", False):
            continue
        patched_fn = patch_transformers_attention(original_fn)
        patched_fn._prefix_sharing_attention_wrapper = True
        manager.patch_item(attention_functions, name, patched_fn)
