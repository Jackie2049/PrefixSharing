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

from typing import Any


def patch_transformers_attention(original_get_interface: Any) -> Any:
    """创建 ``ALL_ATTENTION_FUNCTIONS.get_interface`` 的 PS-aware wrapper。"""

    def ps_aware_get_interface(attn_implementation: str, default: Any = None) -> Any:
        original_fn = original_get_interface(attn_implementation, default)

        def patched_attention(module: Any, query: Any, key: Any, value: Any,
                              attention_mask: Any, *args: Any, **kwargs: Any) -> Any:
            from prefix_sharing.integrations.context import current_prefix_sharing_context

            ctx = current_prefix_sharing_context()
            if ctx is None:
                return original_fn(module, query, key, value, attention_mask, *args, **kwargs)

            from prefix_sharing.integrations.verl_fsdp import PrefixSharingFSDPAttentionRuntime

            layer_id = int(getattr(module, "layer_idx", 0) or 0)
            runtime = PrefixSharingFSDPAttentionRuntime(layer_id=layer_id)
            query_ld = query.transpose(1, 2)
            key_ld = key.transpose(1, 2)
            value_ld = value.transpose(1, 2)
            output_ld = runtime.forward(None, query_ld, key_ld, value_ld)
            # HF attention_interface 接收 [B,H,L,D] Q 但返回 [B,L,H,D] 输出
            # （Qwen2Attention 用 attn_output.reshape(*input_shape, -1) 验证）。
            # runtime 在 [B,L,H,D] 空间工作，输出已是 [B,L,H,D]，无需再 transpose。
            return output_ld, None

        return patched_attention

    return ps_aware_get_interface
