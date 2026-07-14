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

from prefix_sharing.diagnostics import dump_fsdp_attn_output

# per-forward 累积每层 attention 输出，最后一层 flush 成 attn_outputs.pt。
# layer_number == 1 时清空（新 forward 起点），== num_layers 时存盘。
# 与 cmp_diag_verl080.cmp_attn_layer 约定一致：dict {layer_1based: tensor[N, hidden]}。

def patch_transformers_attention(original_getitem: Any) -> Any:
    """创建 ``ALL_ATTENTION_FUNCTIONS.get_interface`` 的 PS-aware wrapper。

    ``ALL_ATTENTION_FUNCTIONS`` 是继承 ``MutableMapping`` 的 GeneralInterface 实例，
    调用 ``.get_interface(attn_implementation)`` 等价于 ``ALL_ATTENTION_FUNCTIONS[attn_implementation]``。
    因此我们 patch ``__getitem__`` 来拦截所有 attention 接口查找。
    """

    import logging as _ps_diag_log
    _ps_diag_log.basicConfig(level=_ps_diag_log.INFO,
                             format='%(asctime)s [PS-diag] %(message)s',
                             datefmt='%H:%M:%S')

    def ps_aware_get_interface(attn_implementation: str, default: Any = None) -> Any:
        original_fn = original_getitem(attn_implementation) if default is None else original_getitem(attn_implementation)

        def patched_attention(module: Any, query: Any, key: Any, value: Any,
                              attention_mask: Any, *args: Any, **kwargs: Any) -> Any:
            from prefix_sharing.integrations.context import current_prefix_sharing_context

            _ps_diag_log.info("patched_attention called module=%s has_layer_idx=%s has_config=%s n_layers=%s",
                              type(module).__name__,
                              hasattr(module, "layer_idx"),
                              hasattr(module, "config"),
                              getattr(getattr(module, "config", None), "num_hidden_layers", "MISSING"))

            ctx = current_prefix_sharing_context()
            if os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
                from prefix_sharing.diagnostics import dump_fsdp_attention_inputs

                _ps_diag_log.info("DIAG: calling dump_fsdp_attention_inputs")
                dump_fsdp_attention_inputs(query, key, value, module)
            if ctx is None:
                result = original_fn(module, query, key, value, attention_mask, *args, **kwargs)
                # ##### [PS-diag] OFF attn output dump（context 不激活 = baseline） #####
                if os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
                    _ps_diag_log.info("DIAG: calling dump_fsdp_attn_output (OFF)")
                    dump_fsdp_attn_output(result, module)
                # ##### [PS-diag] end #####
                return result

            from prefix_sharing.integrations.verl_fsdp import PrefixSharingFSDPAttentionRuntime

            layer_id = int(getattr(module, "layer_idx", 0) or 0)
            num_layers = int(getattr(getattr(module, "config", None), "num_hidden_layers", 0) or 0)
            runtime = PrefixSharingFSDPAttentionRuntime(
                layer_id=layer_id,
                num_layers=num_layers,
            )
            query_ld = query.transpose(1, 2)
            key_ld = key.transpose(1, 2)
            value_ld = value.transpose(1, 2)
            output_ld = runtime.forward(None, query_ld, key_ld, value_ld)
            # HF attention_interface 接收 [B,H,L,D] Q 但返回 [B,L,H,D] 输出
            # （Qwen2Attention 用 attn_output.reshape(*input_shape, -1) 验证）。
            # runtime 在 [B,L,H,D] 空间工作，输出已是 [B,L,H,D]，无需再 transpose。
            # ##### [PS-diag] ON attn output dump（context 激活 = PS 路径） #####
            if os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
                dump_fsdp_attn_output(output_ld, module)
            # ##### [PS-diag] end #####
            return output_ld, None

        return patched_attention

    return ps_aware_get_interface
