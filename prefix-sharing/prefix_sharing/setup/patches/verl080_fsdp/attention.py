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

# per-forward 累积每层 attention 输出，最后一层 flush 成 attn_outputs.pt。
# layer_number == 1 时清空（新 forward 起点），== num_layers 时存盘。
# 与 cmp_diag.cmp_attn_layer 约定一致：dict {layer_1based: tensor[N, hidden]}。
_FSDP_ATTN_BUFFER: dict[int, Any] = {}


def _dump_fsdp_attn_output(output: Any, module: Any) -> None:
    """把 attention 输出累积到 buffer，最后一层 flush 成 attn_outputs.pt。

    output 形态 [B,L,H,D]（o_proj 前），reshape 成 [N, H*D]（N=B*L）对齐 Megatron
    的 [N, hidden] packed 格式。ON（runtime output_ld）和 OFF（original_fn output）
    在本拦截点形态一致，因此 cmp ON-vs-OFF 有效。
    """
    import torch

    from prefix_sharing.tools.diagnostic_dump import _get_dump_dir

    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    if isinstance(output, tuple):
        output = output[0]
    if not hasattr(output, "dim") or output.dim() < 3:
        return
    layer_number = int(getattr(module, "layer_idx", 0) or 0) + 1  # 1-based
    num_layers = int(getattr(getattr(module, "config", None), "num_hidden_layers", 0) or 0)
    if num_layers == 0:
        return
    # [B,L,H,D] -> [N, H*D]
    hidden = output.shape[-1] * output.shape[-2]
    out_2d = output.reshape(-1, hidden).detach().cpu().contiguous()
    if layer_number == 1:
        _FSDP_ATTN_BUFFER.clear()
    _FSDP_ATTN_BUFFER[layer_number] = out_2d
    if layer_number == num_layers:
        # 多 rank（DP）下只有 rank 0 存盘，其余 rank 仅累积后丢弃，避免文件 clobber。
        # 与 _save_tensor 的 _rank0_only() 门控一致：2D/logits dump 已通过 _save_tensor
        # 自动 rank-0 门控；本函数直接 torch.save，需显式补门控。单卡（rank 0 或
        # dist 未初始化）_rank0_only() 恒 True，行为不变。
        from prefix_sharing.tools.diagnostic_dump import _rank0_only

        if _rank0_only():
            torch.save(_FSDP_ATTN_BUFFER, os.path.join(dump_dir, "attn_outputs.pt"))
        _FSDP_ATTN_BUFFER.clear()


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
                    _dump_fsdp_attn_output(result, module)
                # ##### [PS-diag] end #####
                return result

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
            # ##### [PS-diag] ON attn output dump（context 激活 = PS 路径） #####
            if os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
                _dump_fsdp_attn_output(output_ld, module)
            # ##### [PS-diag] end #####
            return output_ld, None

        return patched_attention

    return ps_aware_get_interface
