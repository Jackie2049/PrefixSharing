"""Patch: no_padding_2_padding — prefix-sharing 物理裁剪后修正序列长度

物理裁剪后，模型输出 NestedTensor 的 offsets().diff() 给出每行的 trimmed 长度，
但 data["attention_mask"] 仍反映原始长度，导致
  sum(prompt_lens + response_lens) != values.shape[0]
断言失败。

修复：当 tensor 是 NestedTensor 且 offsets().diff().sum() 与
attention_mask 导出的 sum 不一致时，用 NestedTensor 自身的 offsets
作为 trimmed 序列长度。response 未被裁剪，所以
trimmed_prompt_lens = trimmed_seq_lens - response_lens。

不依赖任何跨进程通信或 metadata，纯靠模型输出的形状推导。

注意：多个模块（padding.py、losses.py、ray_trainer.py）通过
from...import 引用此函数。每个模块的局部引用都需要被 patch。
本 factory 确保所有 patch 共享同一个 original 函数引用，避免链式包装。
"""

from __future__ import annotations

from prefix_sharing.diagnostics import diagnostic_dump_enabled

from typing import Any

import torch
import torch.nn.functional as F

from verl.utils import tensordict_utils as tu

# 缓存真正的原始函数，避免多次 patch 时链式包装（patched-of-patched）。
# 第一次 patch_factory 调用时缓存 original；后续调用直接复用同一个 wrapper。
_cached_original: Any | None = None
_cached_wrapper: Any | None = None


def patch_no_padding_2_padding(original_func: Any) -> Any:
    """创建 no_padding_2_padding 的 patch wrapper。

    多个模块（padding、losses、ray_trainer、distillation losses）
    都有 from...import 的局部引用。PatchSpec 对每个模块分别调用
    本 factory。如果直接用 original_func（可能是已被 patch 的版本），
    会产生链式包装。因此缓存真正的 original，所有模块共享同一个 wrapper。
    """
    global _cached_original, _cached_wrapper

    if _cached_original is None:
        _cached_original = original_func

    # 始终基于真正的 original 创建 wrapper（不是链式包装）
    if _cached_wrapper is None:
        _cached_wrapper = _make_wrapper(_cached_original)

    return _cached_wrapper


def _make_wrapper(original_func: Any) -> Any:

    def patched_no_padding_2_padding(tensor: Any, data: Any) -> Any:
        # ── 先做 PS trimming 检测，不匹配时走 PS-aware 路径 ──
        values = tensor.values() if tensor.is_nested else tensor
        prompt_ids = data["prompts"]
        response_ids = data["responses"]

        max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)

        if prompt_ids.is_nested:
            prompt_lens = prompt_ids.offsets().diff()
            response_lens = response_ids.offsets().diff()
            if max_response_len < 0:
                max_response_len = response_lens.max().item()
        else:
            attention_mask = data["attention_mask"]
            assert not attention_mask.is_nested
            prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
            response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
            max_response_len = response_ids.shape[1]

        sequence_lens = prompt_lens + response_lens
        sequence_offsets = sequence_lens.cumsum(dim=0)

        # ── PS trimming detection ──
        # 模型输出 NestedTensor 的 offsets().diff() 给出每行的实际长度。
        # 如果与 attention_mask 导出的总长度不一致，说明发生了 PS 物理裁剪。
        # 用 trimmed_seq_lens - response_lens 代替 prompt_lens。
        if tensor.is_nested:
            trimmed_seq_lens = tensor.offsets().diff()
            expected_total = sequence_offsets[-1].item()
            actual_total = values.shape[0]
            if expected_total != actual_total:
                # PS trimming detected
                trimmed_prompt_lens = trimmed_seq_lens - response_lens
                sequence_lens = trimmed_prompt_lens + response_lens
                sequence_offsets = sequence_lens.cumsum(dim=0)
                print(
                    f"[PS][nopadding] trimming detected: "
                    f"expected_total={expected_total} actual_total={actual_total} "
                    f"original_prompt_lens={prompt_lens.tolist()} "
                    f"trimmed_prompt_lens={trimmed_prompt_lens.tolist()} "
                    f"response_lens={response_lens.tolist()}"
                )
                # PS path: 不匹配时直接走自己的实现
                # (避免原函数的 assert 再次失败)
                assert sequence_offsets[-1].item() == values.shape[0], (
                    f"[PS] sequence_offsets[-1]={sequence_offsets[-1].item()} "
                    f"!= values.shape[0]={values.shape[0]}"
                )
                response_list = []
                skip_padding = (0, 0) * (values.ndim - 1)
                for resp_len, seq_offset in zip(response_lens, sequence_offsets, strict=True):
                    pad_size = max_response_len - resp_len
                    response_list.append(
                        F.pad(
                            values[seq_offset - resp_len - 1 : seq_offset - 1],
                            (*skip_padding, 0, pad_size),
                        )
                    )
                return torch.stack(response_list, dim=0)

        # ── normal path: 未检测到 PS trimming，走原始函数 ──
        return original_func(tensor, data)

    return patched_no_padding_2_padding