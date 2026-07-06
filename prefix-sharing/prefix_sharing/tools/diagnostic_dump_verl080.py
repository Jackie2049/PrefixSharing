"""verl080 NestedTensor 精度验证数据采集层。

与 v070 diagnostic_dump.py 的关系：
- 底层存盘（``_get_dump_dir`` / ``_save_tensor`` / ``_rank0_only``）复用 v070，不重写。
- 本模块只负责 v080 特有的「NestedTensor → v070 兼容 .pt 格式」转换 + dump。
- dump 出的 .pt 满足 v070 ``cmp_diag.py`` 的格式约定，对比工具原样复用。

dump 内容（第一版，聚焦 loss 相关量）：
  - ``prefix_lens.pt``          各序列前缀长度（ON=plan.prefix_lens, OFF=全0）  ← 对齐锚
  - ``cu_seqlens_q.pt``         attention packed 累积边界 [B+1]（= NestedTensor offsets）
  - ``cu_seqlens_q_logits.pt``  logits packed 累积边界（与上同值）
  - ``logits.pt``               packed logits [N, V//tp]
  - ``logprobs_{tag}.pt``       2D log_probs [B, L_max]（restore 后展开）
  - ``entropy_{tag}.pt``        2D entropy [B, L_max]（可选）

触发：env var ``PREFIX_SHARING_DIAG_DUMP=/path``（同 v070）。ON/OFF 各设一个目录，
跑完用 ``python cmp_diag.py --dir-on ./dump_on --dir-off ./dump_off --tag <tag>`` 对比。

注：``attn_outputs.pt`` / RoPE 留第二阶段（attention patch 双路径 + layer_number
来源待定），第一版聚焦验证 loss 正确性所需的 logits / logp / 元数据。
"""

from __future__ import annotations

from typing import Any

import torch

# 复用 v070 底层存盘工具（与数据结构无关的纯存盘逻辑：rank0 去重 + cpu clone + try/except）
from prefix_sharing.tools.diagnostic_dump import (
    _get_dump_dir,
    _save_tensor,
)


# ════════════════════════════════════════════════════════════════
#  NestedTensor → v070 兼容格式转换
# ════════════════════════════════════════════════════════════════

def nested_to_2d_full(nested: Any, original_lengths: list[int], L_max: int) -> torch.Tensor:
    """NestedTensor (jagged) → 2D ``[B, L_max]``，每行从 0 开始 right-pad 0。

    restore 后的 log_probs/entropy NestedTensor 每行长度 = ``original_lengths[i]``
    （reuser prefix 列已恢复），故 ON/OFF 都用此函数展开到统一 ``[B, L_max]`` 坐标系，
    供 ``cmp_diag.cmp_2d`` 直接逐元素对比。

    Args:
        nested: jagged NestedTensor，行 i 长度 = ``original_lengths[i]``。
        original_lengths: 每行原始（完整）长度。
        L_max: ``max(original_lengths)``，2D 的列数。
    """
    offsets = nested.offsets()
    values = nested.values()
    tail_shape = tuple(values.shape[1:])  # 标量时为 ()，logp/entropy 即标量
    rows = []
    for i in range(len(original_lengths)):
        row = values[offsets[i]:offsets[i + 1]]               # [L_i, ...]
        if len(row) < L_max:
            pad_shape = (L_max - len(row),) + tail_shape
            pad = torch.zeros(pad_shape, dtype=row.dtype, device=row.device)
            row = torch.cat([row, pad], dim=0)
        rows.append(row)
    return torch.stack(rows, dim=0)                           # [B, L_max, ...]


def nested_offsets_to_cu(nested: Any) -> torch.Tensor:
    """NestedTensor offsets → cu_seqlens（累积边界 ``[B+1]``）。

    offsets 本身就是 ``[B+1]`` 累积长度，语义与 v070 的
    ``packed_seq_params.cu_seqlens_q_padded`` 一致。
    """
    return nested.offsets().detach().clone()


def build_attention_mask_2d(original_lengths: list[int], L_max: int) -> torch.Tensor:
    """构造 attention_mask 2D ``[B, L_max]``：每行 ``[0:L_i-1)`` 为 True。

    范围 = 所有 predict 有效的位置。``log_probs[i, j]`` 预测 ``token[j+1]``，故 ``j``
    只有 ``< L_i-1`` 时才有意义——``j=L_i-1`` 预测越界的 ``token[L_i]``（序列到此结束，
    该 token 不存在，PPO loss_mask 也排除此位）。

    用于 log_probs/entropy 的"整体对比"——验证 restore 后 prompt 区 + prompt-last +
    response 区所有 predict 有效位置是否都对（含 interior 复制 + prefix-last 重算 +
    response forward）。与 :func:`build_label_mask_2d` 的区别仅是范围更宽（多覆盖
    prompt 区 predict），两者都不含越界预测位 POS ``L_i-1``。
    """
    B = len(original_lengths)
    mask = torch.zeros(B, L_max, dtype=torch.bool)
    for i, L in enumerate(original_lengths):
        if L > 1:  # L-1>0：至少 2 个 token 才存在 predict
            mask[i, :L - 1] = True
    return mask


def build_label_mask_2d(
    response_lens: list[int],
    original_lengths: list[int],
    L_max: int,
) -> torch.Tensor:
    """构造 label_mask 2D ``[B, L_max]``：``[prompt-last : L_i-1)``，不含 response 末尾。

    从 ``response_lens``（每行 response token 数）+ ``original_lengths`` 推 prompt_len：
      ``prompt_len_i = original_lengths[i] - response_lens[i]``
      ``start = prompt_len_i - 1``（prompt-last，预测 response_0 的位置，prefix-sharing
        restore 关键重算点）
      ``end = original_lengths[i] - 1``（开区间，不含 response 最后一个 token；
        POS L_i-1 无 next token，predict 它的 logp 越界）

    用 response_lens 而非 loss_mask 位置：verl080 padding 后 loss_mask = response_mask
    是 2D left-right padded，坐标系与 restore 后 ``[B, L_max]`` 紧凑坐标系不同；但
    response token 数（= loss_mask 行 sum）与坐标系无关，据此推 prompt_len 最稳。
    dump 的就是最终比较范围，**不依赖 cmp_diag 的 trim**。
    """
    B = len(original_lengths)
    mask = torch.zeros(B, L_max, dtype=torch.bool)
    for i in range(B):
        prompt_len_i = original_lengths[i] - response_lens[i]
        start = prompt_len_i - 1            # prompt-last（预测 response_0 的位置）
        end = original_lengths[i] - 1        # 不含 response 最后一个 token（越界 predict）
        if end > start >= 0:
            mask[i, start:end] = True
    return mask


# ════════════════════════════════════════════════════════════════
#  dump 函数（每个独立判 env var，无 env 直接 return，零副作用）
# ════════════════════════════════════════════════════════════════

def dump_meta_verl080(prefix_lens: list[int], cu_seqlens: torch.Tensor) -> None:
    """存元数据：``prefix_lens`` + ``cu_seqlens_q`` + ``cu_seqlens_q_logits``。

    Args:
        prefix_lens: ON=``plan.prefix_lens``，OFF=``[0]*B``。对齐的关键
            （cmp_diag 的 fallback 对齐路径用它从 OFF full-packed 抽 suffix）。
        cu_seqlens: NestedTensor offsets（累积边界 ``[B+1]``），attention/logits
            packed 共用同一边界（THD packing 按 NestedTensor 顺序）。
    """
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor("prefix_lens.pt", torch.tensor(prefix_lens, dtype=torch.int32), dump_dir)
    _save_tensor("cu_seqlens_q.pt", cu_seqlens, dump_dir)
    # 同值，cmp_diag.cmp_logits_packed 按此名加载
    _save_tensor("cu_seqlens_q_logits.pt", cu_seqlens, dump_dir)


def dump_logits_verl080(logits: torch.Tensor) -> None:
    """存 packed logits，scope=``tp_vocab``：每个 tp rank 存自己的词表片。

    纯 TP 下 logits 是 ``[N, V//tp]``，各 tp rank 不同（vocab 切分）。每个 tp
    rank 存自己的 shard（文件名 ``logits_tp{r}.pt``；``tp_size==1`` 时退化为
    ``logits.pt``，单卡零改动）。dump 路径不引入跨 rank 通信；cmp 侧按
    ``parallel_info.json`` 把 ``_tp{0..tp-1}`` 拼回 full vocab 再比。
    """
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor("logits.pt", logits, dump_dir, scope="tp_vocab")


def dump_logprobs_2d_verl080(logp_2d: torch.Tensor, tag: str) -> None:
    """存 2D log_probs ``[B, L_max]``，文件名 ``logprobs_{tag}.pt``。"""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor(f"logprobs_{tag}.pt", logp_2d, dump_dir)


def dump_attention_mask_verl080(mask_2d: torch.Tensor, tag: str) -> None:
    """存 2D attention_mask ``[B, L_max]``（bool），文件名 ``attention_mask_{tag}.pt``。

    全有效位 mask（prompt + response，去 padding）。cmp_diag 里作外部 mask：
    ``--mask-file dump_off/attention_mask_{tag}.pt --no-trim``。

    带 tag：mask 必须和 logprobs_{tag} 来自同一 forward（同 batch、同 L_max），
    否则 shape 不匹配。tag 与 logprobs 一致（``"old"`` / ``"train"``）。
    """
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor(f"attention_mask_{tag}.pt", mask_2d.to(torch.bool), dump_dir)


def dump_label_mask_verl080(mask_2d: torch.Tensor, tag: str) -> None:
    """存 2D label_mask ``[B, L_max]``（bool），文件名 ``label_mask_{tag}.pt``。

    范围 ``[prompt-last : L_i-1)``，不含 response 末尾（由 :func:`build_label_mask_2d`
    据 response_lens 构造，见该函数 docstring）。dump 的就是最终比较范围，
    cmp_diag_verl080 直接用，无需额外参数。

    带 tag：mask 必须和 logprobs_{tag} 来自同一 forward（同 batch、同 L_max），
    否则 shape 不匹配。tag 与 logprobs 一致（``"old"`` / ``"train"``）。
    """
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor(f"label_mask_{tag}.pt", mask_2d.to(torch.bool), dump_dir)


def dump_entropy_2d_verl080(ent_2d: torch.Tensor | None, tag: str) -> None:
    """存 2D entropy ``[B, L_max]``，文件名 ``entropy_{tag}.pt``。``None`` 时跳过。"""
    dump_dir = _get_dump_dir()
    if dump_dir is None or ent_2d is None:
        return
    _save_tensor(f"entropy_{tag}.pt", ent_2d, dump_dir)


def dump_raw_logits_verl080(raw_output: Any) -> None:
    """从 HF/verl model raw output 中提取 logits 并 dump。

    ON 路径中 logits 是裁剪后的 packed logits；OFF 路径中 logits 是完整 packed
    logits。二者都由 ``cmp_diag_verl080`` 根据 ``prefix_lens`` / ``cu_seqlens`` 做
    suffix 对齐。
    """
    if _get_dump_dir() is None:
        return
    logits = raw_output["logits"] if isinstance(raw_output, dict) else raw_output.logits
    dump_logits_verl080(logits)


def dump_fsdp_on_metadata_verl080(micro_batch: Any, prefix_sharing_plan: Any, tag: str) -> None:
    """Dump FSDP PrefixSharing ON 路径的元数据和原始 batch 对齐锚点。"""
    if _get_dump_dir() is None:
        return

    prefix_lens = list(prefix_sharing_plan.prefix_lens)
    original_lengths = list(prefix_sharing_plan.original_lengths)
    kept_lengths = list(prefix_sharing_plan.kept_lengths_q)

    cu_seqlens = torch.zeros(len(kept_lengths) + 1, dtype=torch.int64)
    for index, length in enumerate(kept_lengths):
        cu_seqlens[index + 1] = cu_seqlens[index] + length
    dump_meta_verl080(prefix_lens, cu_seqlens)

    max_length = max(original_lengths) if original_lengths else 0
    dump_attention_mask_verl080(build_attention_mask_2d(original_lengths, max_length), tag)

    loss_mask = micro_batch.get("loss_mask")
    if loss_mask is not None:
        response_lengths = _response_lengths_from_loss_mask(loss_mask, len(original_lengths))
        dump_label_mask_verl080(
            build_label_mask_2d(response_lengths, original_lengths, max_length),
            tag,
        )
    dump_input_ids_2d_verl080(micro_batch, original_lengths, max_length, tag)


def dump_fsdp_model_output_2d_verl080(
    model_output: dict[str, Any],
    original_lengths: list[int],
    tag: str,
) -> None:
    """Dump restore 后的 FSDP log_probs / entropy 到统一 2D 坐标系。"""
    if _get_dump_dir() is None:
        return
    max_length = max(original_lengths) if original_lengths else 0
    log_probs = model_output.get("log_probs")
    if log_probs is None:
        return
    log_probs_2d = _maybe_nested_to_2d(log_probs, original_lengths, max_length)
    if log_probs_2d is None or log_probs_2d.dim() != 2:
        return
    dump_logprobs_2d_verl080(log_probs_2d, tag)

    entropy = model_output.get("entropy")
    if entropy is not None:
        entropy_2d = _maybe_nested_to_2d(entropy, original_lengths, max_length)
        if entropy_2d is not None and entropy_2d.dim() == 2:
            dump_entropy_2d_verl080(entropy_2d, tag)


def dump_fsdp_baseline_verl080(micro_batch: Any, result: Any, tag: str) -> None:
    """Dump FSDP OFF baseline diagnostics (no prefix-sharing).

    OFF 路径返回 ``(loss, output_dict)``，``output_dict["model_output"]`` 里的
    ``log_probs`` / ``entropy`` 在 ``use_remove_padding=True`` 下通常是 jagged
    NestedTensor；这里统一展开到 ``[B, L_max]``。
    """
    if _get_dump_dir() is None:
        return
    if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], dict):
        output_dict = result[1]
    else:
        return
    model_output = output_dict.get("model_output", {})
    if not model_output:
        return

    original_lengths = _original_lengths_from_input_ids(micro_batch.get("input_ids"))
    if original_lengths is None:
        return

    prefix_lens = [0] * len(original_lengths)
    cu_seqlens = torch.zeros(len(original_lengths) + 1, dtype=torch.int64)
    for index, length in enumerate(original_lengths):
        cu_seqlens[index + 1] = cu_seqlens[index] + length
    dump_meta_verl080(prefix_lens, cu_seqlens)

    max_length = max(original_lengths) if original_lengths else 0
    dump_attention_mask_verl080(build_attention_mask_2d(original_lengths, max_length), tag)

    loss_mask = micro_batch.get("loss_mask")
    if loss_mask is not None:
        response_lengths = _response_lengths_from_loss_mask(loss_mask, len(original_lengths))
        dump_label_mask_verl080(
            build_label_mask_2d(response_lengths, original_lengths, max_length),
            tag,
        )
    dump_input_ids_2d_verl080(micro_batch, original_lengths, max_length, tag)
    dump_fsdp_model_output_2d_verl080(model_output, original_lengths, tag)


def dump_input_ids_2d_verl080(
    micro_batch: Any,
    original_lengths: list[int],
    max_length: int,
    tag: str,
) -> None:
    """Dump 原始 input_ids 到 2D ``[B, L_max]`` 作为 ON/OFF batch 对齐锚。"""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    input_ids = micro_batch.get("input_ids")
    if input_ids is None:
        return
    if _is_nested_tensor(input_ids):
        ids_2d = nested_to_2d_full(input_ids, original_lengths, max_length)
    elif hasattr(input_ids, "dim") and input_ids.dim() == 2:
        ids_2d = input_ids
    else:
        return
    _save_tensor(f"input_ids_{tag}.pt", ids_2d.long().cpu(), dump_dir)


def _maybe_nested_to_2d(value: Any, original_lengths: list[int], max_length: int) -> Any | None:
    if _is_nested_tensor(value):
        return nested_to_2d_full(value, original_lengths, max_length)
    if hasattr(value, "dim"):
        return value
    return None


def _original_lengths_from_input_ids(input_ids: Any) -> list[int] | None:
    if _is_nested_tensor(input_ids):
        return [int(length) for length in input_ids.offsets().diff().tolist()]
    if input_ids is not None and hasattr(input_ids, "dim") and input_ids.dim() == 2:
        return [int(input_ids.shape[1])] * int(input_ids.shape[0])
    return None


def _response_lengths_from_loss_mask(loss_mask: Any, batch_size: int) -> list[int]:
    if _is_nested_tensor(loss_mask):
        offsets = loss_mask.offsets()
        values = loss_mask.values()
        return [int(values[offsets[i]:offsets[i + 1]].sum()) for i in range(batch_size)]
    return loss_mask.sum(dim=-1).long().cpu().tolist()


def _is_nested_tensor(value: Any) -> bool:
    return hasattr(value, "offsets") and hasattr(value, "values")
