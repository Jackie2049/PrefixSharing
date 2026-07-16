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

import contextlib
from typing import Any

import torch

# 复用 v070 底层存盘工具（与数据结构无关的纯存盘逻辑：rank0 去重 + cpu clone + try/except）
from prefix_sharing.tools.diagnostic_dump import (
    _get_dump_dir,
    _save_tensor,
    _stage_last_layer,
    _pp_suffix,
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


def dump_logprobs_2d_verl080(logp_2d: torch.Tensor, tag: str,
                           scope: str = "global") -> None:
    """存 2D log_probs ``[B, L_max]``，文件名 ``logprobs_{tag}.pt``。

    ``scope="pp_last"`` 时仅最后一个 PP stage 落盘（多卡下 logprobs 只在末 stage 产生）。
    """
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor(f"logprobs_{tag}.pt", logp_2d, dump_dir, scope=scope)


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


def dump_entropy_2d_verl080(ent_2d: torch.Tensor | None, tag: str,
                           scope: str = "global") -> None:
    """存 2D entropy ``[B, L_max]``，文件名 ``entropy_{tag}.pt``。``None`` 时跳过。

    ``scope="pp_last"`` 时仅最后一个 PP stage 落盘（多卡下 entropy 只在末 stage 产生）。
    """
    dump_dir = _get_dump_dir()
    if dump_dir is None or ent_2d is None:
        return
    _save_tensor(f"entropy_{tag}.pt", ent_2d, dump_dir, scope=scope)


# ════════════════════════════════════════════════════════════════
#  Post-RoPE Q/K dump (per layer)
# ════════════════════════════════════════════════════════════════

_ROPE_POSTQK_BUFFER: dict[int, dict] | None = None


def dump_rope_postqk_verl080(layer_number: int,
                           rotated_query: torch.Tensor,
                           rotated_key: torch.Tensor,
                           num_layers: int,
                           positions: torch.Tensor | None = None) -> None:
    """Accumulate one layer's post-RoPE Q/K. Auto-flush to ``rope_postqk.pt`` on last layer.

    Format: ``{layer_idx: {"query": [T, H, D], "key": [T, H, D], "positions": [T] or None}}``
    ON  packed 只含 suffix（裁剪后），OFF packed 含完整序列。
    cmp 侧用 prefix_lens + cu_seqlens 做 suffix 对齐后对比（同 attn_output 模式）。
    positions 可选，用于手动排查时的位置回溯。
    """
    global _ROPE_POSTQK_BUFFER
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    if _ROPE_POSTQK_BUFFER is None:
        _ROPE_POSTQK_BUFFER = {}
    entry = {
        "query": rotated_query.detach().cpu().clone(),
        "key": rotated_key.detach().cpu().clone(),
    }
    if positions is not None:
        entry["positions"] = positions.detach().cpu().clone()
    _ROPE_POSTQK_BUFFER[layer_number] = entry
    if layer_number == _stage_last_layer(num_layers):
        _flush_dict_buffer("rope_postqk.pt", _ROPE_POSTQK_BUFFER, dump_dir)
        _ROPE_POSTQK_BUFFER = None


def _flush_dict_buffer(fname: str, buffer: dict, dump_dir: str) -> None:
    """rank0 直接 torch.save 一个 dict buffer。PP-aware gating + suffix。

    不能用 _save_tensor：它对入参做 .detach().cpu().clone()，dict 没 .detach() →
    AttributeError 被其 except 吞掉，文件永不写盘（rope_postqk.pt 曾因此丢失）。
    entries 应在插入时已 detach().cpu().clone()。仿 _flush_attn_buffer。

    Under PP, each stage's tp_rank==0 writes with ``_pp{r}`` suffix;
    assembly later merges all stage files.
    """
    import os as _os
    from prefix_sharing.tools.diagnostic_dump import _should_write_for_scope
    if not _should_write_for_scope("pp_stage"):
        return
    try:
        stem, sep, ext = fname.rpartition(".")
        pp_sfx = _pp_suffix()
        fname_pp = f"{stem}{pp_sfx}{sep}{ext}" if sep else f"{fname}{pp_sfx}"
        torch.save(buffer, _os.path.join(dump_dir, fname_pp))
    except Exception as _e:
        print(f"[PS-diag] {fname} save failed: {_e}", flush=True)


_ROPE_PREQK_BUFFER: dict[int, dict] | None = None


def dump_rope_preqk_verl080(layer_number: int,
                             query: torch.Tensor,
                             key: torch.Tensor,
                             num_layers: int) -> None:
    """Accumulate one layer's pre-RoPE Q/K. Auto-flush to ``rope_preqk.pt`` on last layer.

    Format: ``{layer_idx: {"query": [T, H, D], "key": [T, H, D]}}``。
    旋转前的 Q/K（纯 QKV 投影输出，未加位置编码）。ON/OFF 应逐元素相同——
    同 hidden_states、同 QKV 权重。用来隔离：pre-RoPE 相同但 post-RoPE 不同 →
    问题在 RoPE；pre 就不同 → 问题在上游（hidden_states / 投影）。
    """
    global _ROPE_PREQK_BUFFER
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    if _ROPE_PREQK_BUFFER is None:
        _ROPE_PREQK_BUFFER = {}
    _ROPE_PREQK_BUFFER[layer_number] = {
        "query": query.detach().cpu().clone(),
        "key": key.detach().cpu().clone(),
    }
    if layer_number == _stage_last_layer(num_layers):
        _flush_dict_buffer("rope_preqk.pt", _ROPE_PREQK_BUFFER, dump_dir)
        _ROPE_PREQK_BUFFER = None


_EXPANDED_KV_BUFFER: dict[int, dict] | None = None


def dump_expanded_kv_on(layer_number: int, expanded_key: torch.Tensor,
                        expanded_value: torch.Tensor, num_layers: int) -> None:
    """ON: 累加 build_kv 输出（expanded K/V = prefix 复用 + suffix，全量），满层 flush ``expanded_kv.pt``。

    Format: ``{layer_idx: {"key": [T,H,D], "value": [T,H,D]}}``。这是 ON attention 实际用的
    完整 K/V，应与 OFF ``full_kv.pt`` 逐元素相同——验证 prefix-sharing 的 KV 展开/复用
    是否正确还原了完整 KV。
    """
    global _EXPANDED_KV_BUFFER
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    if _EXPANDED_KV_BUFFER is None:
        _EXPANDED_KV_BUFFER = {}
    _EXPANDED_KV_BUFFER[layer_number] = {
        "key": expanded_key.detach().cpu().clone(),
        "value": expanded_value.detach().cpu().clone(),
    }
    if layer_number == _stage_last_layer(num_layers):
        _flush_dict_buffer("expanded_kv.pt", _EXPANDED_KV_BUFFER, dump_dir)
        _EXPANDED_KV_BUFFER = None


_FULL_KV_BUFFER: dict[int, dict] | None = None


def dump_full_kv_off(layer_number: int, key: torch.Tensor, value: torch.Tensor,
                     num_layers: int) -> None:
    """OFF: 累加完整 K（post-RoPE）/ V，满层 flush ``full_kv.pt``。

    Format: ``{layer_idx: {"key": [T,H,D], "value": [T,H,D]}}``。key 应为 post-RoPE 完整 K
    （与 ON expanded_key 同语义），value 为完整 V。供与 ON expanded_kv 逐元素对比。
    """
    global _FULL_KV_BUFFER
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    if _FULL_KV_BUFFER is None:
        _FULL_KV_BUFFER = {}
    _FULL_KV_BUFFER[layer_number] = {
        "key": key.detach().cpu().clone(),
        "value": value.detach().cpu().clone(),
    }
    if layer_number == _stage_last_layer(num_layers):
        _flush_dict_buffer("full_kv.pt", _FULL_KV_BUFFER, dump_dir)
        _FULL_KV_BUFFER = None


_BUILD_KV_INPUT_V_BUFFER: dict[int, torch.Tensor] | None = None


def dump_build_kv_input_v_on(layer_number: int, value: torch.Tensor,
                             num_layers: int) -> None:
    """ON: build_kv 输入的 V（get_qkv 出来、build_kv 之前的 raw V）。满层 flush ``build_kv_input_v.pt``。

    Format: ``{layer_idx: [T_on, ...]}``。供与 OFF ``full_kv.pt`` 的 V 做 suffix 对比——
    定位 V 是在 build_kv 之前（get_qkv/hidden_states）就偏，还是 build_kv 引入。
    T_on vs T_off 还能看出 ON 有没有把 hidden_states 裁剪成 suffix-only。
    """
    global _BUILD_KV_INPUT_V_BUFFER
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    if _BUILD_KV_INPUT_V_BUFFER is None:
        _BUILD_KV_INPUT_V_BUFFER = {}
    _BUILD_KV_INPUT_V_BUFFER[layer_number] = value.detach().cpu().clone()
    if layer_number == _stage_last_layer(num_layers):
        _flush_dict_buffer("build_kv_input_v.pt", _BUILD_KV_INPUT_V_BUFFER, dump_dir)
        _BUILD_KV_INPUT_V_BUFFER = None


_HIDDEN_STATES_BUFFER: dict[int, torch.Tensor] | None = None


def dump_hidden_states_on(layer_number: int, hidden_states: torch.Tensor,
                          num_layers: int) -> None:
    """Accumulate hidden_states at attention entrance. Auto-flush ``hidden_states.pt`` on last layer.

    Used to verify whether ON/OFF hidden_states are bit-identical for suffix tokens.
    Format: ``{layer_idx: [T, H]}``.
    """
    global _HIDDEN_STATES_BUFFER
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    if _HIDDEN_STATES_BUFFER is None:
        _HIDDEN_STATES_BUFFER = {}
    _HIDDEN_STATES_BUFFER[layer_number] = hidden_states.detach().cpu().clone()
    if layer_number == _stage_last_layer(num_layers):
        _flush_dict_buffer("hidden_states.pt", _HIDDEN_STATES_BUFFER, dump_dir)
        _HIDDEN_STATES_BUFFER = None


@contextlib.contextmanager
def capture_rope_qk(attention_module):
    """Hook apply_rotary_pos_emb（post-RoPE Q/K）+ get_query_key_value_tensors（pre-RoPE Q/K/V）。

    两个 hook，全 in-context：
      - get_qkv 返回值统一截 Q/K/V（pre-squeeze，与 ON 侧 get_qkv+squeeze 之后 dump 同源）。
      - apply_rotary_pos_emb 截 post-RoPE Q/K（返回值）。

    Q/K/V 都从 get_qkv 一次返回取（pre-RoPE），避免 ON 侧 Q/K/V 也走两个不对称来源。
    用法::

        with capture_rope_qk(self) as (qk_caps, qkv_caps):
            result = original_forward(...)
        # qk_caps[0]={"post":Q_post}, qk_caps[1]={"post":K_post}（apply_rotary 返回）
        # qkv_caps[-1] = (Q_pre, K_pre, V_pre)（get_qkv 返回，pre-squeeze）
    """
    import megatron.core.transformer.attention as _attn_mod
    import types as _types

    _orig_arpe = _attn_mod.apply_rotary_pos_emb
    _orig_get_qkv = type(attention_module).get_query_key_value_tensors
    qk_captures: list = []      # apply_rotary 返回（post-RoPE）
    qkv_captures: list = []     # get_qkv 返回 (Q, K, V) pre-squeeze

    def _capturing_arpe(t, *args, **kwargs):
        out = _orig_arpe(t, *args, **kwargs)
        qk_captures.append({"post": out})
        return out

    def _capturing_get_qkv(self_, *args, **kwargs):
        out = _orig_get_qkv(self_, *args, **kwargs)
        try:
            qkv_captures.append(tuple(out[:3]))  # (Q, K, V)
        except Exception:
            pass
        return out

    _attn_mod.apply_rotary_pos_emb = _capturing_arpe
    attention_module.get_query_key_value_tensors = _types.MethodType(
        _capturing_get_qkv, attention_module)
    try:
        yield qk_captures, qkv_captures
    finally:
        _attn_mod.apply_rotary_pos_emb = _orig_arpe
        try:
            del attention_module.get_query_key_value_tensors
        except AttributeError:
            pass
