"""Pure PyTorch reference backend."""

from __future__ import annotations

import math
from typing import Any

import torch

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.observability import PrefixSharingStats
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_ATTENTION_KV,
    PrefixActivationSlotId,
    PrefixAttentionStore,
)


class TorchReferenceBackend:
    capabilities = BackendCapabilities(
        name="torch_ref",
        supports_cpu=True,
        supports_cuda=True,
        supports_cann=True,
        supports_different_q_kv_lengths=True,
        supports_prefix_last_restore=True,
    )

    def validate(self, config: PrefixSharingConfig, model_config: Any | None = None) -> None:
        config.validate(model_config=model_config)

    def apply_rope(
        self,
        query: Any,
        key: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        rope_fn: Any | None = None,
        **_: Any,
    ) -> tuple[Any, Any]:
        if rope_fn is None:
            return query, key
        return rope_fn(query, key, prefix_sharing_plan.q_position_offsets, prefix_sharing_plan.kv_position_offsets)

    def build_kv(
        self,
        key: Any,
        value: Any,
        store: PrefixAttentionStore,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        layer_id: int,
        tp_rank: int = 0,
        stats: PrefixSharingStats | None = None,
    ) -> tuple[Any, Any]:
        layout = packed_batch_layout or PackedBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q)
        # Input K/V still follow the framework's padded packed layout; only
        # valid tokens may enter the store or expanded KV.
        key_rows = _split_packed(key, layout.padded_lengths)
        value_rows = _split_packed(value, layout.padded_lengths)
        expanded_offsets = _cumsum(prefix_sharing_plan.expanded_lengths_kv)
        expanded_total = expanded_offsets[-1]
        expanded_key = key.new_empty((expanded_total, *key.shape[1:]))
        expanded_value = value.new_empty((expanded_total, *value.shape[1:]))
        store_count = 0
        reuse_count = 0
        reuse_hit_count = 0
        reuse_miss_count = 0
        stored_tokens = 0
        reused_prefix_tokens = 0
        # This loop relies on the current online detector invariant that a provider
        # appears before every reuser that loads from it. All rows' QKV tensors have
        # already been produced in parallel by this point; the ordering here only
        # controls KV assembly before attention. Do not reorder or parallelize this
        # loop unless provider dependencies are handled explicitly, e.g. by a
        # topology-aware build phase.
        for batch_index, (key_row, value_row) in enumerate(zip(key_rows, value_rows)):
            valid_length = layout.valid_lengths[batch_index]
            valid_key_row = key_row[:valid_length]
            valid_value_row = value_row[:valid_length]
            expanded_start = expanded_offsets[batch_index]
            expanded_end = expanded_offsets[batch_index + 1]
            expanded_key_row = expanded_key[expanded_start:expanded_end]
            expanded_value_row = expanded_value[expanded_start:expanded_end]
            if not prefix_sharing_plan.is_reuser(batch_index):
                expanded_key_row.copy_(valid_key_row)
                expanded_value_row.copy_(valid_value_row)
                slot_id = PrefixActivationSlotId(
                    prefix_sharing_plan.forward_id,
                    prefix_sharing_plan.micro_batch_id,
                    layer_id,
                    batch_index,
                    PREFIX_STATE_TYPE_ATTENTION_KV,
                    tp_rank,
                )
                # Publish this row's KV so later reusers in this micro-batch can load it.
                store.store(
                    slot_id,
                    key_tensor=expanded_key_row,
                    value_tensor=expanded_value_row,
                    prefix_len=expanded_key_row.shape[0],
                    overwrite=True,
                )
                store_count += 1
                stored_tokens += int(expanded_key_row.shape[0])
            else:
                provider = prefix_sharing_plan.provider_index[batch_index]
                provider_slot_id = PrefixActivationSlotId(
                    prefix_sharing_plan.forward_id,
                    prefix_sharing_plan.micro_batch_id,
                    layer_id,
                    provider,
                    PREFIX_STATE_TYPE_ATTENTION_KV,
                    tp_rank,
                )
                # Load the already-published provider KV before building this reuser's expanded KV.
                reuse_count += 1
                try:
                    entry = store.load(provider_slot_id)
                except KeyError:
                    reuse_miss_count += 1
                    if stats is not None:
                        stats.record_attention_kv_build(
                            layer_id=layer_id,
                            store_count=store_count,
                            reuse_count=reuse_count,
                            reuse_hit_count=reuse_hit_count,
                            reuse_miss_count=reuse_miss_count,
                            stored_tokens=stored_tokens,
                            reused_prefix_tokens=reused_prefix_tokens,
                            expanded_kv_tokens=expanded_total,
                            valid_q_tokens=layout.total_valid_length,
                            padded_q_tokens=layout.total_padded_length,
                        )
                    raise
                reuse_hit_count += 1
                prefix_len = prefix_sharing_plan.prefix_lens[batch_index]
                reused_prefix_tokens += int(prefix_len)
                expanded_key_row[:prefix_len].copy_(entry.key_tensor[:prefix_len])
                expanded_key_row[prefix_len:].copy_(valid_key_row)
                expanded_value_row[:prefix_len].copy_(entry.value_tensor[:prefix_len])
                expanded_value_row[prefix_len:].copy_(valid_value_row)
                own_slot_id = PrefixActivationSlotId(
                    prefix_sharing_plan.forward_id,
                    prefix_sharing_plan.micro_batch_id,
                    layer_id,
                    batch_index,
                    PREFIX_STATE_TYPE_ATTENTION_KV,
                    tp_rank,
                )
                # Publish the expanded reuser KV because a later row may reuse this longer prefix.
                store.store(
                    own_slot_id,
                    key_tensor=expanded_key_row,
                    value_tensor=expanded_value_row,
                    prefix_len=expanded_key_row.shape[0],
                    overwrite=True,
                )
                store_count += 1
                stored_tokens += int(expanded_key_row.shape[0])
        if stats is not None:
            stats.record_attention_kv_build(
                layer_id=layer_id,
                store_count=store_count,
                reuse_count=reuse_count,
                reuse_hit_count=reuse_hit_count,
                reuse_miss_count=reuse_miss_count,
                stored_tokens=stored_tokens,
                reused_prefix_tokens=reused_prefix_tokens,
                expanded_kv_tokens=expanded_total,
                valid_q_tokens=layout.total_valid_length,
                padded_q_tokens=layout.total_padded_length,
            )
        return expanded_key, expanded_value

    def attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        **_: Any,
    ) -> Any:
        batch_layout = packed_batch_layout or PackedBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q)
        
        # QKV从batch拆分到单条序列，便于精度问题定位
        query_rows = _split_packed(query, batch_layout.padded_lengths)
        key_rows = _split_packed(key, prefix_sharing_plan.expanded_lengths_kv)
        value_rows = _split_packed(value, prefix_sharing_plan.expanded_lengths_kv)

        # 逐条序列进行注意力计算
        outputs = []
        for batch_index, (q_row, k_row, v_row) in enumerate(zip(query_rows, key_rows, value_rows)):
            valid_length = batch_layout.valid_lengths[batch_index]
            # padding不参与注意力计算
            q_valid = q_row[:valid_length]
            prefix_len = prefix_sharing_plan.q_position_offsets[batch_index]
            if valid_length == 0:
                outputs.append(torch.zeros_like(q_row))
                continue
            mask = _causal_q_kv_mask(
                q_len=q_valid.shape[0],
                kv_len=k_row.shape[0],
                q_start=prefix_len,
                device=q_valid.device,
            )
            valid_output = _attention_row(q_valid, k_row, v_row, mask)
            if valid_length == q_row.shape[0]:
                outputs.append(valid_output)
                continue
            padded_output = torch.zeros_like(q_row)
            padded_output[:valid_length] = valid_output
            outputs.append(padded_output)
        return torch.cat(outputs, dim=0)


def _split_packed(tensor: Any, lengths: list[int]) -> list[Any]:
    if not lengths:
        return []
    if sum(lengths) != tensor.shape[0]:
        raise ValueError("packed tensor first dimension does not match lengths")
    return list(torch.split(tensor, lengths, dim=0))


def _cumsum(lengths: list[int]) -> list[int]:
    offsets = [0]
    total = 0
    for length in lengths:
        total += int(length)
        offsets.append(total)
    return offsets


def _causal_q_kv_mask(q_len: int, kv_len: int, q_start: int, device: Any) -> Any:
    q_positions = torch.arange(q_start, q_start + q_len, device=device).unsqueeze(1)
    kv_positions = torch.arange(0, kv_len, device=device).unsqueeze(0)
    return kv_positions <= q_positions


def _attention_row(q_row: Any, k_row: Any, v_row: Any, mask: Any) -> Any:
    # 用 ``F.scaled_dot_product_attention`` 替代手写 einsum+softmax：
    # 在 bf16 autocast 下 ``torch.einsum`` 会被降到 bf16（即使输入已 .float()），
    # 导致 softmax 精度严重劣化，误差在残差流里逐层放大；SDPA 不受 autocast 降精度
    # 影响（内部 fp32 累加），与 HF attention 数值一致，且更快。Q/K/V 维持原 dtype。
    import torch.nn.functional as _F

    scale = 1.0 / math.sqrt(q_row.shape[-1])

    if q_row.dim() == 2:
        q4 = q_row.unsqueeze(0)                     # [1, Lq, D]
        k4 = k_row.unsqueeze(0)                     # [1, Lk, D]
        v4 = v_row.unsqueeze(0)
        m4 = mask.unsqueeze(0)                      # [1, Lq, Lk]
        out = _F.scaled_dot_product_attention(q4, k4, v4, attn_mask=m4, scale=scale)
        return out.squeeze(0)

    if q_row.dim() != 3:
        raise ValueError("TorchReferenceBackend attention expects packed rows with 2 or 3 dims")

    # 适配 GQA：把 KV head 复制到与 Q head 数一致（SDPA 旧版本不支持原生 GQA）。
    q_heads = q_row.shape[1]
    kv_heads = k_row.shape[1]
    if q_heads != kv_heads:
        if q_heads % kv_heads != 0:
            raise ValueError("query heads must be a multiple of kv heads for grouped-query attention")
        repeat = q_heads // kv_heads
        k_row = k_row.repeat_interleave(repeat, dim=1)
        v_row = v_row.repeat_interleave(repeat, dim=1)

    # 行内布局 [L, H, D] -> SDPA 期望的 [B=1, H, L, D]
    q4 = q_row.transpose(0, 1).unsqueeze(0).contiguous()
    k4 = k_row.transpose(0, 1).unsqueeze(0).contiguous()
    v4 = v_row.transpose(0, 1).unsqueeze(0).contiguous()
    m4 = mask.unsqueeze(0).unsqueeze(0)             # [1, 1, Lq, Lk] bool
    out = _F.scaled_dot_product_attention(q4, k4, v4, attn_mask=m4, scale=scale)
    return out.squeeze(0).transpose(0, 1)           # 回到 [Lq, H, D]
