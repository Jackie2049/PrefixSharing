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
    PREFIX_STATE_TYPE_DELTANET_STATE,
    PrefixActivationSlotId,
    PrefixAttentionStore,
    PrefixDeltanetStore,
)


class TorchReferenceBackend:
    capabilities = BackendCapabilities(
        name="torch_ref",
        supports_cpu=True,
        supports_cuda=True,
        supports_cann=True,
        supports_different_q_kv_lengths=True,
        supports_prefix_last_restore=True,
        supports_gated_attention=True,
        supports_deltanet_state_reuse=True,
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
        expanded_keys = []
        expanded_values = []
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
            if not prefix_sharing_plan.is_reuser(batch_index):
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
                    key_tensor=valid_key_row,
                    value_tensor=valid_value_row,
                    prefix_len=valid_key_row.shape[0],
                    overwrite=True,
                )
                store_count += 1
                stored_tokens += int(valid_key_row.shape[0])
                expanded_keys.append(valid_key_row)
                expanded_values.append(valid_value_row)
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
                            expanded_kv_tokens=sum(int(row.shape[0]) for row in expanded_keys),
                            valid_q_tokens=layout.total_valid_length,
                            padded_q_tokens=layout.total_padded_length,
                        )
                    raise
                reuse_hit_count += 1
                prefix_len = prefix_sharing_plan.prefix_lens[batch_index]
                reused_prefix_tokens += int(prefix_len)
                expanded_key = torch.cat([entry.key_tensor[:prefix_len], valid_key_row], dim=0)
                expanded_value = torch.cat([entry.value_tensor[:prefix_len], valid_value_row], dim=0)
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
                    key_tensor=expanded_key,
                    value_tensor=expanded_value,
                    prefix_len=expanded_key.shape[0],
                    overwrite=True,
                )
                store_count += 1
                stored_tokens += int(expanded_key.shape[0])
                expanded_keys.append(expanded_key)
                expanded_values.append(expanded_value)
        if stats is not None:
            stats.record_attention_kv_build(
                layer_id=layer_id,
                store_count=store_count,
                reuse_count=reuse_count,
                reuse_hit_count=reuse_hit_count,
                reuse_miss_count=reuse_miss_count,
                stored_tokens=stored_tokens,
                reused_prefix_tokens=reused_prefix_tokens,
                expanded_kv_tokens=sum(int(row.shape[0]) for row in expanded_keys),
                valid_q_tokens=layout.total_valid_length,
                padded_q_tokens=layout.total_padded_length,
            )
        return torch.cat(expanded_keys, dim=0), torch.cat(expanded_values, dim=0)

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

    def gated_attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        gate: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Apply Qwen3.5-style output gate after prefix-expanded attention.

        The gate is derived from the current kept hidden states in the model, so
        prefix sharing must not cache it. This reference helper keeps that
        invariant explicit for future HybridAttention integrations.
        """

        attention_output = self.attention(
            query,
            key,
            value,
            prefix_sharing_plan,
            packed_batch_layout=packed_batch_layout,
            **kwargs,
        )
        if attention_output.shape != gate.shape:
            raise ValueError("gate shape must match attention output shape")
        return attention_output * torch.sigmoid(gate)

    def build_deltanet_states(
        self,
        state_update: Any,
        store: PrefixDeltanetStore,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        layer_id: int,
        tp_rank: int = 0,
    ) -> Any:
        """Build prefix-expanded Qwen3.5 GatedDeltaNet recurrent trajectories.

        This reference uses a cumulative recurrent trajectory to verify the
        critical prefix boundary and autograd semantics; real integrations
        should map the same store entry to the engine's recurrent/conv cache
        params.
        """

        layout = packed_batch_layout or PackedBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q)
        update_rows = _split_packed(state_update, layout.padded_lengths)
        outputs = []
        for batch_index, update_row in enumerate(update_rows):
            valid_length = layout.valid_lengths[batch_index]
            valid_update_row = update_row[:valid_length]
            if not prefix_sharing_plan.is_reuser(batch_index):
                state_trajectory = torch.cumsum(valid_update_row, dim=0)
                slot_id = PrefixActivationSlotId(
                    prefix_sharing_plan.forward_id,
                    prefix_sharing_plan.micro_batch_id,
                    layer_id,
                    batch_index,
                    PREFIX_STATE_TYPE_DELTANET_STATE,
                    tp_rank,
                )
                # Publish provider state so later reusers can start from the
                # exact prefix boundary instead of recomputing the prefix.
                store.store(
                    slot_id,
                    recurrent_state=state_trajectory,
                    prefix_len=state_trajectory.shape[0],
                    overwrite=True,
                )
                outputs.append(_pad_like_row(state_trajectory, update_row))
                continue

            provider = prefix_sharing_plan.provider_index[batch_index]
            provider_slot_id = PrefixActivationSlotId(
                prefix_sharing_plan.forward_id,
                prefix_sharing_plan.micro_batch_id,
                layer_id,
                provider,
                PREFIX_STATE_TYPE_DELTANET_STATE,
                tp_rank,
            )
            # The provider trajectory is indexed at prefix_len - 1 to obtain the
            # reusable state after the shared prefix has been consumed.
            entry = store.load(provider_slot_id)
            prefix_len = prefix_sharing_plan.prefix_lens[batch_index]
            if prefix_len <= 0:
                initial_state = torch.zeros_like(valid_update_row[:1]).squeeze(0)
                provider_prefix_trajectory = valid_update_row[:0]
            else:
                if prefix_len > entry.recurrent_state.shape[0]:
                    raise ValueError("prefix_len exceeds stored provider activation length")
                initial_state = entry.recurrent_state[prefix_len - 1]
                provider_prefix_trajectory = entry.recurrent_state[:prefix_len]
            suffix_trajectory = initial_state + torch.cumsum(valid_update_row, dim=0)
            own_state_trajectory = torch.cat([provider_prefix_trajectory, suffix_trajectory], dim=0)
            own_slot_id = PrefixActivationSlotId(
                prefix_sharing_plan.forward_id,
                prefix_sharing_plan.micro_batch_id,
                layer_id,
                batch_index,
                PREFIX_STATE_TYPE_DELTANET_STATE,
                tp_rank,
            )
            # Publish the expanded reuser trajectory for transitive reuse by a later row.
            store.store(
                own_slot_id,
                recurrent_state=own_state_trajectory,
                prefix_len=own_state_trajectory.shape[0],
                overwrite=True,
            )
            outputs.append(_pad_like_row(suffix_trajectory, update_row))
        return torch.cat(outputs, dim=0)


def _split_packed(tensor: Any, lengths: list[int]) -> list[Any]:
    if not lengths:
        return []
    if sum(lengths) != tensor.shape[0]:
        raise ValueError("packed tensor first dimension does not match lengths")
    return list(torch.split(tensor, lengths, dim=0))


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


def _pad_like_row(valid_row: Any, packed_row: Any) -> Any:
    if valid_row.shape[0] == packed_row.shape[0]:
        return valid_row
    padded_row = torch.zeros_like(packed_row)
    padded_row[: valid_row.shape[0]] = valid_row
    return padded_row
