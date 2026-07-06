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
    PREFIX_STATE_TYPE_DELTANET_STATE,
    PrefixActivationSlotId,
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

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------
    def _ensure_3d_thd(self, tensor: Any, name: str) -> Any:
        if tensor.dim() == 3:
            return tensor
        if tensor.dim() == 2:
            raise ValueError(
                f"{name} has 2 dims {tuple(tensor.shape)}; expected "
                "(total_tokens, num_heads, head_dim)"
            )
        raise ValueError(f"{name} has unexpected rank {tensor.dim()}")

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
        store: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        layer_id: int,
        tp_rank: int = 0,
        stats: PrefixSharingStats | None = None,
    ) -> tuple[Any, Any]:
        """构建去重 s_packed KV。

        框架传入的 key/value 包含所有 tokens 的 K/V（含 prefix 和 suffix）。
        逐 input 提取 suffix K/V，拼接为 s_packed KV，无需 torch.cat 或 store。
        """
        layout = packed_batch_layout or PackedBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q)
        key_rows = _split_packed(key, layout.padded_lengths)
        value_rows = _split_packed(value, layout.padded_lengths)

        s_packed_k = []
        s_packed_v = []
        stored_tokens = 0

        for batch_index in range(prefix_sharing_plan.batch_size):
            valid_length = layout.valid_lengths[batch_index]
            if valid_length == 0:
                continue

            prefix_len = prefix_sharing_plan.prefix_lens[batch_index]
            suffix_len = valid_length
            suffix_k = key_rows[batch_index][:suffix_len]
            suffix_v = value_rows[batch_index][:suffix_len]

            if not prefix_sharing_plan.is_reuser(batch_index):
                # Provider: 整行 K/V 都是 unique suffix
                s_packed_k.append(suffix_k)
                s_packed_v.append(suffix_v)
                stored_tokens += suffix_len
            else:
                # Reuser: prefix 在 s_packed 中已存在（由之前的 provider/reuser 构建），
                # 只追加 suffix
                s_packed_k.append(suffix_k)
                s_packed_v.append(suffix_v)
                stored_tokens += suffix_len

        if stats is not None:
            stats.record_attention_kv_build(
                layer_id=layer_id,
                store_count=prefix_sharing_plan.batch_size,
                reuse_count=0,
                reuse_hit_count=0,
                reuse_miss_count=0,
                stored_tokens=stored_tokens,
                reused_prefix_tokens=0,
                expanded_kv_tokens=stored_tokens,
                valid_q_tokens=layout.total_valid_length,
                padded_q_tokens=layout.total_padded_length,
            )

        return torch.cat(s_packed_k, dim=0), torch.cat(s_packed_v, dim=0)

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
        """s_packed 模式下的统一 attention。

        Q: (total_q, n_heads, d) — per-input suffix tokens, padded per input
        K/V: (s_packed_length, n_kv_heads, d) — s_packed unique KV (去重)
        mask: (total_q, s_packed_length) — 全局 custom causal mask
        """
        plan = prefix_sharing_plan
        layout = packed_batch_layout or PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

        q = self._ensure_3d_thd(query, "query")
        k = self._ensure_3d_thd(key, "key")
        v = self._ensure_3d_thd(value, "value")

        # Q total = sum of padded lengths (since input Q is padded)
        total_q = sum(layout.padded_lengths)
        if q.shape[0] != total_q:
            raise ValueError(
                f"query.shape[0]={q.shape[0]} != total_q={total_q}"
            )
        if k.shape[0] != plan.s_packed_length:
            raise ValueError(
                f"key.shape[0]={k.shape[0]} != s_packed_length={plan.s_packed_length}"
            )
        if total_q == 0 or plan.s_packed_length == 0:
            return torch.zeros_like(q)

        # 全局 custom causal mask (sparse, only for valid Q tokens)
        mask = plan.build_global_custom_mask(q.device)

        # 逐行 attention（每行对应 layout.padded_lengths 中一个元素）
        padded_offset = 0
        valid_q_offset = 0  # 累积的 s_packed Q 位置（仅 valid tokens）
        outputs: list[Any] = []
        for batch_index in range(plan.batch_size):
            padded_len = layout.padded_lengths[batch_index]
            q_len = plan.s_packed_q_lengths[batch_index]

            if q_len == 0:
                # 无 Q token: 全是 padding
                outputs.append(q[padded_offset:padded_offset + padded_len])
                padded_offset += padded_len
                continue

            # 提取 valid Q tokens (从 padded 区间中提取 valid 部分)
            q_valid = q[padded_offset:padded_offset + q_len]
            k_row = k
            v_row = v

            mask_row = mask[valid_q_offset:valid_q_offset + q_len]
            out_valid = _attention_row(q_valid, k_row, v_row, mask_row)

            # Repad to padded_len if needed
            if q_len == padded_len:
                outputs.append(out_valid)
            else:
                padded = torch.zeros(padded_len, *out_valid.shape[1:], dtype=out_valid.dtype, device=out_valid.device)
                padded[:q_len] = out_valid
                outputs.append(padded)

            padded_offset += padded_len
            valid_q_offset += q_len

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
    scale = math.sqrt(q_row.shape[-1])
    if q_row.dim() == 2:
        scores = q_row @ k_row.transpose(-1, -2) / scale
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        return probs @ v_row
    if q_row.dim() != 3:
        raise ValueError("TorchReferenceBackend attention expects packed rows with 2 or 3 dims")

    # 适配GQA
    q_heads = q_row.shape[1]
    kv_heads = k_row.shape[1]
    if q_heads != kv_heads:
        if q_heads % kv_heads != 0:
            raise ValueError("query heads must be a multiple of kv heads for grouped-query attention")
        repeat = q_heads // kv_heads
        k_row = k_row.repeat_interleave(repeat, dim=1)
        v_row = v_row.repeat_interleave(repeat, dim=1)

    # 注意力计算
    scores = torch.einsum("qhd,khd->hqk", q_row, k_row) / scale
    scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v_row)


def _pad_like_row(valid_row: Any, packed_row: Any) -> Any:
    if valid_row.shape[0] == packed_row.shape[0]:
        return valid_row
    padded_row = torch.zeros_like(packed_row)
    padded_row[: valid_row.shape[0]] = valid_row
    return padded_row
