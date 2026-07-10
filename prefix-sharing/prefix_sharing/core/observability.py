"""Prefix-sharing 可观测性统计。

本模块只记录诊断所需的轻量计数，不改变 prefix-sharing 的计算语义。
统计对象绑定在单个 ``PrefixSharingRuntimeContext`` 生命周期内，用来对比
planner 的理论复用收益和 runtime 的真实 KV 复用行为。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from prefix_sharing.core.planner import PrefixSharingPlan

if TYPE_CHECKING:
    # PackedBatchLayout is used only for type annotations (kept_padded_tokens
    # is read via duck-typing at runtime). Importing it eagerly would create a
    # circular import: observability -> backends.packed_layout ->
    # backends.__init__ -> torch_ref -> core.observability.
    from prefix_sharing.backends.packed_layout import PackedBatchLayout


@dataclass
class PrefixSharingLayerStats:
    """单个 attention layer 的真实 KV 复用统计。"""

    layer_id: int
    # 本层写入 PrefixAttentionStore 的 KV 条目数，包括 provider 原始 KV 和 reuser 扩展后的 KV。
    store_count: int = 0
    # 本层尝试复用 provider KV 的次数；正常应等于本层 reuser 数。
    reuse_count: int = 0
    # 本层成功复用 provider KV 的次数，用于确认真实复用路径是否生效。
    reuse_hit_count: int = 0
    # 本层复用 provider KV 失败的次数；非 0 通常表示 provider 顺序或 store key 有问题。
    reuse_miss_count: int = 0
    # 本层写入 PrefixAttentionStore 的 KV token 总数，只统计有效 token，不统计 TP/CP padding。
    stored_tokens: int = 0
    # 本层实际从 provider KV 中复用的 prefix token 总数，是判断复用是否真的发生的核心指标。
    reused_prefix_tokens: int = 0
    # 本层 build_kv 后传给 attention 的 expanded KV token 总数，等于各样本原始有效长度之和。
    expanded_kv_tokens: int = 0
    # 本层裁剪后真实参与 query 计算的有效 token 数，不包含 packed padding 槽位。
    valid_q_tokens: int = 0
    # 本层 packed query tensor 的 token 槽位数，包含 TP/CP padding，可用于判断 padding 是否吞掉收益。
    padded_q_tokens: int = 0


@dataclass
class PrefixSharingStats:
    """单个 micro-batch 的 prefix-sharing expected/actual 诊断统计。"""

    # 当前 forward 的唯一编号，来自 PrefixSharingPlan，用于跨日志关联同一次前向。
    forward_id: int
    # 当前 micro-batch 的编号，来自 PrefixSharingPlan，用于定位 batch 级复用效果。
    micro_batch_id: int
    # 当前 micro-batch 的样本数量。
    batch_size: int
    # 原始有效 token 总数，即未裁剪前 attention_mask 为 true 的 token 数。
    original_tokens: int
    # 裁剪后实际参与计算的有效 token 总数，不包含 packed padding。
    kept_valid_tokens: int
    # TP/CP padding 后 packed tensor 中的 token 槽位总数，包含 padding。
    kept_padded_tokens: int
    # 理论上通过 prefix-sharing 复用的有效 token 数，等于 original_tokens - kept_valid_tokens。
    reused_valid_tokens: int
    # 理论有效 token 复用比例，用于判断性能没有提升是否因为复用比例本身过低。
    reused_valid_token_ratio: float
    # 被其他样本复用的 provider 样本数量。
    provider_count: int
    # 复用其他样本 prefix 的 reuser 样本数量。
    reuser_count: int
    # 当前 micro-batch 内检测到的共享前缀组数量。
    sharing_group_count: int
    # 每层理论应发生的 KV load 次数；正常等于 reuser_count。
    expected_reused_counts_per_layer: int
    # 每层理论应从 provider KV 中加载的 prefix token 总数。
    expected_reused_prefix_tokens_per_layer: int
    # 理论需要执行 prefix-last restore 的位置数。
    expected_restore_count: int
    # runtime 实际执行 prefix-last restore 的位置数。
    actual_restore_count: int = 0
    # 按 layer_id 聚合的 runtime KV 复用统计。
    layers: dict[int, PrefixSharingLayerStats] = field(default_factory=dict)

    @classmethod
    def from_plan(
        cls,
        prefix_sharing_plan: PrefixSharingPlan,
        packed_batch_layout: PackedBatchLayout,
    ) -> "PrefixSharingStats":
        original_tokens = sum(prefix_sharing_plan.original_lengths)
        kept_valid_tokens = sum(prefix_sharing_plan.kept_lengths_q)
        reused_valid_tokens = original_tokens - kept_valid_tokens
        reuser_indices = [
            index
            for index in range(prefix_sharing_plan.batch_size)
            if prefix_sharing_plan.is_reuser(index)
        ]
        sharing_groups = {
            (spec.provider_idx_in_batch, spec.prefix_len)
            for spec in prefix_sharing_plan.reuse_specs
        }
        return cls(
            forward_id=prefix_sharing_plan.forward_id,
            micro_batch_id=prefix_sharing_plan.micro_batch_id,
            batch_size=prefix_sharing_plan.batch_size,
            original_tokens=original_tokens,
            kept_valid_tokens=kept_valid_tokens,
            kept_padded_tokens=packed_batch_layout.total_padded_length,
            reused_valid_tokens=reused_valid_tokens,
            reused_valid_token_ratio=(
                reused_valid_tokens / original_tokens if original_tokens > 0 else 0.0
            ),
            provider_count=sum(prefix_sharing_plan.is_provider),
            reuser_count=len(reuser_indices),
            sharing_group_count=len(sharing_groups),
            expected_reused_counts_per_layer=len(reuser_indices),
            expected_reused_prefix_tokens_per_layer=sum(
                prefix_sharing_plan.prefix_lens[index] for index in reuser_indices
            ),
            # Counts reuser rows restored (one bulk slice per reuser), matching
            # record_restore's per-reuser count. interior specs are no longer
            # restored per-position, so don't count them here.
            expected_restore_count=len(reuser_indices),
        )

    def layer(self, layer_id: int) -> PrefixSharingLayerStats:
        if layer_id not in self.layers:
            self.layers[layer_id] = PrefixSharingLayerStats(layer_id=layer_id)
        return self.layers[layer_id]

    def record_attention_kv_build(
        self,
        *,
        layer_id: int,
        store_count: int,
        reuse_count: int,
        reuse_hit_count: int,
        reuse_miss_count: int,
        stored_tokens: int,
        reused_prefix_tokens: int,
        expanded_kv_tokens: int,
        valid_q_tokens: int,
        padded_q_tokens: int,
    ) -> None:
        layer = self.layer(layer_id)
        layer.store_count += store_count
        layer.reuse_count += reuse_count
        layer.reuse_hit_count += reuse_hit_count
        layer.reuse_miss_count += reuse_miss_count
        layer.stored_tokens += stored_tokens
        layer.reused_prefix_tokens += reused_prefix_tokens
        layer.expanded_kv_tokens += expanded_kv_tokens
        layer.valid_q_tokens += valid_q_tokens
        layer.padded_q_tokens += padded_q_tokens

    def record_restore(self, count: int) -> None:
        self.actual_restore_count += count

    def layer_matches_expected(self, layer_id: int) -> bool:
        layer = self.layers.get(layer_id)
        if layer is None:
            return False
        return (
            layer.reuse_count == self.expected_reused_counts_per_layer
            and layer.reused_prefix_tokens == self.expected_reused_prefix_tokens_per_layer
            and layer.reuse_miss_count == 0
        )
