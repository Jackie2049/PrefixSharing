"""Prefix-sharing observability statistics.

This module only records lightweight counters needed for diagnostics and does
not alter the computational semantics of prefix-sharing. Statistics objects
are bound to the lifetime of a single ``PrefixSharingRuntimeContext`` and are
used to compare the planner's theoretical reuse benefit against the runtime's
actual KV reuse behavior.
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
    """Per-layer KV reuse statistics for a single attention layer."""

    layer_id: int
    # Number of KV entries written to PrefixAttentionStore for this layer,
    # including both provider raw KV and reuser expanded KV.
    store_count: int = 0
    # Number of times this layer attempted to reuse provider KV; should
    # normally equal the number of reusers in this layer.
    reuse_count: int = 0
    # Number of times this layer successfully reused provider KV; used to
    # confirm that the actual reuse path is effective.
    reuse_hit_count: int = 0
    # Number of times this layer failed to reuse provider KV; non-zero
    # usually indicates provider ordering or store key issues.
    reuse_miss_count: int = 0
    # Total KV tokens written to PrefixAttentionStore for this layer, counting
    # only valid tokens (excluding TP/CP padding).
    stored_tokens: int = 0
    # Total prefix tokens actually reused from provider KV in this layer; the
    # key metric for confirming that reuse really happened.
    reused_prefix_tokens: int = 0
    # Total expanded KV tokens passed to attention after build_kv in this
    # layer; equals the sum of each sample's original valid length.
    expanded_kv_tokens: int = 0
    # Valid tokens that actually participate in query computation in this
    # layer, excluding packed padding slots.
    valid_q_tokens: int = 0
    # Token slots in the packed query tensor for this layer, including TP/CP
    # padding; useful for determining whether padding is consuming the benefit.
    padded_q_tokens: int = 0


@dataclass
class PrefixSharingStats:
    """Per-micro-batch prefix-sharing expected/actual diagnostic statistics."""

    # Unique identifier for the current forward pass, from PrefixSharingPlan;
    # used to correlate logs across the same forward pass.
    forward_id: int
    # Micro-batch identifier from PrefixSharingPlan; used to locate batch-level
    # reuse effectiveness.
    micro_batch_id: int
    # Number of samples in the current micro-batch.
    batch_size: int
    # Total original valid tokens (attention_mask == true before trimming).
    original_tokens: int
    # Total valid tokens that participate in computation after trimming,
    # excluding packed padding.
    kept_valid_tokens: int
    # Total token slots in the packed tensor after TP/CP padding, including
    # padding slots.
    kept_padded_tokens: int
    # Theoretically reused valid tokens via prefix-sharing; equals
    # original_tokens - kept_valid_tokens.
    reused_valid_tokens: int
    # Theoretical valid-token reuse ratio; useful for determining whether lack
    # of performance gain is due to a low reuse ratio itself.
    reused_valid_token_ratio: float
    # Number of provider samples that are reused by other samples.
    provider_count: int
    # Number of reuser samples that reuse another sample's prefix.
    reuser_count: int
    # Number of shared prefix groups detected in the current micro-batch.
    sharing_group_count: int
    # Expected KV load count per layer; should normally equal reuser_count.
    expected_reused_counts_per_layer: int
    # Expected total prefix tokens loaded from provider KV per layer.
    expected_reused_prefix_tokens_per_layer: int
    # Theoretical number of positions requiring prefix-last restore.
    expected_restore_count: int
    # Number of prefix-last restores actually executed by the runtime.
    actual_restore_count: int = 0
    # Per-layer runtime KV reuse statistics, aggregated by layer_id.
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
