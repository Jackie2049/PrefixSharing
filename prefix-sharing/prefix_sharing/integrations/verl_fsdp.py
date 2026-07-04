"""verl FSDP integration helpers for PrefixSharing.

The FSDP path follows the same public shape as the Megatron integration:
``build_*`` returns ``(trimmed_micro_batch, PrefixSharingRuntimeState | None)``.
This module deliberately keeps the first version small and CPU-testable; real
verl/FSDP monkey patches should call into these helpers rather than embedding
prefix-sharing semantics in framework code.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from prefix_sharing.backends.factory import get_backend_instance
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import current_prefix_sharing_context
from prefix_sharing.integrations.megatron_attention import IntegrationUnavailable
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo
from prefix_sharing.integrations.patch_manager import PatchHandle, PatchManager
from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState


@dataclass
class VerlFSDPIntegration:
    """Install PrefixSharing helpers for the verl FSDP path."""

    config: PrefixSharingConfig
    backend: Any | None = None

    def install(self, model_config: Any | None = None) -> PatchHandle:
        self.config.validate(model_config=model_config, integrate_mode="verl_fsdp")
        self._ensure_verl_importable()
        return PatchManager().handle()

    @staticmethod
    def _ensure_verl_importable() -> None:
        try:
            importlib.import_module("verl")
        except ModuleNotFoundError as exc:
            raise IntegrationUnavailable("verl is not importable in this environment") from exc


class PrefixSharingFSDPAttentionRuntime:
    """Standalone FSDP attention runtime for PrefixSharing.

    The first version supports dense Q/K/V tensors shaped ``[B, L, H, D]``.
    It packs only the PrefixSharing Q-path kept tokens, delegates KV expansion
    and attention to the configured backend, then scatters computed outputs back
    to their original dense positions. Reuser prefix positions are intentionally
    left zero here; output/logprob restore fills them later.
    """

    def __init__(self, *, layer_id: int = 0) -> None:
        self.layer_id = layer_id

    def forward(self, attn_func: Any, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        del attn_func, args, kwargs
        ctx = current_prefix_sharing_context()
        if ctx is None:
            raise RuntimeError("PrefixSharingFSDPAttentionRuntime requires active prefix_sharing_runtime_context")
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise RuntimeError("PrefixSharing FSDP attention runtime currently expects dense [B, L, H, D] Q/K/V")
        if query.shape != key.shape or query.shape != value.shape:
            raise RuntimeError("query, key, and value must have the same dense shape")

        plan = ctx.prefix_sharing_plan
        packed_query = _pack_dense_qkv(query, plan)
        packed_key = _pack_dense_qkv(key, plan)
        packed_value = _pack_dense_qkv(value, plan)
        expanded_key, expanded_value = ctx.attention_backend.build_kv(
            packed_key,
            packed_value,
            ctx.store,
            plan,
            packed_batch_layout=ctx.packed_batch_layout,
            layer_id=self.layer_id,
            tp_rank=getattr(ctx.parallel_info, "tp_rank", 0),
            stats=ctx.stats,
        )
        packed_output = ctx.attention_backend.attention(
            packed_query,
            expanded_key,
            expanded_value,
            plan,
            packed_batch_layout=ctx.packed_batch_layout,
        )
        return _scatter_packed_output_to_dense(packed_output, query, plan)


def build_prefix_sharing_micro_batch_fsdp(
    batch: Any,
    config: PrefixSharingConfig,
    *,
    model_config: Any | None = None,
    backend: Any | None = None,
) -> tuple[Any, PrefixSharingRuntimeState | None]:
    """Build a trimmed FSDP micro-batch and PrefixSharing runtime state.

    This helper is intentionally framework-light: it expects 2D
    ``input_ids``/``attention_mask`` and returns the original batch unchanged
    when prefix sharing is disabled or no reusable prefix is detected.
    """

    if not config.enable_prefix_sharing:
        return batch, None
    config.validate(model_config=model_config, integrate_mode="verl_fsdp")

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].to(bool)
    if input_ids.dim() != 2 or attention_mask.dim() != 2:
        raise RuntimeError("prefix sharing FSDP path expects 2D input_ids/attention_mask")
    if input_ids.shape != attention_mask.shape:
        raise RuntimeError("input_ids and attention_mask must have the same shape")

    valid_indices = [
        attention_mask[row].nonzero(as_tuple=False).flatten()
        for row in range(input_ids.shape[0])
    ]
    sequences = [
        input_ids[row, indices].detach().cpu().tolist()
        for row, indices in enumerate(valid_indices)
    ]
    prefix_sharing_plan = PrefixSharingPlanner(config).plan(sequences)
    if not prefix_sharing_plan.has_sharing:
        return batch, None

    trimmed_micro_batch = _clone_batch(batch)
    trimmed_attention_mask = attention_mask.clone()
    trimmed_attention_mask[:] = False

    for row, indices in enumerate(valid_indices):
        keep_start, keep_end = prefix_sharing_plan.input_keep_ranges[row]
        kept_indices = indices[keep_start:keep_end]
        trimmed_attention_mask[row, kept_indices] = True

    trimmed_micro_batch["attention_mask"] = trimmed_attention_mask

    if "loss_mask" in trimmed_micro_batch:
        trimmed_loss_mask = trimmed_micro_batch["loss_mask"].to(bool).clone()
        trimmed_loss_mask[:] = False
        for row, indices in enumerate(valid_indices):
            keep_start, keep_end = prefix_sharing_plan.loss_mask_keep_ranges[row]
            trimmed_loss_mask[row, indices[keep_start:keep_end]] = batch["loss_mask"][row, indices[keep_start:keep_end]].to(bool)
        trimmed_micro_batch["loss_mask"] = trimmed_loss_mask

    packed_batch_layout = PackedBatchLayout.from_valid_lengths(prefix_sharing_plan.kept_lengths_q)
    runtime_state = PrefixSharingRuntimeState(
        prefix_sharing_plan=prefix_sharing_plan,
        attention_backend=get_backend_instance(config, backend),
        packed_batch_layout=packed_batch_layout,
        parallel_info=MegatronParallelInfo(),
        kept_position_ids=trimmed_micro_batch.get("position_ids"),
    )
    return trimmed_micro_batch, runtime_state


def restore_prefix_sharing_outputs_2d(
    output: dict[str, Any],
    log_probs_fn: Any,
) -> dict[str, Any]:
    """Restore 2D FSDP outputs after PrefixSharing Q-path trimming.

    Restores both prefix regions:
    - interior prefix columns copy provider logp/entropy/logits/attention output;
    - prefix-last logp is recomputed with provider logits and the reuser label,
      while entropy/logits/attention output are copied from the provider.
    """

    ctx = current_prefix_sharing_context()
    if ctx is None:
        return output
    plan = ctx.prefix_sharing_plan
    if not plan.has_sharing:
        return output

    log_probs = output.get("log_probs")
    if log_probs is None:
        return output

    import torch

    entropy = output.get("entropy")
    logits = output.get("logits")
    attention_output = output.get("attention_output")

    prefix_last_spec_by_reuser = {
        spec.reuse_idx_in_batch: spec for spec in ctx.prefix_last_restore_indices
    }
    restored_reusers = 0

    for reuser_idx in range(1, plan.batch_size):
        prefix_len = plan.prefix_lens[reuser_idx]
        provider_idx = plan.provider_index[reuser_idx]
        if provider_idx == reuser_idx or prefix_len <= 0:
            continue

        if prefix_len - 1 > 0:
            log_probs[reuser_idx, 0:prefix_len - 1] = log_probs[provider_idx, 0:prefix_len - 1]

        if entropy is not None:
            entropy[reuser_idx, 0:prefix_len] = entropy[provider_idx, 0:prefix_len]
        if logits is not None:
            logits[reuser_idx, 0:prefix_len] = logits[provider_idx, 0:prefix_len]
        if attention_output is not None:
            attention_output[reuser_idx, 0:prefix_len] = attention_output[provider_idx, 0:prefix_len]

        prefix_last_spec = prefix_last_spec_by_reuser.get(reuser_idx)
        if prefix_last_spec is not None:
            saved_logits_key = (reuser_idx, prefix_last_spec.target_2d_pos)
            saved_provider_logits = ctx.prefix_last_logits_saved.get(saved_logits_key)
            if saved_provider_logits is None:
                if logits is None:
                    raise KeyError(saved_logits_key)
                saved_provider_logits = logits[
                    provider_idx,
                    prefix_len - 1:prefix_len,
                ]
            reuser_label = torch.tensor(
                [prefix_last_spec.label_value],
                dtype=torch.long,
                device=log_probs.device,
            )
            log_probs[reuser_idx, prefix_len - 1] = log_probs_fn(
                saved_provider_logits,
                reuser_label,
            ).reshape(())
        else:
            log_probs[reuser_idx, prefix_len - 1] = log_probs[provider_idx, prefix_len - 1]

        restored_reusers += 1

    if ctx.stats is not None:
        ctx.stats.record_restore(restored_reusers)
    return output


def _clone_batch(batch: Any) -> Any:
    if hasattr(batch, "clone"):
        try:
            return batch.clone()
        except TypeError:
            pass
    if isinstance(batch, dict):
        return dict(batch)
    return batch.copy()


def _pack_dense_qkv(tensor: Any, plan: Any) -> Any:
    rows = []
    for row, (start, end) in enumerate(plan.input_keep_ranges):
        rows.append(tensor[row, start:end])
    if not rows:
        return tensor.new_empty((0, *tensor.shape[2:]))
    import torch

    return torch.cat(rows, dim=0)


def _scatter_packed_output_to_dense(packed_output: Any, dense_like: Any, plan: Any) -> Any:
    output = dense_like.new_zeros(dense_like.shape)
    cursor = 0
    for row, (start, end) in enumerate(plan.input_keep_ranges):
        length = end - start
        if length <= 0:
            continue
        output[row, start:end] = packed_output[cursor:cursor + length]
        cursor += length
    return output
