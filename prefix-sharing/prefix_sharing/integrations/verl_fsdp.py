"""verl FSDP integration helpers for PrefixSharing.

The FSDP path follows the same public shape as the Megatron integration:
``build_*`` returns ``(trimmed_micro_batch, PrefixSharingRuntimeState | None)``.
The helpers stay framework-light enough for CPU tests, while the explicit
``verl080_fsdp`` patch set wires them into ``FSDPEngineWithLMHead.forward_step``.
"""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from prefix_sharing.backends.factory import get_backend_instance
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import current_prefix_sharing_context
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.integrations.megatron_attention import IntegrationUnavailable
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo
from prefix_sharing.integrations.patch_manager import PatchHandle, PatchManager
from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState
from prefix_sharing.integrations.verl_mcore import _collect_kept_position_rows
from prefix_sharing.integrations.verl_mcore import _extract_seq_from_nested_tensor
from prefix_sharing.integrations.verl_mcore import _is_nested_tensor
from prefix_sharing.integrations.verl_mcore import _trim_nested_batch

_SUPPORTED_TRANSFORMERS_ATTENTIONS = {
    "flash_attention_2",
    "flash_attention_3",
    "sdpa",
    "flex_attention",
    "eager",
}


@dataclass
class VerlFSDPIntegration:
    """Install PrefixSharing helpers for the verl FSDP path."""

    config: PrefixSharingConfig
    backend: Any | None = None

    def install(self, model_config: Any | None = None) -> PatchHandle:
        self.config.validate(model_config=model_config, integrate_mode="verl_fsdp")
        self._ensure_verl_importable()
        return self._install_transformers_attention_patch()

    @staticmethod
    def _ensure_verl_importable() -> None:
        try:
            importlib.import_module("verl")
        except ModuleNotFoundError as exc:
            raise IntegrationUnavailable("verl is not importable in this environment") from exc

    @staticmethod
    def _install_transformers_attention_patch() -> PatchHandle:
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        except ModuleNotFoundError as exc:
            raise IntegrationUnavailable("transformers is not importable in this environment") from exc

        manager = PatchManager()
        for name in list(ALL_ATTENTION_FUNCTIONS.keys()):
            if name in _SUPPORTED_TRANSFORMERS_ATTENTIONS:
                manager.patch_item(
                    ALL_ATTENTION_FUNCTIONS,
                    name,
                    _create_prefix_sharing_attention_wrapper(ALL_ATTENTION_FUNCTIONS[name]),
                )
        return manager.handle()


class PrefixSharingFSDPAttentionRuntime:
    """Standalone FSDP attention runtime for PrefixSharing.

    The first version supports dense Q/K/V tensors shaped ``[B, L, H, D]`` and
    packed single-batch tensors shaped ``[1, T, H, D]`` from verl remove-padding
    prepare. Dense input is packed by kept Q-path tokens and scattered back to
    original positions; packed input is returned in packed shape. Reuser prefix
    positions are restored later by the output/logprob restore step.
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
        if query.shape[0] == 1 and key.shape[0] == 1 and value.shape[0] == 1:
            packed_query = query.squeeze(0)
            packed_key = key.squeeze(0)
            packed_value = value.squeeze(0)
            packed_output = _run_packed_attention_runtime(
                ctx,
                packed_query,
                packed_key,
                packed_value,
                layer_id=self.layer_id,
            )
            return packed_output.unsqueeze(0)
        if query.shape[:2] != key.shape[:2] or query.shape[:2] != value.shape[:2]:
            raise RuntimeError("query, key, and value must share dense batch/sequence dimensions")

        plan = ctx.prefix_sharing_plan
        packed_query = _pack_dense_qkv(query, plan)
        packed_key = _pack_dense_qkv(key, plan)
        packed_value = _pack_dense_qkv(value, plan)
        packed_output = _run_packed_attention_runtime(
            ctx,
            packed_query,
            packed_key,
            packed_value,
            layer_id=self.layer_id,
        )
        return _scatter_packed_output_to_dense(packed_output, query, plan)


def forward_prefix_sharing_fsdp_micro_batch(
    micro_batch: Any,
    model: Any,
    config: PrefixSharingConfig,
    *,
    model_config: Any | None = None,
    backend: Any | None = None,
    temperature: float = 1.0,
    calculate_entropy: bool = False,
    log_probs_fn: Any | None = None,
    entropy_fn: Any | None = None,
    autocast_context: Any | None = None,
) -> dict[str, Any]:
    """Run one dense verl/FSDP-style micro-batch with PrefixSharing.

    This is the executable helper used by fake/local FSDP tests and by engines
    that do not expose prepare hooks:
    prepare the micro-batch, open the runtime context, run the model with a
    PrefixSharing attention runtime, compute token-level outputs, then restore
    reuser prefix columns. Real verl FSDP patching should prefer the engine's
    own ``prepare_model_inputs`` / ``prepare_model_outputs`` path.
    """

    trimmed_micro_batch, runtime_state = build_prefix_sharing_micro_batch_fsdp(
        micro_batch,
        config,
        model_config=model_config,
        backend=backend,
    )
    context = prefix_sharing_runtime_context(runtime_state) if runtime_state is not None else nullcontext(None)
    autocast = autocast_context if autocast_context is not None else nullcontext()

    with context as ctx, autocast:
        model_output = _call_fsdp_model(
            model,
            trimmed_micro_batch,
            prefix_sharing_runtime=PrefixSharingFSDPAttentionRuntime(),
            enable_prefix_sharing=runtime_state is not None,
        )
        logits = _extract_logits(model_output) / float(temperature)
        output = {
            "model_output": model_output,
            "logits": logits.clone(),
        }
        labels = _labels_for_log_probs(micro_batch)
        if labels is not None:
            output["log_probs"] = _compute_log_probs(logits, labels, log_probs_fn)
        if calculate_entropy:
            output["entropy"] = _compute_entropy(logits, entropy_fn)
        attention_output = getattr(model_output, "attention_output", None)
        if attention_output is not None:
            output["attention_output"] = attention_output.clone()

        if ctx is not None:
            _save_prefix_last_logits(ctx, logits)
            if "log_probs" in output:
                restore_prefix_sharing_outputs_2d(output, log_probs_fn or _default_log_probs_fn)
        return output


def build_prefix_sharing_micro_batch_fsdp(
    batch: Any,
    config: PrefixSharingConfig,
    *,
    model_config: Any | None = None,
    backend: Any | None = None,
) -> tuple[Any, PrefixSharingRuntimeState | None]:
    """Build a trimmed FSDP micro-batch and PrefixSharing runtime state.

    This helper is intentionally framework-light: it accepts dense 2D
    ``input_ids``/``attention_mask`` or jagged NestedTensor ``input_ids`` from
    verl remove-padding, and returns the original batch unchanged when prefix
    sharing is disabled or no reusable prefix is detected.
    """

    if not config.enable_prefix_sharing:
        return batch, None
    config.validate(model_config=model_config, integrate_mode="verl_fsdp")

    input_ids = batch["input_ids"]
    is_nested_input = _is_nested_tensor(input_ids)
    if is_nested_input:
        sequences = _extract_seq_from_nested_tensor(input_ids)
        valid_indices = None
        attention_mask = None
    else:
        attention_mask = batch["attention_mask"].to(bool)
        if input_ids.dim() != 2 or attention_mask.dim() != 2:
            raise RuntimeError("prefix sharing FSDP path expects 2D or jagged NestedTensor input_ids")
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

    if is_nested_input:
        trimmed_micro_batch = _trim_nested_batch(batch, prefix_sharing_plan)
        kept_position_rows = _collect_kept_position_rows(
            trimmed_micro_batch,
            prefix_sharing_plan,
            is_nested_tensor=True,
        )
    else:
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
        kept_position_rows = _collect_kept_position_rows(
            trimmed_micro_batch,
            prefix_sharing_plan,
            is_nested_tensor=False,
            attention_mask_bool=attention_mask,
        )

    packed_batch_layout = PackedBatchLayout.from_kept_position_rows(
        kept_position_rows,
        align_size=1,
    )
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


def _run_packed_attention_runtime(
    ctx: Any,
    packed_query: Any,
    packed_key: Any,
    packed_value: Any,
    *,
    layer_id: int,
) -> Any:
    from prefix_sharing.tools.perf_profiler import PerfProfiler

    plan = ctx.prefix_sharing_plan
    profiler = PerfProfiler.current()

    if profiler is not None:
        profiler.start_phase(PerfProfiler.PHASE_ATTN_KV)
    expanded_key, expanded_value = ctx.attention_backend.build_kv(
        packed_key,
        packed_value,
        ctx.store,
        plan,
        packed_batch_layout=ctx.packed_batch_layout,
        layer_id=layer_id,
        tp_rank=getattr(ctx.parallel_info, "tp_rank", 0),
        stats=ctx.stats,
    )
    if profiler is not None:
        profiler.stop_phase(PerfProfiler.PHASE_ATTN_KV)

    if profiler is not None:
        profiler.start_phase(PerfProfiler.PHASE_ATTN_COMPUTE)
    output = ctx.attention_backend.attention(
        packed_query,
        expanded_key,
        expanded_value,
        plan,
        packed_batch_layout=ctx.packed_batch_layout,
    )
    if profiler is not None:
        profiler.stop_phase(PerfProfiler.PHASE_ATTN_COMPUTE)

    return output


def _create_prefix_sharing_attention_wrapper(original_fn: Any) -> Any:
    """Wrap HF attention registry functions with PrefixSharing support."""

    def wrapped(module: Any, query: Any, key: Any, value: Any, attention_mask: Any, *args: Any, **kwargs: Any) -> Any:
        prefix_sharing_runtime = kwargs.pop("prefix_sharing_runtime", None)
        if prefix_sharing_runtime is None:
            return original_fn(module, query, key, value, attention_mask, *args, **kwargs)

        def attn_func(q: Any, k: Any, v: Any, *inner_args: Any, **inner_kwargs: Any) -> Any:
            result = original_fn(module, q, k, v, attention_mask, *inner_args, **inner_kwargs)
            return result[0] if isinstance(result, tuple) else result

        return prefix_sharing_runtime.forward(attn_func, query, key, value, *args, **kwargs), None

    return wrapped


def _call_fsdp_model(
    model: Any,
    micro_batch: Any,
    *,
    prefix_sharing_runtime: PrefixSharingFSDPAttentionRuntime,
    enable_prefix_sharing: bool,
) -> Any:
    model_inputs = {
        key: micro_batch[key]
        for key in ("input_ids", "attention_mask", "position_ids")
        if key in micro_batch
    }
    model_inputs["use_cache"] = False
    if enable_prefix_sharing:
        model_inputs["prefix_sharing_runtime"] = prefix_sharing_runtime
    try:
        return model(**model_inputs)
    except TypeError:
        if not enable_prefix_sharing:
            raise
        # Some local smoke models and older HF modules may not accept unknown
        # kwargs at the top-level forward. In that case the attention patch is
        # expected to obtain the runtime from a framework-specific closure.
        model_inputs.pop("prefix_sharing_runtime", None)
        return model(**model_inputs)


def _extract_logits(model_output: Any) -> Any:
    if isinstance(model_output, dict):
        return model_output["logits"]
    return model_output.logits


def _labels_for_log_probs(micro_batch: Any) -> Any | None:
    if "labels" in micro_batch:
        return micro_batch["labels"]
    if "input_ids" not in micro_batch:
        return None
    import torch

    input_ids = micro_batch["input_ids"]
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = 0
    return labels


def _compute_log_probs(logits: Any, labels: Any, log_probs_fn: Any | None) -> Any:
    return (log_probs_fn or _default_log_probs_fn)(logits, labels)


def _default_log_probs_fn(logits: Any, labels: Any) -> Any:
    import torch

    safe_labels = labels.long().clamp_min(0) % logits.shape[-1]
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)


def _compute_entropy(logits: Any, entropy_fn: Any | None) -> Any:
    if entropy_fn is not None:
        return entropy_fn(logits)
    import torch

    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def _save_prefix_last_logits(ctx: Any, logits: Any) -> None:
    for index in ctx.prefix_last_restore_indices:
        key = (index.reuse_idx_in_batch, index.target_2d_pos)
        ctx.prefix_last_logits_saved[key] = logits[
            index.provider_idx_in_batch,
            index.target_2d_pos:index.target_2d_pos + 1,
        ]


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
