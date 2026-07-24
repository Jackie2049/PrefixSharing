"""Patch: FSDPEngineWithLMHead.forward_step — verl 0.8.0 FSDP 路径。

thin wrapper：读取 prefix_sharing_config，优先复用真实 engine 的
``prepare_model_inputs`` / ``prepare_model_outputs``，在 forward 期间注入
PrefixSharing runtime，并在输出阶段做 interior / prefix-last restore。
本 patch 已覆盖 dense 2D 与 verl remove-padding 后的 jagged NestedTensor
形态；Ulysses SP、fused kernels 等未验证形态仍在配置校验阶段显式拒绝。
"""

from __future__ import annotations

import os
from typing import Any


def patch_fsdp_forward_step(original_forward_step: Any) -> Any:
    """创建 FSDPEngineWithLMHead.forward_step 的 patch wrapper。"""

    def patched_forward_step(self: Any, micro_batch: Any, loss_function: Any, forward_only: bool):
        from prefix_sharing.core.config import PrefixSharingConfig
        from prefix_sharing.integrations.verl_mcore import read_ps_config_from_engine_config
        from prefix_sharing.tools.perf_profiler import PerfProfiler, ProfilerScope

        raw_config = read_ps_config_from_engine_config(self.engine_config)
        ps_config = PrefixSharingConfig.from_raw(raw_config)
        if not ps_config.enable_prefix_sharing:
            # Memory sampling is managed by the step-level ProfilerScope; this
            # path records only the model-forward phase.
            profiler = ProfilerScope.current()
            if profiler is not None:
                forward_phase = (
                    PerfProfiler.PHASE_FORWARD_OLD
                    if forward_only
                    else PerfProfiler.PHASE_FORWARD
                )
                profiler.start_phase(forward_phase)

            # Without diagnostics this path must delegate directly to verl.
            # Diagnostics use the equivalent native flow to expose OFF-baseline
            # raw logits and 2D log probabilities.
            if (
                os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None
                and hasattr(self, "prepare_model_inputs")
                and hasattr(self, "prepare_model_outputs")
            ):
                result = _call_original_like_engine(self, micro_batch, loss_function, forward_only)
                from prefix_sharing.tools.diagnostic_dump import dump_fsdp_baseline_verl080

                diagnostic_tag = "train" if self.module.training else "old"
                dump_fsdp_baseline_verl080(micro_batch, result, diagnostic_tag)
            else:
                result = original_forward_step(self, micro_batch, loss_function, forward_only)

            if profiler is not None:
                profiler.stop_phase(forward_phase)
            return result

        if hasattr(micro_batch, "to"):
            try:
                from verl.utils.device import get_device_id

                micro_batch = micro_batch.to(get_device_id())
            except Exception:
                # 本地单测使用 plain dict / fake engine，不依赖 verl device helper。
                pass

        ulysses_sp_size = _read_runtime_value(
            self.engine_config,
            micro_batch,
            "ulysses_sequence_parallel_size",
            default=1,
        )
        use_fused_kernels = _read_runtime_value(
            self.engine_config,
            micro_batch,
            "use_fused_kernels",
            default=False,
        )
        ps_config.validate(
            model_config={
                "model_type": "text_only_causal_lm",
                "ulysses_sequence_parallel_size": ulysses_sp_size,
                "use_fused_kernels": use_fused_kernels,
            },
            integrate_mode="verl_fsdp",
        )

        if hasattr(self, "prepare_model_inputs") and hasattr(self, "prepare_model_outputs"):
            return _forward_step_with_engine_prepare(
                self, micro_batch, loss_function, forward_only, ps_config,
            )

        from prefix_sharing.integrations.verl_fsdp import forward_prefix_sharing_fsdp_micro_batch

        calculate_entropy = bool(
            _read_runtime_value(self.engine_config, micro_batch, "calculate_entropy", default=False)
        )
        temperature = _read_temperature(micro_batch)
        output = forward_prefix_sharing_fsdp_micro_batch(
            micro_batch,
            self.module,
            ps_config,
            model_config={
                "model_type": "text_only_causal_lm",
                "ulysses_sequence_parallel_size": ulysses_sp_size,
                "use_fused_kernels": use_fused_kernels,
            },
            temperature=temperature,
            calculate_entropy=calculate_entropy,
            entropy_fn=getattr(self, "compute_entropy_from_logits", None),
        )
        model_output = {
            key: value
            for key, value in output.items()
            if key in {"log_probs", "entropy", "logits", "attention_output"}
        }

        if loss_function is not None:
            loss, metrics = loss_function(
                model_output=model_output,
                data=micro_batch,
                dp_group=self.get_data_parallel_group(),
            )
        else:
            assert forward_only, "forward_only must be True when loss_function is None"
            import torch

            device = output["logits"].device
            loss = torch.tensor(1.0, device=device)
            metrics = {}

        return loss, {
            "model_output": model_output,
            "loss": loss.detach().item(),
            "metrics": metrics,
        }

    return patched_forward_step


def _forward_step_with_engine_prepare(
    self: Any,
    micro_batch: Any,
    loss_function: Any,
    forward_only: bool,
    ps_config: Any,
) -> Any:
    import torch
    from contextlib import nullcontext

    from prefix_sharing.integrations.context import current_prefix_sharing_context
    from prefix_sharing.integrations.context import create_prefix_sharing_context
    from prefix_sharing.integrations.verl_fsdp import PrefixSharingFSDPAttentionRuntime
    from prefix_sharing.integrations.verl_fsdp import build_prefix_sharing_micro_batch_fsdp

    from prefix_sharing.tools.perf_profiler import PerfProfiler, ProfilerScope

    profiler = ProfilerScope.current()
    if profiler is not None:
        profiler.start_phase(PerfProfiler.PHASE_PLAN)

    trimmed_micro_batch, ps_state = build_prefix_sharing_micro_batch_fsdp(
        micro_batch,
        ps_config,
        model_config={
            "model_type": "text_only_causal_lm",
            "ulysses_sequence_parallel_size": _read_runtime_value(
                self.engine_config,
                micro_batch,
                "ulysses_sequence_parallel_size",
                default=1,
            ),
            "use_fused_kernels": _read_runtime_value(
                self.engine_config,
                micro_batch,
                "use_fused_kernels",
                default=False,
            ),
        },
    )
    if profiler is not None:
        profiler.stop_phase(PerfProfiler.PHASE_PLAN)  # Detect, plan, and trim on CPU.

    if ps_state is None:
        return _call_original_like_engine(self, trimmed_micro_batch, loss_function, forward_only)

    if os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
        from prefix_sharing.tools.diagnostic_dump import dump_fsdp_on_metadata_verl080

        diagnostic_tag = "train" if self.module.training else "old"
        dump_fsdp_on_metadata_verl080(
            micro_batch,
            ps_state.prefix_sharing_plan,
            diagnostic_tag,
        )

    model_inputs, output_args = self.prepare_model_inputs(micro_batch=trimmed_micro_batch)
    model_inputs["prefix_sharing_runtime"] = PrefixSharingFSDPAttentionRuntime()
    autocast_dtype = getattr(self, "_autocast_dtype", torch.float32)
    device_name = _read_device_name()
    autocast_ctx = (
        nullcontext()
        if autocast_dtype == torch.float32
        else torch.autocast(device_type=device_name, dtype=autocast_dtype)
    )
    # ── Create PS context with manual lifecycle (survives backward for AC) ──
    ctx, ctx_cleanup = create_prefix_sharing_context(ps_state)

    # Set _ps_ctx on every attention module so the attention patch reads
    # the context from the module itself rather than ContextVar (compatible
    # with activation-checkpointing recompute, which bypasses the context
    # manager that set the ContextVar).
    for attention_module in self.module.modules():
        if hasattr(attention_module, "layer_idx") and hasattr(attention_module, "q_proj"):
            attention_module._ps_ctx = ctx

    # Register diagnostic gradient hooks when the diagnostic dump is enabled.
    _register_grad_dump_hooks(self.module, forward_only)

    weight_grad_dump_done: list[bool] = [False]

    def _cleanup_ps_ctx(_module, _grad_input, _grad_output):
        """Release PS state, dump weight gradients, and remove diagnostic hooks."""
        print(f"[cleanup_hook] firing tag={diagnostic_tag} forward_only={forward_only}", flush=True)
        if (
            os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None
            and not forward_only
            and not weight_grad_dump_done[0]
        ):
            from prefix_sharing.tools.diagnostic_dump import dump_weight_grads_verl080

            print(f"[cleanup_hook] dumping weight grads tag={diagnostic_tag}", flush=True)
            dump_weight_grads_verl080(self.module, diagnostic_tag)
            weight_grad_dump_done[0] = True

        ctx_cleanup()
        for module in self.module.modules():
            try:
                del module._ps_ctx
            except AttributeError:
                pass

            grad_hook_handles = getattr(module, "_ps_grad_handles", None)
            if grad_hook_handles is not None:
                for grad_hook_handle in grad_hook_handles:
                    grad_hook_handle.remove()
                module._ps_grad_handles = None

    self.module.register_full_backward_hook(_cleanup_ps_ctx)

    with autocast_ctx:
        if profiler is not None:
            forward_phase = (
                PerfProfiler.PHASE_FORWARD_OLD
                if forward_only
                else PerfProfiler.PHASE_FORWARD
            )
            profiler.start_phase(forward_phase)
        raw_output = self.module(**model_inputs, use_cache=False)
        if profiler is not None:
            profiler.stop_phase(forward_phase)

        if os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
            from prefix_sharing.tools.diagnostic_dump import _get_dp_size, dump_raw_logits_verl080

            dump_raw_logits_verl080(raw_output, dp_aware=_get_dp_size() > 1)

        _save_prefix_last_logits_from_raw_output(raw_output)
        model_output = self.prepare_model_outputs(
            output=raw_output,
            output_args=output_args,
            micro_batch=trimmed_micro_batch,
            logits_processor_func=loss_function,
        )

        if profiler is not None:
            profiler.start_phase(PerfProfiler.PHASE_RESTORE)
        model_output = _restore_engine_model_output(model_output)
        if profiler is not None:
            profiler.stop_phase(PerfProfiler.PHASE_RESTORE)  # CPU-side output restoration.

        if os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
            from prefix_sharing.tools.diagnostic_dump import dump_fsdp_model_output_2d_verl080

            dump_fsdp_model_output_2d_verl080(
                model_output,
                list(ps_state.prefix_sharing_plan.original_lengths),
                diagnostic_tag,
            )

        if loss_function is not None:
            if profiler is not None:
                profiler.start_phase(PerfProfiler.PHASE_LOSS)
            loss, metrics = loss_function(
                model_output=model_output,
                data=micro_batch,
                dp_group=self.get_data_parallel_group(),
            )
            if profiler is not None:
                profiler.stop_phase(PerfProfiler.PHASE_LOSS)
        else:
            assert forward_only, "forward_only must be True when loss_function is None"
            loss = torch.tensor(1.0, device=_infer_output_device(model_output))
            metrics = {}

        return loss, {
            "model_output": model_output,
            "loss": loss.detach().item(),
            "metrics": metrics,
        }


def _register_grad_dump_hooks(model: Any, forward_only: bool) -> None:
    """Register attention-gradient dump hooks for diagnostic backward passes.

    Hooks are attached to module objects so activation-checkpoint recomputation
    does not register duplicates. Forward-only (old-logp) calls do not need
    backward hooks.
    """
    if os.environ.get("PREFIX_SHARING_DIAG_DUMP") is None:
        return
    if forward_only:
        return

    num_layers = int(getattr(getattr(model, "config", None), "num_hidden_layers", 0) or 0)
    if num_layers == 0:
        return

    for attention_module in model.modules():
        if not (hasattr(attention_module, "layer_idx") and hasattr(attention_module, "q_proj")):
            continue

        layer_number = int(attention_module.layer_idx) + 1
        existing_grad_hook_handles = getattr(attention_module, "_ps_grad_handles", None)
        if existing_grad_hook_handles is not None:
            for grad_hook_handle in existing_grad_hook_handles:
                grad_hook_handle.remove()
            existing_grad_hook_handles.clear()
        else:
            attention_module._ps_grad_handles = []

        def _make_grad_hook(layer_number: int):
            def _grad_hook(_module, _grad_input, grad_output):
                from prefix_sharing.tools.diagnostic_dump import dump_attn_grad_verl080

                dump_attn_grad_verl080(grad_output[0], layer_number, num_layers)

            return _grad_hook

        attention_module._ps_grad_handles.append(
            attention_module.register_full_backward_hook(_make_grad_hook(layer_number))
        )


def _call_original_like_engine(self: Any, micro_batch: Any, loss_function: Any, forward_only: bool) -> Any:
    # No sharing detected after planning. Delegate to the original engine
    # implementation shape by calling the unpatched method through the closure
    # is not possible here, so callers must hit the outer wrapper fallback when
    # prefix sharing is disabled. For no-sharing enabled batches we reproduce
    # the normal engine flow without opening a prefix-sharing context.
    import torch
    from contextlib import nullcontext

    # 对齐 verl 原生 forward_step：先把 micro_batch 搬到 device（disable 路径绕过了
    # patched_forward_step 里那段 .to(device)，这里补上，否则 prepare_model_outputs
    # 里 logits/temperature device 不一致）。
    if hasattr(micro_batch, "to"):
        try:
            from verl.utils.device import get_device_id
            micro_batch = micro_batch.to(get_device_id())
        except Exception:
            pass
    model_inputs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)
    autocast_dtype = getattr(self, "_autocast_dtype", torch.float32)
    device_name = _read_device_name()
    autocast_ctx = (
        nullcontext()
        if autocast_dtype == torch.float32
        else torch.autocast(device_type=device_name, dtype=autocast_dtype)
    )
    # Register diagnostic gradient hooks for the OFF baseline when enabled.
    _register_grad_dump_hooks(self.module, forward_only)
    diagnostic_tag = "train" if self.module.training else "old"

    weight_grad_dump_done_off: list[bool] = [False]

    def _cleanup_off(_module, _grad_input, _grad_output):
        """Dump weight gradients and remove diagnostic hooks after backward."""
        print(f"[cleanup_hook_off] firing tag={diagnostic_tag} forward_only={forward_only}", flush=True)
        if (
            os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None
            and not forward_only
            and not weight_grad_dump_done_off[0]
        ):
            from prefix_sharing.tools.diagnostic_dump import dump_weight_grads_verl080

            print(f"[cleanup_hook_off] dumping weight grads tag={diagnostic_tag}", flush=True)
            dump_weight_grads_verl080(self.module, diagnostic_tag)
            weight_grad_dump_done_off[0] = True

        for module in self.module.modules():
            grad_hook_handles = getattr(module, "_ps_grad_handles", None)
            if grad_hook_handles is not None:
                for grad_hook_handle in grad_hook_handles:
                    grad_hook_handle.remove()
                module._ps_grad_handles = None

    self.module.register_full_backward_hook(_cleanup_off)

    from prefix_sharing.tools.perf_profiler import ProfilerScope

    profiler = ProfilerScope.current()
    if profiler is not None:
        forward_phase = (
            profiler.PHASE_FORWARD_OLD
            if forward_only
            else profiler.PHASE_FORWARD
        )
        profiler.start_phase(forward_phase)

    with autocast_ctx:
        raw_output = self.module(**model_inputs, use_cache=False)
        if profiler is not None:
            profiler.stop_phase(forward_phase)

        if os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
            from prefix_sharing.tools.diagnostic_dump import _get_dp_size, dump_raw_logits_verl080

            dump_raw_logits_verl080(raw_output, dp_aware=_get_dp_size() > 1)

        model_output = self.prepare_model_outputs(
            output=raw_output,
            output_args=output_args,
            micro_batch=micro_batch,
            logits_processor_func=loss_function,
        )
        if loss_function is not None:
            loss, metrics = loss_function(
                model_output=model_output,
                data=micro_batch,
                dp_group=self.get_data_parallel_group(),
            )
        else:
            assert forward_only, "forward_only must be True when loss_function is None"
            loss = torch.tensor(1.0, device=_infer_output_device(model_output))
            metrics = {}

        return loss, {
            "model_output": model_output,
            "loss": loss.detach().item(),
            "metrics": metrics,
        }


def _save_prefix_last_logits_from_raw_output(raw_output: Any) -> None:
    from prefix_sharing.integrations.context import current_prefix_sharing_context

    ctx = current_prefix_sharing_context()
    if ctx is None:
        return
    logits = raw_output["logits"] if isinstance(raw_output, dict) else raw_output.logits
    if logits.dim() == 3 and logits.shape[0] == 1:
        packed_logits = logits.squeeze(0)
    elif logits.dim() == 3:
        # Dense [B, L, V] path: save by 2D provider row/column.
        for index in ctx.prefix_last_restore_indices:
            ctx.prefix_last_logits_saved[(index.reuse_idx_in_batch, index.target_2d_pos)] = logits[
                index.provider_idx_in_batch,
                index.target_2d_pos:index.target_2d_pos + 1,
            ]
        return
    else:
        packed_logits = logits
    for index in ctx.prefix_last_restore_indices:
        ctx.prefix_last_logits_saved[(index.reuse_idx_in_batch, index.target_2d_pos)] = packed_logits[
            index.provider_1d_pos:index.provider_1d_pos + 1,
        ]


def _restore_engine_model_output(model_output: dict[str, Any]) -> dict[str, Any]:
    try:
        from verl.utils.torch_functional import logprobs_from_logits
    except Exception:
        logprobs_from_logits = _default_log_probs_fn
    try:
        from verl.utils.torch_functional import entropy_from_logits
    except Exception:
        entropy_from_logits = None

    from prefix_sharing.integrations.verl_mcore import restore_via_2d_unfold_verl080
    from prefix_sharing.integrations.verl_mcore import _is_nested_tensor
    from prefix_sharing.integrations.verl_fsdp import restore_prefix_sharing_outputs_2d

    restored = restore_via_2d_unfold_verl080(
        model_output,
        logprobs_from_logits,
        entropy_from_logits,
    )
    log_probs = restored.get("log_probs")
    if log_probs is not None and not _is_nested_tensor(log_probs):
        return restore_prefix_sharing_outputs_2d(restored, logprobs_from_logits)
    return restored


def _default_log_probs_fn(logits: Any, labels: Any, **_: Any) -> Any:
    import torch

    safe_labels = labels.long().clamp_min(0) % logits.shape[-1]
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)


def _read_device_name() -> str:
    try:
        from verl.utils.device import get_device_name

        return get_device_name()
    except Exception:
        return "cpu"


def _infer_output_device(model_output: dict[str, Any]) -> Any:
    for value in model_output.values():
        if hasattr(value, "device"):
            return value.device
        if hasattr(value, "values"):
            try:
                return value.values().device
            except Exception:
                pass
    return "cpu"


def _read_runtime_value(engine_config: Any, micro_batch: Any, name: str, default: Any) -> Any:
    try:
        from verl.utils import tensordict_utils as tu

        value = tu.get_non_tensor_data(micro_batch, name, default=None)
        if value is not None:
            return value
    except Exception:
        pass
    if isinstance(micro_batch, dict) and name in micro_batch:
        return micro_batch[name]
    return getattr(engine_config, name, default)


def _read_temperature(micro_batch: Any) -> float:
    value = micro_batch.get("temperature", 1.0) if isinstance(micro_batch, dict) else 1.0
    try:
        if hasattr(value, "detach"):
            return float(value.detach().flatten()[0].item())
        return float(value)
    except Exception:
        return 1.0
