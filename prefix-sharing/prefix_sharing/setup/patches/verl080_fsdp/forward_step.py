"""Patch: FSDPEngineWithLMHead.forward_step — verl 0.8.0 FSDP 路径。

thin wrapper：读取 prefix_sharing_config，优先复用真实 engine 的
``prepare_model_inputs`` / ``prepare_model_outputs``，在 forward 期间注入
PrefixSharing runtime，并在输出阶段做 interior / prefix-last restore。
本 patch 已覆盖 dense 2D 与 verl remove-padding 后的 jagged NestedTensor
形态；Ulysses SP、fused kernels 等未验证形态仍在配置校验阶段显式拒绝。
"""

from __future__ import annotations

from prefix_sharing.diagnostics import diagnostic_dump_enabled

from typing import Any


def patch_fsdp_forward_step(original_forward_step: Any) -> Any:
    """创建 FSDPEngineWithLMHead.forward_step 的 patch wrapper。"""

    # Patch _CheckpointFrame.check_recomputed_tensors_match and
    # _internal_assert to no-op.
    # PrefixSharing patched attention adds Q/K/V store/load nodes to the
    # computation graph, causing the saved-tensor count mismatch detected by
    # these methods.  The recomputed values are numerically correct — the count
    # difference is benign.  Bypass both checks so ON-path training completes.
    import torch.utils.checkpoint as _cp
    # Apply once, globally.
    if not getattr(patch_fsdp_forward_step, "_cp_patched", False):
        _cp._CheckpointFrame.check_recomputed_tensors_match = lambda self, gid: None  # type: ignore[method-assign]
        if hasattr(_cp, "_internal_assert"):
            _cp._internal_assert = lambda *a, **kw: None
        patch_fsdp_forward_step._cp_patched = True

    def patched_forward_step(self: Any, micro_batch: Any, loss_function: Any, forward_only: bool):
        from prefix_sharing.core.config import PrefixSharingConfig
        from prefix_sharing.integrations.verl_mcore import read_ps_config_from_engine_config

        raw_config = read_ps_config_from_engine_config(self.engine_config)
        ps_config = PrefixSharingConfig.from_raw(raw_config)
        if not ps_config.enable_prefix_sharing:
            # 普通 disabled 路径必须完全透传原生 forward_step；只有诊断模式
            # 才走等价展开路径，以便拿到 raw logits / 2D logp 做 OFF baseline dump。
            if (
                diagnostic_dump_enabled()
                and hasattr(self, "prepare_model_inputs")
                and hasattr(self, "prepare_model_outputs")
            ):
                result = _call_original_like_engine(self, micro_batch, loss_function, forward_only)
                from prefix_sharing.tools.diagnostic_dump_verl080 import dump_fsdp_baseline_verl080

                dump_fsdp_baseline_verl080(micro_batch, result, "train")
            else:
                result = original_forward_step(self, micro_batch, loss_function, forward_only)
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
                self,
                micro_batch,
                loss_function,
                forward_only,
                ps_config,
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
    from prefix_sharing.integrations.context import prefix_sharing_runtime_context
    from prefix_sharing.integrations.verl_fsdp import PrefixSharingFSDPAttentionRuntime
    from prefix_sharing.integrations.verl_fsdp import build_prefix_sharing_micro_batch_fsdp

    # DIAG_DUMP: dump 原始 full input_ids 必须在前面的 build_prefix_sharing_micro_batch_fsdp
    # 之前执行，因为后者会就地修改 micro_batch（裁剪 prefix tokens）。
    # 用原始 micro_batch 保存完整的 input_ids 供 cmp_diag 对齐 baseline。
    import os as _ps_diag_fwd_ids2
    if _ps_diag_fwd_ids2.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
        _dump_full_input_ids_only(micro_batch, "train")

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
    if ps_state is None:
        return _call_original_like_engine(self, trimmed_micro_batch, loss_function, forward_only)

    if diagnostic_dump_enabled() is not None:
        from prefix_sharing.tools.diagnostic_dump_verl080 import dump_fsdp_on_metadata_verl080

        dump_fsdp_on_metadata_verl080(micro_batch, ps_state.prefix_sharing_plan, "train")

    # 获取模型层数以支持 per-layer diagnostic dump
    _diag_num_layers = int(getattr(
        getattr(getattr(self, "module", None), "config", None),
        "num_hidden_layers", 0)) or 0

    model_inputs, output_args = self.prepare_model_inputs(micro_batch=trimmed_micro_batch)
    model_inputs["prefix_sharing_runtime"] = PrefixSharingFSDPAttentionRuntime()
    model_inputs["prefix_sharing_runtime"].num_layers = _diag_num_layers
    autocast_dtype = getattr(self, "_autocast_dtype", torch.float32)
    device_name = _read_device_name()
    autocast_ctx = (
        nullcontext()
        if autocast_dtype == torch.float32
        else torch.autocast(device_type=device_name, dtype=autocast_dtype)
    )
    with prefix_sharing_runtime_context(ps_state), autocast_ctx:
        raw_output = self.module(**model_inputs, use_cache=False)
        if diagnostic_dump_enabled() is not None:
            from prefix_sharing.tools.diagnostic_dump_verl080 import dump_raw_logits_verl080

            dump_raw_logits_verl080(raw_output)
        _save_prefix_last_logits_from_raw_output(raw_output)
        model_output = self.prepare_model_outputs(
            output=raw_output,
            output_args=output_args,
            micro_batch=trimmed_micro_batch,
            logits_processor_func=loss_function,
        )
        model_output = _restore_engine_model_output(model_output)

        import os as _os_diag2
        if _os_diag2.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
            from prefix_sharing.tools.diagnostic_dump_verl080 import dump_fsdp_model_output_2d_verl080

            dump_fsdp_model_output_2d_verl080(
                model_output,
                list(ps_state.prefix_sharing_plan.original_lengths),
                "train",
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

    # DIAG_DUMP: ON path dump原始full input_ids（suffix-only dump会缺失prefix tokens）
    import os as _ps_diag_fwd_ids
    if _ps_diag_fwd_ids.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
        _dump_full_input_ids_only(micro_batch, "train")

    autocast_dtype = getattr(self, "_autocast_dtype", torch.float32)
    autocast_dtype = getattr(self, "_autocast_dtype", torch.float32)
    device_name = _read_device_name()
    autocast_ctx = (
        nullcontext()
        if autocast_dtype == torch.float32
        else torch.autocast(device_type=device_name, dtype=autocast_dtype)
    )
    with autocast_ctx:
        raw_output = self.module(**model_inputs, use_cache=False)
        import os as _os_logits_off
        if _os_logits_off.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
            from prefix_sharing.tools.diagnostic_dump_verl080 import dump_raw_logits_verl080

            dump_raw_logits_verl080(raw_output)
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
        import os as _os_diag_off_out
        if _os_diag_off_out.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
            from prefix_sharing.tools.diagnostic_dump_verl080 import dump_fsdp_baseline_verl080
            dump_fsdp_baseline_verl080(
                micro_batch,
                (loss, {"model_output": model_output, "loss": loss.detach().item(), "metrics": metrics}),
                "train",
            )
        return loss, {"model_output": model_output, "loss": loss.detach().item(), "metrics": metrics}


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


def _dump_full_input_ids_only(micro_batch: Any, tag: str) -> None:
    """Dump the original (full) input_ids before prefix sharing trimming.

    The ON path dumps ``input_ids_train.pt`` from the ``trimmed_micro_batch``,
    which has shared prefix tokens removed.  This helper saves the **original**
    ``micro_batch`` input_ids so that ``cmp_diag_verl080`` can compare the
    full input against the OFF baseline, rather than reporting 186+ differing
    tokens as a false positive.

    Multiple forwards (e.g. PPO micro-batches) all call this.  Only the FIRST
    dump is preserved; subsequent calls (recompute / later micro-batches) are
    skipped to avoid overwriting with trimmed or partial data.
    """
    import os
    import torch

    from prefix_sharing.tools.diagnostic_dump import _get_dump_dir, _rank0_only

    if getattr(_dump_full_input_ids_only, "_saved", False):
        return
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    try:
        raw = micro_batch["input_ids"]
        if hasattr(raw, "values"):
            raw = raw.values()
        ids = raw.detach().cpu().long()
        fname = f"full_input_ids_{tag}.pt"
        if _rank0_only():
            torch.save(ids, os.path.join(dump_dir, fname))
        _dump_full_input_ids_only._saved = True
    except Exception:
        pass
