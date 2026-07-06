"""Patch: FSDPEngineWithLMHead.forward_step — verl 0.8.0 FSDP 路径。

thin wrapper：读取 prefix_sharing_config，优先复用真实 engine 的
``prepare_model_inputs`` / ``prepare_model_outputs``，在 forward 期间注入
PrefixSharing runtime，并在输出阶段做 interior / prefix-last restore。
本 patch 已覆盖 dense 2D 与 verl remove-padding 后的 jagged NestedTensor
形态；Ulysses SP、fused kernels 等未验证形态仍在配置校验阶段显式拒绝。
"""

from __future__ import annotations

from typing import Any


def patch_fsdp_forward_step(original_forward_step: Any) -> Any:
    """创建 FSDPEngineWithLMHead.forward_step 的 patch wrapper。"""

    def patched_forward_step(self: Any, micro_batch: Any, loss_function: Any, forward_only: bool):
        from prefix_sharing.core.config import PrefixSharingConfig
        from prefix_sharing.integrations.verl_mcore import read_ps_config_from_engine_config

        raw_config = read_ps_config_from_engine_config(self.engine_config)
        ps_config = PrefixSharingConfig.from_raw(raw_config)
        if not ps_config.enable_prefix_sharing:
            # 真实 engine（有 prepare_model_inputs/outputs）：走 _call_original_like_engine，
            # 它与 verl 原生 forward_step forward 逻辑等价，但暴露 raw_output 使 OFF logits dump 可达；
            # fake engine / 非 prepare 风格：仍走原生 original_forward_step 保持兼容。
            if hasattr(self, "prepare_model_inputs") and hasattr(self, "prepare_model_outputs"):
                result = _call_original_like_engine(self, micro_batch, loss_function, forward_only)
            else:
                result = original_forward_step(self, micro_batch, loss_function, forward_only)
            # ##### [PS-diag] OFF dump: FSDP baseline 2D logp/entropy/masks #####
            import os as _os_diag_off
            if _os_diag_off.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
                _dump_fsdp_baseline(micro_batch, result)
            # ##### [PS-diag] end #####
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

    # ##### [PS-diag] dump 元数据 + attention_mask + label_mask（ON/OFF 通用） #####
    import os as _os_diag
    if _os_diag.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
        from prefix_sharing.tools.diagnostic_dump_verl080 import (
            dump_meta_verl080,
            dump_attention_mask_verl080, dump_label_mask_verl080,
            build_attention_mask_2d, build_label_mask_2d,
        )
        _plan_diag = ps_state.prefix_sharing_plan
        _prefix_lens_diag = list(_plan_diag.prefix_lens)
        _orig_lens_diag = list(_plan_diag.original_lengths)
        # FSDP packed path: cu_seqlens = cumsum of kept_lengths_q,  [B+1] int64
        _kept = _plan_diag.kept_lengths_q
        _cu_diag = torch.zeros(len(_kept) + 1, dtype=torch.int64)
        for _i_diag, _l_diag in enumerate(_kept):
            _cu_diag[_i_diag + 1] = _cu_diag[_i_diag] + _l_diag
        dump_meta_verl080(_prefix_lens_diag, _cu_diag)
        _Lmax_diag = max(_orig_lens_diag) if _orig_lens_diag else 0
        dump_attention_mask_verl080(build_attention_mask_2d(_orig_lens_diag, _Lmax_diag), "train")
        _lm_diag = micro_batch.get("loss_mask")
        if _lm_diag is not None:
            _response_lens_diag = _lm_diag.sum(dim=-1).long().cpu().tolist()
            dump_label_mask_verl080(build_label_mask_2d(_response_lens_diag, _orig_lens_diag, _Lmax_diag), "train")
        # dump 原始（未 trim）input_ids 到 2D，供 ON/OFF batch 内容直接对比
        _dump_input_ids_2d(micro_batch, _orig_lens_diag, _Lmax_diag, "train")
    # ##### [PS-diag] dump end #####

    model_inputs, output_args = self.prepare_model_inputs(micro_batch=trimmed_micro_batch)
    model_inputs["prefix_sharing_runtime"] = PrefixSharingFSDPAttentionRuntime()
    autocast_dtype = getattr(self, "_autocast_dtype", torch.float32)
    device_name = _read_device_name()
    autocast_ctx = (
        nullcontext()
        if autocast_dtype == torch.float32
        else torch.autocast(device_type=device_name, dtype=autocast_dtype)
    )
    with prefix_sharing_runtime_context(ps_state), autocast_ctx:
        raw_output = self.module(**model_inputs, use_cache=False)
        # ##### [PS-diag] dump packed logits（ON = 裁剪后 packed，必须在 logp 消耗前） #####
        import os as _os_logits_on
        if _os_logits_on.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
            from prefix_sharing.tools.diagnostic_dump_verl080 import dump_logits_verl080
            _logits_on = raw_output["logits"] if isinstance(raw_output, dict) else raw_output.logits
            dump_logits_verl080(_logits_on)
        # ##### [PS-diag] dump logits end #####
        _save_prefix_last_logits_from_raw_output(raw_output)
        model_output = self.prepare_model_outputs(
            output=raw_output,
            output_args=output_args,
            micro_batch=trimmed_micro_batch,
            logits_processor_func=loss_function,
        )
        model_output = _restore_engine_model_output(model_output)

        # ##### [PS-diag] dump 2D logprobs/entropy（ON=restore后, OFF=原始） #####
        import os as _os_diag2
        if _os_diag2.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
            from prefix_sharing.integrations.verl_mcore import _is_nested_tensor
            from prefix_sharing.tools.diagnostic_dump_verl080 import (
                dump_logprobs_2d_verl080, dump_entropy_2d_verl080, nested_to_2d_full,
            )
            _lp_diag = model_output.get("log_probs")
            if _lp_diag is not None:
                if _is_nested_tensor(_lp_diag):
                    _ol_diag = list(ps_state.prefix_sharing_plan.original_lengths)
                    _Lmax_diag = max(_ol_diag) if _ol_diag else 0
                    _lp_diag = nested_to_2d_full(_lp_diag, _ol_diag, _Lmax_diag)
                if _lp_diag.dim() == 2:
                    dump_logprobs_2d_verl080(_lp_diag, "train")
                    _ent_diag = model_output.get("entropy")
                    if _ent_diag is not None:
                        if _is_nested_tensor(_ent_diag):
                            _ent_diag = nested_to_2d_full(
                                _ent_diag,
                                list(ps_state.prefix_sharing_plan.original_lengths),
                                max(ps_state.prefix_sharing_plan.original_lengths) if ps_state.prefix_sharing_plan.original_lengths else 0,
                            )
                        if _ent_diag.dim() == 2:
                            dump_entropy_2d_verl080(_ent_diag, "train")
        # ##### [PS-diag] dump 2D logprobs/entropy end #####

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
    autocast_dtype = getattr(self, "_autocast_dtype", torch.float32)
    device_name = _read_device_name()
    autocast_ctx = (
        nullcontext()
        if autocast_dtype == torch.float32
        else torch.autocast(device_type=device_name, dtype=autocast_dtype)
    )
    with autocast_ctx:
        raw_output = self.module(**model_inputs, use_cache=False)
        # ##### [PS-diag] dump packed logits（OFF baseline = 完整 packed） #####
        import os as _os_logits_off
        if _os_logits_off.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
            from prefix_sharing.tools.diagnostic_dump_verl080 import dump_logits_verl080
            _logits_off = raw_output["logits"] if isinstance(raw_output, dict) else raw_output.logits
            dump_logits_verl080(_logits_off)
        # ##### [PS-diag] dump logits end #####
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


def _dump_fsdp_baseline(micro_batch: Any, result: Any) -> None:
    """Dump FSDP OFF baseline diagnostics (no prefix-sharing).

    OFF 路径走原生 verl ``forward_step``，返回 ``(loss, output_dict)``，
    ``output_dict["model_output"]`` 里的 ``log_probs``/``entropy`` 在 ``use_remove_padding=True`` 下
    是 NestedTensor（jagged），用 :func:`nested_to_2d_full` 展开到 ``[B, L_max]``。
    """
    import torch

    from prefix_sharing.integrations.verl_mcore import _is_nested_tensor
    from prefix_sharing.tools.diagnostic_dump_verl080 import (
        build_attention_mask_2d, build_label_mask_2d, nested_to_2d_full,
        dump_attention_mask_verl080, dump_entropy_2d_verl080,
        dump_label_mask_verl080, dump_logprobs_2d_verl080, dump_meta_verl080,
    )

    # result = (loss, output_dict); output_dict["model_output"] holds log_probs/entropy
    if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], dict):
        output_dict = result[1]
    else:
        return
    model_output = output_dict.get("model_output", {})
    if not model_output:
        return

    # original_lengths from micro_batch["input_ids"] (pre-trim full lengths)
    _ids = micro_batch.get("input_ids")
    if _is_nested_tensor(_ids):
        _orig_lens = [int(d) for d in _ids.offsets().diff().tolist()]
    elif _ids is not None and hasattr(_ids, "dim") and _ids.dim() == 2:
        # Dense [B, L]: all rows share the same length L (right-padded).
        _orig_lens = [int(_ids.shape[1])] * int(_ids.shape[0])
    else:
        return

    _prefix_lens = [0] * len(_orig_lens)
    _cu = torch.zeros(len(_orig_lens) + 1, dtype=torch.int64)
    for _i, _l in enumerate(_orig_lens):
        _cu[_i + 1] = _cu[_i] + _l
    dump_meta_verl080(_prefix_lens, _cu)

    _Lmax = max(_orig_lens) if _orig_lens else 0
    dump_attention_mask_verl080(build_attention_mask_2d(_orig_lens, _Lmax), "train")
    _lm = micro_batch.get("loss_mask")
    if _lm is not None:
        if _is_nested_tensor(_lm):
            _off = _lm.offsets()
            _val = _lm.values()
            _response_lens = [int(_val[_off[i]:_off[i + 1]].sum()) for i in range(len(_orig_lens))]
        else:
            _response_lens = _lm.sum(dim=-1).long().cpu().tolist()
        dump_label_mask_verl080(build_label_mask_2d(_response_lens, _orig_lens, _Lmax), "train")

    # dump 原始 input_ids 到 2D，供 ON/OFF batch 内容直接对比
    _dump_input_ids_2d(micro_batch, _orig_lens, _Lmax, "train")

    _lp = model_output.get("log_probs")
    if _lp is None:
        return
    if _is_nested_tensor(_lp):
        _lp_2d = nested_to_2d_full(_lp, _orig_lens, _Lmax)
    elif _lp.dim() == 2:
        _lp_2d = _lp
    else:
        return
    dump_logprobs_2d_verl080(_lp_2d, "train")
    _ent = model_output.get("entropy")
    if _ent is not None:
        if _is_nested_tensor(_ent):
            _ent = nested_to_2d_full(_ent, _orig_lens, _Lmax)
        if _ent.dim() == 2:
            dump_entropy_2d_verl080(_ent, "train")


def _dump_input_ids_2d(micro_batch: Any, orig_lens: list[int], l_max: int, tag: str) -> None:
    """Dump 原始（未 trim）input_ids 到 2D ``[B, L_max]``，文件名 ``input_ids_{tag}.pt``。

    用于 ON/OFF 两次 run 的 batch 内容直接逐 token 对比——这是判定 cmp_diag 逐行对比
    是否成立的前提（只有 batch 内容字节级一致，逐元素 logp/entropy 对比才有意义）。
    NestedTensor input_ids 按 original_lengths 展开到统一 [B, L_max]；dense 2D 直接存。
    """
    import torch

    from prefix_sharing.integrations.verl_mcore import _is_nested_tensor
    from prefix_sharing.tools.diagnostic_dump import _get_dump_dir, _save_tensor

    if _get_dump_dir() is None:
        return
    _ids = micro_batch.get("input_ids")
    if _ids is None:
        return
    if _is_nested_tensor(_ids):
        from prefix_sharing.tools.diagnostic_dump_verl080 import nested_to_2d_full
        ids_2d = nested_to_2d_full(_ids, orig_lens, l_max)
    elif hasattr(_ids, "dim") and _ids.dim() == 2:
        ids_2d = _ids
    else:
        return
    _save_tensor(f"input_ids_{tag}.pt", ids_2d.long().cpu(), _get_dump_dir())
