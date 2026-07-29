"""Patch: MegatronEngineWithLMHead.forward_step — verl 0.8.0 engine 架构

thin wrapper：消费 batch → 读 config → 构建状态 → 设 context → 喂回原始 forward_step。

所有业务逻辑（config 读取、batch 构建、layout 计算）由 integrations 层处理，
本 patch 只负责编排调用顺序和设置 runtime context。
"""

from __future__ import annotations

from typing import Any


def _ps_forward_step_probe(event: str, **fields: Any) -> None:
    """Emit a compact rank-aware probe for distributed hang diagnosis."""

    try:
        from prefix_sharing.integrations.parallel_info import get_megatron_parallel_info

        parallel_info = get_megatron_parallel_info()
        rank_prefix = (
            f"global_rank={parallel_info.global_rank} "
            f"tp_rank={parallel_info.tp_rank}/tp_size={parallel_info.tp_size} "
            f"pp_rank={parallel_info.pp_rank}/pp_size={parallel_info.pp_size} "
            f"cp_rank={parallel_info.cp_rank}/cp_size={parallel_info.cp_size}"
        )
    except Exception as exc:
        rank_prefix = f"parallel_info_unavailable={type(exc).__name__}:{exc}"

    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {field_text}" if field_text else ""
    print(f"[PS][forward_step][{rank_prefix}] {event}{suffix}", flush=True)


def _describe_batch(batch: Any) -> str:
    try:
        keys = list(batch.keys())
    except Exception:
        keys = []

    pieces = [f"type={type(batch).__name__}", f"keys={keys[:8]}"]
    for key in ("input_ids", "attention_mask", "position_ids", "labels"):
        try:
            value = batch[key]
        except Exception:
            continue
        shape = getattr(value, "shape", None)
        is_nested = getattr(value, "is_nested", None)
        pieces.append(f"{key}_shape={tuple(shape) if shape is not None else None}")
        if is_nested is not None:
            pieces.append(f"{key}_is_nested={is_nested}")
    return ",".join(pieces)


def _ps_timing_is_enabled() -> bool:
    import os as _os
    return _os.environ.get("PS_TIMING", "0") == "1"


def patch_verl_forward_step(original_forward_step: Any) -> Any:
    """创建 MegatronEngineWithLMHead.forward_step 的 patch wrapper。"""

    def patched_forward_step(
        self,
        batch_iter,
        model,
        logits_processor_func,
        postprocess_micro_batch_func,
    ):
        # ── 获取原始 micro-batch ──
        _ps_forward_step_probe("enter")
        _ps_forward_step_probe("before_next_batch")
        original_batch = next(batch_iter)
        _ps_forward_step_probe("after_next_batch", batch=_describe_batch(original_batch))
        batch_for_forward = original_batch

        # ── 读取配置 ──
        _ps_forward_step_probe("before_read_config")
        from prefix_sharing.integrations.verl_mcore import read_ps_config_from_engine_config
        from prefix_sharing.core.config import PrefixSharingConfig
        ps_config_raw = read_ps_config_from_engine_config(self.engine_config)
        ps_config = PrefixSharingConfig.from_raw(ps_config_raw)
        _ps_forward_step_probe(
            "after_read_config",
            enable_prefix_sharing=ps_config.enable_prefix_sharing,
            backend=ps_config.backend,
        )

        ps_state = None
        if ps_config.enable_prefix_sharing:
            # batch.to(device) 使 tensor 在目标设备上，
            # 原始 forward_step 会再次 batch.to(device)（幂等）
            from verl.utils.megatron_utils import get_device_id
            device_id = get_device_id()
            _ps_forward_step_probe("before_batch_to_device", device_id=device_id)
            batch_on_device = original_batch.to(device_id)
            _ps_forward_step_probe("after_batch_to_device")

            # batch裁剪
            _ps_forward_step_probe("before_prepare_micro_batch")
            from prefix_sharing.integrations.verl_mcore import build_prefix_sharing_micro_batch_verl080
            batch_for_forward, ps_state = build_prefix_sharing_micro_batch_verl080(
                self, batch_on_device, ps_config,
            )
            _ps_forward_step_probe(
                "after_prepare_micro_batch",
                has_runtime_state=ps_state is not None,
                batch=_describe_batch(batch_for_forward),
                layout=(
                    None
                    if ps_state is None
                    else (
                        f"valid={ps_state.packed_batch_layout.valid_lengths},"
                        f"padded={ps_state.packed_batch_layout.padded_lengths},"
                        f"cu={ps_state.packed_batch_layout.cu_seqlens}"
                    )
                ),
            )
        else:
            _ps_forward_step_probe("skip_prepare_prefix_sharing_disabled")

        # ##### [PS-diag] dump 元数据 + attention_mask + label_mask（ON/OFF 通用） #####
        import os as _os
        if _os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
            from prefix_sharing.tools.diagnostic_dump_verl080 import (
                dump_meta_verl080,
                dump_attention_mask_verl080, dump_label_mask_verl080,
                build_attention_mask_2d, build_label_mask_2d,
                nested_offsets_to_cu,
            )
            from prefix_sharing.integrations.verl_mcore import _is_nested_tensor
            _ids_nested = batch_for_forward["input_ids"]
            if _is_nested_tensor(_ids_nested):
                if ps_state is not None:
                    _plan = ps_state.prefix_sharing_plan
                    _prefix_lens = list(_plan.prefix_lens)
                    _orig_lens = list(_plan.original_lengths)
                else:
                    _diffs = _ids_nested.offsets().diff().tolist()
                    _orig_lens = [int(d) for d in _diffs]
                    _prefix_lens = [0] * len(_orig_lens)
                dump_meta_verl080(_prefix_lens, nested_offsets_to_cu(_ids_nested))
                _Lmax_lm = max(_orig_lens) if _orig_lens else 0
                _tag_lm = "train" if model.training else "old"
                dump_attention_mask_verl080(
                    build_attention_mask_2d(_orig_lens, _Lmax_lm), _tag_lm)
                _lm = original_batch.get("loss_mask")
                if _lm is not None:
                    if _is_nested_tensor(_lm):
                        _lm_off = _lm.offsets()
                        _lm_val = _lm.values()
                        _response_lens = [
                            int(_lm_val[_lm_off[i]:_lm_off[i + 1]].sum())
                            for i in range(len(_orig_lens))]
                    else:
                        _response_lens = _lm.sum(dim=-1).long().cpu().tolist()
                    dump_label_mask_verl080(
                        build_label_mask_2d(_response_lens, _orig_lens, _Lmax_lm),
                        _tag_lm)
        # ##### [PS-diag] dump 元数据 + masks end #####

        # ── 构造修改后的 iterator 喂回原始 forward_step ──
        modified_iter = iter([batch_for_forward])

        # ── runtime context ──
        from prefix_sharing.integrations.context import prefix_sharing_runtime_context
        from contextlib import nullcontext

        context_manager = (
            prefix_sharing_runtime_context(ps_state)
            if ps_state is not None
            else nullcontext()
        )

        _ps_forward_step_probe("before_original_forward_step", has_runtime_state=ps_state is not None)
        with context_manager:
            output = original_forward_step(
                self,
                modified_iter,
                model,
                logits_processor_func,
                postprocess_micro_batch_func,
            )
            # v080 restore：在 context 仍激活时重组 reuser prefix 区段。
            if ps_state is not None:
                from prefix_sharing.integrations.verl_mcore import restore_via_2d_unfold_verl080
                from prefix_sharing.integrations.context import current_prefix_sharing_context
                from verl.utils.megatron.tensor_parallel import (
                    vocab_parallel_entropy,
                    vocab_parallel_log_probs_from_logits,
                )
                output_dict, postprocess_fn = output
                output_dict = restore_via_2d_unfold_verl080(
                    output_dict,
                    vocab_parallel_log_probs_from_logits,
                    vocab_parallel_entropy,
                )

                # 释放 vocab 维 logits
                ctx = current_prefix_sharing_context()
                if ctx is not None:
                    ctx.prefix_last_logits_saved.clear()
                output = (output_dict, postprocess_fn)

            # [PS-TIMING] single synchronize + print per-layer/per-mb summary (rank0 only)
            _ps_do_timing = _ps_timing_is_enabled()
            if _ps_do_timing:
                import torch as _t
                _t.npu.synchronize()
                try:
                    from prefix_sharing.integrations.parallel_info import get_megatron_parallel_info
                    _pi = get_megatron_parallel_info()
                    _rank = _pi.global_rank
                except Exception:
                    _rank = 0
                _forward_id = (ps_state.prefix_sharing_plan.forward_id
                               if ps_state is not None else 0)
                if _rank == 0:
                    from prefix_sharing.integrations.megatron_runtime import ps_print_timing_summary
                    ps_print_timing_summary(_forward_id, _rank)

            # ##### [PS-diag] dump 2D logprobs/entropy（ON=restore后, OFF=原始） #####
            import os as _os2
            if _os2.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
                from prefix_sharing.tools.diagnostic_dump_verl080 import (
                    nested_to_2d_full, dump_logprobs_2d_verl080, dump_entropy_2d_verl080,
                )
                from prefix_sharing.integrations.verl_mcore import _is_nested_tensor
                _tag = "train" if model.training else "old"
                _out_dict, _ = output
                _lp = _out_dict.get("log_probs")
                if _is_nested_tensor(_lp):
                    if ps_state is not None:
                        _ol = list(ps_state.prefix_sharing_plan.original_lengths)
                    else:
                        _ol = [int(d) for d in _lp.offsets().diff().tolist()]
                    _Lmax = max(_ol) if _ol else 0
                    dump_logprobs_2d_verl080(nested_to_2d_full(_lp, _ol, _Lmax), _tag)
                    _ent = _out_dict.get("entropy")
                    if _is_nested_tensor(_ent):
                        dump_entropy_2d_verl080(nested_to_2d_full(_ent, _ol, _Lmax), _tag)
            # ##### [PS-diag] dump 2D logprobs/entropy end #####
        _ps_forward_step_probe("after_original_forward_step")
        return output

    return patched_forward_step
