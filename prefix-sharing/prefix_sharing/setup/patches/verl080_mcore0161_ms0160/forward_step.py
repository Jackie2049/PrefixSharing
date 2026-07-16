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
        # batch_iter 来自外层 engine 的 forward_step 调用方。
        # 消费 batch，由 build_prefix_sharing_micro_batch_verl080 进行物理裁剪
        # 返回 trimmed_batch（物理裁剪后的 micro-batch）和 ps_state。
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
        # 仅当 PREFIX_SHARING_DIAG_DUMP 设定时触发，否则零开销。
        # ON: prefix_lens / original_lengths 取自 plan；
        # OFF: prefix_lens 全0、original_lengths 从 input_ids NestedTensor offsets diff 推。
        # cu_seqlens 取送进 forward 的 input_ids NestedTensor offsets（ON=裁剪后 packed 边界, OFF=完整）。
        import os as _diag_os
        if _diag_os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
            from prefix_sharing.tools.diagnostic_dump_verl080 import (
                dump_meta_verl080,
                dump_attention_mask_verl080, dump_label_mask_verl080,
                build_attention_mask_2d, build_label_mask_2d,
                nested_offsets_to_cu,
            )
            from prefix_sharing.integrations.verl_mcore import _is_nested_tensor
            input_ids_nested = batch_for_forward["input_ids"]
            if _is_nested_tensor(input_ids_nested):
                if ps_state is not None:
                    sharing_plan = ps_state.prefix_sharing_plan
                    prefix_lens_list = list(sharing_plan.prefix_lens)
                    original_lengths = list(sharing_plan.original_lengths)
                else:
                    seq_lengths = input_ids_nested.offsets().diff().tolist()
                    original_lengths = [int(length) for length in seq_lengths]
                    prefix_lens_list = [0] * len(original_lengths)
                dump_meta_verl080(prefix_lens_list,
                                 nested_offsets_to_cu(input_ids_nested))
                # attention_mask + label_mask：两种 log_probs 对比范围，都不含越界预测位
                # （POS L_i-1，其 logp 预测不存在的 token[L_i]）。对齐 restore 后 log_probs
                # 的 [B, L_max] 紧凑坐标系。
                #   attention_mask：[0:L_i-1) prompt 区+prompt-last+response 区（整体 restore 验证）
                #   label_mask：[prompt-last:L_i-1) prompt-last+response 区（PPO loss 范围）
                max_seq_len = max(original_lengths) if original_lengths else 0
                # tag 与 logprobs 一致（接入点2 用 model.training 区分 old/train），
                # 保证 mask 和 logprobs_{tag} 来自同一 forward（同 batch、同 L_max）。
                diag_tag = "train" if model.training else "old"
                # attention_mask 仅依赖 original_lengths，不需要 loss_mask。
                dump_attention_mask_verl080(
                    build_attention_mask_2d(original_lengths, max_seq_len), diag_tag)
                # label_mask 用 response_lengths（每行 response token 数）。verl080 padding 后
                # loss_mask = response_mask 是 2D left-right padded（非 NestedTensor，见
                # verl padding.py:71），不能走 nested_to_2d_full；但 response token 数 =
                # loss_mask 行 sum，与坐标系无关，据此推 prompt_len 最稳（2D/NestedTensor 均适用）。
                loss_mask_tensor = original_batch.get("loss_mask")
                if loss_mask_tensor is not None:
                    if _is_nested_tensor(loss_mask_tensor):
                        loss_offsets = loss_mask_tensor.offsets()
                        loss_values = loss_mask_tensor.values()
                        response_lengths = [
                            int(loss_values[loss_offsets[i]:loss_offsets[i + 1]].sum())
                            for i in range(len(original_lengths))]
                    else:
                        # .long() 免 import torch（本文件顶部未导入 torch）；
                        # .cpu() 防御 on-device tensor 的 tolist()
                        response_lengths = loss_mask_tensor.sum(dim=-1).long().cpu().tolist()
                    dump_label_mask_verl080(
                        build_label_mask_2d(response_lengths, original_lengths, max_seq_len),
                        diag_tag)
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
            # forward_step 返回 (output_dict, partial(postprocess_func))，
            # 解包处理 output_dict 再重包。restore_via_2d_unfold_verl080 内部
            # 会检查 context / restore_indices，无 restore 需求时 early return。
            #
            # PP guard: 非末 stage 返回的是 tensor（hidden_states），不是 dict；
            # restore 只在末 stage（output 是含 log_probs 的 dict）才有意义。
            if ps_state is not None:
                from prefix_sharing.integrations.verl_mcore import restore_via_2d_unfold_verl080
                from prefix_sharing.integrations.context import current_prefix_sharing_context
                from verl.utils.megatron.tensor_parallel import (
                    vocab_parallel_entropy,
                    vocab_parallel_log_probs_from_logits,
                )
                output_dict, postprocess_fn = output
                if isinstance(output_dict, dict):
                    output_dict = restore_via_2d_unfold_verl080(
                        output_dict,
                        vocab_parallel_log_probs_from_logits,
                        vocab_parallel_entropy,
                    )
                    # 释放 vocab 维 logits（占用大，只在 context 生命周期内持有，
                    # restore 已消费完毕）。clear 职责在此，不在包装函数内。
                    ctx = current_prefix_sharing_context()
                    if ctx is not None:
                        ctx.prefix_last_logits_saved.clear()
                output = (output_dict, postprocess_fn)
            # ##### [PS-diag] dump 2D logprobs/entropy（ON=restore后, OFF=原始） #####
            # restore 后（ON）或原始 forward（OFF）的 log_probs/entropy 都是 NestedTensor，
            # 每行长度 = original_lengths[i]，展开到统一 [B, L_max] 供 cmp_diag.cmp_2d 逐元素对比。
            #
            # PP guard: 非末 stage 返回 tensor 而非 dict；跳过 get() 避免 AttributeError。
            import os as _diag_os
            if _diag_os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
                from prefix_sharing.tools.diagnostic_dump_verl080 import (
                    nested_to_2d_full, dump_logprobs_2d_verl080, dump_entropy_2d_verl080,
                )
                from prefix_sharing.integrations.verl_mcore import _is_nested_tensor
                # 对齐 v070: tag = "old" if forward_only else "train"。
                # forward_step 拿不到 forward_only，用 model.training 等价区分
                # （eval_mode→training=False→"old" 对应 old_logp 阶段；
                #  train_mode→training=True→"train" 对应 update_actor 阶段）。
                # 这样一次 run 自动产出 logprobs_old + logprobs_train 两份，不互相覆盖。
                diag_tag = "train" if model.training else "old"
                output_dict_raw, _ = output
                if isinstance(output_dict_raw, dict):
                    log_probs_nested = output_dict_raw.get("log_probs")
                    if _is_nested_tensor(log_probs_nested):
                        if ps_state is not None:
                            original_lengths = list(
                                ps_state.prefix_sharing_plan.original_lengths)
                        else:
                            original_lengths = [
                                int(length) for length
                                in log_probs_nested.offsets().diff().tolist()]
                        max_seq_len = max(original_lengths) if original_lengths else 0
                        dump_logprobs_2d_verl080(
                            nested_to_2d_full(log_probs_nested, original_lengths, max_seq_len),
                            diag_tag, scope="pp_last")
                        entropy_nested = output_dict_raw.get("entropy")
                        if _is_nested_tensor(entropy_nested):
                            dump_entropy_2d_verl080(
                                nested_to_2d_full(entropy_nested, original_lengths, max_seq_len),
                                diag_tag, scope="pp_last")
            # ##### [PS-diag] dump 2D logprobs/entropy end #####
        _ps_forward_step_probe("after_original_forward_step")
        return output

    return patched_forward_step
