"""Patch: MegatronEngineWithLMHead.forward_step — verl 0.8.0 engine architecture

Thin wrapper: consume batch → read config → build state → set context → feed
back to original forward_step.

All business logic (config reading, batch construction, layout computation)
is handled by the integrations layer. This patch only orchestrates the call
sequence and sets up the runtime context.
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
    """Create a patch wrapper for MegatronEngineWithLMHead.forward_step."""

    def patched_forward_step(
        self,
        batch_iter,
        model,
        logits_processor_func,
        postprocess_micro_batch_func,
    ):
        # ── Retrieve original micro-batch ──
        # batch_iter comes from the outer engine's forward_step caller.
        # Consume the batch; build_prefix_sharing_micro_batch_verl080 performs
        # physical trimming. Returns trimmed_batch (physically trimmed
        # micro-batch) and ps_state.
        _ps_forward_step_probe("enter")
        _ps_forward_step_probe("before_next_batch")
        original_batch = next(batch_iter)
        _ps_forward_step_probe("after_next_batch", batch=_describe_batch(original_batch))
        batch_for_forward = original_batch

        # ── Read configuration ──
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
            # batch.to(device) ensures tensors are on the target device.
            # The original forward_step will call batch.to(device) again
            # (idempotent).
            from verl.utils.megatron_utils import get_device_id
            device_id = get_device_id()
            _ps_forward_step_probe("before_batch_to_device", device_id=device_id)
            batch_on_device = original_batch.to(device_id)
            _ps_forward_step_probe("after_batch_to_device")

            # Batch trimming
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

        # ##### [PS-diag] dump metadata + attention_mask + label_mask (shared by ON/OFF) #####
        # Only triggered when PREFIX_SHARING_DIAG_DUMP is set; zero overhead otherwise.
        # ON: prefix_lens / original_lengths are taken from the plan;
        # OFF: prefix_lens are all-zero, original_lengths derived from
        #      input_ids NestedTensor offsets diff.
        # cu_seqlens are taken from the input_ids NestedTensor offsets fed into
        # forward (ON = trimmed packed boundaries, OFF = full).
        import os as _os
        if _os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
            from prefix_sharing.tools.diagnostic_dump import (
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
                # attention_mask + label_mask: two comparison ranges for log_probs,
                # both excluding the out-of-bounds prediction position (POS L_i-1,
                # whose logp predicts the non-existent token[L_i]). Aligned to the
                # compact [B, L_max] coordinate system of restored log_probs.
                #   attention_mask: [0:L_i-1) prompt region + prompt-last + response
                #                   region (full restore verification)
                #   label_mask:     [prompt-last:L_i-1) prompt-last + response region
                #                   (PPO loss scope)
                _Lmax_lm = max(_orig_lens) if _orig_lens else 0
                # tag aligned with logprobs (entry point 2 uses model.training to
                # distinguish old/train), ensuring mask and logprobs_{tag} come from
                # the same forward pass (same batch, same L_max).
                _tag_lm = "train" if model.training else "old"
                # attention_mask only depends on _orig_lens; no loss_mask needed.
                dump_attention_mask_verl080(
                    build_attention_mask_2d(_orig_lens, _Lmax_lm), _tag_lm)
                # label_mask uses response_lens (number of response tokens per row).
                # After verl080 padding, loss_mask = response_mask is 2D left-right
                # padded (not a NestedTensor, see verl padding.py:71), so it cannot
                # go through nested_to_2d_full; however the response token count =
                # loss_mask row sum is coordinate-system agnostic, making it the
                # most robust way to derive prompt_len (works for both 2D and
                # NestedTensor).
                _lm = original_batch.get("loss_mask")
                if _lm is not None:
                    if _is_nested_tensor(_lm):
                        _lm_off = _lm.offsets()
                        _lm_val = _lm.values()
                        _response_lens = [
                            int(_lm_val[_lm_off[i]:_lm_off[i + 1]].sum())
                            for i in range(len(_orig_lens))]
                    else:
                        # .long() avoids importing torch (not imported at file top);
                        # .cpu() guards against on-device tensor .tolist()
                        _response_lens = _lm.sum(dim=-1).long().cpu().tolist()
                    dump_label_mask_verl080(
                        build_label_mask_2d(_response_lens, _orig_lens, _Lmax_lm),
                        _tag_lm)
        # ##### [PS-diag] dump metadata + masks end #####

        # ── Build modified iterator to feed back into original forward_step ──
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
            # v080 restore: reassemble reuser prefix segments while context is
            # still active. forward_step returns (output_dict,
            # partial(postprocess_func)); unpack, process output_dict, then
            # repackage. restore_via_2d_unfold_verl080 internally checks
            # context / restore_indices and does an early return when no
            # restore is needed.
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
                # Release vocab-dimension logits (high memory usage; only held
                # during context lifetime — restore has already consumed them).
                # The clear responsibility belongs here, not in the wrapper
                # function.
                ctx = current_prefix_sharing_context()
                if ctx is not None:
                    ctx.prefix_last_logits_saved.clear()
                output = (output_dict, postprocess_fn)
            # ##### [PS-diag] dump 2D logprobs/entropy (ON=post-restore, OFF=original) #####
            # After restore (ON) or from original forward (OFF), log_probs / entropy
            # are both NestedTensors with per-row length = original_lengths[i].
            # Expand to a uniform [B, L_max] for cmp_diag.cmp_2d element-wise
            # comparison.
            import os as _os2
            if _os2.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
                from prefix_sharing.tools.diagnostic_dump import (
                    nested_to_2d_full, dump_logprobs_2d_verl080, dump_entropy_2d_verl080,
                )
                from prefix_sharing.integrations.verl_mcore import _is_nested_tensor
                # Aligned with v070: tag = "old" if forward_only else "train".
                # forward_step does not have access to forward_only, so use
                # model.training as an equivalent distinction:
                #   eval_mode → training=False → "old"  (old_logp phase)
                #   train_mode → training=True → "train" (update_actor phase)
                # A single run thus produces logprobs_old + logprobs_train without
                # overwriting each other.
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
