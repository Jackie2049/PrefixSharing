"""verl Megatron actor integration helpers.

This module keeps the Megatron/MCore batch construction and restore helpers.
Production monkey-patching is owned by ``prefix_sharing.setup`` patch sets;
this module no longer exposes standalone patch installer classes.

Both paths share the same core logic (plan -> trim -> layout -> state).

``VerlMCoreBatchAdapter`` is framework-light and testable locally. It turns a
verl-style micro-batch payload into prefix-sharing metadata plus trimmed
inputs/labels/masks, and it assembles restored logprobs after forward.
"""

from __future__ import annotations

from typing import Any

from prefix_sharing.backends.factory import get_backend_instance
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import current_prefix_sharing_context
from prefix_sharing.integrations.parallel_info import get_megatron_parallel_info
from prefix_sharing.integrations.runtime_state import PrefixSharingRuntimeState
from prefix_sharing.integrations.verl_utils import collect_kept_position_rows
from prefix_sharing.integrations.verl_utils import extract_seq_from_dense_tensor
from prefix_sharing.integrations.verl_utils import extract_seq_from_nested_tensor
from prefix_sharing.integrations.verl_utils import is_nested_tensor
from prefix_sharing.integrations.verl_utils import trim_redundant_prefix_in_nested_tensor
from prefix_sharing.integrations.verl_utils import trim_plain_batch_thd
from prefix_sharing.integrations.verl_utils import read_ps_config_from_engine_config


def restore_reuser_prefix_columns_2d(
    output: dict[str, Any],
    vocab_parallel_log_probs_fn: Any,
    vocab_parallel_entropy_fn: Any = None,
) -> dict[str, Any]:
    """Restore reuser prefix columns in 2D space — build_kv-style slice + concat.

    Mirrors :meth:`TorchReferenceBackend.build_kv`
    (``torch.cat([provider_kv[:prefix_len], own_suffix])``): instead of writing
    each prefix token one scalar at a time, the whole prefix interval is sliced
    off the **direct provider's already-restored 2D row** and only the single
    prefix-last logprob is recomputed.

    Per ``reuser_idx`` with direct provider ``provider_idx = provider_index[reuser_idx]``
    and ``P = prefix_lens[reuser_idx]`` (columns are identity-mapped in the
    unfolded 2D tensor, so ``target_2d_pos`` == column):

    - **interior ``[0, P-2]``**: ``log_probs[reuser_idx, 0:P-1] = log_probs[provider_idx, 0:P-1]``
      (bulk copy). Identical across the shared prefix (same logits + labels),
      and ``provider_idx`` was restored earlier in the batch-order loop, so its
      row already holds correct values — no per-position provider resolution
      needed.
    - **prefix-last ``P-1``**: recompute ``log_probs[reuser_idx, P-1]`` from the
      saved provider logits + the reuser's own first-suffix label (differs from
      the provider's). When the reuser has no suffix (``suffix_len == 0``) the
      planner emits no prefix-last spec; that column is masked downstream, so
      the provider's value is copied as a safe placeholder.
    - **entropy ``[0, P-1]``**: ``entropy[reuser_idx, 0:P] = entropy[provider_idx, 0:P]``
      (whole prefix copied, including prefix-last — entropy is label-independent).

    Rows are visited in ``range(B)`` order so a provider is always restored
    before any reuser that reads it (the same online-detector invariant
    ``build_kv`` relies on).

    Args:
        output: Output dict with ``log_probs`` [B, L] and optionally
            ``entropy`` [B, L] in 2D space (unfolded from the trimmed
            NestedTensor by :func:`restore_via_2d_unfold_verl080`).
        vocab_parallel_log_probs_fn: ``logits [1, V//tp]``, ``label [1]`` →
            scalar; used only for the prefix-last recompute.
        vocab_parallel_entropy_fn: Retained for call-site compatibility;
            unused (entropy is copied, never recomputed).

    Returns:
        ``output`` with ``log_probs`` and ``entropy`` mutated in-place.
    """

    ctx = current_prefix_sharing_context()
    if ctx is None:
        return output
    plan = ctx.prefix_sharing_plan
    # Guard on reuser presence (not on prefix_last_restore_indices): a batch
    # whose reusers all have suffix_len == 0 emits no prefix-last spec but still
    # needs its interior prefix columns restored.
    if not plan.has_sharing:
        return output

    import torch

    log_probs = output.get("log_probs")
    if log_probs is None:
        return output
    entropy = output.get("entropy")

    provider_index = plan.provider_index
    prefix_lens = plan.prefix_lens

    # reuser row → its prefix-last restore spec (one per reuser-with-suffix;
    # interior positions have no spec — they are bulk-sliced below).
    prefix_last_spec_by_reuser = {
        spec.reuse_idx_in_batch: spec for spec in ctx.prefix_last_restore_indices
    }

    restored_reusers = 0
    # Row 0 is always a provider (nothing precedes it to reuse), so start at 1.
    # A reuser's provider always has a smaller batch index (online-detector
    # invariant), so it is already restored when we reach reuser_idx.
    for reuser_idx in range(1, len(prefix_lens)):
        prefix_len = prefix_lens[reuser_idx]
        if provider_index[reuser_idx] == reuser_idx or prefix_len <= 0:
            continue  # provider / non-reuser: row already complete
        provider_idx = provider_index[reuser_idx]

        # interior [0, prefix_len-2]: bulk-copy from the provider's restored row.
        if prefix_len - 1 > 0:
            log_probs[reuser_idx, 0:prefix_len - 1] = log_probs[provider_idx, 0:prefix_len - 1]

        # prefix-last (position prefix_len-1): recompute with the reuser's label.
        prefix_last_spec = prefix_last_spec_by_reuser.get(reuser_idx)
        if prefix_last_spec is not None:
            saved_logits_key = (reuser_idx, prefix_last_spec.target_2d_pos)
            saved_provider_logits = ctx.prefix_last_logits_saved[saved_logits_key]  # [1, V//tp]
            reuser_label = torch.tensor(
                [prefix_last_spec.label_value], dtype=torch.long, device=log_probs.device,
            )  # [1]
            log_probs[reuser_idx, prefix_len - 1] = vocab_parallel_log_probs_fn(
                saved_provider_logits, reuser_label,
            ).reshape(())
        else:
            # suffix_len == 0: no prefix-last spec; column is masked downstream.
            log_probs[reuser_idx, prefix_len - 1] = log_probs[provider_idx, prefix_len - 1]

        # entropy [0, prefix_len-1]: whole prefix copied (label-independent).
        if entropy is not None:
            entropy[reuser_idx, 0:prefix_len] = entropy[provider_idx, 0:prefix_len]

        restored_reusers += 1

    if ctx.stats is not None:
        ctx.stats.record_restore(restored_reusers)
    return output


# ═══════════════════════════════════════════════════════════════
# v080 restore 包装：NestedTensor → 2D left-pad → 复用 2D restore → 压回
# ═══════════════════════════════════════════════════════════════


def restore_via_2d_unfold_verl080(
    output: dict,
    vocab_parallel_log_probs_fn: Any,
    vocab_parallel_entropy_fn: Any = None,
) -> dict:
    """v080 restore 包装：NestedTensor → 2D left-pad → 复用 restore_reuser_prefix_columns_2d → 压回。

    v080 物理裁剪后 reuser NestedTensor 行只含 suffix 区段，prefix 区段（含
    prefix-last）被物理删除。本函数在 forward_step 出口（context 仍激活、provider
    prefix-last logits 已存于 ``ctx.prefix_last_logits_saved``）完成重组：

    1. 展开裁剪后 NestedTensor 各行为完整 2D ``[B, L_max]``（reuser prefix 区段
       left-pad 0，尾部 right-pad 0 到 L_max）
    2. 复用 :func:`restore_reuser_prefix_columns_2d`：interior 整段从直接
       provider 的已恢复 2D 行 bulk 切片复制，prefix-last 用存的 logits +
       ``index.label_value`` 重算
    3. 按各 ``original_lengths`` 切片压回 NestedTensor (jagged)

    列映射为 identity：left-pad 后 valid-content 的 0-based 偏移即 2D 列号，
    ``target_2d_pos`` 直接当列索引用，无需 ``valid_indices`` / 列映射表。

    Must be called inside ``prefix_sharing_runtime_context`` (reads
    ``current_prefix_sharing_context``), after the vocab_logprobs patch has saved
    provider prefix-last logits into ``ctx.prefix_last_logits_saved``.

    Args:
        output: forward_step 返回的 output_dict，含 ``"log_probs"`` NestedTensor
            （裁剪后 jagged），可选 ``"entropy"`` NestedTensor。**不含** tuple 外层
            （tuple 解包由调用方负责）。
        vocab_parallel_log_probs_fn: 用于 prefix-last logp 重算。
        vocab_parallel_entropy_fn: 可选，当前未直接使用（entropy 走复制路径——
            interior 和 prefix-last 都从 provider 复制，不重算）。

    Returns:
        ``output``（``log_probs``/``entropy`` 被替换为重组后的 NestedTensor）。
    """

    ctx = current_prefix_sharing_context()
    if ctx is None:
        return output
    plan = ctx.prefix_sharing_plan
    # Guard on reuser presence, not on prefix_last_restore_indices: a batch
    # whose reusers all have suffix_len == 0 has no prefix-last spec but still
    # needs interior prefix columns restored.
    if not plan.has_sharing:
        return output

    log_probs_nested = output.get("log_probs")
    if log_probs_nested is None or not is_nested_tensor(log_probs_nested):
        return output
    entropy_nested = output.get("entropy")
    has_entropy = entropy_nested is not None and is_nested_tensor(entropy_nested)

    original_lengths = plan.original_lengths
    input_keep_ranges = plan.input_keep_ranges
    B = len(original_lengths)
    if B == 0:
        return output
    L_max = max(original_lengths)

    # --- Step 1: 展开裁剪后 NestedTensor → 完整 2D [B, L_max] ---
    log_probs_2d, entropy_2d = _unfold_trimmed_nested_to_2d(
        log_probs_nested,
        entropy_nested if has_entropy else None,
        original_lengths,
        input_keep_ranges,
        L_max,
        B,
    )

    # --- Step 2: 复用 restore_reuser_prefix_columns_2d ---
    # build_kv 式区间拼接：interior 整段从直接 provider 的已恢复 2D 行切片，
    # prefix-last 用 index.label_value + saved logits 重算。identity 列映射
    # （target_2d_pos 即 2D 列号，无 left padding）。
    output_2d: dict[str, Any] = {"log_probs": log_probs_2d}
    if entropy_2d is not None:
        output_2d["entropy"] = entropy_2d
    output_2d = restore_reuser_prefix_columns_2d(
        output_2d,
        vocab_parallel_log_probs_fn,
        vocab_parallel_entropy_fn,
    )

    # --- Step 3: 按各 original_lengths 压回 NestedTensor ---
    output["log_probs"] = _fold_2d_to_nested(output_2d["log_probs"], original_lengths)
    if entropy_2d is not None:
        output["entropy"] = _fold_2d_to_nested(output_2d["entropy"], original_lengths)

    num_prefix_last = len(ctx.prefix_last_restore_indices)
    print(
        f"[PS][restore_verl080] unfolded B={B} L_max={L_max}, "
        f"restored reusers={num_prefix_last} (prefix-last entries; "
        f"interior handled by bulk slice)",
        flush=True,
    )
    return output


def _unfold_trimmed_nested_to_2d(
    log_probs_nested: Any,
    entropy_nested: Any,
    original_lengths: list[int],
    input_keep_ranges: list,
    L_max: int,
    B: int,
) -> tuple[Any, Any | None]:
    """展开裁剪后 NestedTensor → 完整 2D [B, L_max]（reuser prefix left-pad 0）。

    裁剪后各行：
      - provider (keep_start=0): 完整 [prefix | suffix]，长度 = original_lengths[i]
      - reuser  (keep_start=prefix_len>0): 仅 [suffix]，长度 = original_lengths[i]-prefix_len

    展开后每行恢复成 [prefix_zeros | suffix]，再 right-pad 0 到 L_max。
    left-pad 的 zeros 不在 autograd 图里，但 restore 会覆盖 prefix 区段（interior
    复制 provider、prefix-last 重算），最终值在图里。right-pad 尾部在压回时丢弃。
    """
    import torch

    log_probs_offsets = log_probs_nested.offsets()
    log_probs_values = log_probs_nested.values()
    if entropy_nested is not None:
        entropy_offsets = entropy_nested.offsets()
        entropy_values = entropy_nested.values()

    log_probs_rows: list[Any] = []
    entropy_rows: list[Any] | None = [] if entropy_nested is not None else None

    for seq_idx in range(B):
        orig_len = original_lengths[seq_idx]
        prefix_len = input_keep_ranges[seq_idx][0]

        log_probs_suffix = log_probs_values[log_probs_offsets[seq_idx]:log_probs_offsets[seq_idx + 1]]
        log_probs_rows.append(_build_padded_row(log_probs_suffix, prefix_len, orig_len, L_max))

        if entropy_nested is not None:
            entropy_suffix = entropy_values[entropy_offsets[seq_idx]:entropy_offsets[seq_idx + 1]]
            entropy_rows.append(_build_padded_row(entropy_suffix, prefix_len, orig_len, L_max))

    log_probs_2d = torch.stack(log_probs_rows, dim=0)
    entropy_2d = torch.stack(entropy_rows, dim=0) if entropy_rows else None
    return log_probs_2d, entropy_2d


def _build_padded_row(
    suffix_data: Any, prefix_len: int, orig_len: int, L_max: int,
) -> Any:
    """构造一行完整 2D ``[prefix_zeros | suffix]`` right-pad 0 到 L_max。"""
    import torch

    device = suffix_data.device
    dtype = suffix_data.dtype
    tail_shape = tuple(suffix_data.shape[1:])
    pieces: list[Any] = []
    if prefix_len > 0:
        pieces.append(torch.zeros((prefix_len,) + tail_shape, dtype=dtype, device=device))
    pieces.append(suffix_data)
    row = torch.cat(pieces, dim=0)  # [orig_len, ...]
    if orig_len < L_max:
        pad = torch.zeros((L_max - orig_len,) + tail_shape, dtype=dtype, device=device)
        row = torch.cat([row, pad], dim=0)
    return row


def _fold_2d_to_nested(tensor_2d: Any, original_lengths: list[int]) -> Any:
    """完整 2D [B, L_max] → NestedTensor (jagged)，按各 original_lengths 切片。"""
    import torch

    rows = [tensor_2d[seq_idx, :original_lengths[seq_idx]] for seq_idx in range(len(original_lengths))]
    values = torch.cat(rows, dim=0) if rows else tensor_2d.reshape(0, *tensor_2d.shape[2:])
    offsets = torch.tensor(
        [0] + [sum(original_lengths[: idx + 1]) for idx in range(len(original_lengths))],
        dtype=torch.long,
        device=tensor_2d.device,
    )
    if hasattr(torch.nested, "nested_tensor_from_jagged"):
        return torch.nested.nested_tensor_from_jagged(values, offsets)
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def build_prefix_sharing_micro_batch_verl080(
    engine_self: Any,
    batch: Any,
    ps_config: PrefixSharingConfig,
) -> tuple[Any, PrefixSharingRuntimeState | None]:
    """verl 0.8.0 engine 架构下的 prefix-sharing micro-batch 构建。

    MCore/THD 路径：NestedTensor 裁剪后按 kept 区段展开为 packed layout，
    2D 路径：物理裁剪 input_ids/position_ids，通过 attention_mask 标记 valid。

    参数 ps_config 已由调用方通过 PrefixSharingConfig.from_raw() 解析完成，
    不需要再次 from_raw。

    核心原则：2D + attention_mask 为主路径，NestedTensor 路径仅在
    GPU + use_remove_padding=True 时作为可选优化。
    NPU 不支持 torch.nested，所有 NPU 场景都走 2D 路径。
    """
    # ── PATH 1: prefix sharing disabled ──
    if not ps_config.enable_prefix_sharing:
        print("[PS][prepare] PATH 1: prefix sharing disabled")
        return batch, None

    # ── 阶段 1: 配置校验 ──
    use_remove_padding = getattr(engine_self.engine_config, "use_remove_padding", True)
    ps_config.validate_for_engine(use_remove_padding=use_remove_padding)

    # ── 阶段 2: 拒绝不支持的特性 ──
    try:
        from verl.utils import tensordict_utils as tu
        use_fused = tu.get_non_tensor_data(batch, "use_fused_kernels", default=False)
    except Exception:
        use_fused = False
    if use_fused:
        raise RuntimeError("prefix sharing phase 1 requires fused kernels disabled")
    if getattr(engine_self.engine_config, "dynamic_context_parallel", False):
        raise RuntimeError("prefix sharing phase 1 does not support dynamic context parallel")

    # ── 阶段 3: 从 batch 提取序列 ──
    # NestedTensor → 从 offsets/values 提取；kept_position_rows 由 trim 直接返回。
    # Plain 2D → 从 attention_mask.nonzero() 提取，并保留 valid_indices
    # 供 trim 后的 collect_kept_position_rows 使用。
    input_ids = batch["input_ids"]
    is_nested_input = is_nested_tensor(input_ids)
    attention_mask_bool_for_layout = None

    if is_nested_input:
        sequences = extract_seq_from_nested_tensor(input_ids)
    else:
        # plain 2D tensor（需要 attention_mask）
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            print("[PS][prepare] PATH 4: plain 2D batch without attention_mask")
            return batch, None
        attention_mask_bool = attention_mask.to(bool)
        valid_indices = [
            attention_mask_bool[row].nonzero(as_tuple=False).flatten()
            for row in range(input_ids.shape[0])
        ]
        sequences = extract_seq_from_dense_tensor(input_ids, valid_indices)

    # ── 阶段 4: 前缀共享规划 ──
    plan = PrefixSharingPlanner(ps_config).plan(sequences)
    if not plan.has_sharing:
        print("[PS][prepare] no prefix sharing detected")
        return batch, None

    # ── 阶段 5: 物理裁剪 batch ──
    #   NestedTensor path: 裁剪 input_ids/position_ids/loss_mask 以匹配 kept 区段。
    #   2D path: 只改 attention_mask（Megatron 从 mask 动态重算 packed），
    #   v080 THD 路径用 preprocess_thd_engine(input_ids) 直接处理数据，
    #   不看 attention_mask。必须物理裁剪 input_ids/position_ids。
    if is_nested_input:
        trimmed_batch, kept_position_rows = trim_redundant_prefix_in_nested_tensor(batch, plan)
    else:
        trimmed_batch = trim_plain_batch_thd(batch, plan, valid_indices)
        kept_position_rows = collect_kept_position_rows(
            trimmed_batch, plan, is_nested_input,
            valid_indices=valid_indices,
        )

    # ── 阶段 6: 构建 layout ──
    parallel_info = get_megatron_parallel_info()
    align_size = (
        parallel_info.tp_size * parallel_info.cp_size * 2
        if parallel_info.cp_size > 1
        else parallel_info.tp_size
    )
    packed_layout = PackedBatchLayout.from_kept_position_rows(
        kept_position_rows,
        align_size=int(align_size),
    )

    # ── 阶段 7: 构建 state ──
    state = PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        attention_backend=get_backend_instance(ps_config),
        packed_batch_layout=packed_layout,
        parallel_info=parallel_info,
    )

    print(
        f"[PS][prepare] PATH 6: sharing detected, plan={plan}, layout={packed_layout}"
    )

    return trimmed_batch, state

