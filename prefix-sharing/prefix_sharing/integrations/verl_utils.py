"""Shared helpers for verl FSDP and MCore PrefixSharing integrations."""

from __future__ import annotations

from typing import Any

import torch

from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.integrations.context import current_prefix_sharing_context


def read_ps_config_from_engine_config(engine_config: Any) -> Any | None:
    """Read PrefixSharing config from a verl 0.8 style engine config.

    Internal ``prefix_sharing_config`` remains a compatibility escape hatch.
    The public-facing verl path should prefer ``use_prefix_grouper`` plus
    ``prefix_grouper.mode=arbitrary_prefix``.
    """

    override = getattr(engine_config, "override_transformer_config", None)
    if override is not None:
        if isinstance(override, dict):
            explicit_config = override.get("prefix_sharing_config")
        else:
            explicit_config = getattr(override, "prefix_sharing_config", None)
        if explicit_config is not None:
            return explicit_config

    explicit_config = getattr(engine_config, "prefix_sharing_config", None)
    if explicit_config is not None:
        return explicit_config

    return read_ps_config_from_prefix_grouper(engine_config)


def read_ps_config_from_prefix_grouper(engine_config: Any) -> dict[str, Any] | None:
    use_prefix_grouper = _read_actor_value(engine_config, "use_prefix_grouper", False)
    if not use_prefix_grouper:
        return None

    prefix_grouper_config = _read_actor_value(engine_config, "prefix_grouper", None)
    mode = _read_actor_value(prefix_grouper_config, "mode", "prompt_only")
    normalized_mode = str(mode or "prompt_only").strip().lower()

    if normalized_mode in {"prompt_only", "prompt-only", "prefix_grouper"}:
        return {"enable_prefix_sharing": False}
    if normalized_mode not in {"arbitrary_prefix", "arbitrary-prefix", "prefix_sharing"}:
        raise ValueError(
            "prefix_grouper.mode must be one of: prompt_only, arbitrary_prefix"
        )

    values: dict[str, Any] = {"enable_prefix_sharing": True}
    for field_name in (
        "detector",
        "backend",
        "min_prefix_len",
        "min_group_size",
        "boundary_strategy",
        "validate_precision",
        "integrate_mode",
        "model_type",
    ):
        field_value = _read_actor_value(prefix_grouper_config, field_name, None)
        if field_value is not None:
            values[field_name] = field_value
    return values


def clone_batch(batch: Any) -> Any:
    if hasattr(batch, "clone"):
        try:
            return batch.clone()
        except TypeError:
            pass
    if hasattr(batch, "copy"):
        return batch.copy()
    if isinstance(batch, dict):
        return dict(batch)
    raise TypeError("unsupported batch type for prefix sharing")


def _read_actor_bool(config: Any, dotted_name: str, default: bool) -> bool:
    value = _read_actor_value(config, dotted_name, default)
    return bool(value)


def _read_actor_value(config: Any, dotted_name: str, default: Any) -> Any:
    current = config
    for part in dotted_name.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part, default)
        else:
            getter = getattr(current, "get", None)
            if callable(getter):
                current = getter(part, default)
            else:
                current = getattr(current, part, default)
    return current


def trim_redundant_prefix_in_nested_tensor(
    batch: Any, prefix_sharing_plan: PrefixSharingPlan
) -> tuple[Any, list[Any]]:
    """Trim redundant prefix tokens in NestedTensor inputs so that their computation
    are skipped and not performed.

    Args:
        batch: Input batch with NestedTensor ``input_ids``.
        prefix_sharing_plan: PrefixSharingPlan with keep ranges.

    Returns:
        ``(trimmed_batch, kept_position_rows)``. ``kept_position_rows`` is the
        per-row position-id slice used to build the trimmed NestedTensor, so
        callers do not need to unpack it again.
    """

    trimmed_batch = clone_batch(batch)
    input_ids = batch["input_ids"]
    position_ids = batch["position_ids"]
    
    # trim input_ids
    trimmed_batch["input_ids"] = _trim_nested_tensor(input_ids, prefix_sharing_plan)

    # trim position_ids
    if is_nested_tensor(position_ids):
        kept_position_rows = _trim_nested_rows(position_ids, prefix_sharing_plan)
    else:
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask_bool = attention_mask.to(bool)
        else:
            attention_mask_bool = torch.ones(
                position_ids.shape[0], position_ids.shape[1],
                dtype=torch.bool, device=position_ids.device,
            )
        kept_position_rows = _trim_2d_tensor(
            position_ids, prefix_sharing_plan, attention_mask_bool
        )
    trimmed_batch["position_ids"] = torch.nested.as_nested_tensor(
        kept_position_rows, layout=torch.jagged
    )

    # trim loss_mask
    loss_mask = batch.get("loss_mask")
    if loss_mask is not None and is_nested_tensor(loss_mask):
        trimmed_batch["loss_mask"] = _trim_nested_tensor(loss_mask, prefix_sharing_plan)

    return trimmed_batch, kept_position_rows


def trim_redundant_prefix_in_dense_tensor(
    batch: Any,
    prefix_sharing_plan: PrefixSharingPlan,
    valid_indices: list[Any],
) -> tuple[Any, list[Any]]:
    """Trim redundant prefix tokens from a dense 2D batch by masking.

    Unlike ``trim_plain_batch_thd`` which physically removes kept tokens,
    this helper keeps ``input_ids`` and ``position_ids`` dense and only
    masks the kept positions in ``attention_mask`` and ``loss_mask``.
    The position-id rows are trimmed to jagged tensors for packed layout
    construction, but the original 2D ``position_ids`` tensor is left
    unchanged so downstream dense-path code can still index it.

    Args:
        batch: Input batch with 2D ``input_ids``, ``position_ids`` and
            ``attention_mask``.
        prefix_sharing_plan: Prefix sharing plan with keep ranges.
        valid_indices: Pre-computed nonzero indices from ``attention_mask``.

    Returns:
        ``(trimmed_batch, kept_position_rows)``.
    """

    trimmed_batch = clone_batch(batch)
    attention_mask = batch["attention_mask"].to(bool)
    trimmed_attention_mask = attention_mask.clone()
    trimmed_attention_mask[:] = False

    for row, indices in enumerate(valid_indices):
        keep_start, keep_end = prefix_sharing_plan.input_keep_ranges[row]
        kept_indices = indices[keep_start:keep_end]
        trimmed_attention_mask[row, kept_indices] = True
    trimmed_batch["attention_mask"] = trimmed_attention_mask

    if "loss_mask" in trimmed_batch:
        trimmed_loss_mask = trimmed_batch["loss_mask"].to(bool).clone()
        trimmed_loss_mask[:] = False
        for row, indices in enumerate(valid_indices):
            keep_start, keep_end = prefix_sharing_plan.loss_mask_keep_ranges[row]
            trimmed_loss_mask[row, indices[keep_start:keep_end]] = (
                batch["loss_mask"][row, indices[keep_start:keep_end]].to(bool)
            )
        trimmed_batch["loss_mask"] = trimmed_loss_mask

    kept_position_rows = _trim_2d_tensor(
        batch["position_ids"], prefix_sharing_plan, attention_mask
    )

    return trimmed_batch, kept_position_rows


def trim_plain_batch_thd(batch: Any, plan: PrefixSharingPlan, valid_indices: list[Any] | None = None) -> Any:
    """Physically trim a plain 2D tensor batch for verl 0.8 THD paths.

    Args:
        batch: Input batch with 2D tensors.
        plan: Prefix sharing plan with keep ranges.
        valid_indices: Pre-computed nonzero indices from attention_mask.
            If None, will compute from batch["attention_mask"].
    """

    input_ids = batch["input_ids"]
    position_ids = batch["position_ids"]

    if valid_indices is None:
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask_bool = attention_mask.to(bool)
        else:
            attention_mask_bool = torch.ones(
                input_ids.shape[0], input_ids.shape[1],
                dtype=torch.bool, device=input_ids.device,
            )
        valid_indices = [
            attention_mask_bool[row].nonzero(as_tuple=False).flatten()
            for row in range(input_ids.shape[0])
        ]

    kept_id_rows = []
    kept_pos_rows = []

    for row in range(input_ids.shape[0]):
        indices = valid_indices[row]
        keep_start, keep_end = plan.input_keep_ranges[row]
        kept_indices = indices[keep_start:keep_end]
        kept_id_rows.append(input_ids[row, kept_indices])
        kept_pos_rows.append(position_ids[row, kept_indices])

    trimmed_batch = clone_batch(batch)
    trimmed_batch["input_ids"] = torch.nested.as_nested_tensor(kept_id_rows, layout=torch.jagged)
    trimmed_batch["position_ids"] = torch.nested.as_nested_tensor(kept_pos_rows, layout=torch.jagged)

    loss_mask = batch.get("loss_mask")
    if loss_mask is not None:
        kept_loss_rows = []
        for row in range(loss_mask.shape[0]):
            indices = valid_indices[row]
            keep_start, keep_end = plan.input_keep_ranges[row]
            kept_indices = indices[keep_start:keep_end]
            kept_loss_rows.append(loss_mask[row, kept_indices])
        trimmed_batch["loss_mask"] = torch.nested.as_nested_tensor(
            kept_loss_rows, layout=torch.jagged
        )

    return trimmed_batch


def _trim_nested_rows(nested_tensor: Any, prefix_sharing_plan: PrefixSharingPlan) -> list[Any]:
    """Trim each sequence of a NestedTensor to its keep range in the plan.

    Returns a list of 1D tensors (one per sequence), not packed into NestedTensor.
    """

    offsets = nested_tensor.offsets()
    values = nested_tensor.values()

    sliced = []
    for i in range(len(prefix_sharing_plan.input_keep_ranges)):
        seq_values = values[offsets[i]:offsets[i + 1]]
        keep_start, keep_end = prefix_sharing_plan.input_keep_ranges[i]
        sliced.append(seq_values[keep_start:keep_end])

    return sliced


def _trim_nested_tensor(nested_tensor: Any, prefix_sharing_plan: PrefixSharingPlan) -> Any:
    """Trim each sequence of a NestedTensor and pack into a jagged NestedTensor."""

    return torch.nested.as_nested_tensor(
        _trim_nested_rows(nested_tensor, prefix_sharing_plan), layout=torch.jagged
    )


def _trim_2d_tensor(
    tensor_2d: Any,
    prefix_sharing_plan: PrefixSharingPlan,
    attention_mask_bool: Any,
) -> list[Any]:
    """Trim each row of a 2D dense tensor to its keep range in the plan.

    Returns a list of 1D tensors (one per row), not packed into NestedTensor.
    """

    kept_rows = []
    for row in range(tensor_2d.shape[0]):
        indices = attention_mask_bool[row].nonzero(as_tuple=False).flatten()
        keep_start, keep_end = prefix_sharing_plan.input_keep_ranges[row]
        kept_indices = indices[keep_start:keep_end]
        kept_rows.append(tensor_2d[row, kept_indices])
    return kept_rows


def collect_kept_position_rows(
    trimmed_batch: Any,
    prefix_sharing_plan: PrefixSharingPlan,
    is_nested_input: bool,
    attention_mask_bool: Any | None = None,
    valid_indices: list[Any] | None = None,
) -> list[Any]:
    """Collect per-row kept position ids from a trimmed verl batch.

    Args:
        trimmed_batch: Batch after trimming.
        plan: Prefix sharing plan.
        is_nested_input: Whether position_ids is a NestedTensor.
        attention_mask_bool: Boolean attention mask (deprecated, use valid_indices).
        valid_indices: Pre-computed nonzero indices. If provided, skips
            attention_mask_bool computation.
    """

    position_ids = trimmed_batch["position_ids"]

    if is_nested_input or is_nested_tensor(position_ids):
        offsets = position_ids.offsets()
        values = position_ids.values()
        return [values[offsets[i]:offsets[i + 1]] for i in range(len(prefix_sharing_plan.input_keep_ranges))]

    # 2D tensor — need valid_indices to locate valid column indices
    if valid_indices is None:
        if attention_mask_bool is None:
            raise ValueError(
                "valid_indices or attention_mask_bool is required when position_ids is 2D tensor; "
                "keep_range is a sequence offset, not a column index"
            )
        valid_indices = [
            attention_mask_bool[i].nonzero(as_tuple=False).flatten()
            for i in range(len(prefix_sharing_plan.input_keep_ranges))
        ]

    rows = []
    for i in range(len(prefix_sharing_plan.input_keep_ranges)):
        indices = valid_indices[i]
        keep_start, keep_end = prefix_sharing_plan.input_keep_ranges[i]
        kept_indices = indices[keep_start:keep_end]
        rows.append(position_ids[i, kept_indices])
    return rows


def is_nested_tensor(tensor: Any) -> bool:
    return (
        hasattr(tensor, "offsets")
        and callable(tensor.offsets)
        and hasattr(tensor, "values")
        and callable(tensor.values)
    )


def _copy_tensors_to_cpu_lists(tensors: list[Any]) -> list[list[int]]:
    """Copy device tensors to Python int lists with a single sync point.

    Fast path (GPU / NPU): stage all slices into pinned host buffers with
    non-blocking copies, then synchronize the source device once — avoiding
    one pipeline stall per tensor. Backends without pinned-memory support
    (older torch_npu, CPU, …) fall back to plain per-tensor copies: correct
    everywhere, just without the batched-copy optimization.

    Device handling is dispatched via ``getattr(torch, device.type)`` so this
    works on any accelerator backend without hardcoding ``torch.cuda``.
    """

    if not tensors:
        return []

    device = tensors[0].device
    accel = getattr(torch, device.type, None)
    if accel is None or not hasattr(accel, "synchronize"):
        # CPU and unknown backends: plain copies, no accelerator calls.
        return [t.cpu().tolist() for t in tensors]

    try:
        pinned_buffers = [
            torch.empty_like(t, device="cpu", pin_memory=True) for t in tensors
        ]
    except RuntimeError:
        # Backend has no pinned-memory allocator (e.g. old torch_npu):
        # fall back to the original per-tensor synchronous copies.
        return [t.cpu().tolist() for t in tensors]

    for buf, t in zip(pinned_buffers, tensors):
        buf.copy_(t, non_blocking=True)

    # Single sync point on the device the tensors actually live on.
    accel.synchronize(device)

    return [buf.tolist() for buf in pinned_buffers]


def extract_seq_from_nested_tensor(nested_tensor: Any) -> list[list[int]]:
    """Extract token sequences from a NestedTensor (jagged layout)."""

    offsets = nested_tensor.offsets()
    values = nested_tensor.values()

    tensor_slices = [
        values[offsets[i]:offsets[i + 1]].detach()
        for i in range(offsets.numel() - 1)
    ]
    return _copy_tensors_to_cpu_lists(tensor_slices)


def extract_seq_from_dense_tensor(
    input_ids: Any,
    valid_indices: list[Any],
) -> list[list[int]]:
    """Extract token sequences from a dense 2D device tensor with batched async copy.

    Uses a batched async copy: instead of calling .cpu().tolist() per row
    (which syncs each time), all device→CPU copies are queued non-blocking,
    then synchronized once (CUDA/NPU fast path; other backends fall back to
    plain copies).

    Args:
        input_ids: 2D device tensor of shape [batch_size, seq_len].
        valid_indices: Pre-computed nonzero indices per row.

    Returns:
        List of token ID lists (CPU Python ints).
    """

    tensor_slices = [
        input_ids[row, indices].detach()
        for row, indices in enumerate(valid_indices)
    ]
    return _copy_tensors_to_cpu_lists(tensor_slices)


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
