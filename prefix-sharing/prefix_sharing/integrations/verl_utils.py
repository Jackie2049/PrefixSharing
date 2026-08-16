"""Shared helpers for verl FSDP and MCore PrefixSharing integrations."""

from __future__ import annotations

from typing import Any

import torch

from prefix_sharing.core.planner import PrefixSharingPlan


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
