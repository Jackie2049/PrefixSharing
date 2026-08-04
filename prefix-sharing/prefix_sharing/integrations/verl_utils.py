"""Shared helpers for verl FSDP and MCore PrefixSharing integrations."""

from __future__ import annotations

from typing import Any

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

    return _prefix_sharing_config_from_prefix_grouper(engine_config)


def _prefix_sharing_config_from_prefix_grouper(engine_config: Any) -> dict[str, Any] | None:
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


def _clone_batch(batch: Any) -> Any:
    if hasattr(batch, "clone"):
        return batch.clone()
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


def _trim_nested_batch(batch: Any, plan: PrefixSharingPlan) -> Any:
    """Physically trim a NestedTensor batch for verl 0.8 THD paths."""

    import torch

    trimmed_batch = _clone_batch(batch)

    input_ids = batch["input_ids"]
    position_ids = batch["position_ids"]

    trimmed_ids_seqs = _slice_nested_sequences(input_ids, plan)
    new_input_ids = torch.nested.nested_tensor(trimmed_ids_seqs, layout=torch.jagged)
    trimmed_batch["input_ids"] = new_input_ids

    if _is_nested_tensor(position_ids):
        trimmed_pos_seqs = _slice_nested_sequences(position_ids, plan)
        new_position_ids = torch.nested.nested_tensor(trimmed_pos_seqs, layout=torch.jagged)
    else:
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask_bool = attention_mask.to(bool)
        else:
            attention_mask_bool = torch.ones(
                position_ids.shape[0], position_ids.shape[1],
                dtype=torch.bool, device=position_ids.device,
            )
        trimmed_pos_seqs = _slice_2d_position_rows(
            position_ids, plan, attention_mask_bool,
        )
        new_position_ids = torch.nested.nested_tensor(trimmed_pos_seqs, layout=torch.jagged)
    trimmed_batch["position_ids"] = new_position_ids

    loss_mask = batch.get("loss_mask")
    if loss_mask is not None and _is_nested_tensor(loss_mask):
        trimmed_loss_seqs = _slice_nested_sequences(loss_mask, plan)
        trimmed_batch["loss_mask"] = torch.nested.nested_tensor(
            trimmed_loss_seqs, layout=torch.jagged
        )

    return trimmed_batch


def _trim_plain_batch_thd(batch: Any, plan: PrefixSharingPlan, valid_indices: list[Any] | None = None) -> Any:
    """Physically trim a plain 2D tensor batch for verl 0.8 THD paths.

    Args:
        batch: Input batch with 2D tensors.
        plan: Prefix sharing plan with keep ranges.
        valid_indices: Pre-computed nonzero indices from attention_mask.
            If None, will compute from batch["attention_mask"].
    """

    import torch

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

    trimmed_batch = _clone_batch(batch)
    trimmed_batch["input_ids"] = torch.nested.nested_tensor(kept_id_rows, layout=torch.jagged)
    trimmed_batch["position_ids"] = torch.nested.nested_tensor(kept_pos_rows, layout=torch.jagged)

    loss_mask = batch.get("loss_mask")
    if loss_mask is not None:
        kept_loss_rows = []
        for row in range(loss_mask.shape[0]):
            indices = valid_indices[row]
            keep_start, keep_end = plan.input_keep_ranges[row]
            kept_indices = indices[keep_start:keep_end]
            kept_loss_rows.append(loss_mask[row, kept_indices])
        trimmed_batch["loss_mask"] = torch.nested.nested_tensor(
            kept_loss_rows, layout=torch.jagged
        )

    return trimmed_batch


def _slice_nested_sequences(nested_tensor: Any, plan: PrefixSharingPlan) -> list[Any]:
    offsets = nested_tensor.offsets()
    values = nested_tensor.values()

    sliced = []
    for i in range(len(plan.input_keep_ranges)):
        seq_values = values[offsets[i]:offsets[i + 1]]
        keep_start, keep_end = plan.input_keep_ranges[i]
        sliced.append(seq_values[keep_start:keep_end])

    return sliced


def _slice_2d_position_rows(
    position_ids: Any,
    plan: PrefixSharingPlan,
    attention_mask_bool: Any,
) -> list[Any]:
    kept_rows = []
    for row in range(position_ids.shape[0]):
        indices = attention_mask_bool[row].nonzero(as_tuple=False).flatten()
        keep_start, keep_end = plan.input_keep_ranges[row]
        kept_indices = indices[keep_start:keep_end]
        kept_rows.append(position_ids[row, kept_indices])
    return kept_rows


def _collect_kept_position_rows(
    trimmed_batch: Any,
    plan: PrefixSharingPlan,
    is_nested_tensor: bool,
    attention_mask_bool: Any | None = None,
    valid_indices: list[Any] | None = None,
) -> list[Any]:
    """Collect per-row kept position ids from a trimmed verl batch.

    Args:
        trimmed_batch: Batch after trimming.
        plan: Prefix sharing plan.
        is_nested_tensor: Whether position_ids is a NestedTensor.
        attention_mask_bool: Boolean attention mask (deprecated, use valid_indices).
        valid_indices: Pre-computed nonzero indices. If provided, skips
            attention_mask_bool computation.
    """

    position_ids = trimmed_batch["position_ids"]

    if is_nested_tensor or _is_nested_tensor(position_ids):
        offsets = position_ids.offsets()
        values = position_ids.values()
        return [values[offsets[i]:offsets[i + 1]] for i in range(len(plan.input_keep_ranges))]

    # 2D tensor — need valid_indices to locate valid column indices
    if valid_indices is None:
        if attention_mask_bool is None:
            raise ValueError(
                "valid_indices or attention_mask_bool is required when position_ids is 2D tensor; "
                "keep_range is a sequence offset, not a column index"
            )
        valid_indices = [
            attention_mask_bool[i].nonzero(as_tuple=False).flatten()
            for i in range(len(plan.input_keep_ranges))
        ]

    rows = []
    for i in range(len(plan.input_keep_ranges)):
        indices = valid_indices[i]
        keep_start, keep_end = plan.input_keep_ranges[i]
        kept_indices = indices[keep_start:keep_end]
        rows.append(position_ids[i, kept_indices])
    return rows


def _is_nested_tensor(tensor: Any) -> bool:
    return (
        hasattr(tensor, "offsets")
        and callable(tensor.offsets)
        and hasattr(tensor, "values")
        and callable(tensor.values)
    )


def _extract_seq_from_nested_tensor(nested_tensor: Any) -> list[list[int]]:
    offsets = nested_tensor.offsets()
    values = nested_tensor.values()
    sequences = []
    for i in range(offsets.numel() - 1):
        seq = values[offsets[i]:offsets[i + 1]].detach().cpu().tolist()
        sequences.append(seq)
    return sequences
