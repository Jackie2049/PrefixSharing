"""Unified diagnostic dump for ON/OFF precision comparison — verl080 FSDP + Megatron.

Single env var ``PREFIX_SHARING_DIAG_DUMP`` controls the dump directory.

Covers both Megatron and FSDP paths. Core infrastructure (``_get_dump_dir``,
``_save_tensor``, ``_rank0_only``, buffer helpers) is shared; path-specific
functions are prefixed (``dump_fsdp_*`` for FSDP, ``dump_*_verl080`` for
Megatron verl080). NestedTensor ↔ 2D conversions are shared.

Usage:
    export PREFIX_SHARING_DIAG_DUMP=/path/to/dump_on   # ON  run
    export PREFIX_SHARING_DIAG_DUMP=/path/to/dump_off  # OFF run
    # Then compare with cmp_diag.py
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import torch

_log = logging.getLogger(__name__)

# ── Internal state ──────────────────────────────────────────────
_DUMP_DIR: str | None = None
_META_SAVED: set[str] = set()
_ATTN_BUFFER: dict[int, torch.Tensor] | None = None
_ROPE_BUFFER: dict[int, dict] | None = None
_ROPE_FREQS_BUFFER: dict[int, torch.Tensor] | None = None


def _get_dump_dir() -> str | None:
    """Read env var once, create directory on first call (cached)."""
    global _DUMP_DIR
    if _DUMP_DIR is not None:
        return _DUMP_DIR
    path = os.environ.get("PREFIX_SHARING_DIAG_DUMP")
    if path:
        _DUMP_DIR = path
        os.makedirs(path, exist_ok=True)
    return _DUMP_DIR


def _rank0_only() -> bool:
    """Returns True on rank 0 (or if distributed is not initialized)."""
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:
        pass
    return True


# ── Parallel-topology-aware dumping (TP / SP / PP) ──────────────

_TENSOR_SCOPES: dict[str, str] = {
    "logits": "tp_vocab",
    "attn_outputs": "pp_stage",
    "rope_postqk": "pp_stage",
    "rope_preqk": "pp_stage",
    "rope_freqs": "pp_stage",
    "expanded_kv": "pp_stage",
    "full_kv": "pp_stage",
    "build_kv_input_v": "pp_stage",
    "hidden_states": "pp_stage",
}

_PARALLEL_INFO_CACHE: Any = None
_MANIFEST_WRITTEN: set[str] = set()


def _cached_parallel_info() -> Any:
    """Read MegatronParallelInfo once and cache (tp/pp/cp ranks + sizes)."""
    global _PARALLEL_INFO_CACHE
    if _PARALLEL_INFO_CACHE is not None:
        return _PARALLEL_INFO_CACHE
    try:
        from prefix_sharing.integrations.parallel_info import (
            get_megatron_parallel_info,
        )
        _PARALLEL_INFO_CACHE = get_megatron_parallel_info()
    except Exception:
        _PARALLEL_INFO_CACHE = None
    return _PARALLEL_INFO_CACHE


def _stage_last_layer(num_layers_global: int) -> int:
    """Return the last global layer number owned by the current PP stage."""
    parallel_info = _cached_parallel_info()
    if parallel_info is None or parallel_info.pp_size <= 1:
        return num_layers_global
    base = num_layers_global // parallel_info.pp_size
    remainder = num_layers_global % parallel_info.pp_size
    if parallel_info.pp_rank < remainder:
        return (parallel_info.pp_rank + 1) * (base + 1)
    else:
        return remainder * (base + 1) + (parallel_info.pp_rank - remainder + 1) * base


def _pp_suffix() -> str:
    """Return ``'_pp{r}'`` when ``pp_size > 1``, else ``''``."""
    parallel_info = _cached_parallel_info()
    if parallel_info is not None and parallel_info.pp_size > 1:
        return f"_pp{parallel_info.pp_rank}"
    return ""


def _should_write_for_scope(scope: str) -> bool:
    """Return True if this rank should dump data for the given scope."""
    parallel_info = _cached_parallel_info()
    if scope == "global":
        return _rank0_only()
    if scope == "tp_vocab":
        return True
    if scope == "pp_last":
        if parallel_info is not None and not parallel_info.is_pipeline_last_stage:
            return False
        return parallel_info is None or parallel_info.tp_rank == 0
    if scope == "pp_stage":
        if parallel_info is None or parallel_info.pp_size <= 1:
            return _rank0_only()
        return parallel_info.tp_rank == 0
    return _rank0_only()


def _with_suffix(name: str, suffix: str) -> str:
    """Insert a rank suffix before the extension: logits.pt → logits_tp0.pt."""
    if not suffix:
        return name
    stem, sep, ext = name.rpartition(".")
    return f"{stem}{suffix}{sep}{ext}" if sep else f"{name}{suffix}"


def _ensure_manifest(dump_dir: str, parallel_info: Any) -> None:
    """Write parallel_info.json once: topology + scope map."""
    if dump_dir in _MANIFEST_WRITTEN:
        return
    _MANIFEST_WRITTEN.add(dump_dir)
    if parallel_info is not None and parallel_info.pp_size > 1:
        if parallel_info.tp_rank != 0:
            return
    elif not _rank0_only():
        return
    import json
    manifest = {
        "tp_size": getattr(parallel_info, "tp_size", 1) if parallel_info else 1,
        "pp_size": getattr(parallel_info, "pp_size", 1) if parallel_info else 1,
        "cp_size": getattr(parallel_info, "cp_size", 1) if parallel_info else 1,
        "global_rank_of_dumper": getattr(parallel_info, "global_rank", 0) if parallel_info else 0,
        "scopes": dict(_TENSOR_SCOPES),
    }
    try:
        with open(os.path.join(dump_dir, "parallel_info.json"),
                  "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)
        _log.warning("parallel_info.json saved (tp=%d pp=%d)",
                     manifest["tp_size"], manifest["pp_size"])
    except Exception as exc:
        _log.warning("parallel_info.json save failed: %s", exc)


# ── Generic helpers ─────────────────────────────────────────────

def _save_tensor(name: str, tensor: torch.Tensor, dump_dir: str,
                 scope: str = "global") -> bool:
    """Save a tensor to ``dump_dir/<name>``, scope-aware."""
    parallel_info = _cached_parallel_info()
    _ensure_manifest(dump_dir, parallel_info)

    if not _should_write_for_scope(scope):
        return False

    if scope == "tp_vocab" and parallel_info is not None and parallel_info.tp_size > 1:
        filename = _with_suffix(name, f"_tp{parallel_info.tp_rank}")
    else:
        if scope == "tp_vocab":
            scope = "global"
        filename = name
    try:
        filepath = os.path.join(dump_dir, filename)
        torch.save(tensor.detach().cpu().clone(), filepath)
        _log.warning("%s saved (%s, scope=%s)", filename, tensor.shape, scope)
        return True
    except Exception as exc:
        _log.warning("%s save failed: %s", filename, exc)
        return False


def _save_meta(packed_seq_params: Any,
               prefix_lens_list: list[int],
               dump_dir: str,
               meta_key: str = "attn") -> None:
    """Save cu_seqlens + prefix_lens (dedup per meta_key across one run)."""
    global _META_SAVED
    cumulative_key = f"cu_{meta_key}"
    prefix_lens_key = "pl"
    if not _rank0_only():
        return
    try:
        if cumulative_key not in _META_SAVED:
            cumulative_seq_lens = packed_seq_params.cu_seqlens_q_padded.detach().cpu().clone()
            cumulative_filename = "cu_seqlens_q.pt" if meta_key == "attn" else f"cu_seqlens_q_{meta_key}.pt"
            torch.save(cumulative_seq_lens, os.path.join(dump_dir, cumulative_filename))
            _log.warning("%s saved (%s)", cumulative_filename, cumulative_seq_lens.shape)
            _META_SAVED.add(cumulative_key)
    except Exception as exc:
        _log.warning("cu_seqlens (%s) save failed: %s", meta_key, exc)

    try:
        if prefix_lens_key not in _META_SAVED:
            prefix_lens_tensor = torch.tensor(prefix_lens_list, dtype=torch.int32)
            torch.save(prefix_lens_tensor, os.path.join(dump_dir, "prefix_lens.pt"))
            _log.warning("prefix_lens.pt saved (%s)", prefix_lens_tensor.shape)
            _META_SAVED.add(prefix_lens_key)
    except Exception as exc:
        _log.warning("prefix_lens.pt save failed: %s", exc)


# ── Per-layer buffer helpers ────────────────────────────────────

def _add_to_attn_buffer(layer_number: int, tensor: torch.Tensor) -> None:
    """Accumulate one layer's attention output into the global buffer."""
    global _ATTN_BUFFER
    if _ATTN_BUFFER is None:
        _ATTN_BUFFER = {}
    _ATTN_BUFFER[layer_number] = tensor.detach().cpu().clone()


def _flush_attn_buffer(dump_dir: str) -> None:
    """Write accumulated attn_outputs dict to disk and clear buffer."""
    global _ATTN_BUFFER
    if _ATTN_BUFFER is None:
        return
    if not _should_write_for_scope("pp_stage"):
        _ATTN_BUFFER = None
        return
    try:
        moved = {k: v.detach().cpu().clone() for k, v in _ATTN_BUFFER.items()}
        filename = f"attn_outputs{_pp_suffix()}.pt"
        torch.save(moved, os.path.join(dump_dir, filename))
        _log.warning("%s saved (%d layers)", filename, len(moved))
        _ATTN_BUFFER = None
    except Exception as exc:
        _log.warning("attn_outputs.pt save failed: %s", exc)


def _add_to_rope_buffer(layer_number: int, rotated_query: torch.Tensor,
                        rotated_key: torch.Tensor,
                        positions: torch.Tensor | None = None) -> None:
    """Accumulate one layer's RoPE encoding into the global buffer."""
    global _ROPE_BUFFER
    if _ROPE_BUFFER is None:
        _ROPE_BUFFER = {}
    _ROPE_BUFFER[layer_number] = {
        "query": rotated_query.detach().cpu().clone(),
        "key": rotated_key.detach().cpu().clone(),
        "positions": positions.detach().cpu().clone() if positions is not None else None,
    }


def _flush_rope_buffer(dump_dir: str) -> None:
    """Write accumulated rope_postqk dict to disk and clear buffer."""
    global _ROPE_BUFFER
    if _ROPE_BUFFER is None:
        return
    if not _should_write_for_scope("pp_stage"):
        _ROPE_BUFFER = None
        return
    try:
        filename = f"rope_postqk{_pp_suffix()}.pt"
        torch.save(_ROPE_BUFFER, os.path.join(dump_dir, filename))
        _log.warning("%s saved (%d layers)", filename, len(_ROPE_BUFFER))
        _ROPE_BUFFER = None
    except Exception as exc:
        _log.warning("rope_postqk.pt save failed: %s", exc)


def _flush_dict_buffer(filename: str, buffer: dict, dump_dir: str) -> None:
    """rank0 torch.save a dict buffer. PP-aware gating + suffix."""
    if not _should_write_for_scope("pp_stage"):
        return
    try:
        stem, separator, extension = filename.rpartition(".")
        pp_suffix_str = _pp_suffix()
        pp_filename = f"{stem}{pp_suffix_str}{separator}{extension}" if separator else f"{filename}{pp_suffix_str}"
        torch.save(buffer, os.path.join(dump_dir, pp_filename))
    except Exception as exc:
        print(f"[PS-diag] {filename} save failed: {exc}", flush=True)


# ════════════════════════════════════════════════════════════════
#  NestedTensor → 2D conversion helpers (shared Megatron + FSDP)
# ════════════════════════════════════════════════════════════════

def nested_to_2d_full(nested: Any, original_lengths: list[int], max_seq_len: int) -> torch.Tensor:
    """NestedTensor (jagged) → 2D ``[B, max_seq_len]``, each row right-padded with 0."""
    offsets = nested.offsets()
    values = nested.values()
    tail_shape = tuple(values.shape[1:])
    rows = []
    for seq_idx in range(len(original_lengths)):
        row = values[offsets[seq_idx]:offsets[seq_idx + 1]]
        if len(row) < max_seq_len:
            pad_shape = (max_seq_len - len(row),) + tail_shape
            pad = torch.zeros(pad_shape, dtype=row.dtype, device=row.device)
            row = torch.cat([row, pad], dim=0)
        rows.append(row)
    return torch.stack(rows, dim=0)


def nested_offsets_to_cu(nested: Any) -> torch.Tensor:
    """NestedTensor offsets → cu_seqlens ``[B+1]``."""
    return nested.offsets().detach().clone()


def build_attention_mask_2d(original_lengths: list[int], max_seq_len: int) -> torch.Tensor:
    """Build attention_mask 2D ``[batch_size, max_seq_len]``: row ``[0:seq_len-1)`` = True."""
    batch_size = len(original_lengths)
    mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    for seq_idx, seq_len in enumerate(original_lengths):
        if seq_len > 1:
            mask[seq_idx, :seq_len - 1] = True
    return mask


def build_label_mask_2d(
    response_lens: list[int],
    original_lengths: list[int],
    max_seq_len: int,
) -> torch.Tensor:
    """Build label_mask 2D ``[batch_size, max_seq_len]``: ``[prompt-last : seq_len-1)``."""
    batch_size = len(original_lengths)
    mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    for seq_idx in range(batch_size):
        prompt_len = original_lengths[seq_idx] - response_lens[seq_idx]
        start = prompt_len - 1
        end = original_lengths[seq_idx] - 1
        if end > start >= 0:
            mask[seq_idx, start:end] = True
    return mask


def _is_nested_tensor(value: Any) -> bool:
    return hasattr(value, "offsets") and hasattr(value, "values")


# ════════════════════════════════════════════════════════════════
#  Shared dump helpers (tagged 2D outputs, used by both paths)
# ════════════════════════════════════════════════════════════════

def dump_meta_verl080(prefix_lens: list[int], cu_seqlens: torch.Tensor) -> None:
    """Save metadata: prefix_lens + cu_seqlens_q + cu_seqlens_q_logits."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor("prefix_lens.pt", torch.tensor(prefix_lens, dtype=torch.int32), dump_dir)
    _save_tensor("cu_seqlens_q.pt", cu_seqlens, dump_dir)
    _save_tensor("cu_seqlens_q_logits.pt", cu_seqlens, dump_dir)


def dump_logits_verl080(logits: torch.Tensor) -> None:
    """Save packed logits, scope=tp_vocab."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor("logits.pt", logits, dump_dir, scope="tp_vocab")


def dump_logprobs_2d_verl080(logp_2d: torch.Tensor, tag: str,
                             scope: str = "global") -> None:
    """Save 2D log_probs ``[B, L_max]``."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor(f"logprobs_{tag}.pt", logp_2d, dump_dir, scope=scope)


def dump_entropy_2d_verl080(ent_2d: torch.Tensor | None, tag: str,
                            scope: str = "global") -> None:
    """Save 2D entropy ``[B, L_max]``. ``None`` skips."""
    dump_dir = _get_dump_dir()
    if dump_dir is None or ent_2d is None:
        return
    _save_tensor(f"entropy_{tag}.pt", ent_2d, dump_dir, scope=scope)


def dump_attention_mask_verl080(mask_2d: torch.Tensor, tag: str) -> None:
    """Save 2D attention_mask ``[B, L_max]`` (bool)."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor(f"attention_mask_{tag}.pt", mask_2d.to(torch.bool), dump_dir)


def dump_label_mask_verl080(mask_2d: torch.Tensor, tag: str) -> None:
    """Save 2D label_mask ``[B, L_max]`` (bool)."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor(f"label_mask_{tag}.pt", mask_2d.to(torch.bool), dump_dir)


def dump_raw_logits_verl080(raw_output: Any) -> None:
    """Extract and dump logits from HF/verl model raw output."""
    if _get_dump_dir() is None:
        return
    logits = raw_output["logits"] if isinstance(raw_output, dict) else raw_output.logits
    dump_logits_verl080(logits)


# ════════════════════════════════════════════════════════════════
#  Per-layer dump functions (Megatron path — invasive in attention)
# ════════════════════════════════════════════════════════════════

def dump_rope_freqs(q_freqs: torch.Tensor, layer_number: int,
                    num_layers: int) -> None:
    """Accumulate per-token RoPE angles (ON/OFF shared). Auto-flush on last layer."""
    global _ROPE_FREQS_BUFFER
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    if _ROPE_FREQS_BUFFER is None:
        _ROPE_FREQS_BUFFER = {}
    _ROPE_FREQS_BUFFER[layer_number] = q_freqs.detach().cpu().clone()
    if layer_number == _stage_last_layer(num_layers):
        if _should_write_for_scope("pp_stage"):
            try:
                filename = f"rope_freqs{_pp_suffix()}.pt"
                torch.save(_ROPE_FREQS_BUFFER, os.path.join(dump_dir, filename))
                _log.warning("%s saved (%d layers)", filename, len(_ROPE_FREQS_BUFFER))
            except Exception as exc:
                _log.warning("rope_freqs.pt save failed: %s", exc)
        _ROPE_FREQS_BUFFER = None


def dump_rope_postqk_verl080(layer_number: int,
                             rotated_query: torch.Tensor,
                             rotated_key: torch.Tensor,
                             num_layers: int,
                             positions: torch.Tensor | None = None) -> None:
    """Accumulate one layer's post-RoPE Q/K. Auto-flush to rope_postqk.pt."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _add_to_rope_buffer(layer_number, rotated_query, rotated_key, positions)
    if layer_number == _stage_last_layer(num_layers):
        _flush_rope_buffer(dump_dir)


_ROPE_PREQK_BUFFER: dict[int, dict] | None = None


def dump_rope_preqk_verl080(layer_number: int,
                            query: torch.Tensor,
                            key: torch.Tensor,
                            num_layers: int) -> None:
    """Accumulate one layer's pre-RoPE Q/K. Auto-flush to rope_preqk.pt."""
    global _ROPE_PREQK_BUFFER
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    if _ROPE_PREQK_BUFFER is None:
        _ROPE_PREQK_BUFFER = {}
    _ROPE_PREQK_BUFFER[layer_number] = {
        "query": query.detach().cpu().clone(),
        "key": key.detach().cpu().clone(),
    }
    if layer_number == _stage_last_layer(num_layers):
        _flush_dict_buffer("rope_preqk.pt", _ROPE_PREQK_BUFFER, dump_dir)
        _ROPE_PREQK_BUFFER = None


_EXPANDED_KV_BUFFER: dict[int, dict] | None = None


def dump_expanded_kv_on(layer_number: int, expanded_key: torch.Tensor,
                        expanded_value: torch.Tensor, num_layers: int) -> None:
    """ON: accumulate build_kv output K/V. Auto-flush to expanded_kv.pt."""
    global _EXPANDED_KV_BUFFER
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    if _EXPANDED_KV_BUFFER is None:
        _EXPANDED_KV_BUFFER = {}
    _EXPANDED_KV_BUFFER[layer_number] = {
        "key": expanded_key.detach().cpu().clone(),
        "value": expanded_value.detach().cpu().clone(),
    }
    if layer_number == _stage_last_layer(num_layers):
        _flush_dict_buffer("expanded_kv.pt", _EXPANDED_KV_BUFFER, dump_dir)
        _EXPANDED_KV_BUFFER = None


_FULL_KV_BUFFER: dict[int, dict] | None = None


def dump_full_kv_off(layer_number: int, key: torch.Tensor, value: torch.Tensor,
                     num_layers: int) -> None:
    """OFF: accumulate full K/V. Auto-flush to full_kv.pt."""
    global _FULL_KV_BUFFER
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    if _FULL_KV_BUFFER is None:
        _FULL_KV_BUFFER = {}
    _FULL_KV_BUFFER[layer_number] = {
        "key": key.detach().cpu().clone(),
        "value": value.detach().cpu().clone(),
    }
    if layer_number == _stage_last_layer(num_layers):
        _flush_dict_buffer("full_kv.pt", _FULL_KV_BUFFER, dump_dir)
        _FULL_KV_BUFFER = None


_BUILD_KV_INPUT_V_BUFFER: dict[int, torch.Tensor] | None = None


def dump_build_kv_input_v_on(layer_number: int, value: torch.Tensor,
                             num_layers: int) -> None:
    """ON: accumulate raw V before build_kv. Auto-flush to build_kv_input_v.pt."""
    global _BUILD_KV_INPUT_V_BUFFER
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    if _BUILD_KV_INPUT_V_BUFFER is None:
        _BUILD_KV_INPUT_V_BUFFER = {}
    _BUILD_KV_INPUT_V_BUFFER[layer_number] = value.detach().cpu().clone()
    if layer_number == _stage_last_layer(num_layers):
        _flush_dict_buffer("build_kv_input_v.pt", _BUILD_KV_INPUT_V_BUFFER, dump_dir)
        _BUILD_KV_INPUT_V_BUFFER = None


_HIDDEN_STATES_BUFFER: dict[int, torch.Tensor] | None = None


def dump_hidden_states_on(layer_number: int, hidden_states: torch.Tensor,
                          num_layers: int) -> None:
    """Accumulate hidden_states at attention entrance. Auto-flush to hidden_states.pt."""
    global _HIDDEN_STATES_BUFFER
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    if _HIDDEN_STATES_BUFFER is None:
        _HIDDEN_STATES_BUFFER = {}
    _HIDDEN_STATES_BUFFER[layer_number] = hidden_states.detach().cpu().clone()
    if layer_number == _stage_last_layer(num_layers):
        _flush_dict_buffer("hidden_states.pt", _HIDDEN_STATES_BUFFER, dump_dir)
        _HIDDEN_STATES_BUFFER = None


def dump_attn_on(
    output_tensor: torch.Tensor,
    packed_seq_params: Any,
    prefix_sharing_plan: Any,
    layer_number: int,
    num_layers: int,
) -> None:
    """ON-mode attention hook — accumulate per-layer attn output."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _add_to_attn_buffer(layer_number, output_tensor)
    if layer_number == 1:
        _save_meta(packed_seq_params, list(prefix_sharing_plan.prefix_lens),
                   dump_dir, meta_key="attn")
    if layer_number == _stage_last_layer(num_layers):
        _flush_attn_buffer(dump_dir)


def dump_attn_off(
    output_tensor: torch.Tensor,
    packed_seq_params: Any,
    layer_number: int,
    batch_size: int,
    num_layers: int,
) -> None:
    """OFF-mode attention hook — accumulate per-layer attn output."""
    if packed_seq_params is None or not hasattr(packed_seq_params, 'cu_seqlens_q_padded'):
        return
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _add_to_attn_buffer(layer_number, output_tensor)
    if layer_number == 1:
        _save_meta(packed_seq_params, [0] * batch_size, dump_dir, meta_key="attn")
    if layer_number == _stage_last_layer(num_layers):
        _flush_attn_buffer(dump_dir)


def dump_logits(
    logits: torch.Tensor,
    packed_seq_params: Any,
    prefix_lens_list: list[int],
) -> None:
    """Dump model output logits [B, N, vocab//tp]."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor("logits.pt", logits, dump_dir)
    _save_meta(packed_seq_params, prefix_lens_list, dump_dir, meta_key="logits")


def dump_position_ids(position_ids: torch.Tensor) -> None:
    """Dump packed position_ids [N]."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    _save_tensor("position_ids.pt", position_ids, dump_dir)


# ════════════════════════════════════════════════════════════════
#  Megatron-specific: capture_rope_qk context manager
# ════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def capture_rope_qk(attention_module):
    """Hook apply_rotary_pos_emb + get_query_key_value_tensors for per-layer dump."""
    import megatron.core.transformer.attention as _attention_module
    import types as _types_module

    _original_apply_rotary_pos_emb = _attention_module.apply_rotary_pos_emb
    _original_get_query_key_value_tensors = type(attention_module).get_query_key_value_tensors
    qk_captures: list = []
    qkv_captures: list = []

    def _capturing_apply_rotary(tensor, *args, **kwargs):
        output = _original_apply_rotary_pos_emb(tensor, *args, **kwargs)
        qk_captures.append({"post": output})
        return output

    def _capturing_get_qkv(self_, *args, **kwargs):
        output = _original_get_query_key_value_tensors(self_, *args, **kwargs)
        try:
            qkv_captures.append(tuple(output[:3]))
        except Exception:
            pass
        return output

    _attention_module.apply_rotary_pos_emb = _capturing_apply_rotary
    attention_module.get_query_key_value_tensors = _types_module.MethodType(
        _capturing_get_qkv, attention_module)
    try:
        yield qk_captures, qkv_captures
    finally:
        _attention_module.apply_rotary_pos_emb = _original_apply_rotary_pos_emb
        try:
            del attention_module.get_query_key_value_tensors
        except AttributeError:
            pass


# ════════════════════════════════════════════════════════════════
#  FSDP-specific dump functions
# ════════════════════════════════════════════════════════════════

def dump_fsdp_on_metadata_verl080(micro_batch: Any, prefix_sharing_plan: Any, tag: str) -> None:
    """Dump FSDP PrefixSharing ON-path metadata and batch alignment anchors."""
    if _get_dump_dir() is None:
        return

    prefix_lens = list(prefix_sharing_plan.prefix_lens)
    original_lengths = list(prefix_sharing_plan.original_lengths)
    kept_lengths = list(prefix_sharing_plan.kept_lengths_q)

    cu_seqlens = torch.zeros(len(kept_lengths) + 1, dtype=torch.int64)
    for index, length in enumerate(kept_lengths):
        cu_seqlens[index + 1] = cu_seqlens[index] + length
    dump_meta_verl080(prefix_lens, cu_seqlens)

    max_length = max(original_lengths) if original_lengths else 0
    dump_attention_mask_verl080(build_attention_mask_2d(original_lengths, max_length), tag)

    loss_mask = micro_batch.get("loss_mask")
    if loss_mask is not None:
        response_lengths = _response_lengths_from_loss_mask(loss_mask, len(original_lengths))
        dump_label_mask_verl080(
            build_label_mask_2d(response_lengths, original_lengths, max_length),
            tag,
        )
    dump_input_ids_2d_verl080(micro_batch, original_lengths, max_length, tag)


def dump_fsdp_model_output_2d_verl080(
    model_output: dict[str, Any],
    original_lengths: list[int],
    tag: str,
) -> None:
    """Dump restored FSDP log_probs/entropy to unified 2D coordinate system."""
    if _get_dump_dir() is None:
        return
    max_length = max(original_lengths) if original_lengths else 0
    log_probs = model_output.get("log_probs")
    if log_probs is None:
        return
    log_probs_2d = _maybe_nested_to_2d(log_probs, original_lengths, max_length)
    if log_probs_2d is None or log_probs_2d.dim() != 2:
        return
    dump_logprobs_2d_verl080(log_probs_2d, tag)

    entropy = model_output.get("entropy")
    if entropy is not None:
        entropy_2d = _maybe_nested_to_2d(entropy, original_lengths, max_length)
        if entropy_2d is not None and entropy_2d.dim() == 2:
            dump_entropy_2d_verl080(entropy_2d, tag)


def dump_fsdp_baseline_verl080(micro_batch: Any, result: Any, tag: str) -> None:
    """Dump FSDP OFF baseline diagnostics (no prefix-sharing)."""
    if _get_dump_dir() is None:
        return
    if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], dict):
        output_dict = result[1]
    else:
        return
    model_output = output_dict.get("model_output", {})
    if not model_output:
        return

    original_lengths = _original_lengths_from_input_ids(micro_batch.get("input_ids"))
    if original_lengths is None:
        return

    prefix_lens = [0] * len(original_lengths)
    cu_seqlens = torch.zeros(len(original_lengths) + 1, dtype=torch.int64)
    for index, length in enumerate(original_lengths):
        cu_seqlens[index + 1] = cu_seqlens[index] + length
    dump_meta_verl080(prefix_lens, cu_seqlens)

    max_length = max(original_lengths) if original_lengths else 0
    dump_attention_mask_verl080(build_attention_mask_2d(original_lengths, max_length), tag)

    loss_mask = micro_batch.get("loss_mask")
    if loss_mask is not None:
        response_lengths = _response_lengths_from_loss_mask(loss_mask, len(original_lengths))
        dump_label_mask_verl080(
            build_label_mask_2d(response_lengths, original_lengths, max_length),
            tag,
        )
    dump_input_ids_2d_verl080(micro_batch, original_lengths, max_length, tag)
    dump_fsdp_model_output_2d_verl080(model_output, original_lengths, tag)


def dump_input_ids_2d_verl080(
    micro_batch: Any,
    original_lengths: list[int],
    max_length: int,
    tag: str,
) -> None:
    """Dump raw input_ids to 2D [B, L_max] as ON/OFF batch alignment anchor."""
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    input_ids = micro_batch.get("input_ids")
    if input_ids is None:
        return
    if _is_nested_tensor(input_ids):
        ids_2d = nested_to_2d_full(input_ids, original_lengths, max_length)
    elif hasattr(input_ids, "dim") and input_ids.dim() == 2:
        ids_2d = input_ids
    else:
        return
    _save_tensor(f"input_ids_{tag}.pt", ids_2d.long().cpu(), dump_dir)


# ── FSDP internal helpers ───────────────────────────────────────

def _maybe_nested_to_2d(value: Any, original_lengths: list[int], max_length: int) -> Any | None:
    if _is_nested_tensor(value):
        return nested_to_2d_full(value, original_lengths, max_length)
    if hasattr(value, "dim"):
        return value
    return None


def _original_lengths_from_input_ids(input_ids: Any) -> list[int] | None:
    if _is_nested_tensor(input_ids):
        return [int(length) for length in input_ids.offsets().diff().tolist()]
    if input_ids is not None and hasattr(input_ids, "dim") and input_ids.dim() == 2:
        return [int(input_ids.shape[1])] * int(input_ids.shape[0])
    return None


def _response_lengths_from_loss_mask(loss_mask: Any, batch_size: int) -> list[int]:
    if _is_nested_tensor(loss_mask):
        offsets = loss_mask.offsets()
        values = loss_mask.values()
        return [int(values[offsets[i]:offsets[i + 1]].sum()) for i in range(batch_size)]
    return loss_mask.sum(dim=-1).long().cpu().tolist()


# ════════════════════════════════════════════════════════════════
#  Backward-compatible aliases for Megatron path callers
# ════════════════════════════════════════════════════════════════

def dump_rope_freqs_on(q_freqs: torch.Tensor, layer_number: int,
                       num_layers: int) -> None:
    """Backward-compatible alias for Megatron ON path → ``dump_rope_freqs``."""
    dump_rope_freqs(q_freqs, layer_number, num_layers)


def dump_rope_freqs_off(q_pos_emb: torch.Tensor, layer_number: int,
                        num_layers: int) -> None:
    """Backward-compatible alias for Megatron OFF path → ``dump_rope_freqs``.

    Stores the raw RoPE angle table. The cmp tool expects per-token frequencies
    in ``rope_freqs.pt``; this wrapper passes through as-is for compatibility.
    """
    dump_rope_freqs(q_pos_emb, layer_number, num_layers)
