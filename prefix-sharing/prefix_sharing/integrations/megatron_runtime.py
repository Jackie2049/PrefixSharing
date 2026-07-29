"""Runtime hook used by the minimal Megatron attention patch."""

from __future__ import annotations

from typing import Any

import torch

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.integrations.context import current_prefix_sharing_context
from prefix_sharing.utils import ensure_global_packed_token_lengths


# ---------------------------------------------------------------------------
# Per-micro-batch timing accumulator (controlled by PS_TIMING env var)
# ---------------------------------------------------------------------------
_ps_timing_enabled = None
_ps_timing_torch = None
_ps_timing_events: dict[str, list[tuple[int, Any, Any]]] = {}  # category -> [(layer_id, start_ev, end_ev), ...]
_ps_timing_mb_events: dict[str, tuple[Any, Any]] = {}  # category -> (start_ev, end_ev) for per-microbatch timing


def _ps_timing_is_enabled() -> bool:
    global _ps_timing_enabled
    if _ps_timing_enabled is None:
        _ps_timing_enabled = __import__("os").environ.get("PS_TIMING", "0") == "1"
    return _ps_timing_enabled


def _ps_timing_get_torch() -> Any:
    global _ps_timing_torch
    if _ps_timing_torch is None:
        _ps_timing_torch = __import__("torch")
    return _ps_timing_torch


def _ps_timing_record(category: str, layer_id: int) -> Any:
    """Create+record start event; return (start_ev, end_ev). Caller must record end_ev."""
    t = _ps_timing_get_torch()
    start_ev = t.npu.Event(enable_timing=True)
    end_ev = t.npu.Event(enable_timing=True)
    start_ev.record()
    _ps_timing_events.setdefault(category, []).append((layer_id, start_ev, end_ev))
    return end_ev


def _ps_timing_end(end_ev: Any) -> None:
    end_ev.record()


def _ps_timing_record_mb(category: str) -> Any:
    """Record per-microbatch-level timing event. Returns end_ev for caller to record."""
    t = _ps_timing_get_torch()
    start_ev = t.npu.Event(enable_timing=True)
    end_ev = t.npu.Event(enable_timing=True)
    start_ev.record()
    _ps_timing_mb_events[category] = (start_ev, end_ev)
    return end_ev


def _ps_timing_end_mb(end_ev: Any) -> None:
    end_ev.record()


def _ps_timing_reset() -> None:
    """Reset all timing accumulators for the next micro-batch."""
    global _ps_timing_events, _ps_timing_mb_events
    _ps_timing_events = {}
    _ps_timing_mb_events = {}


def ps_print_timing_summary(forward_id: int, global_rank: Any, extra: dict[str, float] | None = None) -> None:
    """Print accumulated per-layer + per-microbatch timing summary.

    Call this after ``torch.npu.synchronize()`` to ensure all events have landed.
    """
    global _ps_timing_events, _ps_timing_mb_events
    if not _ps_timing_is_enabled():
        return

    has_layer = bool(_ps_timing_events)
    has_mb = bool(_ps_timing_mb_events)

    if not has_layer and not has_mb:
        return

    # --- Per-layer breakdown ---
    if has_layer:
        per_layer: dict[int, dict[str, float]] = {}
        for category, ev_list in _ps_timing_events.items():
            for layer_id, start_ev, end_ev in ev_list:
                elapsed = start_ev.elapsed_time(end_ev)
                per_layer.setdefault(layer_id, {})[category] = elapsed

        for layer_id in sorted(per_layer):
            parts = per_layer[layer_id]
            line = (
                "[PS-TIMING-BREAKDOWN] forward={} rank={} layer={} ".format(forward_id, global_rank, layer_id)
                + " ".join("{}={:.3f}ms".format(k, v) for k, v in sorted(parts.items()))
            )
            print(line)

    # --- Per-category totals ---
    totals: dict[str, float] = {}

    # Per-layer categories
    for category, ev_list in _ps_timing_events.items():
        totals[category + "_sum"] = sum(start_ev.elapsed_time(end_ev) for _, start_ev, end_ev in ev_list)

    # Per-microbatch categories
    for category, (start_ev, end_ev) in _ps_timing_mb_events.items():
        totals[category] = start_ev.elapsed_time(end_ev)

    # Compute forward_total from per-layer sums (all layer-scoped categories)
    layer_categories = {"qkv", "rope", "build_kv", "mask", "fa", "output_proj",
                        "b_total"}
    forward_total = sum(v for k, v in totals.items() if k.replace("_sum", "") in layer_categories)

    # Compute restore_total from restore sub-phases
    restore_categories = {"restore_unfold", "restore_bulk", "restore_recompute", "restore_pack"}
    restore_total = sum(v for k, v in totals.items() if k in restore_categories)

    # Compute mb_total = pre_forward + forward + logprobs + restore
    pre_forward_categories = {"detect", "plan", "trim", "layout"}
    pre_forward_total = sum(v for k, v in totals.items() if k in pre_forward_categories)
    logprobs_total = totals.get("save_logits", 0.0)
    mb_total = pre_forward_total + forward_total + logprobs_total + restore_total

    # Determine mode (totals keys have _sum suffix for per-layer categories)
    _ps_categories = {"rope_sum", "build_kv_sum", "mask_sum", "fa_sum", "output_proj_sum"}
    has_ps = any(k in _ps_categories or k.startswith("ps_") for k in totals)
    mode = "ps" if has_ps else "baseline"

    total_line = (
        "[PS-TIMING-TOTALS] forward={} rank={} mode={} ".format(forward_id, global_rank, mode)
        + " ".join("{}={:.3f}ms".format(k, v) for k, v in sorted(totals.items()))
    )
    total_line += " forward_total={:.3f}ms restore_total={:.3f}ms mb_total={:.3f}ms".format(
        forward_total, restore_total, mb_total)
    if extra:
        total_line += " " + " ".join("{}={:.3f}ms".format(k, v) for k, v in sorted(extra.items()))
    print(total_line)

    # Reset for next micro-batch
    _ps_timing_events = {}
    _ps_timing_mb_events = {}


def prefix_attention(
    attention_module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    rotary_pos_emb: Any,
    packed_seq_params: Any,
) -> tuple[Any, Any] | None:
    """Run prefix-sharing attention when a runtime context is active.

    Returns ``None`` for the normal Megatron path. When active, this function
    owns RoPE, KV expansion, causal masking, and output projection.
    """
    # 读取并校验前缀共享上下文 prefix_sharing_context
    prefix_sharing_context = current_prefix_sharing_context()
    if prefix_sharing_context is None:
        return None
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        raise RuntimeError("prefix sharing phase 1 requires packed_seq_params.qkv_format='thd'")
    if rotary_pos_emb is None:
        raise RuntimeError("prefix sharing phase 1 requires rotary_pos_emb")
    if prefix_sharing_context.packed_batch_layout.packed_position_ids is None:
        raise RuntimeError("prefix sharing context is missing packed_position_ids")

    # 确保 QKV 符合 THD packing格式
    packed_batch_layout = prefix_sharing_context.packed_batch_layout
    ensure_global_packed_token_lengths(
        {
            "query_length": query.shape[0],
            "key_length": key.shape[0],
            "value_length": value.shape[0],
        },
        total_padded_length=packed_batch_layout.total_padded_length,
        context="attention hook",
    )

    # QK位置编码
    #   mcore v0.16.1 的 RoPE 需要 cu_seqlens, mscale, cp_group 等入参
    #       returns cu_seqlens for verl 0.8.0 (mcore 0.16.1)
    #       returns None/defaults for verl 0.7.0 (mcore 0.12.1 ~ 0.15.x)
    cu_seqlens_q = _extract_cu_seqlens(packed_seq_params, "cu_seqlens_q_padded", "cu_seqlens_q")
    cu_seqlens_kv = _extract_cu_seqlens(packed_seq_params, "cu_seqlens_kv_padded", "cu_seqlens_kv")
    mscale = _get_yarn_mscale(attention_module)
    cp_group = _get_cp_group(attention_module)
    q_pos_emb, k_pos_emb = _unpack_rotary_pos_emb(rotary_pos_emb)

    # [PS-TIMING] RoPE
    _ps_rope_end_ev = None
    if _ps_timing_is_enabled():
        _ps_rope_end_ev = _ps_timing_record("rope", int(getattr(attention_module, "layer_number", 0) or 0))

    query, key = _apply_positioned_rope(
        attention_module,
        query,
        key,
        q_pos_emb,
        k_pos_emb,
        packed_batch_layout.packed_position_ids,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        mscale=mscale,
        cp_group=cp_group,
    )

    if _ps_rope_end_ev is not None:
        _ps_timing_end(_ps_rope_end_ev)

    parallel_info = prefix_sharing_context.parallel_info
    layer_id = int(getattr(attention_module, "layer_number", 0) or 0)
    seq_parallel = getattr(getattr(attention_module, "config", None), "sequence_parallel", None)
    print(
        f"[PS][attention][global_rank={parallel_info.global_rank} tp_rank={parallel_info.tp_rank}/"
        f"tp_size={parallel_info.tp_size}(sequence_parallel={seq_parallel}) pp_rank={parallel_info.pp_rank}/pp_size={parallel_info.pp_size} layer={layer_id}] "
        f"enter prefix-sharing path: query_token_length={query.shape[0]} "
        f"total_padded_length={packed_batch_layout.total_padded_length} query_shape={tuple(query.shape)}, "
        f"key_shape={tuple(key.shape)}, value_shape={tuple(value.shape)}, valid_lengths={packed_batch_layout.valid_lengths}, "
        f"padded_lengths={packed_batch_layout.padded_lengths}, cu_seqlens={packed_batch_layout.cu_seqlens}"
    )

    # [PS-TIMING] build_kv
    _ps_buildkv_end_ev = None
    if _ps_timing_is_enabled():
        _ps_buildkv_end_ev = _ps_timing_record("build_kv", layer_id)

    # 前缀共享：provider 存储激活值，reuser 拼接激活值
    attention_backend = prefix_sharing_context.attention_backend or TorchReferenceBackend()
    expanded_key, expanded_value = attention_backend.build_kv(
        key,
        value,
        prefix_sharing_context.store,
        prefix_sharing_context.prefix_sharing_plan,
        packed_batch_layout=packed_batch_layout,
        layer_id=layer_id,
        tp_rank=parallel_info.tp_rank,
        stats=prefix_sharing_context.stats,
    )

    if _ps_buildkv_end_ev is not None:
        _ps_timing_end(_ps_buildkv_end_ev)

    print(
        f"[PS][attention][global_rank={parallel_info.global_rank} tp_rank={parallel_info.tp_rank}/"
        f"tp_size={parallel_info.tp_size}(sequence_parallel={seq_parallel}) pp_rank={parallel_info.pp_rank}/pp_size={parallel_info.pp_size} layer={layer_id}] "
        f"built expanded kv: expanded_key_shape={tuple(expanded_key.shape)}, expanded_value_shape={tuple(expanded_value.shape)}"
    )

    # 注意力计算 (attention timing is logged inside flash_atten_npu.py as PS-TIMING mask/fa)
    core_attn_out = attention_backend.attention(
        query,
        expanded_key,
        expanded_value,
        prefix_sharing_context.prefix_sharing_plan,
        packed_batch_layout=packed_batch_layout,
        attention_mask=attention_mask,
        layer_id=layer_id,
    )
    core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

    # [PS-TIMING] output_proj
    _ps_proj_end_ev = None
    if _ps_timing_is_enabled():
        _ps_proj_end_ev = _ps_timing_record("output_proj", layer_id)

    output = attention_module.linear_proj(core_attn_out)  # (tensor, bias) tuple

    if _ps_proj_end_ev is not None:
        _ps_timing_end(_ps_proj_end_ev)

    ######### prefix-sharing diag: ON attention_output (per-layer) #########
    try:
        from prefix_sharing.tools.diagnostic_dump import dump_attn_on
        dump_attn_on(output[0], packed_seq_params, prefix_sharing_context.prefix_sharing_plan,
                     attention_module.layer_number,
                     attention_module.config.num_layers)
    except Exception as e:
        print(f"last-attn dump (ON) failed: {e}")
    ######### prefix-sharing diag: ON attention_output (per-layer) #########
    # ---

    return output


def _apply_positioned_rope(
    attention_module: Any,
    query: Any,
    key: Any,
    q_pos_emb: Any,
    k_pos_emb: Any,
    packed_position_ids: Any,
    *,
    cu_seqlens_q: Any | None = None,
    cu_seqlens_kv: Any | None = None,
    mscale: float | None = None,
    cp_group: Any | None = None,
) -> tuple[Any, Any]:
    """Apply RoPE using packed_position_ids, with optional v0.16.1 API params.

    v070 (mcore <= 0.15.x): cu_seqlens=None, no mscale/cp_group.
    v0.16.1+ (mcore 0.16.1): cu_seqlens from packed_seq_params, mscale for
    yarn models, cp_group for context parallel.

    Backward compatible: mscale and cp_group are only passed to
    apply_rotary_pos_emb when they differ from defaults, so v0.15.x
    (which doesn't have these kwargs) won't get a TypeError.
    """
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

    positions = packed_position_ids.to(device=query.device, dtype=torch.long)
    max_needed = positions.max().item() + 1

    if q_pos_emb is not None and max_needed > q_pos_emb.shape[0]:
        dim_half = q_pos_emb.shape[-1] // 2
        step = q_pos_emb[1:2, :, :, :dim_half] - q_pos_emb[0:1, :, :, :dim_half]
        extra_positions = torch.arange(
            q_pos_emb.shape[0], max_needed,
            device=q_pos_emb.device, dtype=q_pos_emb.dtype,
        )
        extra_angles = extra_positions[:, None, None, None] * step
        extra_emb = torch.cat([extra_angles, extra_angles], dim=-1)
        q_pos_emb = torch.cat([q_pos_emb, extra_emb], dim=0)
    if k_pos_emb is not None and max_needed > k_pos_emb.shape[0]:
        dim_half = k_pos_emb.shape[-1] // 2
        step = k_pos_emb[1:2, :, :, :dim_half] - k_pos_emb[0:1, :, :, :dim_half]
        extra_positions = torch.arange(
            k_pos_emb.shape[0], max_needed,
            device=k_pos_emb.device, dtype=k_pos_emb.dtype,
        )
        extra_angles = extra_positions[:, None, None, None] * step
        extra_emb = torch.cat([extra_angles, extra_angles], dim=-1)
        k_pos_emb = torch.cat([k_pos_emb, extra_emb], dim=0)

    def _rope_kwargs(_unused_cu_seqlens: Any | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"config": attention_module.config, "cu_seqlens": None}
        if mscale is not None and mscale != 1.0:
            kwargs["mscale"] = mscale
        return kwargs

    if q_pos_emb is not None:
        q_freqs = q_pos_emb.index_select(0, positions)
        ######### prefix-sharing diag: ON rope_freqs (per-layer) #########
        try:
            from prefix_sharing.tools.diagnostic_dump import dump_rope_freqs_on
            dump_rope_freqs_on(q_freqs, attention_module.layer_number,
                               attention_module.config.num_layers)
        except Exception as e:
            print(f"rope_freqs_on dump failed: {e}")
        ######### prefix-sharing diag: ON rope_freqs (per-layer) #########
        query = apply_rotary_pos_emb(
            query.unsqueeze(1),
            q_freqs,
            **_rope_kwargs(cu_seqlens_q),
        ).squeeze(1)
    if k_pos_emb is not None:
        k_freqs = k_pos_emb.index_select(0, positions)
        key = apply_rotary_pos_emb(
            key.unsqueeze(1),
            k_freqs,
            **_rope_kwargs(cu_seqlens_kv),
        ).squeeze(1)
    return query, key


# ═══════════════════════════════════════
# v0.16.1 API helpers (backward compatible with v070)
# ═══════════════════════════════════════


def _unpack_rotary_pos_emb(rotary_pos_emb: Any) -> tuple[Any, Any]:
    if isinstance(rotary_pos_emb, (tuple, list)) and len(rotary_pos_emb) == 2:
        return rotary_pos_emb[0], rotary_pos_emb[1]
    return rotary_pos_emb, rotary_pos_emb


def _extract_cu_seqlens(packed_seq_params: Any, primary_attr: str, fallback_attr: str) -> Any | None:
    if packed_seq_params is None:
        return None
    val = getattr(packed_seq_params, primary_attr, None)
    if val is None:
        val = getattr(packed_seq_params, fallback_attr, None)
    return val


def _get_yarn_mscale(attention_module: Any) -> float:
    try:
        from megatron.core.transformer.attention import _yarn_get_concentration_factor_from_config
        return float(_yarn_get_concentration_factor_from_config(attention_module.config))
    except (ImportError, AttributeError):
        return 1.0


def _get_cp_group(attention_module: Any) -> Any | None:
    pg_collection = getattr(attention_module, "pg_collection", None)
    if pg_collection is None:
        return None
    return getattr(pg_collection, "cp", None)
