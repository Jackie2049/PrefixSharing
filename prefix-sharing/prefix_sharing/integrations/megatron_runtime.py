"""Runtime hook used by the minimal Megatron attention patch."""

from __future__ import annotations

from typing import Any

import torch

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.integrations.context import current_prefix_sharing_context
from prefix_sharing.utils import ensure_global_packed_token_lengths



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
    print("\n\n\nsuccess come into def prefix_attention\n\n\n")

    # 读取并校验前缀共享上下文 prefix_sharing_context
    prefix_sharing_context = current_prefix_sharing_context()
    if prefix_sharing_context is None:
        print("\n\n\nprefix_sharing_context is None\n\n\n")
        return None
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        raise RuntimeError("prefix sharing phase 1 requires packed_seq_params.qkv_format='thd'")
    if rotary_pos_emb is None:
        raise RuntimeError("prefix sharing phase 1 requires rotary_pos_emb")
    if prefix_sharing_context.packed_batch_layout.packed_position_ids is None:
        raise RuntimeError("prefix sharing context is missing packed_position_ids")

    # 确保 QKV 符合 THD packing格式
    packed_batch_layout = prefix_sharing_context.packed_batch_layout
    cp_view = packed_batch_layout.context_parallel
    if cp_view is None:
        ensure_global_packed_token_lengths(
            {
                "query_length": query.shape[0],
                "key_length": key.shape[0],
                "value_length": value.shape[0],
            },
            total_padded_length=packed_batch_layout.total_padded_length,
            context="attention hook",
        )
        rope_position_ids = packed_batch_layout.packed_position_ids
        expected_token_length = packed_batch_layout.total_padded_length
    else:
        expected_token_length = cp_view.local_total_padded_length
        actual_lengths = {
            "query_length": query.shape[0],
            "key_length": key.shape[0],
            "value_length": value.shape[0],
        }
        mismatched = {
            name: value for name, value in actual_lengths.items()
            if int(value) != int(expected_token_length)
        }
        if mismatched:
            raise RuntimeError(
                "[prefix-sharing] CP attention hook requires CP-local THD token length. "
                f"expected_local_total_padded_length={expected_token_length}, "
                f"actual={actual_lengths}, cp_rank={cp_view.cp_rank}, cp_size={cp_view.cp_size}. "
                "This path does not support SP/CP-local variants with different hook tensor layout."
            )
        rope_position_ids = cp_view.local_position_ids

    # QK位置编码
    #   mcore v0.16.1 的 RoPE 需要 cu_seqlens, mscale, cp_group 等入参
    #       returns cu_seqlens for verl 0.8.0 (mcore 0.16.1)
    #       returns None/defaults for verl 0.7.0 (mcore 0.12.1 ~ 0.15.x) 
    cu_seqlens_q = _extract_cu_seqlens(packed_seq_params, "cu_seqlens_q_padded", "cu_seqlens_q")
    cu_seqlens_kv = _extract_cu_seqlens(packed_seq_params, "cu_seqlens_kv_padded", "cu_seqlens_kv")
    mscale = _get_yarn_mscale(attention_module)
    cp_group = _get_cp_group(attention_module)
    q_pos_emb, k_pos_emb = _unpack_rotary_pos_emb(rotary_pos_emb)
    
    query, key = _apply_positioned_rope(
        attention_module,
        query,
        key,
        q_pos_emb,
        k_pos_emb,
        rope_position_ids,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        mscale=mscale,
        cp_group=cp_group,
    )

    parallel_info = prefix_sharing_context.parallel_info
    layer_id = int(getattr(attention_module, "layer_number", 0) or 0)
    print("\n\n\ntry to build kv\n\n\n")
    seq_parallel = getattr(getattr(attention_module, "config", None), "sequence_parallel", None)
    print(
        f"[PS][attention][global_rank={parallel_info.global_rank} tp_rank={parallel_info.tp_rank}/"
        f"tp_size={parallel_info.tp_size}(sequence_parallel={seq_parallel}) pp_rank={parallel_info.pp_rank}/pp_size={parallel_info.pp_size} layer={layer_id}] "
        f"enter prefix-sharing path: query_token_length={query.shape[0]} "
        f"total_padded_length={packed_batch_layout.total_padded_length} expected_token_length={expected_token_length} "
        f"cp_rank={parallel_info.cp_rank}/cp_size={parallel_info.cp_size} query_shape={tuple(query.shape)}, "
        f"key_shape={tuple(key.shape)}, value_shape={tuple(value.shape)}, valid_lengths={packed_batch_layout.valid_lengths}, "
        f"padded_lengths={packed_batch_layout.padded_lengths}, cu_seqlens={packed_batch_layout.cu_seqlens}"
    )

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
        cp_rank=parallel_info.cp_rank,
        stats=prefix_sharing_context.stats,
    )
    print(
        f"[PS][attention][global_rank={parallel_info.global_rank} tp_rank={parallel_info.tp_rank}/"
        f"tp_size={parallel_info.tp_size}(sequence_parallel={seq_parallel}) pp_rank={parallel_info.pp_rank}/pp_size={parallel_info.pp_size} layer={layer_id}] "
        f"built expanded kv: expanded_key_shape={tuple(expanded_key.shape)}, expanded_value_shape={tuple(expanded_value.shape)}"
    )

    # 注意力计算
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
    output = attention_module.linear_proj(core_attn_out)  # (tensor, bias) tuple

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

    # 当 packed_position_ids 所需要的最大 position id 超过了 q_pos_emb / k_pos_emb 的当前长度时，
    # 就需要对 q_pos_emb / k_pos_emb 进行扩展。
    # THD 模式下生成的 pos_emb 仅覆盖 positions 0 .. max_seqlen_q-1 这段范围，
    # 这个长度往往不够用，因为 prefix-sharing 会保留原始的 position_ids
    #（例如后缀可能从 position 75 开始）。
    #
    # RoPE 具有线性性质：freqs[p] = p * inv_freq。
    # 因此可以通过 pos_emb[1] - pos_emb[0] 恢复出 step（即 inv_freq），
    # 从而生成缺失的高位置频率向量。
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

    # Build kwargs for apply_rotary_pos_emb.
    # Only include version-specific params when they're provided,
    # to maintain backward compat with v070 (mcore <= 0.15.x).
    def _rope_kwargs(_unused_cu_seqlens: Any | None) -> dict[str, Any]:
        """Build kwargs for apply_rotary_pos_emb.

        NOTE: cu_seqlens is ALWAYS set to None here because we pre-selected the
        correct frequencies via index_select(0, packed_position_ids) above.
        Passing real cu_seqlens would trigger the THD RoPE path, which re-splits
        sequences and calls torch.cat — unnecessary double work and breaks on
        NPU (aclnnCat failure).
        """
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
    """解包 rotary_pos_emb，兼容 mcore 版本差异。

    mcore 0.12.1 ~ 0.15.x: (q_pos_emb, k_pos_emb) tuple
    mcore 0.16.1+:             单 tensor（Q/K 共用）
    """
    if isinstance(rotary_pos_emb, (tuple, list)) and len(rotary_pos_emb) == 2:
        return rotary_pos_emb[0], rotary_pos_emb[1]
    return rotary_pos_emb, rotary_pos_emb


def _extract_cu_seqlens(packed_seq_params: Any, primary_attr: str, fallback_attr: str) -> Any | None:
    """Extract cu_seqlens from packed_seq_params, preferring padded version.

    Returns None for v070 (mcore <= 0.15.x) where these attributes don't exist.
    """
    if packed_seq_params is None:
        return None
    val = getattr(packed_seq_params, primary_attr, None)
    if val is None:
        val = getattr(packed_seq_params, fallback_attr, None)
    return val


def _get_yarn_mscale(attention_module: Any) -> float:
    """Get yarn mscale from attention module config (v0.16.1+).

    Returns 1.0 for v070 (mcore <= 0.15.x) where this function doesn't exist.
    """
    try:
        from megatron.core.transformer.attention import _yarn_get_concentration_factor_from_config
        return float(_yarn_get_concentration_factor_from_config(attention_module.config))
    except (ImportError, AttributeError):
        return 1.0


def _get_cp_group(attention_module: Any) -> Any | None:
    """Get context parallel group from attention module (v0.16.1+).

    Returns None for v070 (mcore <= 0.15.x) where pg_collection doesn't exist.
    """
    pg_collection = getattr(attention_module, "pg_collection", None)
    if pg_collection is None:
        return None
    return getattr(pg_collection, "cp", None)
