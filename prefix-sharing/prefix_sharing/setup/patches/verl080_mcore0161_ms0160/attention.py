"""Patch: Attention.forward — thin wrapper

无 context → 调用原始 forward
有 context → QKV + THD squeeze → delegate to integrations.prefix_attention

业务逻辑（RoPE、KV expansion、attention 计算）全部由 integrations 层处理，
本 patch 只负责 QKV 提取（attention module 交互）和 THD squeeze（格式适配）。
"""

from __future__ import annotations

from typing import Any


def patch_megatron_attention(original_forward: Any) -> Any:
    """创建 Attention.forward 的 patch wrapper。"""

    def patched_forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        from prefix_sharing.integrations.context import current_prefix_sharing_context

        ctx = current_prefix_sharing_context()
        if ctx is None:
            # ── normal path: 调用原始 forward ──
            _result = original_forward(
                self,
                hidden_states,
                attention_mask,
                key_value_states=key_value_states,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                inference_params=inference_params,
            )
            # ##### [PS-diag] OFF attn_outputs + rope_freqs_off dump #####
            # OFF 走原始 forward，不经 prefix_attention/_apply_positioned_rope，
            # 所以 ON 路径里的 dump_attn_on/dump_rope_freqs_on 不会触发。
            # 这里在 OFF 分支补 dump，让 cmp_diag 的 attn/RoPE 对比有 OFF ground truth。
            # v070 是直接改 megatron attention 源码在 forward 内部 dump；v080 用 patch
            # wrapper 在 forward 返回后 dump output + 入参 rotary_pos_emb 解包出 angle table，
            # 语义等价（唯一拿不到的是 rope_emb rotated q/k，在 forward 内部，但 rope_freqs
            # angle table 已够验证 RoPE）。
            import os as _os
            if _os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None:
                from prefix_sharing.tools.diagnostic_dump import (
                    dump_attn_off, dump_rope_freqs_off,
                )
                from prefix_sharing.integrations.megatron_runtime import _unpack_rotary_pos_emb
                _attn_out = _result[0] if isinstance(_result, tuple) else _result
                _bs = (
                    len(packed_seq_params.cu_seqlens_q_padded) - 1
                    if (packed_seq_params is not None
                        and hasattr(packed_seq_params, "cu_seqlens_q_padded"))
                    else 0
                )
                dump_attn_off(_attn_out, packed_seq_params,
                              self.layer_number, _bs, self.config.num_layers)
                if rotary_pos_emb is not None:
                    _q_pos_emb, _ = _unpack_rotary_pos_emb(rotary_pos_emb)
                    dump_rope_freqs_off(_q_pos_emb, self.layer_number, self.config.num_layers)
            # ##### [PS-diag] OFF attn_outputs + rope_freqs_off dump end #####
            return _result

        # ── prefix-sharing path ──
        # phase 1: training, THD, no fusion, no output gate

        # QKV extraction — attention module interaction, not business logic
        query, key, value = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            split_qkv=True,
            output_gate=False,
        )
        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # delegate to verified integrations code
        from prefix_sharing.integrations.megatron_runtime import (
            prefix_attention,
        )

        result = prefix_attention(
            self,
            query,
            key,
            value,
            attention_mask,
            rotary_pos_emb,
            packed_seq_params,
        )
        if result is not None:
            return result

        # fallback: should not reach here if context is active,
        # but return original forward as safety net
        return original_forward(
            self,
            hidden_states,
            attention_mask,
            key_value_states=key_value_states,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            inference_params=inference_params,
        )

    return patched_forward