"""Patch: Attention.forward — thin wrapper

No context → call original forward
Has context → QKV + THD squeeze → delegate to integrations.prefix_attention

Business logic (RoPE, KV expansion, attention computation) is entirely
handled by the integrations layer. This patch only handles QKV extraction
(attention module interaction) and THD squeeze (format adaptation).
"""

from __future__ import annotations

from prefix_sharing.diagnostics import diagnostic_dump_enabled

from typing import Any


def patch_megatron_attention(original_forward: Any) -> Any:
    """Create a patch wrapper for Attention.forward."""

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
            # ── normal path: call original forward ──
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
            # OFF path calls original forward and does not go through
            # prefix_attention/_apply_positioned_rope, so dump_attn_on /
            # dump_rope_freqs_on in the ON path will not fire.
            # Dump here in the OFF branch so that cmp_diag has OFF ground truth
            # for attn / RoPE comparison.
            # v070 modified megatron attention source to dump inside forward;
            # v080 uses a patch wrapper to dump output + unpack rotary_pos_emb
            # angle table after forward returns — semantically equivalent (the
            # only thing unavailable is the rope_emb rotated q/k inside forward,
            # but the rope_freqs angle table is sufficient for RoPE verification).
            if diagnostic_dump_enabled() is not None:
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