"""Patch: vocab_parallel_log_probs_from_logits — thin wrapper.

No context → call original function directly
Has context → call original + save provider prefix-last vocab-dimension logits
              (including autograd graph)

Business logic (restore reassembly) is done by
``restore_via_2d_unfold_verl080`` at the forward_step exit. This patch only
saves the provider logits needed for prefix-last recomputation into
``ctx.prefix_last_logits_saved`` while the packed 1D logits are still intact
and the context is active, for later restore use.

Save condition: each entry in ``ctx.prefix_last_restore_indices`` corresponds
to one reuser's prefix-last (one per reuser). Save the vocab-dimension logits
at each provider's direct position for later restore-side prefix-last logprob
recomputation (interior segments are handled by restore-side build_kv-style
bulk slicing and do not read logits).
"""

from __future__ import annotations

from typing import Any


def patch_megatron_vocab(original_fn: Any) -> Any:
    """Create a patch wrapper for vocab_parallel_log_probs_from_logits."""

    def patched_fn(logits, labels):
        # ##### [PS-diag] dump logits (both ON/OFF; must happen before original_fn) #####
        # logits shape: [N, V//tp] (or [N,1,V//tp]); cmp_diag.cmp_logits_packed
        # will reshape to token-major [N,V] and align using cu_seqlens + prefix_lens.
        import os as _os
        _diag_on = _os.environ.get("PREFIX_SHARING_DIAG_DUMP") is not None
        if _diag_on:
            from prefix_sharing.tools.diagnostic_dump import dump_logits_verl080
            dump_logits_verl080(logits)
        # ##### [PS-diag] dump logits end #####

        from prefix_sharing.integrations.context import current_prefix_sharing_context

        ctx = current_prefix_sharing_context()

        # ── Save provider prefix-last logits (must happen before original_fn!) ──
        # original_fn = -vocab_parallel_cross_entropy; its forward modifies
        # logits in-place:
        #   (1) logits -= logits_max        (megatron cross_entropy.py:45)
        #   (2) torch.exp(logits, out=...)  (cross_entropy.py:64-65)
        # After the call, logits have been overwritten with exp(L-max) garbage.
        # Cloning after original_fn would save garbage values, and restore-side
        # recomputation logp(exp(L-max), label) ≠ logp(L, label) would be
        # completely wrong. Must clone the original logits before original_fn
        # (same applies to dump).
        if ctx is not None and ctx.prefix_last_restore_indices:
            # logits may be [N, V//tp] or [N, 1, V//tp]; unify to 2D view.
            # N = total packed 1D length after trimming (provider rows include
            # the prefix-last token).
            logits_2d = logits.view(-1, logits.size(-1))

            # ##### [PS-diag] verify packed coordinate alignment (is logits N valid or padded) #####
            if _diag_on:
                _layout = ctx.packed_batch_layout
                print(
                    f"[PS-diag][packed-align] logits_N={logits_2d.shape[0]} "
                    f"total_padded={_layout.total_padded_length} "
                    f"total_valid={_layout.total_valid_length} "
                    f"has_padding={_layout.has_padding}",
                    flush=True,
                )
                for _idx in ctx.prefix_last_restore_indices:
                    print(
                        f"[PS-diag][packed-align] reuser={_idx.reuse_idx_in_batch} "
                        f"provider={_idx.provider_idx_in_batch} "
                        f"provider_1d_pos={_idx.provider_1d_pos} "
                        f"target_2d_pos={_idx.target_2d_pos}",
                        flush=True,
                    )
            # ##### [PS-diag] verify packed coordinate alignment end #####

            for index in ctx.prefix_last_restore_indices:
                # Each entry corresponds to one reuser's prefix-last; save the
                # vocab-dimension logits at its provider position.
                pos = index.provider_1d_pos
                key = (index.reuse_idx_in_batch, index.target_2d_pos)
                if pos < 0:
                    # Should never happen: prefix-last must fall within the
                    # direct provider's packed segment (see
                    # _build_prefix_last_restore_indices docs). Raise to expose
                    # the issue rather than letting downstream restore silently
                    # KeyError.
                    raise RuntimeError(
                        f"[vocab_logprobs] prefix-last spec got provider_1d_pos<0; "
                        f"key={key} provider_1d_pos={pos}. "
                        f"prefix-last should be within the direct provider's packed segment."
                    )
                # Clone to preserve the autograd graph (restore recomputes logp
                # via backward pass; detach is forbidden).
                saved = logits_2d[pos:pos + 1, :].clone()  # [1, V//tp]
                if saved.shape[0] == 0:
                    raise RuntimeError(
                        f"[vocab_logprobs] empty logits slice: key={key} "
                        f"pos={pos} N={logits_2d.shape[0]} — strict resolve "
                        f"returned an out-of-bounds packed position"
                    )
                # Key convention: (reuser_row, target_2d_pos), aligned with
                # restore_reuser_prefix_columns_2d's saved_key = (reuser_row, valid_col).
                ctx.prefix_last_logits_saved[key] = saved

        # Call original function (after this, logits are in-place modified to
        # exp(L-max), but dump/save have already completed)
        log_probs = original_fn(logits, labels)
        return log_probs

    return patched_fn
