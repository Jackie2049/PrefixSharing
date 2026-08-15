"""Patch: no_padding_2_padding — correct sequence length after prefix-sharing
physical trimming.

After physical trimming, the model output NestedTensor's offsets().diff()
gives the trimmed length per row, but data["attention_mask"] still reflects
the original length, causing the assertion failure:
  sum(prompt_lens + response_lens) != values.shape[0]

Fix: when the tensor is a NestedTensor and offsets().diff().sum() disagrees
with the sum derived from attention_mask, use the NestedTensor's own offsets
as the trimmed sequence lengths. Responses are not trimmed, so
trimmed_prompt_lens = trimmed_seq_lens - response_lens.

No cross-process communication or metadata is needed — purely derived from
the model output shape.

Note: multiple modules (padding.py, losses.py, ray_trainer.py) reference this
function via from...import. Each module's local reference must be patched.
This factory ensures all patches share the same original function reference,
avoiding chained wrapping.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from verl.utils import tensordict_utils as tu

# Cache the true original function to avoid chained wrapping (patched-of-patched)
# when patching multiple times. Cache the original on the first patch_factory
# call; reuse the same wrapper on subsequent calls.
_cached_original: Any | None = None
_cached_wrapper: Any | None = None


def patch_no_padding_2_padding(original_func: Any) -> Any:
    """Create a patch wrapper for no_padding_2_padding.

    Multiple modules (padding, losses, ray_trainer, distillation losses) all
    hold from...import local references. PatchSpec calls this factory once per
    module. Using original_func directly (which may already be a patched
    version) would produce chained wrapping. Therefore we cache the true
    original so all modules share a single wrapper.
    """
    global _cached_original, _cached_wrapper

    if _cached_original is None:
        _cached_original = original_func

    # Always create the wrapper from the true original (not chained wrapping)
    if _cached_wrapper is None:
        _cached_wrapper = _make_wrapper(_cached_original)

    return _cached_wrapper


def _make_wrapper(original_func: Any) -> Any:

    def patched_no_padding_2_padding(tensor: Any, data: Any) -> Any:
        # ── PS trimming detection first; take PS-aware path on mismatch ──
        values = tensor.values() if tensor.is_nested else tensor
        prompt_ids = data["prompts"]
        response_ids = data["responses"]

        max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)

        if prompt_ids.is_nested:
            prompt_lens = prompt_ids.offsets().diff()
            response_lens = response_ids.offsets().diff()
            if max_response_len < 0:
                max_response_len = response_lens.max().item()
        else:
            attention_mask = data["attention_mask"]
            assert not attention_mask.is_nested
            prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
            response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
            max_response_len = response_ids.shape[1]

        sequence_lens = prompt_lens + response_lens
        sequence_offsets = sequence_lens.cumsum(dim=0)

        # ── PS trimming detection ──
        # The model output NestedTensor's offsets().diff() gives the actual
        # length per row. If this disagrees with the total length from
        # attention_mask, PS physical trimming has occurred.
        # Substitute trimmed_seq_lens - response_lens for prompt_lens.
        if tensor.is_nested:
            trimmed_seq_lens = tensor.offsets().diff()
            expected_total = sequence_offsets[-1].item()
            actual_total = values.shape[0]
            if expected_total != actual_total:
                # PS trimming detected
                trimmed_prompt_lens = trimmed_seq_lens - response_lens
                sequence_lens = trimmed_prompt_lens + response_lens
                sequence_offsets = sequence_lens.cumsum(dim=0)
                print(
                    f"[PS][nopadding] trimming detected: "
                    f"expected_total={expected_total} actual_total={actual_total} "
                    f"original_prompt_lens={prompt_lens.tolist()} "
                    f"trimmed_prompt_lens={trimmed_prompt_lens.tolist()} "
                    f"response_lens={response_lens.tolist()}"
                )
                # PS path: on mismatch, use our own implementation
                # (to avoid the original function's assert failing again)
                assert sequence_offsets[-1].item() == values.shape[0], (
                    f"[PS] sequence_offsets[-1]={sequence_offsets[-1].item()} "
                    f"!= values.shape[0]={values.shape[0]}"
                )
                response_list = []
                skip_padding = (0, 0) * (values.ndim - 1)
                for resp_len, seq_offset in zip(response_lens, sequence_offsets, strict=True):
                    pad_size = max_response_len - resp_len
                    response_list.append(
                        F.pad(
                            values[seq_offset - resp_len - 1 : seq_offset - 1],
                            (*skip_padding, 0, pad_size),
                        )
                    )
                return torch.stack(response_list, dim=0)

        # ── normal path: no PS trimming detected, call original function ──
        return original_func(tensor, data)

    return patched_no_padding_2_padding
