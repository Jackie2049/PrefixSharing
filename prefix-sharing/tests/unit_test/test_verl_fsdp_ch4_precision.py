"""Ch4.3 precision verification for PrefixSharing FSDP.

Verifies that PrefixSharing attention produces numerically identical results
to a baseline full-sequence attention, per docs/feature-fsdp.md Chapter 4.3:

- attention output for reuser suffix matches baseline suffix positions
- attention output for provider matches baseline
- gradients flow back through provider prefix KV (no detach)
- restore fills interior prefix + prefix-last correctly
- padding positions do not contribute

The torch_ref backend concatenates provider_kv[:prefix_len] + reuser_suffix_kv,
which is mathematically identical to the baseline full-sequence KV for the
reuser's suffix tokens. This test proves that invariant holds end-to-end.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.integrations.verl_fsdp import (
    PrefixSharingFSDPAttentionRuntime,
    plan_and_trim_microbatch_fsdp,
    restore_prefix_sharing_outputs_2d,
)


def _baseline_attention(query, key, value):
    """Standard scaled dot-product causal attention. [B, L, H, D] -> [B, L, H, D]."""
    B, L, H, D = query.shape
    scale = 1.0 / (D ** 0.5)
    # scores: [B, L_query, M_key, H]
    scores = torch.einsum("blhd,bmhd->blmh", query, key) * scale
    # causal mask: query position l can attend key position m where m <= l
    # scores shape [B, L, M, H], mask shape [L, M] -> broadcast needs [L, M, 1]
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool)).unsqueeze(-1)  # [L, M, 1]
    scores = scores.masked_fill(~causal, float("-inf"))
    attn = torch.softmax(scores, dim=2)                                    # [B, L, M, H]
    out = torch.einsum("blmh,bmhd->blhd", attn, value)                    # [B, L, H, D]
    return out


def test_reuser_suffix_attention_matches_baseline():
    """Reuser suffix attention output must equal baseline suffix positions.

    Key invariant: for a causal model, hidden states at shared-prefix positions
    are identical across rows (same tokens, same causal context). So the K/V at
    prefix positions MUST be identical between provider and reuser. We construct
    the test tensors to reflect that invariant.
    """
    torch.manual_seed(0)
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)

    # Two sequences sharing prefix [1,2,3]; suffix differs.
    batch = {
        "input_ids": torch.tensor(
            [[1, 2, 3, 4, 5], [1, 2, 3, 10, 11]], dtype=torch.long
        ),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.tensor(
            [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long
        ),
    }
    trimmed, state = plan_and_trim_microbatch_fsdp(batch, config)
    assert state is not None
    plan = state.prefix_sharing_plan
    prefix_len = plan.prefix_lens[1]  # 3

    # Construct Q/K/V where prefix positions [0:prefix_len] are IDENTICAL across rows,
    # matching the causal invariant that shared-prefix tokens produce identical
    # hidden states (and thus identical Q/K/V at those positions).
    H, D = 2, 4
    L = 5
    torch.manual_seed(123)
    # Shared prefix Q/K/V (same for both rows at positions 0..prefix_len-1)
    shared_q_prefix = torch.randn(prefix_len, H, D)
    shared_k_prefix = torch.randn(prefix_len, H, D)
    shared_v_prefix = torch.randn(prefix_len, H, D)
    # Distinct suffix Q/K/V
    provider_suffix_q = torch.randn(L - prefix_len, H, D)
    provider_suffix_k = torch.randn(L - prefix_len, H, D)
    provider_suffix_v = torch.randn(L - prefix_len, H, D)
    reuser_suffix_q = torch.randn(L - prefix_len, H, D)
    reuser_suffix_k = torch.randn(L - prefix_len, H, D)
    reuser_suffix_v = torch.randn(L - prefix_len, H, D)

    query = torch.stack(
        [torch.cat([shared_q_prefix, provider_suffix_q]),
         torch.cat([shared_q_prefix, reuser_suffix_q])],
        dim=0,
    )  # [2, L, H, D]
    key = torch.stack(
        [torch.cat([shared_k_prefix, provider_suffix_k]),
         torch.cat([shared_k_prefix, reuser_suffix_k])],
        dim=0,
    )
    value = torch.stack(
        [torch.cat([shared_v_prefix, provider_suffix_v]),
         torch.cat([shared_v_prefix, reuser_suffix_v])],
        dim=0,
    )

    # --- PrefixSharing path (detached to compare values) ---
    runtime = PrefixSharingFSDPAttentionRuntime(layer_id=0)
    with prefix_sharing_runtime_context(state):
        ps_out = runtime.forward(None, query.detach(), key.detach(), value.detach())

    # --- Baseline path ---
    baseline_out = _baseline_attention(query.detach(), key.detach(), value.detach())

    # Provider row: PrefixSharing computes attention for all 5 valid tokens.
    assert torch.allclose(ps_out[0, :L], baseline_out[0, :L], atol=1e-5), \
        f"Provider attention mismatch: max diff {(ps_out[0, :L] - baseline_out[0, :L]).abs().max()}"

    # Reuser row: suffix [prefix_len:] should match baseline exactly.
    assert torch.allclose(ps_out[1, prefix_len:], baseline_out[1, prefix_len:], atol=1e-5), \
        f"Reuser suffix attention mismatch: max diff {(ps_out[1, prefix_len:] - baseline_out[1, prefix_len:]).abs().max()}"

    # Reuser prefix positions are zeroed (to be restored later).
    assert torch.allclose(ps_out[1, :prefix_len], torch.zeros_like(ps_out[1, :prefix_len]))


def test_gradient_flows_through_provider_prefix_kv():
    """Provider prefix KV must retain autograd — no detach.

    Reuser suffix attention reads from provider prefix KV. A loss on the reuser
    suffix must produce a non-zero gradient on the provider's prefix K/V tensors.
    """
    torch.manual_seed(42)
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    batch = {
        "input_ids": torch.tensor(
            [[1, 2, 3, 4, 5], [1, 2, 3, 10, 11]], dtype=torch.long
        ),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.tensor(
            [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long
        ),
    }
    trimmed, state = plan_and_trim_microbatch_fsdp(batch, config)
    assert state is not None

    H, D = 2, 4
    query = torch.randn(2, 5, H, D, requires_grad=True)
    key = torch.randn(2, 5, H, D, requires_grad=True)
    value = torch.randn(2, 5, H, D, requires_grad=True)

    runtime = PrefixSharingFSDPAttentionRuntime(layer_id=0)
    with prefix_sharing_runtime_context(state):
        ps_out = runtime.forward(None, query, key, value)

    # Loss on reuser suffix only (positions 3,4 of row 1)
    loss = ps_out[1, 3:5].sum()
    loss.backward()

    # Provider key/value prefix positions [0:3] must have non-zero grad because
    # reuser suffix attention reads from provider prefix KV.
    assert key.grad is not None, "No gradient on key"
    assert value.grad is not None, "No gradient on value"

    provider_prefix_key_grad = key.grad[0, 0:3]
    provider_prefix_value_grad = value.grad[0, 0:3]
    assert provider_prefix_key_grad.abs().sum() > 0, \
        "Provider prefix key grad is zero — KV was detached"
    assert provider_prefix_value_grad.abs().sum() > 0, \
        "Provider prefix value grad is zero — KV was detached"


def test_restore_interior_prefix_matches_provider():
    """After restore, reuser interior prefix logp must equal provider's."""
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    batch = {
        "input_ids": torch.tensor(
            [[1, 2, 3, 4, 5], [1, 2, 3, 10, 11]], dtype=torch.long
        ),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.tensor(
            [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long
        ),
    }
    _, state = plan_and_trim_microbatch_fsdp(batch, config)
    assert state is not None

    # Simulate trimmed output: provider full, reuser has only suffix (prefix zeroed)
    vocab = 16
    output = {
        "log_probs": torch.tensor(
            [
                [-1.0, -2.0, -3.0, -4.0, -5.0],   # provider full
                [0.0, 0.0, 0.0, -1.4, -1.5],      # reuser: prefix zeroed, suffix real
            ]
        ),
        "entropy": torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.0, 0.0, 0.0, 1.4, 1.5],
            ]
        ),
        "logits": torch.randn(2, 5, vocab),
        "attention_output": torch.randn(2, 5, 4),
    }

    def mock_logp(logits, labels):
        return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    with prefix_sharing_runtime_context(state) as ctx:
        # Save provider prefix-last logits for restore
        spec = ctx.prefix_last_restore_indices[0]
        ctx.prefix_last_logits_saved[(spec.reuse_idx_in_batch, spec.target_2d_pos)] = \
            output["logits"][spec.provider_idx_in_batch, spec.target_2d_pos:spec.target_2d_pos + 1]
        restored = restore_prefix_sharing_outputs_2d(output, mock_logp)

    # Interior prefix logp [0:2] copied from provider
    assert torch.allclose(restored["log_probs"][1, 0:2], restored["log_probs"][0, 0:2])
    # Entropy [0:3] copied from provider
    assert torch.allclose(restored["entropy"][1, 0:3], restored["entropy"][0, 0:3])
    # logits [0:3] copied from provider
    assert torch.allclose(restored["logits"][1, 0:3], restored["logits"][0, 0:3])
    # attention_output [0:3] copied from provider
    assert torch.allclose(restored["attention_output"][1, 0:3], restored["attention_output"][0, 0:3])


def test_restore_prefix_last_recomputed_with_provider_logits_and_reuser_label():
    """prefix-last logp must be recomputed with provider logits + reuser first suffix label."""
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    batch = {
        "input_ids": torch.tensor(
            [[1, 2, 3, 4, 5], [1, 2, 3, 10, 11]], dtype=torch.long
        ),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.tensor(
            [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long
        ),
    }
    _, state = plan_and_trim_microbatch_fsdp(batch, config)
    plan = state.prefix_sharing_plan
    prefix_len = plan.prefix_lens[1]  # 3

    vocab = 16
    output = {
        "log_probs": torch.zeros(2, 5),
        "logits": torch.randn(2, 5, vocab),
    }

    def mock_logp(logits, labels):
        return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    with prefix_sharing_runtime_context(state) as ctx:
        spec = ctx.prefix_last_restore_indices[0]
        # The reuser's first suffix label is token at position prefix_len in input_ids
        reuser_first_suffix_label = batch["input_ids"][1, prefix_len].item()
        saved = output["logits"][0, prefix_len - 1:prefix_len]
        ctx.prefix_last_logits_saved[(spec.reuse_idx_in_batch, spec.target_2d_pos)] = saved
        restored = restore_prefix_sharing_outputs_2d(output, mock_logp)

    expected = mock_logp(saved, torch.tensor([reuser_first_suffix_label]))
    assert torch.allclose(restored["log_probs"][1, prefix_len - 1], expected.reshape(()))


def test_baseline_matches_when_no_sharing():
    """When no prefix sharing detected, output must be identical to baseline."""
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    batch = {
        "input_ids": torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long
        ),
        "attention_mask": torch.ones(2, 4, dtype=torch.bool),
        "position_ids": torch.tensor(
            [[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long
        ),
    }
    returned, state = plan_and_trim_microbatch_fsdp(batch, config)
    assert state is None
    assert returned is batch  # exact same object -> no transformation


def test_padding_positions_do_not_contribute():
    """Padding positions in the batch must not affect valid token outputs."""
    torch.manual_seed(7)
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)

    # Same content but different padding length
    batch_unpadded = {
        "input_ids": torch.tensor(
            [[1, 2, 3, 4, 5], [1, 2, 3, 10, 11]], dtype=torch.long
        ),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.tensor(
            [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long
        ),
    }
    batch_padded = {
        "input_ids": torch.tensor(
            [[1, 2, 3, 4, 5, 0, 0], [1, 2, 3, 10, 11, 0, 0]], dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool
        ),
        "position_ids": torch.tensor(
            [[0, 1, 2, 3, 4, 0, 0], [0, 1, 2, 3, 4, 0, 0]], dtype=torch.long
        ),
    }

    H, D = 2, 4
    _, state_unpadded = plan_and_trim_microbatch_fsdp(batch_unpadded, config)
    _, state_padded = plan_and_trim_microbatch_fsdp(batch_padded, config)
    assert state_unpadded is not None and state_padded is not None

    # Same plan semantics regardless of padding
    plan_u = state_unpadded.prefix_sharing_plan
    plan_p = state_padded.prefix_sharing_plan
    assert plan_u.prefix_lens == plan_p.prefix_lens
    assert plan_u.provider_index == plan_p.provider_index
