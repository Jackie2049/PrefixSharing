from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.integrations.verl_fsdp import (
    PrefixSharingFSDPAttentionRuntime,
    build_prefix_sharing_micro_batch_fsdp,
    restore_prefix_sharing_outputs_2d,
)
from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState


def _mock_log_probs_fn(logits, labels):
    logp = torch.log_softmax(logits.float(), dim=-1)
    labels = labels.long() % logits.size(-1)
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def test_build_prefix_sharing_micro_batch_fsdp_returns_trimmed_batch_and_runtime_state():
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 10, 11, 0],
                [1, 2, 3, 20, 21, 22],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        ),
        "position_ids": torch.tensor(
            [
                [0, 1, 2, 3, 4, 0],
                [0, 1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [
                [2, 3, 10, 11, -100, -100],
                [2, 3, 20, 21, 22, -100],
            ],
            dtype=torch.long,
        ),
        "loss_mask": torch.tensor(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
            ],
            dtype=torch.bool,
        ),
    }

    trimmed_batch, runtime_state = build_prefix_sharing_micro_batch_fsdp(batch, config)

    assert runtime_state is not None
    assert isinstance(runtime_state, PrefixSharingRuntimeState)
    plan = runtime_state.prefix_sharing_plan
    assert plan.has_sharing
    assert plan.provider_index == [0, 0]
    assert plan.prefix_lens == [0, 3]
    assert plan.input_keep_ranges == [(0, 5), (3, 6)]
    assert runtime_state.packed_batch_layout == PackedBatchLayout.from_valid_lengths([5, 3])

    assert trimmed_batch is not batch
    assert torch.equal(trimmed_batch["attention_mask"][0], batch["attention_mask"][0])
    assert trimmed_batch["attention_mask"][1].tolist() == [False, False, False, True, True, True]
    assert trimmed_batch["loss_mask"][1].tolist() == [False, False, False, True, True, False]
    assert torch.equal(trimmed_batch["position_ids"][1, 3:6], torch.tensor([3, 4, 5]))


def test_build_prefix_sharing_micro_batch_fsdp_returns_none_when_no_sharing():
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.bool),
        "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
    }

    returned_batch, runtime_state = build_prefix_sharing_micro_batch_fsdp(batch, config)

    assert returned_batch is batch
    assert runtime_state is None


def test_restore_prefix_sharing_outputs_2d_restores_interior_last_logits_entropy_and_attention_output():
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 10, 11, 0],
                [1, 2, 3, 20, 21, 22],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        ),
        "position_ids": torch.tensor(
            [
                [0, 1, 2, 3, 4, 0],
                [0, 1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
        ),
    }
    _, runtime_state = build_prefix_sharing_micro_batch_fsdp(batch, config)
    assert runtime_state is not None

    vocab = 5
    hidden = 4
    output = {
        "log_probs": torch.tensor(
            [
                [-0.1, -0.2, -0.3, -0.4, -0.5, 0.0],
                [0.0, 0.0, 0.0, -1.3, -1.4, -1.5],
            ]
        ),
        "entropy": torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.0],
                [0.0, 0.0, 0.0, 1.3, 1.4, 1.5],
            ]
        ),
        "logits": torch.arange(2 * 6 * vocab, dtype=torch.float32).reshape(2, 6, vocab),
        "attention_output": torch.arange(2 * 6 * hidden, dtype=torch.float32).reshape(2, 6, hidden),
    }
    original_reuser_suffix_logits = output["logits"][1, 3:].clone()
    original_reuser_suffix_attention = output["attention_output"][1, 3:].clone()

    with prefix_sharing_runtime_context(runtime_state) as ctx:
        restore_index = ctx.prefix_last_restore_indices[0]
        saved_logits = output["logits"][restore_index.provider_idx_in_batch, restore_index.target_2d_pos:restore_index.target_2d_pos + 1]
        ctx.prefix_last_logits_saved[(restore_index.reuse_idx_in_batch, restore_index.target_2d_pos)] = saved_logits
        restored = restore_prefix_sharing_outputs_2d(output, _mock_log_probs_fn)

    # interior prefix logp/entropy copied from provider.
    assert torch.allclose(restored["log_probs"][1, 0:2], restored["log_probs"][0, 0:2])
    assert torch.allclose(restored["entropy"][1, 0:3], restored["entropy"][0, 0:3])

    # prefix-last logp recomputed using provider logits and reuser first suffix label (token 20 -> 0 mod vocab).
    expected = torch.log_softmax(saved_logits.float(), dim=-1)[0, 20 % vocab]
    assert torch.allclose(restored["log_probs"][1, 2], expected)

    # logits and attention output for the whole prefix copied from provider.
    assert torch.allclose(restored["logits"][1, 0:3], restored["logits"][0, 0:3])
    assert torch.allclose(restored["attention_output"][1, 0:3], restored["attention_output"][0, 0:3])

    # suffix part stays untouched.
    assert torch.allclose(restored["logits"][1, 3:], original_reuser_suffix_logits)
    assert torch.allclose(restored["attention_output"][1, 3:], original_reuser_suffix_attention)


def test_prefix_sharing_fsdp_attention_runtime_scatter_dense_outputs():
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 10, 11, 0],
                [1, 2, 3, 20, 21, 22],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        ),
        "position_ids": torch.tensor(
            [
                [0, 1, 2, 3, 4, 0],
                [0, 1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
        ),
    }
    _, runtime_state = build_prefix_sharing_micro_batch_fsdp(batch, config)
    assert runtime_state is not None

    torch.manual_seed(1)
    query = torch.randn(2, 6, 2, 4)
    key = torch.randn(2, 6, 2, 4)
    value = torch.randn(2, 6, 2, 4)

    runtime = PrefixSharingFSDPAttentionRuntime(layer_id=7)
    with prefix_sharing_runtime_context(runtime_state) as ctx:
        dense_output = runtime.forward(None, query, key, value)
        assert ctx.stats.layers[7].reuse_hit_count == 1

    assert dense_output.shape == query.shape
    # Provider valid tokens are computed; provider padding remains zero.
    assert not torch.allclose(dense_output[0, 0:5], torch.zeros_like(dense_output[0, 0:5]))
    assert torch.allclose(dense_output[0, 5], torch.zeros_like(dense_output[0, 5]))
    # Reuser prefix is not computed on Q path and is restored later.
    assert torch.allclose(dense_output[1, 0:3], torch.zeros_like(dense_output[1, 0:3]))
    # Reuser suffix is computed.
    assert not torch.allclose(dense_output[1, 3:6], torch.zeros_like(dense_output[1, 3:6]))
