"""Tests for the gather-based KV expansion (prefix_sharing.backends.kv_gather).

Verifies that :func:`build_kv_via_gather` is a drop-in replacement for the
copy-loop assembly in :meth:`TorchReferenceBackend.build_kv`:

* identical expanded K/V values (bitwise — both copy the same source rows);
* equivalent gradients w.r.t. the packed K/V inputs;
* no store at all: the function has no ``store`` parameter (reuse chains
  resolve in packed coordinates at index-build time; the store's only
  readers were the copy loop and the DeltaNet path);
* identical stats accounting;
* transitive reuse chains and TP-padded layouts.

CPU-only; no CUDA/flash-attn dependency.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.kv_gather import build_kv_via_gather, get_kv_gather_index
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.prefix_store import PrefixAttentionStore


def _make_plan(batch_sizes, prefix_lens):
    """Build a PrefixSharingPlan with controlled provider/reuser layout.

    Same construction as tests/unit_test/test_torch_ref_backend.py: provider
    rows have prefix_lens[i]==0; reuser rows share the nearest preceding
    provider's first prefix_lens[i] tokens, so the trie detector produces the
    requested layout (including transitive chains, e.g. [0, 3, 5]).
    """
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1)
    planner = PrefixSharingPlanner(config)
    sequences = []
    next_token = 100
    provider_seqs = {}
    for i, (size, p) in enumerate(zip(batch_sizes, prefix_lens)):
        if p == 0:
            seq = list(range(next_token, next_token + size))
            next_token += size
            provider_seqs[i] = seq
            sequences.append(seq)
        else:
            provider_idx = max(j for j in range(i) if prefix_lens[j] == 0)
            provider_seq = provider_seqs[provider_idx]
            suffix = list(range(next_token, next_token + size - p))
            next_token += size - p
            sequences.append(provider_seq[:p] + suffix)
    return planner.plan(sequences)


def _make_layout(plan, align_size=1):
    rows = [torch.zeros(length, dtype=torch.long) for length in plan.kept_lengths_q]
    return PackedBatchLayout.from_kept_position_rows(rows, align_size=align_size)


def _run_both(key, value, plan, layout, *, layer_id=0, tp_rank=0):
    """Run copy-based and gather-based build_kv on identical inputs."""
    ref_backend = TorchReferenceBackend()
    copy_k, copy_v = ref_backend.build_kv(
        key, value, PrefixAttentionStore(), plan,
        packed_batch_layout=layout, layer_id=layer_id, tp_rank=tp_rank,
    )
    gather_k, gather_v = build_kv_via_gather(
        key, value, plan,
        packed_batch_layout=layout, layer_id=layer_id,
    )
    return (copy_k, copy_v), (gather_k, gather_v)


def _random_kv(total, num_heads=2, head_dim=8, dtype=torch.float32, seed=42, requires_grad=False):
    torch.manual_seed(seed)
    key = torch.randn(total, num_heads, head_dim, dtype=dtype, requires_grad=requires_grad)
    value = torch.randn(total, num_heads, head_dim, dtype=dtype, requires_grad=requires_grad)
    return key, value


# ------------------------------------------------------------------
# value equality vs the copy-based reference
# ------------------------------------------------------------------


def test_gather_matches_copy_simple_reuse():
    plan = _make_plan([6, 5], [0, 3])
    layout = _make_layout(plan)
    key, value = _random_kv(layout.total_padded_length)

    (copy_k, copy_v), (gather_k, gather_v) = _run_both(key, value, plan, layout)

    assert torch.equal(gather_k, copy_k)
    assert torch.equal(gather_v, copy_v)


def test_gather_matches_copy_transitive_chain():
    # row0=provider(8), row1 reuses row0(prefix=3), row2 reuses row1's expanded(prefix=5)
    plan = _make_plan([8, 7, 6], [0, 3, 5])
    layout = _make_layout(plan)
    key, value = _random_kv(layout.total_padded_length)

    (copy_k, copy_v), (gather_k, gather_v) = _run_both(key, value, plan, layout)

    assert torch.equal(gather_k, copy_k)
    assert torch.equal(gather_v, copy_v)


def test_gather_matches_copy_multi_providers_and_reusers():
    plan = _make_plan([8, 7, 6, 4, 5], [0, 3, 5, 0, 2])
    layout = _make_layout(plan)
    key, value = _random_kv(layout.total_padded_length)

    (copy_k, copy_v), (gather_k, gather_v) = _run_both(key, value, plan, layout)

    assert torch.equal(gather_k, copy_k)
    assert torch.equal(gather_v, copy_v)


def test_gather_matches_copy_no_sharing():
    plan = _make_plan([4, 6], [0, 0])
    layout = _make_layout(plan)
    key, value = _random_kv(layout.total_padded_length)

    (copy_k, copy_v), (gather_k, gather_v) = _run_both(key, value, plan, layout)

    assert torch.equal(gather_k, copy_k)
    assert torch.equal(gather_v, copy_v)


def test_gather_matches_copy_with_tp_padding():
    plan = _make_plan([5, 4, 6], [0, 3, 2])
    layout = _make_layout(plan, align_size=4)
    assert layout.has_padding
    key, value = _random_kv(layout.total_padded_length)

    (copy_k, copy_v), (gather_k, gather_v) = _run_both(key, value, plan, layout)

    # Expanded output follows semantic lengths (padding stripped), bitwise equal.
    assert gather_k.shape[0] == sum(plan.expanded_lengths_kv)
    assert torch.equal(gather_k, copy_k)
    assert torch.equal(gather_v, copy_v)


def test_gather_matches_copy_transitive_chain_with_padding():
    plan = _make_plan([8, 7, 6, 4], [0, 3, 5, 0])
    layout = _make_layout(plan, align_size=4)
    key, value = _random_kv(layout.total_padded_length)

    (copy_k, copy_v), (gather_k, gather_v) = _run_both(key, value, plan, layout)

    assert torch.equal(gather_k, copy_k)
    assert torch.equal(gather_v, copy_v)


def test_gather_matches_copy_suffix_only_prefix_row():
    # reuser whose sequence is entirely prefix (suffix_len == 0 is excluded by
    # the planner's restore logic but build_kv must still assemble correctly)
    plan = _make_plan([6, 3], [0, 3])
    layout = _make_layout(plan)
    key, value = _random_kv(layout.total_padded_length)

    (copy_k, copy_v), (gather_k, gather_v) = _run_both(key, value, plan, layout)

    assert torch.equal(gather_k, copy_k)
    assert torch.equal(gather_v, copy_v)


# ------------------------------------------------------------------
# gradient equivalence
# ------------------------------------------------------------------


@pytest.mark.parametrize("batch_sizes,prefix_lens", [
    ([6, 5], [0, 3]),
    ([8, 7, 6], [0, 3, 5]),        # transitive chain
    ([8, 7, 6, 4, 5], [0, 3, 5, 0, 2]),
])
def test_gather_gradients_match_copy(batch_sizes, prefix_lens):
    plan = _make_plan(batch_sizes, prefix_lens)
    layout = _make_layout(plan)
    total = layout.total_padded_length

    key1, value1 = _random_kv(total, dtype=torch.float64, requires_grad=True)
    key2 = key1.detach().clone().requires_grad_(True)
    value2 = value1.detach().clone().requires_grad_(True)

    ref_backend = TorchReferenceBackend()
    copy_k, copy_v = ref_backend.build_kv(
        key1, value1, PrefixAttentionStore(), plan,
        packed_batch_layout=layout, layer_id=0,
    )
    gather_k, gather_v = build_kv_via_gather(
        key2, value2, plan,
        packed_batch_layout=layout, layer_id=0,
    )

    # Exercise a non-uniform gradient so per-token weighting differs.
    torch.manual_seed(7)
    grad_k = torch.randn_like(copy_k)
    grad_v = torch.randn_like(copy_v)
    copy_k.backward(grad_k)
    copy_v.backward(grad_v)
    gather_k.backward(grad_k)
    gather_v.backward(grad_v)

    torch.testing.assert_close(key2.grad, key1.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(value2.grad, value1.grad, rtol=1e-12, atol=1e-12)


def test_gather_backward_kernel_shape_is_single_scatter():
    """The whole point of the gather path: one autograd node per tensor.

    Expanded-KV graph must contain a single index_select node per K/V
    (whose backward is one index_add), not a chain of CopySlices nodes.
    """
    plan = _make_plan([8, 7, 6], [0, 3, 5])
    layout = _make_layout(plan)
    key, value = _random_kv(layout.total_padded_length, requires_grad=True)

    gather_k, _ = build_kv_via_gather(
        key, value, plan,
        packed_batch_layout=layout, layer_id=0,
    )
    assert gather_k.grad_fn is not None
    assert "IndexSelect" in type(gather_k.grad_fn).__name__


# ------------------------------------------------------------------
# store contract: the gather path has no store parameter at all
# ------------------------------------------------------------------


def test_gather_has_no_store_parameter():
    """Gather resolves reuse chains in packed coordinates; the store's only
    readers were the copy loop and the DeltaNet path.  Lock the contract:
    build_kv_via_gather does not accept a store."""
    import inspect

    assert "store" not in inspect.signature(build_kv_via_gather).parameters


# ------------------------------------------------------------------
# stats accounting
# ------------------------------------------------------------------


class _RecordingStats:
    def __init__(self):
        self.calls = []

    def record_attention_kv_build(self, **kwargs):
        self.calls.append(kwargs)


def test_gather_stats_match_copy():
    plan = _make_plan([8, 7, 6], [0, 3, 5])
    layout = _make_layout(plan)
    key, value = _random_kv(layout.total_padded_length)

    copy_stats = _RecordingStats()
    gather_stats = _RecordingStats()
    TorchReferenceBackend().build_kv(
        key, value, PrefixAttentionStore(), plan,
        packed_batch_layout=layout, layer_id=0, stats=copy_stats,
    )
    build_kv_via_gather(
        key, value, plan,
        packed_batch_layout=layout, layer_id=0, stats=gather_stats,
    )

    assert len(copy_stats.calls) == len(gather_stats.calls) == 1
    assert gather_stats.calls[0] == copy_stats.calls[0]


# ------------------------------------------------------------------
# index caching
# ------------------------------------------------------------------


def test_gather_index_cached_per_plan_and_layout():
    plan = _make_plan([8, 7, 6], [0, 3, 5])
    layout = _make_layout(plan)

    first = get_kv_gather_index(plan, layout, torch.device("cpu"))
    second = get_kv_gather_index(plan, layout, torch.device("cpu"))
    assert first is second  # same tensor object → built once, shared by all layers

    other_layout = _make_layout(plan)
    third = get_kv_gather_index(plan, other_layout, torch.device("cpu"))
    assert third is not first  # different layout object → rebuild
    assert torch.equal(third, first)
