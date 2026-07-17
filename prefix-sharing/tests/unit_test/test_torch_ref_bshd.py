"""Tests for the BSHD torch reference backend.

The most important test here is ``test_end_to_end_matches_per_row_reference``:
it wires plan → layout → build_kv → attention together and compares against an
independent per-row Python reference computed in absolute coordinates.  This is
the integration seam that a component-level test cannot cover — the first BSHD
implementation passed all its component tests while the integrated path was
numerically wrong (mask built in trimmed/left-aligned coordinates while the
tensors were in absolute/padded coordinates).
"""

from __future__ import annotations

import math

import torch

from prefix_sharing.backends.batched_layout import BatchedBatchLayout
from prefix_sharing.backends.torch_ref_bshd import (
    TorchReferenceBackendBshd,
    _build_bshd_attention_mask,
    _kept_q_row_mask,
)
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.prefix_store import PrefixAttentionStore


def _shared_plan():
    """2 sequences, share 3-token prefix, lengths 5 and 6."""
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    )
    return planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]],
        forward_id=1, micro_batch_id=1,
    )


def _chained_plan():
    """3 sequences with transitive reuse: row1 reuses row0, row2 reuses row1."""
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
    )
    return planner.plan(
        [[1, 2, 3, 4, 10], [1, 2, 3, 4, 20, 21], [1, 2, 3, 4, 20, 30]],
        forward_id=2, micro_batch_id=1,
    )


class TestBuildKvBshd:
    def test_provider_stored_and_reuser_expanded(self):
        """Provider KV stored; reuser KV = cat(provider_prefix, own_suffix)."""
        plan = _shared_plan()
        # plan: prefix_lens=[0, 3], expanded_lengths_kv=[5, 6]
        backend = TorchReferenceBackendBshd()
        store = PrefixAttentionStore()
        B, S, H, D = 2, 6, 1, 8
        torch.manual_seed(0)
        key = torch.randn(B, S, H, D)
        value = torch.randn(B, S, H, D)
        layout = BatchedBatchLayout(valid_lengths=[5, 6], seq_length=S)

        expanded_k, expanded_v = backend.build_kv(
            key, value, store, plan,
            packed_batch_layout=layout, layer_id=0, tp_rank=0,
        )

        expected_max_kv = max(plan.expanded_lengths_kv)  # 6
        assert expanded_k.shape == (B, expected_max_kv, H, D)
        assert expanded_v.shape == (B, expected_max_kv, H, D)

        # Provider row: all valid tokens stored/expanded as-is.
        assert torch.equal(expanded_k[0, :5], key[0, :5])
        assert torch.equal(expanded_v[0, :5], value[0, :5])

        # Reuser row: first 3 = provider prefix, last 3 = own suffix
        # (absolute columns [3, 6) of the untrimmed padded input).
        assert torch.equal(expanded_k[1, :3], key[0, :3])
        assert torch.equal(expanded_k[1, 3:6], key[1, 3:6])
        assert torch.equal(expanded_v[1, :3], value[0, :3])
        assert torch.equal(expanded_v[1, 3:6], value[1, 3:6])

        # Padding region is zero.
        assert torch.all(expanded_k[0, 5:] == 0)

    def test_transitive_reuse(self):
        """A reuser's expanded KV is published so a later row can reuse it."""
        plan = _chained_plan()
        assert plan.provider_index[1] == 0
        assert plan.provider_index[2] == 1  # reuses row1's longer prefix
        backend = TorchReferenceBackendBshd()
        store = PrefixAttentionStore()
        B, S, H, D = 3, 6, 1, 8
        torch.manual_seed(0)
        key = torch.randn(B, S, H, D)
        value = torch.randn(B, S, H, D)
        layout = BatchedBatchLayout(valid_lengths=[5, 6, 6], seq_length=S)

        expanded_k, expanded_v = backend.build_kv(
            key, value, store, plan,
            packed_batch_layout=layout, layer_id=0, tp_rank=0,
        )

        # Row2 prefix [0, 5) must equal row1's expanded KV [0, 5):
        # [0,4) from row0, [4,5) from row1's own suffix.
        assert torch.equal(expanded_k[2, :4], key[0, :4])
        assert torch.equal(expanded_k[2, 4:5], key[1, 4:5])
        assert torch.equal(expanded_k[2, 5:6], key[2, 5:6])

    def test_empty_batch(self):
        plan = _shared_plan()
        store = PrefixAttentionStore()
        backend = TorchReferenceBackendBshd()
        k = torch.randn(0, 4, 1, 8)
        v = torch.randn(0, 4, 1, 8)
        expanded_k, expanded_v = backend.build_kv(
            k, v, store, plan,
            packed_batch_layout=BatchedBatchLayout(valid_lengths=[], seq_length=4),
            layer_id=0, tp_rank=0,
        )
        assert expanded_k.shape[0] == 0
        assert expanded_v.shape[0] == 0


class TestBuildBshdAttentionMask:
    def test_provider_causal_absolute(self):
        plan = _shared_plan()
        mask = _build_bshd_attention_mask(
            plan=plan,
            valid_lengths=[5, 6],
            expanded_kv_lengths=[5, 6],
            max_q=6, max_kv=6, batch_size=2,
            device=torch.device("cpu"),
        )
        assert mask.shape == (2, 1, 6, 6)
        # Provider (row 0): causal within [5, 5]; pad row 5 fully masked.
        for q in range(5):
            for kv in range(5):
                assert mask[0, 0, q, kv].item() == (kv > q)
        assert mask[0, 0, 5, :].all()
        assert mask[0, 0, :, 5].all()

    def test_reuser_absolute_coordinates(self):
        """Reuser suffix Q rows live at [prefix_len, valid) — NOT left-aligned.

        This is the regression test for the first implementation's coordinate
        mismatch: the mask must open rows [3, 6), not rows [0, 3).
        """
        plan = _shared_plan()
        mask = _build_bshd_attention_mask(
            plan=plan,
            valid_lengths=[5, 6],
            expanded_kv_lengths=[5, 6],
            max_q=6, max_kv=6, batch_size=2,
            device=torch.device("cpu"),
        )
        reuser = mask[1, 0]

        # Prefix Q rows [0, 3): fully masked (their output is zeroed later).
        assert reuser[:3, :].all()

        # Suffix Q rows [3, 6): prefix KV cols [0, 3) all visible.
        assert not reuser[3:6, :3].any()

        # Suffix KV cols [3, 6): causal within the suffix
        # (suffix q index j = abs_q - 3 sees suffix kv index k = abs_kv - 3 <= j).
        for abs_q in range(3, 6):
            for abs_kv in range(3, 6):
                j, k = abs_q - 3, abs_kv - 3
                assert reuser[abs_q, abs_kv].item() == (k > j), (
                    f"abs_q={abs_q} abs_kv={abs_kv} should be {'masked' if k > j else 'visible'}"
                )

    def test_kept_q_row_mask(self):
        plan = _shared_plan()
        keep = _kept_q_row_mask(plan, [5, 6], 6, torch.device("cpu"))
        assert keep[0].tolist() == [True] * 5 + [False]
        assert keep[1].tolist() == [False] * 3 + [True] * 3


class TestAttentionBshd:
    def test_output_shape_and_zeroed_rows(self):
        plan = _shared_plan()
        backend = TorchReferenceBackendBshd()
        store = PrefixAttentionStore()
        B, S, H, D = 2, 6, 2, 8
        torch.manual_seed(0)
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, S, 1, D)
        v = torch.randn(B, S, 1, D)
        layout = BatchedBatchLayout(valid_lengths=[5, 6], seq_length=S)

        ek, ev = backend.build_kv(k, v, store, plan, packed_batch_layout=layout, layer_id=0)
        out = backend.attention(q, ek, ev, plan, packed_batch_layout=layout)

        assert out.shape == (B, S, H, D)
        # Reuser prefix rows [0, 3) and provider pad row 5 are exactly zero.
        assert torch.all(out[1, :3] == 0)
        assert torch.all(out[0, 5:] == 0)
        # Kept rows are non-zero (random inputs — measure of sanity).
        assert out[1, 3:].abs().sum() > 0
        assert out[0, :5].abs().sum() > 0

    def test_gqa_head_expansion(self):
        """H_q=4, H_kv=2 must expand via repeat_interleave (not only MQA)."""
        plan = _shared_plan()
        backend = TorchReferenceBackendBshd()
        store = PrefixAttentionStore()
        B, S, D = 2, 6, 8
        torch.manual_seed(0)
        q = torch.randn(B, S, 4, D)
        k = torch.randn(B, S, 2, D)
        v = torch.randn(B, S, 2, D)
        layout = BatchedBatchLayout(valid_lengths=[5, 6], seq_length=S)

        ek, ev = backend.build_kv(k, v, store, plan, packed_batch_layout=layout, layer_id=0)
        out = backend.attention(q, ek, ev, plan, packed_batch_layout=layout)
        assert out.shape == (B, S, 4, D)

    def test_gqa_invalid_head_ratio_rejected(self):
        plan = _shared_plan()
        backend = TorchReferenceBackendBshd()
        q = torch.randn(2, 6, 3, 8)
        k = torch.randn(2, 6, 2, 8)
        v = torch.randn(2, 6, 2, 8)
        try:
            backend.attention(q, k, v, plan)
        except ValueError as exc:
            assert "multiple of kv heads" in str(exc)
        else:
            raise AssertionError("expected ValueError for 3 q heads vs 2 kv heads")

    def test_end_to_end_matches_per_row_reference(self):
        """plan → layout → build_kv → attention vs independent per-row reference.

        Reference semantics (absolute coordinates):
          provider row: causal attention over its own valid tokens [0, valid)
          reuser row:   suffix rows [prefix_len, valid) attend provider prefix KV
                        [0, prefix_len) + own suffix KV causally.
        """
        torch.manual_seed(7)
        plan = _shared_plan()
        backend = TorchReferenceBackendBshd()
        store = PrefixAttentionStore()
        B, S, H, D = 2, 6, 2, 8
        # Untrimmed padded inputs: row0 valid [0,5), row1 valid [0,6).
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, S, 1, D)
        v = torch.randn(B, S, 1, D)
        layout = BatchedBatchLayout(valid_lengths=[5, 6], seq_length=S)

        ek, ev = backend.build_kv(k, v, store, plan, packed_batch_layout=layout, layer_id=0)
        out = backend.attention(q, ek, ev, plan, packed_batch_layout=layout)

        # ── independent per-row reference ──
        def ref_row(q_rows, k_rows, v_rows):
            """q_rows [Tq,H,D], k_rows/v_rows [Tk,1,D], causal bottom-right."""
            Tk = k_rows.shape[0]
            Tq = q_rows.shape[0]
            offset = Tk - Tq  # q row j sees kv [0, offset + j]
            outs = []
            for j in range(Tq):
                kv_k = k_rows[: offset + j + 1]  # [n,1,D]
                kv_v = v_rows[: offset + j + 1]
                scores = torch.einsum("hd,nhd->hn", q_rows[j], kv_k) / math.sqrt(D)
                probs = torch.softmax(scores, dim=-1)
                outs.append(torch.einsum("hn,nhd->hd", probs, kv_v))
            return torch.stack(outs, dim=0)

        # Provider row 0: full causal over [0, 5).
        ref_provider = ref_row(q[0, :5], k[0, :5], v[0, :5])
        assert torch.allclose(out[0, :5], ref_provider, atol=1e-5), (
            f"provider mismatch: {(out[0, :5] - ref_provider).abs().max().item()}"
        )

        # Reuser row 1: suffix Q rows [3, 6) over expanded KV
        # [provider prefix k[0,:3]] + [own suffix k[1,3:6]].
        ref_k_exp = torch.cat([k[0, :3], k[1, 3:6]], dim=0)
        ref_v_exp = torch.cat([v[0, :3], v[1, 3:6]], dim=0)
        ref_reuser = ref_row(q[1, 3:6], ref_k_exp, ref_v_exp)
        assert torch.allclose(out[1, 3:6], ref_reuser, atol=1e-5), (
            f"reuser mismatch: {(out[1, 3:6] - ref_reuser).abs().max().item()}"
        )

    def test_end_to_end_chained_reuse(self):
        """Transitive chain: row2's attention must see row0+row1 KV correctly."""
        torch.manual_seed(11)
        plan = _chained_plan()
        backend = TorchReferenceBackendBshd()
        store = PrefixAttentionStore()
        B, S, H, D = 3, 6, 1, 8
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, S, H, D)
        v = torch.randn(B, S, H, D)
        layout = BatchedBatchLayout(valid_lengths=[5, 6, 6], seq_length=S)

        ek, ev = backend.build_kv(k, v, store, plan, packed_batch_layout=layout, layer_id=0)
        out = backend.attention(q, ek, ev, plan, packed_batch_layout=layout)

        # Row2 suffix Q rows [5, 6) (prefix_len=5): see [0,4) row0 + [4,5) row1 + self causal.
        # Suffix length is 1 → only row 5, attends all 6 expanded KV columns.
        ref_scores = torch.einsum("hd,nhd->hn", q[2, 5], ek[2, :6]) / math.sqrt(D)
        ref_probs = torch.softmax(ref_scores, dim=-1)
        ref_out = torch.einsum("hn,nhd->hd", ref_probs, ev[2, :6])
        assert torch.allclose(out[2, 5], ref_out, atol=1e-5)
        # Prefix rows zeroed.
        assert torch.all(out[2, :5] == 0)
