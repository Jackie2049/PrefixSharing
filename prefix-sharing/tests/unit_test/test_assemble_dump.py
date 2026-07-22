"""Tests for assemble_dump.py — multi-rank (DP/TP) shard assembly."""

import json
import os

import torch
import pytest

from prefix_sharing.tools.assemble_dump import assemble


def _write_manifest(dir_path: str, *, tp: int = 1, pp: int = 1, dp: int = 1) -> None:
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, "parallel_info.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "tp_size": tp,
                "pp_size": pp,
                "cp_size": 1,
                "dp_size": dp,
                "global_rank_of_dumper": 0,
                "scopes": {"logits": "tp_vocab", "attn_outputs": "pp_stage"},
            },
            f,
        )


def test_dp_logits_concat(tmp_path):
    """DP packed logits [1, N_rank, V] merge on the token axis (dim 1)."""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_manifest(str(raw), dp=2)

    log0 = torch.randn(1, 5, 8)
    log1 = torch.randn(1, 3, 8)
    torch.save(log0, str(raw / "logits_dp0.pt"))
    torch.save(log1, str(raw / "logits_dp1.pt"))

    assemble(str(raw), str(out))

    merged = torch.load(str(out / "logits.pt"), weights_only=True)
    assert merged.shape == (1, 8, 8)
    assert torch.allclose(merged, torch.cat([log0, log1], dim=1))


def test_dp_logits_rejects_non_packed_shape(tmp_path):
    """DP logits must preserve the single-card [1, N, V] dump contract."""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_manifest(str(raw), dp=2)
    torch.save(torch.randn(5, 8), str(raw / "logits_dp0.pt"))
    torch.save(torch.randn(3, 8), str(raw / "logits_dp1.pt"))

    with pytest.raises(ValueError, match=r"must have shape \[1, N, V\]"):
        assemble(str(raw), str(out))


def test_single_card_fast_path(tmp_path):
    """DP=TP=PP=1: all files copied verbatim."""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_manifest(str(raw), dp=1)
    log = torch.randn(4, 8)
    torch.save(log, str(raw / "logits.pt"))

    assemble(str(raw), str(out))

    merged = torch.load(str(out / "logits.pt"), weights_only=True)
    assert torch.allclose(merged, log)


def test_tp_logits_concat(tmp_path):
    """TP>1: logits_tp{t}.pt concatenated on the vocab axis (dim -1)."""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_manifest(str(raw), tp=2)
    torch.save(torch.randn(4, 6), str(raw / "logits_tp0.pt"))
    torch.save(torch.randn(4, 6), str(raw / "logits_tp1.pt"))

    assemble(str(raw), str(out))

    merged = torch.load(str(out / "logits.pt"), weights_only=True)
    assert merged.shape == (4, 12)


def test_dp_nested_per_layer_dict_merge(tmp_path):
    """Nested values merge recursively without blocking later DP artifacts."""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_manifest(str(raw), dp=2)
    rank0 = {
        1: {"query": torch.randn(3, 2, 4), "key": torch.randn(3, 2, 4), "positions": None},
        2: {"query": torch.randn(3, 2, 4), "key": torch.randn(3, 2, 4), "positions": None},
    }
    rank1 = {
        1: {"query": torch.randn(5, 2, 4), "key": torch.randn(5, 2, 4), "positions": None},
        2: {"query": torch.randn(5, 2, 4), "key": torch.randn(5, 2, 4), "positions": None},
    }
    grad0 = {1: torch.randn(3, 2, 4)}
    grad1 = {1: torch.randn(5, 2, 4)}
    logits0 = torch.randn(1, 3, 8)
    logits1 = torch.randn(1, 5, 8)
    torch.save(rank0, str(raw / "rope_postqk_dp0.pt"))
    torch.save(rank1, str(raw / "rope_postqk_dp1.pt"))
    torch.save(grad0, str(raw / "attn_grads_dp0.pt"))
    torch.save(grad1, str(raw / "attn_grads_dp1.pt"))
    torch.save(logits0, str(raw / "logits_dp0.pt"))
    torch.save(logits1, str(raw / "logits_dp1.pt"))

    assemble(str(raw), str(out))

    merged = torch.load(str(out / "rope_postqk.pt"), weights_only=True)
    assert merged[1]["query"].shape == (8, 2, 4)
    assert merged[1]["key"].shape == (8, 2, 4)
    assert merged[1]["positions"] is None
    assert torch.allclose(merged[2]["query"], torch.cat([rank0[2]["query"], rank1[2]["query"]], dim=0))

    merged_grads = torch.load(str(out / "attn_grads.pt"), weights_only=True)
    assert torch.allclose(merged_grads[1], torch.cat([grad0[1], grad1[1]], dim=0))
    merged_logits = torch.load(str(out / "logits.pt"), weights_only=True)
    assert torch.allclose(merged_logits, torch.cat([logits0, logits1], dim=1))


def test_dp_cu_seqlens_offset_adjust(tmp_path):
    """DP cu_seqlens: per-rank [0,...] concatenated with cumulative offset.

    rank0 [0,3,8] (seqs 3,5) + rank1 [0,2,6] (seqs 2,4) → [0,3,8,10,14].
    """
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_manifest(str(raw), dp=2)
    torch.save(torch.tensor([0, 3, 8]), str(raw / "cu_seqlens_q_dp0.pt"))
    torch.save(torch.tensor([0, 2, 6]), str(raw / "cu_seqlens_q_dp1.pt"))

    assemble(str(raw), str(out))

    merged = torch.load(str(out / "cu_seqlens_q.pt"), weights_only=True)
    assert merged.tolist() == [0, 3, 8, 10, 14]


def test_dp_cu_seqlens_three_ranks(tmp_path):
    """Three DP ranks: every intermediate boundary must survive the merge."""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_manifest(str(raw), dp=3)
    torch.save(torch.tensor([0, 3, 8]), str(raw / "cu_seqlens_q_dp0.pt"))   # seqs 3,5
    torch.save(torch.tensor([0, 2, 6]), str(raw / "cu_seqlens_q_dp1.pt"))   # seqs 2,4
    torch.save(torch.tensor([0, 1, 4]), str(raw / "cu_seqlens_q_dp2.pt"))   # seqs 1,3

    assemble(str(raw), str(out))

    merged = torch.load(str(out / "cu_seqlens_q.pt"), weights_only=True)
    assert merged.tolist() == [0, 3, 8, 10, 14, 15, 18]
