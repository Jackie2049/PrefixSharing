from __future__ import annotations

import torch

from prefix_sharing.tools import cmp_diag_verl080 as cmp


def _save_metadata(directory, *, prefix_lens, cu_seqlens):
    torch.save(torch.tensor(prefix_lens), directory / "prefix_lens.pt")
    tensor = torch.tensor(cu_seqlens)
    torch.save(tensor, directory / "cu_seqlens_q.pt")
    torch.save(tensor, directory / "cu_seqlens_q_logits.pt")


def test_all_layer_attention_comparison_fails_for_a_diverging_layer(tmp_path):
    on_dir, off_dir = tmp_path / "on", tmp_path / "off"
    on_dir.mkdir()
    off_dir.mkdir()
    _save_metadata(on_dir, prefix_lens=[0], cu_seqlens=[0, 2])
    _save_metadata(off_dir, prefix_lens=[0], cu_seqlens=[0, 2])
    torch.save({1: torch.tensor([[1.0, 0.0], [0.0, 1.0]])}, on_dir / "attn_outputs.pt")
    torch.save({1: torch.tensor([[0.0, 1.0], [1.0, 0.0]])}, off_dir / "attn_outputs.pt")

    result = cmp.cmp_attn_layer(str(on_dir), str(off_dir), layer=None)

    assert result is not None
    assert not result.passed


def test_reuser_first_suffix_positions_use_trimmed_and_full_coordinates():
    on_meta = {"prefix_lens": torch.tensor([0, 4]), "cu_seqlens": torch.tensor([0, 8, 12])}
    off_meta = {"prefix_lens": torch.tensor([0, 0]), "cu_seqlens": torch.tensor([0, 8, 16])}

    assert cmp._reuser_first_suffix_positions(on_meta, off_meta) == [(1, 8, 12)]


def test_input_ids_preflight_fails_when_replay_batch_differs(tmp_path):
    on_dir, off_dir = tmp_path / "on", tmp_path / "off"
    on_dir.mkdir()
    off_dir.mkdir()
    torch.save(torch.tensor([[1, 2]]), on_dir / "input_ids_train.pt")
    torch.save(torch.tensor([[1, 3]]), off_dir / "input_ids_train.pt")

    result = cmp.cmp_input_ids(str(on_dir), str(off_dir), "train")

    assert not result.passed
    assert result.metrics["different_tokens"] == 1
