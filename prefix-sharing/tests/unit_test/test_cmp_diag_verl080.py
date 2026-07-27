"""Precision comparison tests for verl080 diagnostic (cmp_diag.py)."""
from __future__ import annotations

import torch

from prefix_sharing.tools.cmp_diag import cmp_attn_layer
from prefix_sharing import diagnostics


def test_all_layer_attention_comparison_fails_for_a_diverging_layer(tmp_path):
    on_dir, off_dir = tmp_path / "on", tmp_path / "off"
    on_dir.mkdir()
    off_dir.mkdir()
    # save metadata needed by cmp_attn_layer
    for d in (on_dir, off_dir):
        torch.save(torch.tensor([0]), d / "prefix_lens.pt")
        torch.save(torch.tensor([0, 2]), d / "cu_seqlens_q.pt")
        torch.save(torch.tensor([0, 2]), d / "cu_seqlens_q_logits.pt")
    torch.save({1: torch.tensor([[1.0, 0.0], [0.0, 1.0]])}, on_dir / "attn_outputs.pt")
    torch.save({1: torch.tensor([[0.0, 1.0], [1.0, 0.0]])}, off_dir / "attn_outputs.pt")

    result = cmp_attn_layer(str(on_dir), str(off_dir), layer=None)

    assert result is not None
    assert not result.passed


def test_fsdp_attention_input_dump_is_disabled_without_env(monkeypatch):
    monkeypatch.delenv("PREFIX_SHARING_DIAG_DUMP", raising=False)
    module = type("Module", (), {"layer_idx": 0, "config": type("Config", (), {"num_hidden_layers": 1})()})()

    diagnostics.dump_fsdp_attention_inputs(
        torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2), module
    )

    assert diagnostics._FSDP_ATTN_INPUT_BUFFER == {}
