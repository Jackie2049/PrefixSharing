"""Tests for diagnostic_dump DP-size / DP-rank detection (FSDP vs Megatron).

Regression guard: ``get_megatron_parallel_info()`` returns a default object
(tp=1, pp=1) even when Megatron mpu is not initialized (pure FSDP), so the DP
detector must not rely on ``parallel_info is not None`` — it would force
dp_size=1 and lose per-rank shards on multi-GPU FSDP.
"""

import prefix_sharing.tools.diagnostic_dump as dd


def _make_parallel_info(tp_size=1, pp_size=1, tp_rank=0):
    return type(
        "PI",
        (),
        {"tp_size": tp_size, "pp_size": pp_size, "tp_rank": tp_rank},
    )()


def test_is_megatron_parallel():
    assert not dd._is_megatron_parallel(None)
    assert not dd._is_megatron_parallel(_make_parallel_info(1, 1))   # FSDP / pure DP
    assert dd._is_megatron_parallel(_make_parallel_info(2, 1))       # Megatron TP
    assert dd._is_megatron_parallel(_make_parallel_info(1, 2))       # Megatron PP


def test_get_dp_size_fsdp(monkeypatch):
    """FSDP (tp=1, pp=1): dp_size = torch.distributed world_size, not 1."""
    monkeypatch.setattr(dd, "_DP_SIZE_CACHE", None)
    monkeypatch.setattr(dd, "_cached_parallel_info", lambda: _make_parallel_info(1, 1))
    monkeypatch.setattr(dd.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(dd.torch.distributed, "get_world_size", lambda: 2)
    assert dd._get_dp_size() == 2


def test_get_dp_size_megatron_tp(monkeypatch):
    """Megatron TP>1: dp_size = 1 (DP data not shard-visible at the TP-rank-0 writer)."""
    monkeypatch.setattr(dd, "_DP_SIZE_CACHE", None)
    monkeypatch.setattr(dd, "_cached_parallel_info", lambda: _make_parallel_info(2, 1))
    monkeypatch.setattr(dd.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(dd.torch.distributed, "get_world_size", lambda: 8)
    assert dd._get_dp_size() == 1


def test_get_dp_rank_fsdp(monkeypatch):
    """FSDP: dp_rank = torch.distributed rank (each GPU gets its own shard suffix)."""
    monkeypatch.setattr(dd, "_DP_RANK_CACHE", None)
    monkeypatch.setattr(dd, "_cached_parallel_info", lambda: _make_parallel_info(1, 1))
    monkeypatch.setattr(dd.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(dd.torch.distributed, "get_rank", lambda: 1)
    assert dd._get_dp_rank() == 1


def test_dump_weight_grads(tmp_path, monkeypatch):
    """``dump_weight_grads_verl080`` saves every parameter gradient to a single .pt file."""
    import os

    import torch

    monkeypatch.setenv("PREFIX_SHARING_DIAG_DUMP", str(tmp_path))
    monkeypatch.setattr(dd, "_DUMP_DIR", None)

    model = torch.nn.Linear(4, 2)
    loss = model(torch.randn(3, 4)).sum()
    loss.backward()

    dd.dump_weight_grads_verl080(model, tag="train")

    dump_path = tmp_path / "weight_grads_train.pt"
    assert dump_path.exists()
    grads = torch.load(dump_path, map_location="cpu", weights_only=True)
    assert "weight" in grads
    assert "bias" in grads
    assert grads["weight"].shape == (2, 4)
    assert grads["bias"].shape == (2,)


def test_cmp_weight_grads(tmp_path):
    """``cmp_weight_grads.compare`` reports cosine similarity per parameter."""
    import torch
    from prefix_sharing.tools.cmp_weight_grads import compare

    on_dir = tmp_path / "on"
    off_dir = tmp_path / "off"
    on_dir.mkdir()
    off_dir.mkdir()

    torch.save(
        {"weight": torch.ones(2, 4), "bias": torch.ones(2)},
        on_dir / "weight_grads_train.pt",
    )
    torch.save(
        {"weight": torch.ones(2, 4), "bias": torch.ones(2)},
        off_dir / "weight_grads_train.pt",
    )

    ret = compare(str(on_dir), str(off_dir), tag="train")
    assert ret == 0

    # Mismatched gradients should fail the default threshold.
    torch.save(
        {"weight": -torch.ones(2, 4), "bias": torch.ones(2)},
        off_dir / "weight_grads_train.pt",
    )
    ret = compare(str(on_dir), str(off_dir), tag="train")
    assert ret == 1
