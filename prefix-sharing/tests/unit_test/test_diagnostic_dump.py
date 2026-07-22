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
