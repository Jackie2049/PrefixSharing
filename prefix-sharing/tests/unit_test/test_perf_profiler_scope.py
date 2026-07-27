"""Unit tests for the v2.0 ProfilerScope (three-level perf profiling)."""

from __future__ import annotations

import csv
import json
import os
import time

import pytest

from prefix_sharing.tools.perf_profiler import (
    PerfProfiler,
    ProfilerScope,
    _is_per_layer_phase,
)


@pytest.fixture()
def perf_dir(tmp_path):
    return str(tmp_path / "perf_results")


def _make_scope(perf_dir: str, **kwargs) -> ProfilerScope:
    return ProfilerScope(perf_dir, step_id=0, memory_interval=0.01, **kwargs)


def test_per_layer_phase_name_detection():
    assert _is_per_layer_phase("attn.kv.l0")
    assert _is_per_layer_phase("attn.comp.l23")
    assert not _is_per_layer_phase("attn.kv")
    assert not _is_per_layer_phase("fwd")
    assert not _is_per_layer_phase("ps.plan")


def test_create_if_enabled_disabled_by_default(monkeypatch):
    monkeypatch.delenv("PREFIX_SHARING_PERF_PROFILE", raising=False)
    monkeypatch.delenv("PREFIX_SHARING_PERF_DIR", raising=False)
    assert ProfilerScope.create_if_enabled(0) is None


def test_create_if_enabled_dir_alone_enables(perf_dir, monkeypatch):
    """Setting PREFIX_SHARING_PERF_DIR alone (no PROFILE switch) enables profiling."""
    monkeypatch.delenv("PREFIX_SHARING_PERF_PROFILE", raising=False)
    monkeypatch.setenv("PREFIX_SHARING_PERF_DIR", perf_dir)
    scope = ProfilerScope.create_if_enabled(0)
    assert scope is not None
    assert scope.perf_dir == perf_dir


def test_create_if_enabled_reads_env(perf_dir, monkeypatch):
    monkeypatch.setenv("PREFIX_SHARING_PERF_PROFILE", "1")
    monkeypatch.setenv("PREFIX_SHARING_PERF_DIR", perf_dir)
    monkeypatch.setenv("PREFIX_SHARING_PERF_MEMORY_INTERVAL", "0.01")
    scope = ProfilerScope.create_if_enabled(3, kind="logp")
    assert scope is not None
    assert scope.step_id == 3
    assert scope.kind == "logp"
    assert scope.perf_dir == perf_dir
    assert scope.per_layer_enabled is True

    monkeypatch.setenv("PREFIX_SHARING_PERF_PER_LAYER", "0")
    scope = ProfilerScope.create_if_enabled(0)
    assert scope is not None
    assert scope.per_layer_enabled is False


def test_scope_sets_current_and_resets(perf_dir):
    assert ProfilerScope.current() is None
    scope = _make_scope(perf_dir)
    with scope:
        assert ProfilerScope.current() is scope
        # legacy accessor sees the scope too (shared ContextVar)
        assert PerfProfiler.current() is scope
    assert ProfilerScope.current() is None


def test_micro_batch_aggregation_and_csv(perf_dir):
    scope = _make_scope(perf_dir)
    with scope:
        scope.begin_minibatch(0)
        for micro_idx in range(2):
            scope.begin_micro_batch(micro_idx)
            scope.start_phase(PerfProfiler.PHASE_FORWARD)
            time.sleep(0.001)
            scope.stop_phase(PerfProfiler.PHASE_FORWARD)
            scope.start_phase(PerfProfiler.PHASE_BACKWARD)
            scope.stop_phase(PerfProfiler.PHASE_BACKWARD)
            scope.record_per_layer(0, "attn.kv", 0.001)
            scope.record_per_layer(0, "attn.comp", 0.002)
            scope.record_per_layer(1, "attn.kv", 0.003)
            scope.end_micro_batch()
        # mini-batch level phase (outside any micro-batch)
        scope.start_phase(PerfProfiler.PHASE_UPDATE)
        scope.stop_phase(PerfProfiler.PHASE_UPDATE)
        scope.end_minibatch()

    step_dir = os.path.join(perf_dir, "step_0")
    timing_csv = os.path.join(step_dir, "microbatch_timing_train.rank0.csv")
    per_layer_csv = os.path.join(step_dir, "per_layer_attention_train.rank0.csv")
    summary_json = os.path.join(step_dir, "summary_train.rank0.json")

    # ── microbatch_timing.csv ──
    with open(timing_csv, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    assert header[:4] == ["step_id", "mini_batch_idx", "micro_batch_idx", "forward_only"]
    assert "fwd_ms" in header and "bwd_ms" in header
    assert len(rows) == 3  # header + 2 micro-batches
    assert rows[1][1] == "0" and rows[2][2] == "1"
    assert float(rows[1][header.index("fwd_ms")]) > 0.0

    # ── per_layer_attention.csv ──
    with open(per_layer_csv, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["step_id", "mini_batch_idx", "micro_batch_idx", "layer_id", "phase", "duration_ms"]
    # 2 micro-batches × (layer0: kv+comp, layer1: kv) = 6 rows
    assert len(rows) == 7

    # ── summary.json ──
    with open(summary_json, encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["step_id"] == 0
    assert summary["kind"] == "train"
    assert summary["num_mini_batches"] == 1
    assert summary["num_micro_batches"] == 2
    mb = summary["mini_batch_stats"][0]
    assert mb["mini_batch_idx"] == 0
    assert "fwd" in mb["micro_batch_timing"]
    assert set(mb["micro_batch_timing"]["fwd"]) >= {"min_ms", "avg_ms", "max_ms", "p50_ms", "p99_ms"}
    assert "update" in mb["minibatch_phase_ms"]
    pl = summary["per_layer_attention_summary"]
    # cross-layer statistics: avg/min/max over the layer axis
    kv_stats = pl["phases"]["attn.kv"]
    assert kv_stats["num_layers"] == 2
    # layer0: 0.001s x 2 micro-batches = 2ms; layer1: 0.003s x 2 = 6ms
    assert kv_stats["total_ms_per_layer"] == {"0": 2.0, "1": 6.0}
    assert kv_stats["total_avg_ms_per_layer"] == {"0": 1.0, "1": 3.0}
    assert kv_stats["avg_layer_ms"] == 4.0
    assert kv_stats["min_layer"] == {"layer_id": 0, "total_ms": 2.0}
    assert kv_stats["max_layer"] == {"layer_id": 1, "total_ms": 6.0}
    comp_stats = pl["phases"]["attn.comp"]
    # layer0: 0.002s x 2 = 4ms; layer1: no samples = 0
    assert comp_stats["total_ms_per_layer"] == {"0": 4.0, "1": 0.0}
    assert comp_stats["total_avg_ms_per_layer"] == {"0": 2.0, "1": 0.0}
    assert comp_stats["max_layer"]["layer_id"] == 0

    # ── micro_batch_memory (per-micro-batch peak + avg) ──
    mb_stat = summary["mini_batch_stats"][0]
    mbm = mb_stat["micro_batch_memory"]
    assert set(mbm.keys()) == {"0", "1"}
    for key in ("0", "1"):
        entry = mbm[key]
        assert set(entry.keys()) == {
            "peak_allocated_gib", "peak_reserved_gib",
            "avg_allocated_gib", "avg_reserved_gib", "num_samples",
        }


def test_per_layer_phase_name_detection_v2():
    """attn.on / attn.off per-layer names are also detected and excluded from timing."""
    assert _is_per_layer_phase("attn.on.l0")
    assert _is_per_layer_phase("attn.off.l23")
    assert not _is_per_layer_phase("attn.on")
    assert not _is_per_layer_phase("attn.off")


def test_per_layer_disabled(perf_dir):
    scope = _make_scope(perf_dir, per_layer=False)
    with scope:
        scope.begin_micro_batch(0)
        scope.record_per_layer(0, "attn.kv", 0.001)
        scope.end_micro_batch()
    per_layer_csv = os.path.join(perf_dir, "step_0", "per_layer_attention_train.rank0.csv")
    assert not os.path.exists(per_layer_csv)


def test_train_and_logp_scopes_share_step_without_clobbering(perf_dir):
    """logp and train scopes write distinct artifacts in the same step directory."""
    for kind, forward_only in (("logp", True), ("train", False)):
        scope = _make_scope(perf_dir, kind=kind)
        with scope:
            scope.begin_micro_batch(0, forward_only=forward_only)
            scope.start_phase(PerfProfiler.PHASE_FORWARD)
            scope.stop_phase(PerfProfiler.PHASE_FORWARD)
            scope.end_micro_batch()

    logp_csv = os.path.join(perf_dir, "step_0", "microbatch_timing_logp.rank0.csv")
    train_csv = os.path.join(perf_dir, "step_0", "microbatch_timing_train.rank0.csv")
    with open(logp_csv, newline="") as f:
        logp_rows = list(csv.reader(f))
    with open(train_csv, newline="") as f:
        train_rows = list(csv.reader(f))
    assert len(logp_rows) == 2 and logp_rows[1][3] == "1"
    assert len(train_rows) == 2 and train_rows[1][3] == "0"

    # per-kind summaries do not clobber each other
    assert os.path.exists(os.path.join(perf_dir, "step_0", "summary_logp.rank0.json"))
    assert os.path.exists(os.path.join(perf_dir, "step_0", "summary_train.rank0.json"))


def test_repeated_scope_same_step_replaces_stale_artifact(perf_dir):
    """Re-running a step replaces its per-kind CSV instead of duplicating rows."""
    for _ in range(2):
        scope = _make_scope(perf_dir, kind="train")
        with scope:
            scope.begin_micro_batch(0)
            scope.start_phase(PerfProfiler.PHASE_FORWARD)
            scope.stop_phase(PerfProfiler.PHASE_FORWARD)
            scope.end_micro_batch()

    timing_csv = os.path.join(perf_dir, "step_0", "microbatch_timing_train.rank0.csv")
    with open(timing_csv, newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2  # header + current run only


def test_exception_inside_scope_still_saves(perf_dir):
    scope = _make_scope(perf_dir)
    with pytest.raises(RuntimeError):
        with scope:
            scope.begin_micro_batch(0)
            scope.start_phase(PerfProfiler.PHASE_FORWARD)
            scope.stop_phase(PerfProfiler.PHASE_FORWARD)
            scope.end_micro_batch()
            raise RuntimeError("boom")
    assert ProfilerScope.current() is None
    assert os.path.exists(os.path.join(perf_dir, "step_0", "summary_train.rank0.json"))


def test_create_if_enabled_reads_attn_memory_env(perf_dir, monkeypatch):
    monkeypatch.setenv("PREFIX_SHARING_PERF_PROFILE", "1")
    monkeypatch.setenv("PREFIX_SHARING_PERF_DIR", perf_dir)
    monkeypatch.setenv("PREFIX_SHARING_PERF_ATTN_MEMORY", "1")
    scope = ProfilerScope.create_if_enabled(0)
    assert scope is not None
    assert scope.attn_memory_enabled is True

    monkeypatch.setenv("PREFIX_SHARING_PERF_ATTN_MEMORY", "0")
    scope = ProfilerScope.create_if_enabled(0)
    assert scope is not None
    assert scope.attn_memory_enabled is False


def test_attention_memory_disabled_by_default(perf_dir):
    """No attention_memory CSV is produced when attn_memory is disabled."""
    scope = _make_scope(perf_dir, attn_memory=False)
    with scope:
        scope.begin_minibatch(0)
        scope.begin_micro_batch(0)
        scope.start_phase(PerfProfiler.PHASE_ATTN_ON)
        scope.stop_phase(PerfProfiler.PHASE_ATTN_ON)
        scope.end_micro_batch()
        scope.end_minibatch()

    attn_mem_csv = os.path.join(perf_dir, "step_0", "attention_memory_train.rank0.csv")
    assert not os.path.exists(attn_mem_csv)


def test_attention_memory_csv_and_summary(perf_dir):
    """Attention memory sampling produces CSV rows and summary entries from tagged samples."""
    from prefix_sharing.tools.training_monitor import MemoryMonitor, MemorySnapshot

    scope = _make_scope(perf_dir, attn_memory=True)
    with scope:
        scope.begin_minibatch(0)
        scope.begin_micro_batch(0)
        scope.start_phase(PerfProfiler.PHASE_ATTN_ON)
        # Simulate background samples that fall inside the attn.on window.
        scope._monitor._samples.extend([
            MemorySnapshot(1.0, 1.0, 1.5, phase="attn.on"),
            MemorySnapshot(1.1, 2.0, 2.5, phase="attn.on"),
        ])
        scope.stop_phase(PerfProfiler.PHASE_ATTN_ON)
        scope.end_micro_batch()
        scope.end_minibatch()

    step_dir = os.path.join(perf_dir, "step_0")
    attn_mem_csv = os.path.join(step_dir, "attention_memory_train.rank0.csv")
    summary_json = os.path.join(step_dir, "summary_train.rank0.json")

    assert os.path.exists(attn_mem_csv)
    with open(attn_mem_csv, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == [
        "step_id", "mini_batch_idx", "micro_batch_idx", "phase",
        "avg_allocated_gb", "peak_allocated_gb",
        "avg_reserved_gb", "peak_reserved_gb", "num_samples",
    ]
    assert len(rows) == 2
    assert rows[1][3] == "attn.on"
    assert float(rows[1][4]) == 1.5  # avg allocated
    assert float(rows[1][5]) == 2.0  # peak allocated
    assert float(rows[1][6]) == 2.0  # avg reserved
    assert float(rows[1][7]) == 2.5  # peak reserved
    assert int(rows[1][8]) == 2      # num_samples

    with open(summary_json, encoding="utf-8") as f:
        summary = json.load(f)
    attn_summary = summary["attention_memory_summary"]
    assert "attn.on" in attn_summary
    assert attn_summary["attn.on"]["count"] == 1
    assert attn_summary["attn.on"]["avg_allocated_gb"] == 1.5
    assert attn_summary["attn.on"]["peak_allocated_gb"] == 2.0


def test_attention_memory_filters_untagged_samples(perf_dir):
    """Only samples tagged with the active phase contribute to attention memory stats."""
    from prefix_sharing.tools.training_monitor import MemorySnapshot

    scope = _make_scope(perf_dir, attn_memory=True)
    with scope:
        scope.begin_minibatch(0)
        scope.begin_micro_batch(0)
        scope.start_phase(PerfProfiler.PHASE_ATTN_OFF)
        scope._monitor._samples.extend([
            MemorySnapshot(1.0, 1.0, 1.0, phase="attn.off"),
            MemorySnapshot(1.1, 9.0, 9.0, phase=""),
            MemorySnapshot(1.2, 2.0, 2.0, phase="attn.off"),
        ])
        scope.stop_phase(PerfProfiler.PHASE_ATTN_OFF)
        scope.end_micro_batch()
        scope.end_minibatch()

    attn_mem_csv = os.path.join(perf_dir, "step_0", "attention_memory_train.rank0.csv")
    with open(attn_mem_csv, newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2
    # Untagged 9.0 sample should be ignored.
    assert float(rows[1][4]) == 1.5  # avg allocated = (1+2)/2
    assert float(rows[1][5]) == 2.0  # peak allocated
    assert int(rows[1][8]) == 2      # num_samples


def test_memory_trace_csv_tags_batch_idx(tmp_path):
    """memory_trace CSV carries mini/micro-batch and phase columns for per-micro-batch analysis."""
    from prefix_sharing.tools.training_monitor import MemoryMonitor, MemorySnapshot

    mon = MemoryMonitor(interval=0.01)
    mon._samples = [
        MemorySnapshot(1.0, 1.5, 2.0, mini_batch_idx=0, micro_batch_idx=0, phase="attn.on"),
        MemorySnapshot(2.0, 1.6, 2.0, mini_batch_idx=0, micro_batch_idx=-1, phase=""),
    ]
    path = str(tmp_path / "memory_trace.csv")
    mon.save_to_csv(path)
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["timestamp", "mini_batch_idx", "micro_batch_idx", "phase", "allocated_gb", "reserved_gb"]
    assert rows[1] == ["1.0", "0", "0", "attn.on", "1.5", "2.0"]
    assert rows[2] == ["2.0", "0", "-1", "", "1.6", "2.0"]
