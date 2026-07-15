"""Performance profiler for PrefixSharing FSDP path.

Reuses ``MemoryMonitor`` and ``Stopwatch`` from ``training_monitor.py``,
adding FSDP-specific phase definitions, a ContextVar-based accessor, and
unified save/summary methods.

Usage::

    from prefix_sharing.tools.perf_profiler import PerfProfiler

    profiler = PerfProfiler(memory_interval=0.05)
    profiler.start_memory()

    profiler.start_phase(PerfProfiler.PHASE_PLAN)
    ... detect + plan + trim ...
    profiler.stop_phase(PerfProfiler.PHASE_PLAN)

    profiler.start_phase(PerfProfiler.PHASE_FORWARD)
    ... model forward ...
    profiler.stop_phase(PerfProfiler.PHASE_FORWARD)

    profiler.stop_memory()

    profiler.save("/path/to/perf_results")
    print(profiler.summary())

Activation: set env ``PREFIX_SHARING_PERF_PROFILE=1``.  When disabled,
all public methods are safe no-ops.
"""

from __future__ import annotations

import json
import os
import threading
from contextvars import ContextVar
from typing import Any

from prefix_sharing.tools.training_monitor import (
    MemoryMonitor,
    MemorySnapshot,
    Stopwatch,
)

# ── ContextVar for zero-signature-change access ──────────────────
_profiler_context: ContextVar[PerfProfiler | None] = ContextVar(
    "perf_profiler_context",
    default=None,
)


class PerfProfiler:
    """Unified performance profiler for PrefixSharing FSDP micro-batches.

    Wraps ``MemoryMonitor`` (interval HBM sampling) and ``Stopwatch``
    (hierarchical phase timing).  Use ``PerfProfiler.current()`` to
    obtain the active instance from anywhere on the call stack.
    """

    # ── Phase name constants ─────────────────────────────────────
    PHASE_PLAN = "ps.plan"            # detect + plan + batch trim (CPU)
    PHASE_FORWARD = "fwd"             # model forward (includes attention)
    PHASE_BACKWARD = "bwd"            # model backward (via autograd)
    PHASE_ATTN_KV = "attn.kv"         # build_kv: store provider + load reuser
    PHASE_ATTN_COMPUTE = "attn.comp"  # attention computation (FA kernel)
    PHASE_RESTORE = "ps.restore"      # output logprob/entropy restore (CPU)
    PHASE_LOSS = "loss"               # loss function computation

    def __init__(
        self,
        memory_interval: float = 0.05,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._memory_monitor = MemoryMonitor(interval=memory_interval) if enabled else None
        self._stopwatch = Stopwatch() if enabled else None
        self._micro_batch_count = 0
        self._lock = threading.Lock()

    # ── public API ───────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def memory_monitor(self) -> MemoryMonitor | None:
        return self._memory_monitor

    @property
    def stopwatch(self) -> Stopwatch | None:
        return self._stopwatch

    # -- memory ----------------------------------------------------

    def start_memory(self) -> None:
        """Begin background HBM sampling."""
        if self._memory_monitor is not None:
            self._memory_monitor.start()

    def stop_memory(self) -> None:
        """Stop background HBM sampling and join the thread."""
        if self._memory_monitor is not None:
            self._memory_monitor.stop()

    def snapshot_memory(self) -> MemorySnapshot | None:
        """One-shot memory query (does not start sampling loop)."""
        if self._memory_monitor is not None:
            return self._memory_monitor.snapshot()
        return None

    # -- timing ----------------------------------------------------

    def start_phase(self, name: str) -> None:
        """Begin timing *name*."""
        if self._stopwatch is not None:
            self._stopwatch.start(name)

    def stop_phase(self, name: str) -> float:
        """End timing *name*, record a sample, return elapsed seconds."""
        if self._stopwatch is not None:
            return self._stopwatch.stop(name)
        return 0.0

    def lap(self, name: str = "") -> None:
        """Record an instantaneous checkpoint."""
        if self._stopwatch is not None:
            self._stopwatch.lap(name)

    # -- lifecycle -------------------------------------------------

    def begin_micro_batch(self) -> None:
        """Call at the start of each micro-batch.

        Starts memory sampling and resets stopwatch samples (but keeps
        accumulated counters so per-step or cumulative queries work).
        """
        with self._lock:
            self._micro_batch_count += 1
        if self._stopwatch is not None:
            self._stopwatch.reset()

    def end_micro_batch(self) -> None:
        """Call at the end of each micro-batch.

        Stops memory sampling so that the captured trace covers exactly
        the micro-batch time window.
        """
        if self._memory_monitor is not None:
            self._memory_monitor.stop()

    # -- summary / save --------------------------------------------

    def summary(self) -> dict:
        """Aggregate memory + timing stats.

        Returns::

            {
                "memory": {"peak_allocated_gib": ..., "avg_allocated_gib": ...,
                           "peak_reserved_gib": ..., "avg_reserved_gib": ...,
                           "num_samples": ...},
                "timing": {"phases": {"ps.plan": {"total_s":..., "count":..., "avg_s":...}, ...}}
            }
        """
        memory_summary: dict = {}
        if self._memory_monitor is not None:
            memory_summary = self._memory_monitor.summary()
        timing_summary: dict = {}
        if self._stopwatch is not None:
            timing_summary = self._stopwatch.summary()
        return {
            "memory": memory_summary,
            "timing": timing_summary,
            "micro_batch_count": self._micro_batch_count,
        }

    def save(self, dir_path: str, tag: str = "") -> None:
        """Save memory CSV, timing CSV, and summary JSON to *dir_path*.

        Uses *tag* to distinguish ON vs OFF runs, e.g. ``tag="ps_on"``.

        Produces:
          - ``memory_trace{_tag}.csv``
          - ``timing_trace{_tag}.csv``
          - ``summary{_tag}.json``
        """
        os.makedirs(dir_path, exist_ok=True)
        suffix = f"_{tag}" if tag else ""

        # Summary JSON
        summary_path = os.path.join(dir_path, f"summary{suffix}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2, ensure_ascii=False)
        print(f"[PerfProfiler] summary saved to {summary_path}")

        # Memory CSV (only if samples exist)
        if self._memory_monitor is not None and self._memory_monitor._samples:
            try:
                self._memory_monitor.save_to_csv(
                    os.path.join(dir_path, f"memory_trace{suffix}.csv")
                )
            except RuntimeError:
                pass  # no samples, skip

        # Timing CSV
        if self._stopwatch is not None:
            sw = self._stopwatch
            if sw._samples:
                sw.save_to_csv(
                    os.path.join(dir_path, f"timing_trace{suffix}.csv")
                )

    # ── ContextVar access ─────────────────────────────────────────

    @staticmethod
    def current() -> PerfProfiler | None:
        """Return the active ``PerfProfiler`` for the current context.

        ``None`` when ``PREFIX_SHARING_PERF_PROFILE`` is not set or
        ``training_monitor_context`` has not been entered.
        """
        return _profiler_context.get()

    def __enter__(self) -> PerfProfiler:
        token = _profiler_context.set(self)
        self._context_token = token
        return self

    def __exit__(self, *args: Any) -> None:
        _profiler_context.reset(self._context_token)

    @staticmethod
    def create_if_enabled(
        memory_interval: float = 0.05,
    ) -> PerfProfiler | None:
        """Factory: return a PerfProfiler if ``PREFIX_SHARING_PERF_PROFILE=1``.

        Otherwise return ``None`` — callers can use ``if profiler:``
        guards throughout.
        """
        if os.environ.get("PREFIX_SHARING_PERF_PROFILE", "").strip() in (
            "1", "true", "True", "yes", "on",
        ):
            return PerfProfiler(memory_interval=memory_interval, enabled=True)
        return None
