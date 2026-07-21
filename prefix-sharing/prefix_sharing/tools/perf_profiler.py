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

import csv
import json
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from prefix_sharing.tools.training_monitor import (
    MemoryMonitor,
    MemorySnapshot,
    Stopwatch,
)

# ── ContextVar for zero-signature-change access ──────────────────
# Holds either a legacy ``PerfProfiler`` or a v2.0 ``ProfilerScope``.
_profiler_context: ContextVar[Any] = ContextVar(
    "perf_profiler_context",
    default=None,
)

_PER_LAYER_NAME_SUFFIX = (".kv.l", ".comp.l")


class PerfProfiler:
    """Unified performance profiler for PrefixSharing FSDP micro-batches.

    Wraps ``MemoryMonitor`` (interval HBM sampling) and ``Stopwatch``
    (hierarchical phase timing).  Use ``PerfProfiler.current()`` to
    obtain the active instance from anywhere on the call stack.
    """

    # ── Phase name constants ─────────────────────────────────────
    PHASE_PLAN = "ps.plan"            # detect + plan + batch trim (CPU)
    PHASE_FORWARD = "fwd"             # model forward (training)
    PHASE_FORWARD_OLD = "fwd.old"     # model forward (old_logp / forward_only)
    PHASE_BACKWARD = "bwd"            # model backward (via autograd)
    PHASE_ATTN_KV = "attn.kv"         # build_kv: store provider + load reuser
    PHASE_ATTN_COMPUTE = "attn.comp"  # attention computation (FA kernel)
    PHASE_ATTN_PACK = "attn.pack"     # QKV pack dense→packed (cross-layer accumulated)
    PHASE_ATTN_UNPACK = "attn.unpack"  # output scatter packed→dense (cross-layer accumulated)
    PHASE_RESTORE = "ps.restore"      # output logprob/entropy restore (CPU)
    PHASE_LOSS = "loss"               # loss function computation
    PHASE_UPDATE = "update"           # optimizer step (mini-batch level)
    PHASE_ATTN_OFF = "attn.off"       # original HF attention compute (OFF / baseline)

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
    def current() -> Any:
        """Return the active profiler for the current context.

        May be a legacy ``PerfProfiler`` or a v2.0 ``ProfilerScope``
        (both expose ``start_phase``/``stop_phase``).  ``None`` when
        ``PREFIX_SHARING_PERF_PROFILE`` is not set or no profiler
        context has been entered.
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


# ────────────────────────────────────────────────────────────────
# v2.0 — ProfilerScope: three-level (step → mini-batch → micro-batch)
# lifecycle-managed profiler.
# ────────────────────────────────────────────────────────────────


def _is_per_layer_phase(name: str) -> bool:
    """True for per-layer phase names like ``attn.kv.l3`` / ``attn.comp.l12``."""
    return any(tag in name for tag in _PER_LAYER_NAME_SUFFIX)


@dataclass
class MicroBatchSnapshot:
    """Snapshot of a single micro-batch within a step."""

    step_id: int
    mini_batch_idx: int
    micro_batch_idx: int
    forward_only: bool
    timing: dict[str, float] = field(default_factory=dict)           # phase → elapsed_s (excl. per-layer names)
    per_layer: dict[int, dict[str, float]] = field(default_factory=dict)  # layer_id → {"attn.kv": s, "attn.comp": s}
    memory_peak_allocated_gb: float = 0.0
    memory_peak_reserved_gb: float = 0.0


class ProfilerScope:
    """Context manager bounding one profiled step (``with`` block).

    Replaces the v1.0 implicit ``start_memory``/``stop_memory``/``save``
    lifecycle: memory sampling starts on ``__enter__`` and artifacts are
    written under ``{perf_dir}/step_{step_id}/`` on ``__exit__``.

    Hierarchy (markers are inserted at the actual verl call sites):
      - step:        ``__enter__`` / ``__exit__``        (engine_workers.train_mini_batch / infer_batch)
      - mini-batch:  ``begin_minibatch`` / ``end_minibatch``
      - micro-batch: ``begin_micro_batch`` / ``end_micro_batch``  (forward_backward_batch)

    Phase timing outside an active micro-batch (e.g. ``update`` around
    ``optimizer_step``) is accumulated at mini-batch level.
    """

    # Re-export phase constants so call sites can use a single class.
    PHASE_PLAN = PerfProfiler.PHASE_PLAN
    PHASE_FORWARD = PerfProfiler.PHASE_FORWARD
    PHASE_FORWARD_OLD = PerfProfiler.PHASE_FORWARD_OLD
    PHASE_BACKWARD = PerfProfiler.PHASE_BACKWARD
    PHASE_ATTN_KV = PerfProfiler.PHASE_ATTN_KV
    PHASE_ATTN_COMPUTE = PerfProfiler.PHASE_ATTN_COMPUTE
    PHASE_ATTN_PACK = PerfProfiler.PHASE_ATTN_PACK
    PHASE_ATTN_UNPACK = PerfProfiler.PHASE_ATTN_UNPACK
    PHASE_RESTORE = PerfProfiler.PHASE_RESTORE
    PHASE_LOSS = PerfProfiler.PHASE_LOSS
    PHASE_UPDATE = PerfProfiler.PHASE_UPDATE
    PHASE_ATTN_OFF = PerfProfiler.PHASE_ATTN_OFF

    def __init__(
        self,
        perf_dir: str,
        step_id: int,
        *,
        kind: str = "train",
        memory_interval: float = 0.05,
        per_layer: bool = True,
        rank: int = 0,
    ) -> None:
        self.perf_dir = perf_dir
        self.step_id = step_id
        self.kind = kind  # "train" | "logp" — old_logp runs in a separate RPC call
        self.per_layer_enabled = per_layer
        self.rank = rank
        self._monitor = MemoryMonitor(interval=memory_interval)
        self._stopwatch = Stopwatch()
        self._snapshots: list[MicroBatchSnapshot] = []
        self._current_snapshot: MicroBatchSnapshot | None = None
        self._current_mb_idx = -1
        self._minibatch_phases: dict[int, dict[str, float]] = {}  # mb_idx → phase → elapsed_s
        self._per_layer: dict[int, dict[str, float]] = {}
        self._sample_cursor = 0
        self._step_start = 0.0
        self._total_elapsed_s = 0.0
        self._context_token: Any = None

    # ── context manager ──────────────────────────────────────────

    def __enter__(self) -> ProfilerScope:
        self._step_start = time.perf_counter()
        self._monitor.start()
        self._context_token = _profiler_context.set(self)
        return self

    def __exit__(self, *args: Any) -> None:
        # If the profiled region raised during a micro-batch, keep the partial
        # timing/memory data rather than silently dropping that last snapshot.
        if self._current_snapshot is not None:
            self.end_micro_batch()
        # Forced sample so the trace covers the window end (typically
        # right after the last backward).
        self._force_memory_sample()
        self._monitor.stop()
        if self._context_token is not None:
            _profiler_context.reset(self._context_token)
            self._context_token = None
        self._total_elapsed_s = time.perf_counter() - self._step_start
        try:
            self._save_step_artifacts()
        except Exception as exc:  # profiling must never break training
            print(f"[ProfilerScope] failed to save step artifacts: {exc}")

    # ── level markers ────────────────────────────────────────────

    def begin_minibatch(self, mini_batch_idx: int) -> None:
        self._current_mb_idx = mini_batch_idx
        self._minibatch_phases.setdefault(mini_batch_idx, {})

    def end_minibatch(self) -> None:
        pass  # mini-batch stats are aggregated at save time

    def begin_micro_batch(self, micro_batch_idx: int, *, forward_only: bool = False) -> None:
        self._stopwatch.reset()
        self._per_layer = {}
        self._sample_cursor = len(self._monitor._samples)
        self._current_snapshot = MicroBatchSnapshot(
            step_id=self.step_id,
            # logp scopes have no explicit begin_minibatch → default to 0
            mini_batch_idx=max(self._current_mb_idx, 0),
            micro_batch_idx=micro_batch_idx,
            forward_only=forward_only,
        )

    def end_micro_batch(self) -> None:
        snap = self._current_snapshot
        if snap is None:
            return
        # Forced snapshot at the window end (typically right after backward)
        # so short micro-batches still have a peak reading.
        window = self._monitor._samples[self._sample_cursor:]
        window.append(self._monitor.snapshot())
        snap.memory_peak_allocated_gb = max((s.allocated_gb for s in window), default=0.0)
        snap.memory_peak_reserved_gb = max((s.reserved_gb for s in window), default=0.0)
        snap.timing = {
            name: sum(values)
            for name, values in self._stopwatch._samples.items()
            if not _is_per_layer_phase(name)
        }
        snap.per_layer = self._per_layer
        self._snapshots.append(snap)
        self._current_snapshot = None

    # ── phase timing (same surface as PerfProfiler) ──────────────

    def start_phase(self, name: str) -> None:
        self._stopwatch.start(name)

    def stop_phase(self, name: str) -> float:
        elapsed = self._stopwatch.stop(name)
        if self._current_snapshot is None and self._current_mb_idx >= 0:
            # Outside a micro-batch (e.g. ``update``) → mini-batch level.
            phases = self._minibatch_phases.setdefault(self._current_mb_idx, {})
            phases[name] = phases.get(name, 0.0) + elapsed
        return elapsed

    def is_phase_active(self, name: str) -> bool:
        """Return whether *name* is currently running in this scope."""
        return name in self._stopwatch._running

    def record_per_layer(self, layer_id: int, phase: str, elapsed_s: float) -> None:
        if not self.per_layer_enabled:
            return
        per_layer = self._per_layer.setdefault(layer_id, {})
        per_layer[phase] = per_layer.get(phase, 0.0) + elapsed_s

    # ── memory lifecycle is scope-managed; keep the v1.0 method
    # surface so existing call sites (forward_step.py) are no-ops. ──

    def start_memory(self) -> None:
        pass

    def stop_memory(self) -> None:
        pass

    def snapshot_memory(self) -> MemorySnapshot:
        return self._monitor.snapshot()

    # ── ContextVar access ────────────────────────────────────────

    @staticmethod
    def current() -> ProfilerScope | None:
        """Return the active scope, or ``None`` outside a profiled step."""
        active = _profiler_context.get()
        return active if isinstance(active, ProfilerScope) else None

    @staticmethod
    def create_if_enabled(
        step_id: int,
        *,
        kind: str = "train",
        perf_dir: str | None = None,
    ) -> ProfilerScope | None:
        """Factory: return a ProfilerScope if ``PREFIX_SHARING_PERF_PROFILE=1``.

        Configuration via env:
          - ``PREFIX_SHARING_PERF_DIR`` — output root (default ``./perf_results``)
          - ``PREFIX_SHARING_PERF_MEMORY_INTERVAL`` — sampling interval s (default 0.05)
          - ``PREFIX_SHARING_PERF_PER_LAYER`` — ``0`` disables per-layer attention timing
        """
        if os.environ.get("PREFIX_SHARING_PERF_PROFILE", "").strip() not in (
            "1", "true", "True", "yes", "on",
        ):
            return None
        perf_dir = perf_dir or os.environ.get("PREFIX_SHARING_PERF_DIR", "./perf_results")
        try:
            interval = float(os.environ.get("PREFIX_SHARING_PERF_MEMORY_INTERVAL", "0.05"))
        except ValueError:
            interval = 0.05
        per_layer = os.environ.get("PREFIX_SHARING_PERF_PER_LAYER", "1").strip() not in ("0", "false", "False")
        return ProfilerScope(
            perf_dir,
            step_id,
            kind=kind,
            memory_interval=interval,
            per_layer=per_layer,
            rank=_resolve_rank(),
        )

    # ── persistence ──────────────────────────────────────────────

    def _force_memory_sample(self) -> None:
        try:
            self._monitor._samples.append(self._monitor.snapshot())
        except Exception:
            pass

    def _save_step_artifacts(self) -> None:
        out_dir = os.path.join(self.perf_dir, f"step_{self.step_id}")
        os.makedirs(out_dir, exist_ok=True)
        suffix = f"rank{self.rank}"
        self._save_microbatch_timing_csv(out_dir, suffix)
        self._save_per_layer_csv(out_dir, suffix)
        self._save_memory_csv(out_dir, suffix)
        self._save_step_summary_json(out_dir, suffix)
        self._update_latest_link()

    def _save_microbatch_timing_csv(self, out_dir: str, suffix: str) -> None:
        if not self._snapshots:
            return
        phases = [
            PerfProfiler.PHASE_PLAN,
            PerfProfiler.PHASE_FORWARD,
            PerfProfiler.PHASE_FORWARD_OLD,
            PerfProfiler.PHASE_ATTN_PACK,
            PerfProfiler.PHASE_ATTN_KV,
            PerfProfiler.PHASE_ATTN_COMPUTE,
            PerfProfiler.PHASE_ATTN_UNPACK,
            PerfProfiler.PHASE_ATTN_OFF,
            PerfProfiler.PHASE_BACKWARD,
            PerfProfiler.PHASE_RESTORE,
            PerfProfiler.PHASE_LOSS,
        ]
        path = os.path.join(out_dir, f"microbatch_timing_{self.kind}.{suffix}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["step_id", "mini_batch_idx", "micro_batch_idx", "forward_only"]
                + [f"{p}_ms" for p in phases]
                + ["peak_allocated_gb", "peak_reserved_gb"]
            )
            for snap in self._snapshots:
                writer.writerow(
                    [snap.step_id, snap.mini_batch_idx, snap.micro_batch_idx, int(snap.forward_only)]
                    + [round(snap.timing.get(p, 0.0) * 1e3, 3) for p in phases]
                    + [round(snap.memory_peak_allocated_gb, 3), round(snap.memory_peak_reserved_gb, 3)]
                )
        print(f"[ProfilerScope] wrote {len(self._snapshots)} micro-batch rows to {path}")

    def _save_per_layer_csv(self, out_dir: str, suffix: str) -> None:
        rows = [
            (snap, layer_id, phase, elapsed)
            for snap in self._snapshots
            for layer_id, phases in sorted(snap.per_layer.items())
            for phase, elapsed in phases.items()
        ]
        if not rows:
            return
        path = os.path.join(out_dir, f"per_layer_attention_{self.kind}.{suffix}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step_id", "mini_batch_idx", "micro_batch_idx", "layer_id", "phase", "duration_ms"])
            for snap, layer_id, phase, elapsed in rows:
                writer.writerow([
                    snap.step_id, snap.mini_batch_idx, snap.micro_batch_idx,
                    layer_id, phase, round(elapsed * 1e3, 3),
                ])
        print(f"[ProfilerScope] wrote {len(rows)} per-layer rows to {path}")

    def _save_memory_csv(self, out_dir: str, suffix: str) -> None:
        if not self._monitor._samples:
            return
        self._monitor.save_to_csv(os.path.join(out_dir, f"memory_trace_{self.kind}.{suffix}.csv"))

    def _save_step_summary_json(self, out_dir: str, suffix: str) -> None:
        path = os.path.join(out_dir, f"summary_{self.kind}.{suffix}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._build_summary(), f, indent=2, ensure_ascii=False)
        print(f"[ProfilerScope] summary saved to {path}")

    def _update_latest_link(self) -> None:
        link = os.path.join(self.perf_dir, "latest")
        target = f"step_{self.step_id}"
        try:
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(target, link)
        except OSError:
            pass  # Windows without symlink privilege — skip silently

    # ── aggregation ──────────────────────────────────────────────

    @staticmethod
    def _phase_stats(values_s: list[float]) -> dict:
        values_ms = sorted(v * 1e3 for v in values_s)
        n = len(values_ms)
        p50 = values_ms[n // 2] if n % 2 else (values_ms[n // 2 - 1] + values_ms[n // 2]) / 2
        p99 = values_ms[min(n - 1, max(0, int(n * 0.99) - 1))]
        return {
            "min_ms": round(values_ms[0], 3),
            "avg_ms": round(sum(values_ms) / n, 3),
            "max_ms": round(values_ms[-1], 3),
            "p50_ms": round(p50, 3),
            "p99_ms": round(p99, 3),
        }

    def _build_summary(self) -> dict:
        mini_batch_stats = []
        mb_indices = sorted({s.mini_batch_idx for s in self._snapshots} | set(self._minibatch_phases))
        for mb_idx in mb_indices:
            snaps = [s for s in self._snapshots if s.mini_batch_idx == mb_idx]
            phase_names = sorted({p for s in snaps for p in s.timing})
            timing_stats = {
                p: self._phase_stats([s.timing[p] for s in snaps if p in s.timing])
                for p in phase_names
            }
            mb_phases = {
                p: round(v * 1e3, 3) for p, v in self._minibatch_phases.get(mb_idx, {}).items()
            }
            mini_batch_stats.append({
                "mini_batch_idx": mb_idx,
                "num_micro_batches": len(snaps),
                "micro_batch_timing": timing_stats,
                "minibatch_phase_ms": mb_phases,
            })

        layer_kv: dict[int, float] = {}
        layer_comp: dict[int, float] = {}
        for snap in self._snapshots:
            for layer_id, phases in snap.per_layer.items():
                layer_kv[layer_id] = layer_kv.get(layer_id, 0.0) + phases.get(PerfProfiler.PHASE_ATTN_KV, 0.0)
                layer_comp[layer_id] = layer_comp.get(layer_id, 0.0) + phases.get(PerfProfiler.PHASE_ATTN_COMPUTE, 0.0)
        per_layer_summary: dict[str, Any] = {}
        if layer_kv or layer_comp:
            num_layers = max(set(layer_kv) | set(layer_comp)) + 1
            kv_totals = [round(layer_kv.get(i, 0.0) * 1e3, 3) for i in range(num_layers)]
            comp_totals = [round(layer_comp.get(i, 0.0) * 1e3, 3) for i in range(num_layers)]
            slowest_kv = max(layer_kv, key=layer_kv.get) if layer_kv else None
            slowest_comp = max(layer_comp, key=layer_comp.get) if layer_comp else None
            per_layer_summary = {
                "total_kv_ms": kv_totals,
                "total_comp_ms": comp_totals,
                "slowest_kv_layer": (
                    {"layer_id": slowest_kv, "total_ms": round(layer_kv[slowest_kv] * 1e3, 3)}
                    if slowest_kv is not None else None
                ),
                "slowest_comp_layer": (
                    {"layer_id": slowest_comp, "total_ms": round(layer_comp[slowest_comp] * 1e3, 3)}
                    if slowest_comp is not None else None
                ),
            }

        memory_summary = self._monitor.summary()
        return {
            "step_id": self.step_id,
            "kind": self.kind,
            "rank": self.rank,
            "num_mini_batches": len(mb_indices),
            "num_micro_batches": len(self._snapshots),
            "total_elapsed_s": round(self._total_elapsed_s, 4),
            "memory": memory_summary,
            "mini_batch_stats": mini_batch_stats,
            "per_layer_attention_summary": per_layer_summary,
        }


def _resolve_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0
