"""Training monitoring utilities for prefix sharing profiling.

Provides ``MemoryMonitor`` (continuous device memory sampling) and
``Stopwatch`` (lap / start-stop timing with step-context support) for
debugging HBM usage and bottlenecks during prefix-sharing training.

Both classes support ``save_to_csv()`` to dump full time-series data to file
(not to logs), usable in **verl + Megatron** micro-batch pipelines:

* ``MemoryMonitor`` — background thread samples ``allocated_gb`` / ``reserved_gb``
  at a fixed interval.
* ``Stopwatch`` — hierarchical phase timing (minibatch → microbatch_fwd /
  microbatch_bwd → update) with optional **step_id** context for correlation
  across the training loop.

Typical usage::

    from prefix_sharing.tools.training_monitor import (
        MemoryMonitor,
        Stopwatch,
        training_monitor_context,
    )

    mon = MemoryMonitor(interval=0.1)
    sw = Stopwatch()
    mon.start()

    for step_id in range(num_steps):
        sw.set_step_context(step_id)
        sw.start("minibatch_fwd_bwd")

        for mb_idx in range(num_microbatches):
            sw.start("microbatch_fwd")
            ...  # Megatron forward on this microbatch
            sw.stop("microbatch_fwd")

            sw.start("microbatch_bwd")
            ...  # Megatron backward on this microbatch
            sw.stop("microbatch_bwd")

        sw.stop("minibatch_fwd_bwd")

        sw.start("update")
        ...  # optimizer.step()
        sw.stop("update")

    mon.stop()

    mon.save_to_csv("memory_trace.csv")
    sw.save_to_csv("timing_trace.csv")
"""

from __future__ import annotations

import csv
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Literal


# ---------------------------------------------------------------------------
# ContextVar – 无需修改函数签名即可在嵌套调用链中访问 mon / sw
# ---------------------------------------------------------------------------


@dataclass
class TrainingMonitorState:
    """Holds the active ``MemoryMonitor`` and ``Stopwatch`` for the current training step.

    Accessed via ``current_memory_monitor()`` / ``current_stopwatch()`` from
    any nested function without passing parameters.
    """

    mon: MemoryMonitor | None
    sw: Stopwatch | None


_monitor_ctx: ContextVar[TrainingMonitorState | None] = ContextVar(
    "training_monitor_context",
    default=None,
)


def current_memory_monitor() -> MemoryMonitor | None:
    """Return the active ``MemoryMonitor`` (or ``None`` outside a monitoring context)."""
    state = _monitor_ctx.get()
    return state.mon if state is not None else None


def current_stopwatch() -> Stopwatch | None:
    """Return the active ``Stopwatch`` (or ``None`` outside a monitoring context)."""
    state = _monitor_ctx.get()
    return state.sw if state is not None else None


@contextmanager
def training_monitor_context(
    mon: MemoryMonitor | None = None,
    sw: Stopwatch | None = None,
) -> Iterator[TrainingMonitorState]:
    """Context manager that makes *mon* and *sw* available via ``current_*()`` getters.

    Usage (in ``update_policy``, no signature changes needed)::

        mon = MemoryMonitor(interval=0.05)
        sw = Stopwatch()
        with training_monitor_context(mon, sw):
            mon.start()
            for data in dataloader:
                ...   # inside forward_step, call current_stopwatch().start("microbatch")
            mon.stop()
            ... print stats ...
    """
    state = TrainingMonitorState(mon=mon, sw=sw)
    token = _monitor_ctx.set(state)
    try:
        yield state
    finally:
        _monitor_ctx.reset(token)


# ---------------------------------------------------------------------------
# MemoryMonitor
# ---------------------------------------------------------------------------


@dataclass
class MemorySnapshot:
    """A single memory-poll sample."""

    timestamp: float
    allocated_gb: float  # memory_allocated()  – tensor 占用
    reserved_gb: float   # memory_reserved()  – 分配器保留（含缓存）


class MemoryMonitor:
    """Background-thread memory sampler for CUDA / NPU devices.

    Each sample records two metrics:

    * **allocated** – memory actively holding tensors (``memory_allocated()``)
    * **reserved**  – memory reserved by the caching allocator (``memory_reserved()``)

    Call ``start()`` before the region of interest and ``stop()`` after::

        mon = MemoryMonitor(interval=0.05)   # sample every 50 ms
        mon.start()
        ... forward + backward ...
        mon.stop()
        s = mon.summary()
        print(
            f"Peak  allocated={s['peak_allocated_gib']:.1f} GiB  "
            f"reserved={s['peak_reserved_gib']:.1f} GiB"
        )
    """

    def __init__(
        self,
        device_type: Literal["cuda", "npu"] | None = None,
        interval: float = 0.1,
    ) -> None:
        self.interval = interval
        self._samples: list[MemorySnapshot] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self.device_type = self._resolve_device(device_type)

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin background sampling."""
        self._samples.clear()
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background sampling and join the thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)

    def snapshot(self) -> MemorySnapshot:
        """One-shot query of all memory metrics without starting the thread."""
        return self._sample_once()

    # -- per-metric peaks -------------------------------------------------

    @property
    def peak_allocated_gb(self) -> float:
        if not self._samples:
            return 0.0
        return max(s.allocated_gb for s in self._samples)

    @property
    def peak_reserved_gb(self) -> float:
        if not self._samples:
            return 0.0
        return max(s.reserved_gb for s in self._samples)

    def summary(self) -> dict:
        """Aggregate memory metrics.

        Returns keys ``peak_{allocated,reserved}_gib``,
        ``avg_{allocated,reserved}_gib``, plus metadata.
        """
        if not self._samples:
            return {"device_type": self.device_type, "num_samples": 0}
        al = [s.allocated_gb for s in self._samples]
        rs = [s.reserved_gb for s in self._samples]
        return {
            "device_type": self.device_type,
            "num_samples": len(self._samples),
            "interval_s": self.interval,
            "peak_allocated_gib": round(max(al), 3),
            "peak_reserved_gib": round(max(rs), 3),
            "avg_allocated_gib": round(sum(al) / len(al), 3),
            "avg_reserved_gib": round(sum(rs) / len(rs), 3),
        }

    def save_to_csv(self, path: str) -> None:
        """Save all memory samples to a CSV file (no log output).

        Columns: ``timestamp``, ``allocated_gb``, ``reserved_gb``.

        Args:
            path: Output CSV file path.
        """
        if not self._samples:
            raise RuntimeError("[MemoryMonitor] No samples to save. Did you forget to call start()/stop()?")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "allocated_gb", "reserved_gb"])
            for s in self._samples:
                writer.writerow([s.timestamp, s.allocated_gb, s.reserved_gb])
        print(f"[MemoryMonitor] Saved {len(self._samples)} samples to {path}")

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _resolve_device(self, device_type: str | None) -> str | None:
        import torch as _torch

        if device_type is not None:
            return device_type
        try:
            if _torch.npu.is_available():
                return "npu"
        except Exception:
            pass
        if _torch.cuda.is_available():
            return "cuda"
        return None  # CPU – memory monitoring disabled

    def _sample_once(self) -> MemorySnapshot:
        import torch as _torch

        if self.device_type == "npu":
            allocated = _torch.npu.memory_allocated()
            reserved = _torch.npu.memory_reserved()
        elif self.device_type == "cuda":
            allocated = _torch.cuda.memory_allocated()
            reserved = _torch.cuda.memory_reserved()
        else:
            return MemorySnapshot(time.time(), 0.0, 0.0)

        return MemorySnapshot(
            timestamp=time.time(),
            allocated_gb=allocated / (1024**3),
            reserved_gb=reserved / (1024**3),
        )

    def _sample_loop(self) -> None:
        while self._running:
            self._samples.append(self._sample_once())
            time.sleep(self.interval)


# ---------------------------------------------------------------------------
# Stopwatch
# ---------------------------------------------------------------------------


class Stopwatch:
    """Lightweight timing for training phases with distribution support.

    Supports three usage modes:

    **Start / stop** – for repeatable phases (e.g. per-microbatch timing)::

        sw = Stopwatch()
        for _ in range(n):
            sw.start("microbatch")
            ... forward + backward ...
            sw.stop("microbatch")
        sw.start("update")
        ... optimizer ...
        sw.stop("update")

        print(sw.summary())
        # {
        #   "phases": {
        #     "microbatch": {"total_s": 8.2, "count": 4, "avg_s": 2.05,
        #                    "min_s": 1.98, "max_s": 2.11, "median_s": 2.04,
        #                    "p99_s": 2.108, "stddev_s": 0.05},
        #     "update":     {"total_s": 4.3, "count": 1, ...}
        #   }
        # }

    **Sub-phase nesting** — time sub-stages within a microbatch::

        sw.start("microbatch")
        sw.start("forward")
        ... forward ...
        sw.stop("forward")          # recorded as a sample of "forward"
        sw.start("backward")
        ... backward ...
        sw.stop("backward")
        sw.stop("microbatch")       # also recorded as a sample of "microbatch"
        # total_s of "microbatch" == sum of all microbatch durations
        # (sub-phases are a separate dimension, not subtracted)

    **Lap** – for linear checkpoints::

        sw.lap("forward_done")
        sw.lap("backward_done")
    """

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = OrderedDict()  # name → [durations]
        self._sample_step_ids: dict[str, list[int | None]] = OrderedDict()  # name → [step_id]
        self._running: dict[str, float] = OrderedDict()        # name → start_time
        self._laps: list[tuple[str, float]] = []               # (name, timestamp)
        self._current_step_id: int | None = None               # set via set_step_context()

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def lap(self, name: str = "") -> None:
        """Record an instantaneous checkpoint with label *name*."""
        self._laps.append((name, time.perf_counter()))

    def set_step_context(self, step_id: int | None) -> None:
        """Set the current training step id.

        Subsequent ``stop()`` calls will tag their samples with this step_id,
        allowing correlation between timing data and training steps.
        """
        self._current_step_id = step_id

    def reset_step_context(self) -> None:
        """Clear the current step context (next samples get ``None`` step id)."""
        self._current_step_id = None

    def start(self, name: str) -> None:
        """Begin timing for *name*.

        Supports nesting: calling ``start('forward')`` while ``start('microbatch')``
        is already running works fine — each named timer runs independently.
        """
        self._running[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """End timing for *name*, record an individual sample, and return elapsed seconds.

        If a step context was set via ``set_step_context()``, the sample is
        tagged with that step id for later CSV output.
        """
        start = self._running.pop(name, None)
        if start is None:
            print(f"Stopwatch.stop('{name}'): timer was never started")
            return 0.0
        elapsed = time.perf_counter() - start
        self._samples.setdefault(name, [])
        self._samples[name].append(elapsed)
        self._sample_step_ids.setdefault(name, [])
        self._sample_step_ids[name].append(self._current_step_id)
        return elapsed

    def reset(self) -> None:
        """Clear all recorded samples, laps, running timers, and step context."""
        self._samples.clear()
        self._sample_step_ids.clear()
        self._laps.clear()
        self._running.clear()
        self._current_step_id = None

    def elapsed(self, name: str) -> float:
        """Return total cumulative time for phase *name*."""
        return sum(self._samples.get(name, []))

    def raw_samples(self, name: str) -> list[float]:
        """Return the individual duration list for *name* (for custom analysis)."""
        return list(self._samples.get(name, []))

    def summary(self) -> dict:
        """Aggregate timing stats with distribution for each phase.

        Returns::

            {
                "total_s": 12.5,
                "phases": {
                    "microbatch": {"total_s", "count", "avg_s",
                                   "min_s", "max_s", "median_s",
                                   "p99_s", "stddev_s"},
                    ...
                }
            }
        """
        phases: OrderedDict[str, dict] = OrderedDict()
        for name, samples in self._samples.items():
            cnt = len(samples)
            if cnt == 0:
                continue
            total_s = sum(samples)
            sorted_s = sorted(samples)
            avg_s = total_s / cnt
            stddev_s = (sum((s - avg_s) ** 2 for s in samples) / cnt) ** 0.5
            p99_idx = max(0, int(cnt * 0.99) - 1)

            phases[name] = {
                "total_s": round(total_s, 4),
                "count": cnt,
                "avg_s": round(avg_s, 4),
                "min_s": round(sorted_s[0], 4),
                "max_s": round(sorted_s[-1], 4),
                "median_s": round(sorted_s[cnt // 2], 4),
                "p99_s": round(sorted_s[p99_idx], 4),
                "stddev_s": round(stddev_s, 4),
            }

        return {
            "total_s": round(sum(sum(v) for v in self._samples.values()), 4),
            "phases": phases,
        }

    def print_lap_timeline(self, logger_fn=None) -> None:
        """Pretty-print the lap timeline (useful for a single microbatch walk-through).

        *logger_fn* is called instead of ``print`` if given (e.g. ``logger.warning``).
        """
        if len(self._laps) < 2:
            return
        out = []
        t0 = self._laps[0][1]
        for name, ts in self._laps:
            delta = ts - t0
            out.append(f"    [{delta:8.4f}s] {name}")
        text = "\n".join(out)
        if logger_fn:
            logger_fn(text)
        else:
            print(text)

    def save_to_csv(self, path: str) -> None:
        """Save all stopwatch samples to a CSV file (no log output).

        Columns: ``step_id``, ``phase``, ``sample_index``, ``duration_s``.

        The ``step_id`` column is populated only if ``set_step_context()`` was
        called before the corresponding ``stop()``; otherwise it is empty.

        Use this in a **verl + Megatron** loop to dump minibatch / microbatch
        timing for offline analysis::

            sw.save_to_csv("timing_trace.csv")
        """
        if not self._samples:
            raise RuntimeError("[Stopwatch] No samples to save.")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step_id", "phase", "sample_index", "duration_s"])
            for phase in self._samples:
                durations = self._samples[phase]
                step_ids = self._sample_step_ids.get(phase, [None] * len(durations))
                for idx, (d, sid) in enumerate(zip(durations, step_ids)):
                    writer.writerow([sid if sid is not None else "", phase, idx, round(d, 6)])
        print(f"[Stopwatch] Saved {sum(len(v) for v in self._samples.values())} phase samples to {path}")
