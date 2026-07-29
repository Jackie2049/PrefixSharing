"""Monkey-patch training monitor into verl PPO trainer.

Adds MemoryMonitor + Stopwatch to the training loop without modifying
verl source code.  Controlled via environment variables:

    ENABLE_TRAINING_MONITOR=1              # enable
    TRAINING_MONITOR_INTERVAL=0.1           # memory sampling interval (default 0.1s)
    TRAINING_MONITOR_SAVE_DIR=/tmp          # CSV output directory (default /tmp)

Usage (in the launch script):

    export ENABLE_TRAINING_MONITOR=1
    export TRAINING_MONITOR_INTERVAL=0.1
    export TRAINING_MONITOR_SAVE_DIR=/home/ma-user/work/l00561472/metrics

Call ``patch_training_monitor()`` early (before the training loop),
e.g. from ``main_ppo.py`` or a custom entry point::

    from prefix_sharing.tools.patch_training_monitor import patch_training_monitor
    patch_training_monitor()
"""

from __future__ import annotations

import os
import time
from typing import Any

from prefix_sharing.tools.training_monitor import MemoryMonitor, Stopwatch


_mon: MemoryMonitor | None = None
_sw: Stopwatch | None = None
_save_dir: str = "/tmp"
_enabled: bool = False


def is_enabled() -> bool:
    return _enabled


def get_monitor() -> MemoryMonitor | None:
    return _mon


def get_stopwatch() -> Stopwatch | None:
    return _sw


def patch_training_monitor(trainer: Any | None = None) -> None:
    """Install training monitor patches.

    If *trainer* is provided (a ``RayPPOTrainer`` instance), patches are
    applied immediately.  Otherwise, patches are deferred via an import
    hook on ``verl.trainer.ppo.ray_trainer``.

    Call this once before the training loop starts.
    """
    global _mon, _sw, _save_dir, _enabled

    if not os.getenv("ENABLE_TRAINING_MONITOR"):
        return

    interval = float(os.getenv("TRAINING_MONITOR_INTERVAL", "0.1"))
    _save_dir = os.getenv("TRAINING_MONITOR_SAVE_DIR", "/tmp")
    _mon = MemoryMonitor(interval=interval)
    _sw = Stopwatch()
    _enabled = True

    print(
        f"[TrainingMonitor] Enabled: interval={interval}s, save_dir={_save_dir}"
    )

    if trainer is not None:
        _apply_patches(trainer)
    else:
        _install_import_hook()


def _apply_patches(trainer: Any) -> None:
    """Patch the trainer instance directly."""
    _patch_fit(trainer)
    print("[TrainingMonitor] Patches applied to trainer instance.")


# ---------------------------------------------------------------------------
# import hook — deferred patching when ray_trainer is imported
# ---------------------------------------------------------------------------

_original_import = __builtins__.__import__ if isinstance(__builtins__, dict) else __import__


def _install_import_hook() -> None:
    """Install a __import__ hook that patches RayPPOTrainer.fit when ray_trainer is first loaded."""
    import builtins

    _builtin_import = builtins.__import__

    def _patched_import(name, *args, **kwargs):
        module = _builtin_import(name, *args, **kwargs)
        if name == "verl.trainer.ppo.ray_trainer":
            _patch_fit_on_module(module)
        return module

    builtins.__import__ = _patched_import
    print("[TrainingMonitor] Import hook installed (will patch ray_trainer.fit on first import).")


def _patch_fit_on_module(module: Any) -> None:
    """Patch the ``RayPPOTrainer.fit`` method on a module object."""
    # Try both class names
    for cls_name in ("RayPPOTrainer", "RayMegatronTrainer"):
        cls = getattr(module, cls_name, None)
        if cls is not None:
            _patch_fit_on_class(cls)
            print(f"[TrainingMonitor] Patched {cls_name}.fit.")


def _patch_fit_on_class(cls: type) -> None:
    """Wrap ``cls.fit`` to start/stop monitors and tag step timing."""
    original_fit = cls.fit

    def patched_fit(self, *args, **kwargs):
        mon = _mon
        sw = _sw

        if mon is not None:
            mon.start()
        if sw is not None:
            sw.reset()

        # --- patch the *_update_actor / *_update_critic methods ---
        _patch_update_methods(self, sw)

        try:
            result = original_fit(self, *args, **kwargs)
        finally:
            if mon is not None:
                mon.stop()
                ts = time.strftime("%Y%m%d_%H%M%S")
                os.makedirs(_save_dir, exist_ok=True)
                mem_path = os.path.join(_save_dir, f"memory_trace_{ts}.csv")
                time_path = os.path.join(_save_dir, f"timing_trace_{ts}.csv")
                try:
                    mon.save_to_csv(mem_path)
                except Exception as e:
                    print(f"[TrainingMonitor] Failed to save memory trace: {e}")
                try:
                    sw.save_to_csv(time_path)
                except Exception as e:
                    print(f"[TrainingMonitor] Failed to save timing trace: {e}")
                print(
                    f"[TrainingMonitor] Peak allocated={mon.peak_allocated_gb:.1f}GB, "
                    f"reserved={mon.peak_reserved_gb:.1f}GB"
                )
                phases = sw.summary().get("phases", {})
                if phases:
                    phase_strs = []
                    for k, v in phases.items():
                        phase_strs.append(f"{k}={v['total_s']:.1f}s")
                    print(f"[TrainingMonitor] Phase summary: "
                          f"total={sw.summary()['total_s']:.1f}s, "
                          f"{', '.join(phase_strs)}")
        return result

    cls.fit = patched_fit


def _patch_update_methods(trainer: Any, sw: Stopwatch | None) -> None:
    """Patch ``_update_actor`` and ``_update_critic`` to add stopwatch timing."""
    if sw is None:
        return

    # --- _update_actor ---
    if hasattr(trainer, "_update_actor"):
        _orig_update_actor = trainer._update_actor

        def _patched_update_actor(batch):
            sw.start("update_actor")
            try:
                return _orig_update_actor(batch)
            finally:
                sw.stop("update_actor")

        trainer._update_actor = _patched_update_actor

    # --- _update_critic ---
    if hasattr(trainer, "_update_critic"):
        _orig_update_critic = trainer._update_critic

        def _patched_update_critic(batch):
            sw.start("update_critic")
            try:
                return _orig_update_critic(batch)
            finally:
                sw.stop("update_critic")

        trainer._update_critic = _patched_update_critic

    # --- _compute_old_log_prob ---
    if hasattr(trainer, "_compute_old_log_prob"):
        _orig_old_log_prob = trainer._compute_old_log_prob

        def _patched_old_log_prob(batch):
            sw.start("old_log_prob")
            try:
                return _orig_old_log_prob(batch)
            finally:
                sw.stop("old_log_prob")

        trainer._compute_old_log_prob = _patched_old_log_prob


# ---------------------------------------------------------------------------
# convenience — call directly
# ---------------------------------------------------------------------------

def _patch_fit(trainer: Any) -> None:
    """Patch a trainer instance that has already been constructed."""
    for cls_name in ("RayPPOTrainer", "RayMegatronTrainer"):
        if cls_name in type(trainer).__name__:
            _patch_fit_on_class(type(trainer))
            _patch_update_methods(trainer, _sw)
            return
    # Fallback: patch the instance directly
    _patch_fit_on_class(type(trainer))
    _patch_update_methods(trainer, _sw)
