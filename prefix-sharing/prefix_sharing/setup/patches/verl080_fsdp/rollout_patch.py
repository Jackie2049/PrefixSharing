"""Patch: RayPPOTrainer.fit → intercept rollout generate_sequences.

Triggered by env vars:
- PREFIX_SHARING_CAPTURE_ROLLOUT=/path/save.json   → capture first rollout
- PREFIX_SHARING_FIXED_ROLLOUT=/path/load.json     → inject fixed data
- PREFIX_SHARING_PERF_DIR=/path                    → also dump step-level verl
  timing/throughput metrics to {PERF_DIR}/step_{i}/verl_metrics.json

Wraps both async_rollout_manager and actor_rollout_wg to cover all rollout paths.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _apply_rollout_env(rollout_obj: Any) -> None:
    """Read env vars and apply capture or fixed-injection to *rollout_obj*."""
    capture_path = os.environ.get("PREFIX_SHARING_CAPTURE_ROLLOUT", "").strip()
    fixed_path = os.environ.get("PREFIX_SHARING_FIXED_ROLLOUT", "").strip()

    if capture_path:
        from prefix_sharing.tools.inject_fixed_rollout import patch_capture_rollout
        patch_capture_rollout(rollout_obj, capture_path)
    elif fixed_path:
        from prefix_sharing.tools.inject_fixed_rollout import patch_fixed_rollout
        patch_fixed_rollout(rollout_obj, fixed_path)


def _wrap_logger_for_perf_dir(trainer: Any) -> None:
    """If PREFIX_SHARING_PERF_DIR is set, wrap trainer.logger.log so that each
    call also dumps the metrics dict to ``{PERF_DIR}/step_{i}/verl_metrics.json``.

    This co-locates verl's coarse step-level timing (timing_s/*, perf/*) with
    the profiler's fine-grained per-micro-batch summaries under the same dir.
    """
    perf_dir = os.environ.get("PREFIX_SHARING_PERF_DIR", "").strip()
    if not perf_dir:
        return

    logger = getattr(trainer, "logger", None)
    if logger is None or getattr(logger, "_ps_verl_metrics_wrapped", False):
        return

    orig_log = logger.log

    def _log_with_dump(data: dict, step: int, *args: Any, **kwargs: Any) -> Any:
        try:
            step_dir = Path(perf_dir) / f"step_{step}"
            step_dir.mkdir(parents=True, exist_ok=True)
            out_path = step_dir / "verl_metrics.json"
            serializable = {}
            for k, v in data.items():
                try:
                    json.dumps(v)
                    serializable[k] = v
                except (TypeError, ValueError):
                    serializable[k] = str(v)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
        except Exception:
            pass  # never let metrics dumping crash training
        return orig_log(data, step, *args, **kwargs)

    logger.log = _log_with_dump
    logger._ps_verl_metrics_wrapped = True


def patch_ray_trainer_fit(original_fit: Any) -> Any:
    """Wrap RayPPOTrainer.fit to intercept generate_sequences on both rollout objects."""

    def patched_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        actor_rollout_wg = getattr(self, "actor_rollout_wg", None)
        async_mgr = getattr(self, "async_rollout_manager", None)

        if actor_rollout_wg is not None and hasattr(actor_rollout_wg, "generate_sequences"):
            _apply_rollout_env(actor_rollout_wg)
        if async_mgr is not None and hasattr(async_mgr, "generate_sequences"):
            _apply_rollout_env(async_mgr)

        _wrap_logger_for_perf_dir(self)

        return original_fit(self, *args, **kwargs)

    return patched_fit
