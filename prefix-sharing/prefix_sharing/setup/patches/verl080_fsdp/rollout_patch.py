"""Patch: RayPPOTrainer.fit → intercept rollout generate_sequences.

Triggered by env vars:
- PREFIX_SHARING_CAPTURE_ROLLOUT=/path/save.json   → capture first rollout
- PREFIX_SHARING_FIXED_ROLLOUT=/path/load.json     → inject fixed data

Wraps both async_rollout_manager and actor_rollout_wg to cover all rollout paths.
"""

from __future__ import annotations

import os
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


def patch_ray_trainer_fit(original_fit: Any) -> Any:
    """Wrap RayPPOTrainer.fit to intercept generate_sequences on both rollout objects."""

    def patched_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        actor_rollout_wg = getattr(self, "actor_rollout_wg", None)
        async_mgr = getattr(self, "async_rollout_manager", None)

        if actor_rollout_wg is not None and hasattr(actor_rollout_wg, "generate_sequences"):
            _apply_rollout_env(actor_rollout_wg)
        if async_mgr is not None and hasattr(async_mgr, "generate_sequences"):
            _apply_rollout_env(async_mgr)

        return original_fit(self, *args, **kwargs)

    return patched_fit
