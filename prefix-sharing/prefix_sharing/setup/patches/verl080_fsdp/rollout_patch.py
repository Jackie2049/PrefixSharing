"""Patch the verl080 FSDP rollout boundary for fixed rollout experiments."""

from __future__ import annotations

import os
from typing import Any


def _apply_capture_env(rollout_obj: Any, json_path: str) -> None:
    """Apply capture to rollout_obj at the point it actually has generate_sequences."""
    from prefix_sharing.tools.inject_fixed_rollout import patch_capture_rollout

    patch_capture_rollout(rollout_obj, json_path)


def _apply_replay_env(rollout_obj: Any, json_path: str) -> None:
    """Apply replay to rollout_obj at the point it actually has generate_sequences."""
    from prefix_sharing.tools.inject_fixed_rollout import patch_fixed_rollout

    patch_fixed_rollout(rollout_obj, json_path)


def _apply_rollout_env(rollout_obj: Any) -> None:
    capture_path = os.environ.get("PREFIX_SHARING_CAPTURE_ROLLOUT", "").strip()
    fixed_path = os.environ.get("PREFIX_SHARING_FIXED_ROLLOUT", "").strip()
    if capture_path and fixed_path:
        raise ValueError(
            "PREFIX_SHARING_CAPTURE_ROLLOUT and PREFIX_SHARING_FIXED_ROLLOUT are mutually exclusive"
        )
    if capture_path:
        _apply_capture_env(rollout_obj, capture_path)
    elif fixed_path:
        _apply_replay_env(rollout_obj, fixed_path)


def patch_ray_trainer_fit(original_fit: Any) -> Any:
    """Bind capture/replay to the rollout objects once per trainer ``fit`` call."""

    def patched_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Bind to async_rollout_manager if it exists
        if hasattr(self, "async_rollout_manager") and self.async_rollout_manager is not None:
            _apply_rollout_env(self.async_rollout_manager)
        # Also bind to actor_rollout_wg if it exists (fallback for sync trainers)
        if hasattr(self, "actor_rollout_wg") and self.actor_rollout_wg is not None:
            _apply_rollout_env(self.actor_rollout_wg)
        return original_fit(self, *args, **kwargs)

    return patched_fit
