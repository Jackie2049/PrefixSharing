"""Patch the verl080 FSDP rollout boundary for fixed rollout experiments."""

from __future__ import annotations

import os
from typing import Any


def _apply_rollout_env(rollout_obj: Any) -> None:
    capture_path = os.environ.get("PREFIX_SHARING_CAPTURE_ROLLOUT", "").strip()
    fixed_path = os.environ.get("PREFIX_SHARING_FIXED_ROLLOUT", "").strip()
    if capture_path and fixed_path:
        raise ValueError(
            "PREFIX_SHARING_CAPTURE_ROLLOUT and PREFIX_SHARING_FIXED_ROLLOUT are mutually exclusive"
        )
    if capture_path:
        from prefix_sharing.tools.inject_fixed_rollout import patch_capture_rollout

        patch_capture_rollout(rollout_obj, capture_path)
    elif fixed_path:
        from prefix_sharing.tools.inject_fixed_rollout import patch_fixed_rollout

        patch_fixed_rollout(rollout_obj, fixed_path)


def patch_ray_trainer_fit(original_fit: Any) -> Any:
    """Bind capture/replay to the rollout objects once per trainer ``fit`` call."""

    def patched_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        for name in ("actor_rollout_wg", "async_rollout_manager"):
            rollout_obj = getattr(self, name, None)
            if rollout_obj is not None and hasattr(rollout_obj, "generate_sequences"):
                _apply_rollout_env(rollout_obj)
        return original_fit(self, *args, **kwargs)

    return patched_fit
