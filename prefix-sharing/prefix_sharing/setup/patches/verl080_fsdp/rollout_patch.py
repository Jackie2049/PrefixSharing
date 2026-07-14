"""Patch the verl080 FSDP rollout boundary for fixed rollout experiments.

Note: capture/replay via PREFIX_SHARING_CAPTURE_ROLLOUT and PREFIX_SHARING_FIXED_ROLLOUT
is now directly injected in verl/trainer/ppo/ray_trainer.py at the existing
"prefix-sharing: inject data" injection point (fit() entry). This patch is only
needed for the RolloutSkip path (rollout.skip.enable) and for the original
USE_FIXED_ROLLOUT env var support.
"""

from __future__ import annotations

import os
from typing import Any


def patch_ray_trainer_fit(original_fit: Any) -> Any:
    """Legacy stub — capture/replay now injected directly in ray_trainer.py."""

    def patched_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        return original_fit(self, *args, **kwargs)

    return patched_fit
