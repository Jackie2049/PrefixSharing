"""Patch: verl.trainer.main_ppo.run_ppo — fixed rollout + synthetic prefix injection.

Uses default-argument capture so Ray can serialise the patched function.
"""

from __future__ import annotations

import functools

import os
from typing import Any


def _inject_fixed_data(trainer_self: Any) -> None:
    """Check env vars and patch generate_sequences on the trainer."""
    json_path = os.environ.get("USE_FIXED_ROLLOUT", None)
    if json_path:
        from prefix_sharing.tools.inject_fixed_rollout import patch_fixed_rollout
        patch_fixed_rollout(trainer_self, json_path=json_path)

    synthetic_json = os.environ.get("USE_SYNTHETIC_PREFIX", None)
    if synthetic_json:
        from prefix_sharing.tools.inject_synthetic_prefix import patch_synthetic_prefix
        batch_size = trainer_self.config.data.get(
            "gen_batch_size", trainer_self.config.data.train_batch_size
        )
        patch_synthetic_prefix(
            trainer_self,
            json_path=synthetic_json,
            batch_size=batch_size,
            max_prompt_length=trainer_self.config.data.max_prompt_length,
            max_response_length=trainer_self.config.data.max_response_length,
        )


def _patched_fit(trainer_self: Any, *, _original_fit: Any) -> Any:
    """Replacement for RayPPOTrainer.fit — inject fixed data then call original."""
    _inject_fixed_data(trainer_self)
    return _original_fit(trainer_self)


def create_run_ppo_patch(original_run_ppo: Any) -> Any:
    """Return a patched run_ppo with original captured as default argument."""
    import verl.trainer.ppo.ray_trainer as rt

    _original_fit_captured = rt.RayPPOTrainer.fit

    def patched_run_ppo(
        config: Any,
        task_runner_class: Any = None,
        *,
        _orig=original_run_ppo,
        _fit=_original_fit_captured,
        _patched=_patched_fit,
    ) -> None:
        import verl.trainer.ppo.ray_trainer as _rt

        # Temporarily override fit
        _rt.RayPPOTrainer.fit = functools.partial(_patched, _original_fit=_fit)
        try:
            return _orig(config, task_runner_class)
        finally:
            _rt.RayPPOTrainer.fit = _fit

    return patched_run_ppo
