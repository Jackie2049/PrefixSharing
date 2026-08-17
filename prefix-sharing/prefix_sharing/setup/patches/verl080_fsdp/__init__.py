"""module: prefix_sharing.setup.patches.verl080_fsdp

verl 0.8.0 FSDP patch set. This is the default entry for the FSDP open-source line.
It can be selected via the compat matrix, or installed explicitly with
``prefix_sharing.setup.install("verl080_fsdp")``.
"""

from __future__ import annotations

from prefix_sharing.setup.patch_installer import PatchSpec

from .forward_step import (
    patch_forward_backward_batch_for_diag_dump,
    patch_fsdp_forward_step,
)
from .perf_profiler_patch import (
    patch_forward_backward_batch,
    patch_infer_batch,
    patch_train_batch,
    patch_train_mini_batch,
)
from .rollout_patch import patch_ray_trainer_fit


PATCH_SET: list[PatchSpec] = [
    # ═══════════════════════════════════════════════════════════════
    # PrefixSharing entrypoints
    # ═══════════════════════════════════════════════════════════════

    # FSDPEngineWithLMHead.forward_step → verl080_fsdp.patch_fsdp_forward_step
    # patch forward_step() for PrefixSharing workflows
    PatchSpec(
        module_name="verl.workers.engine.fsdp.transformer_impl",
        target_getter=lambda mod: (
            getattr(mod, "FSDPEngineWithLMHead"),
            "forward_step",
        ),
        patch_factory=patch_fsdp_forward_step,
        description=("FSDPEngineWithLMHead.forward_step → verl080_fsdp.patch_fsdp_forward_step: "
                     "patch forward_step() for PrefixSharing workflows"),
        # verl FSDP engine is lazy-loaded only when the actor is created; eager is required.
        eager=True,
    ),

    # ═══════════════════════════════════════════════════════════════
    # Debugging functionality, precision and performance
    # ═══════════════════════════════════════════════════════════════

    # RayPPOTrainer.fit → verl080_fsdp.patch_ray_trainer_fit
    # patch fit() for capturing rollout results or loading fixed rollout data
    PatchSpec(
        module_name="verl.trainer.ppo.ray_trainer",
        target_getter=lambda mod: (mod.RayPPOTrainer, "fit"),
        patch_factory=patch_ray_trainer_fit,
        description=(
            "RayPPOTrainer.fit → verl080_fsdp.patch_ray_trainer_fit: "
            "patch fit() for capturing rollout results or loading fixed rollout data"
        ),
        # ray_trainer is imported by main_ppo at startup; eager ensures the patch is in place.
        eager=True,
    ),

    # TrainingWorker.train_mini_batch → step-level ProfilerScope (kind=train)
    # patch train_mini_batch() for step-level performance profiling
    PatchSpec(
        module_name="verl.workers.engine_workers",
        target_getter=lambda mod: (mod.TrainingWorker, "train_mini_batch"),
        patch_factory=patch_train_mini_batch,
        description=("TrainingWorker.train_mini_batch → verl080_fsdp.patch_train_mini_batch: "
                     "patch train_mini_batch() for step-level performance profiling"),
        eager=True,
    ),

    # BaseEngine.train_batch is called from train_mini_batch.
    # patch train_batch() for optimizer-step performance profiling
    PatchSpec(
        module_name="verl.workers.engine.base",
        target_getter=lambda mod: (mod.BaseEngine, "train_batch"),
        patch_factory=patch_train_batch,
        description=("BaseEngine.train_batch → verl080_fsdp.patch_train_batch: "
                     "patch train_batch() for optimizer-step performance profiling"),
        eager=True,
    ),

    # FSDPEngine.forward_backward_batch → micro-batch ProfilerScope markers
    # patch forward_backward_batch() for micro-batch performance profiling
    PatchSpec(
        module_name="verl.workers.engine.fsdp.transformer_impl",
        target_getter=lambda mod: (mod.FSDPEngine, "forward_backward_batch"),
        patch_factory=patch_forward_backward_batch,
        description=("FSDPEngine.forward_backward_batch → verl080_fsdp.patch_forward_backward_batch: "
                     "patch forward_backward_batch() for micro-batch performance profiling"),
        eager=True,
    ),

    # FSDPEngine.forward_backward_batch → dump weight gradients after backward
    # patch forward_backward_batch() for precision dumps
    PatchSpec(
        module_name="verl.workers.engine.fsdp.transformer_impl",
        target_getter=lambda mod: (mod.FSDPEngine, "forward_backward_batch"),
        patch_factory=patch_forward_backward_batch_for_diag_dump,
        description=("FSDPEngine.forward_backward_batch → "
                     "verl080_fsdp.patch_forward_backward_batch_for_diag_dump: "
                     "patch forward_backward_batch() for weight-gradient diagnostic dumps"),
        eager=True,
    ),

    # TrainingWorker.infer_batch → verl080_fsdp.patch_infer_batch
    # patch infer_batch() for step-level performance profiling
    PatchSpec(
        module_name="verl.workers.engine_workers",
        target_getter=lambda mod: (mod.TrainingWorker, "infer_batch"),
        patch_factory=patch_infer_batch,
        description=("TrainingWorker.infer_batch → verl080_fsdp.patch_infer_batch: "
                     "patch infer_batch() for step-level performance profiling"),
        eager=True,
    ),
]

# ═══════════════════════════════════════════════════════════════
# PrefixSharing Attention
# ═══════════════════════════════════════════════════════════════

# Attention patch: directly modify ALL_ATTENTION_FUNCTIONS dict (same approach as verl's PrefixGrouper).
# Cannot use PatchSpec because get_interface is not an attribute/key of AttentionInterface.
from .attention import install_attention_patch
install_attention_patch()
