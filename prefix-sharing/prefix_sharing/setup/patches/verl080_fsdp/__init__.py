"""verl 0.8.0 FSDP patch set.

Patch targets:
1. FSDPEngineWithLMHead.forward_step → dense FSDP PrefixSharing forward helper
2. Profiler injection for performance validation (without modifying verl source)
3. Rollout / fixed-data injection helpers

This patch set is the default entry point for the FSDP open-source line.
It can be auto-selected via the compatibility matrix, or explicitly installed
via ``prefix_sharing.setup.install("verl080_fsdp")``.
"""

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
    # 1. Fix / feature injection: PrefixSharing core functionality and debug helpers
    # ═══════════════════════════════════════════════════════════════
    PatchSpec(
        module_name="verl.workers.engine.fsdp.transformer_impl",
        target_getter=lambda mod: (
            getattr(mod, "FSDPEngineWithLMHead"),
            "forward_step",
        ),
        patch_factory=patch_fsdp_forward_step,
        description="FSDPEngineWithLMHead.forward_step → PrefixSharing dense FSDP helper",
        eager=True,  # verl FSDP engine is lazy-loaded only when actor is instantiated; must trigger eagerly
    ),
    PatchSpec(
        module_name="verl.trainer.ppo.ray_trainer",
        target_getter=lambda mod: (mod.RayPPOTrainer, "fit"),
        patch_factory=patch_ray_trainer_fit,
        description=(
            "RayPPOTrainer.fit → intercept actor_rollout_wg + async_rollout_manager "
            "for PREFIX_SHARING_CAPTURE_ROLLOUT / PREFIX_SHARING_FIXED_ROLLOUT"
        ),
        eager=True,  # ray_trainer is imported by main_ppo at startup; eager ensures patch is in place
    ),
    # ═══════════════════════════════════════════════════════════════
    # 2. Performance validation: ProfilerScope layered profiling (replaces
    #    invasive source modifications in the original verl codebase)
    # ═══════════════════════════════════════════════════════════════
    PatchSpec(
        module_name="verl.workers.engine.base",
        target_getter=lambda mod: (mod.BaseEngine, "train_batch"),
        patch_factory=patch_train_batch,
        description="BaseEngine.train_batch → time optimizer_step as PHASE_UPDATE via ProfilerScope",
        eager=True,
    ),
    PatchSpec(
        module_name="verl.workers.engine.fsdp.transformer_impl",
        target_getter=lambda mod: (mod.FSDPEngine, "forward_backward_batch"),
        patch_factory=patch_forward_backward_batch,
        description="FSDPEngine.forward_backward_batch → micro-batch profiling without source edit",
        eager=True,
    ),
    PatchSpec(
        module_name="verl.workers.engine.fsdp.transformer_impl",
        target_getter=lambda mod: (mod.FSDPEngine, "forward_backward_batch"),
        patch_factory=patch_forward_backward_batch_for_diag_dump,
        description="FSDPEngine.forward_backward_batch → dump weight gradients after backward",
        eager=True,
    ),
    PatchSpec(
        module_name="verl.workers.engine_workers",
        target_getter=lambda mod: (mod.TrainingWorker, "train_mini_batch"),
        patch_factory=patch_train_mini_batch,
        description="TrainingWorker.train_mini_batch → step-level ProfilerScope (kind=train)",
        eager=True,
    ),
    PatchSpec(
        module_name="verl.workers.engine_workers",
        target_getter=lambda mod: (mod.TrainingWorker, "infer_batch"),
        patch_factory=patch_infer_batch,
        description="TrainingWorker.infer_batch → step-level ProfilerScope (kind=logp)",
        eager=True,
    ),
]

# Attention patch: directly modify ALL_ATTENTION_FUNCTIONS dict (same approach as verl's PrefixGrouper).
# Cannot use PatchSpec because get_interface is not an attribute/key of AttentionInterface.
from .attention import install_attention_patch
install_attention_patch()
