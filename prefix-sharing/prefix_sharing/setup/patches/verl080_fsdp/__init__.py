"""verl 0.8.0 FSDP patch set.

Patch 目标：
1. FSDPEngineWithLMHead.forward_step → dense FSDP PrefixSharing forward helper
2. 性能验证相关 profiler 注入（不修改 verl 源码）
3. rollout/fixed-data 注入辅助

当前 patch set 是 FSDP 开源线的显式开发入口，需通过
``prefix_sharing.setup.install("verl080_fsdp")`` 安装；不要依赖默认兼容矩阵
自动选择，避免与 Megatron/MindSpeed patch set 混用。
"""

from prefix_sharing.setup.registry import PatchSpec

from .forward_step import patch_fsdp_forward_step
from .perf_profiler_patch import (
    patch_forward_backward_batch,
    patch_infer_batch,
    patch_train_batch,
    patch_train_mini_batch,
)
from .rollout_patch import patch_ray_trainer_fit


PATCH_SET: list[PatchSpec] = [
    # ═══════════════════════════════════════════════════════════════
    # 一、fix / feature 注入：PrefixSharing 核心功能与调试辅助
    # ═══════════════════════════════════════════════════════════════
    PatchSpec(
        module_name="verl.workers.engine.fsdp.transformer_impl",
        target_getter=lambda mod: (
            getattr(mod, "FSDPEngineWithLMHead"),
            "forward_step",
        ),
        patch_factory=patch_fsdp_forward_step,
        description="FSDPEngineWithLMHead.forward_step → PrefixSharing dense FSDP helper",
        eager=True,  # verl FSDP engine 仅在 actor 实例化时 lazy-load，必须 eager 触发
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
    # 二、性能验证：ProfilerScope 分层 profiling（对应原 verl 源码中的侵入式修改）
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
# 属于 fix/feature 注入。
from .attention import install_attention_patch
install_attention_patch()
