"""verl 0.8.0 FSDP patch set.

Patch 目标：
1. FSDPEngineWithLMHead.forward_step → dense FSDP PrefixSharing forward helper

当前 patch set 是 FSDP 开源线的显式开发入口，需通过
``prefix_sharing.setup.install("verl080_fsdp")`` 安装；不要依赖默认兼容矩阵
自动选择，避免与 Megatron/MindSpeed patch set 混用。
"""

from prefix_sharing.setup.registry import PatchSpec

from .forward_step import patch_fsdp_forward_step
from .rollout_patch import patch_ray_trainer_fit


PATCH_SET: list[PatchSpec] = [
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
    # Attention patch is applied directly to ALL_ATTENTION_FUNCTIONS dict below;
    # PatchSpec-based approach does not work because get_interface is not a dict key
    # in this transformers version (AttentionInterface has __getitem__ not __getattr__).
    # PatchSpec(
    #     module_name="transformers.modeling_utils",
    #     target_getter=lambda mod: (
    #         mod.ALL_ATTENTION_FUNCTIONS,
    #         "get_interface",
    #     ),
    #     patch_factory=patch_transformers_attention,
    #     description=(
    #         "ALL_ATTENTION_FUNCTIONS.get_interface → PrefixSharing-aware "
    #         "(HF attention KV store/load on Q-path kept tokens)"
    #     ),
    #     eager=True,  # transformers 在 worker 启动早期就加载，必须 eager 立即 patch
    # ),
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
]

# Attention patch: directly modify ALL_ATTENTION_FUNCTIONS dict (same approach as verl's PrefixGrouper).
# Cannot use PatchSpec because get_interface is not an attribute/key of AttentionInterface.
from .attention import install_attention_patch
install_attention_patch()
