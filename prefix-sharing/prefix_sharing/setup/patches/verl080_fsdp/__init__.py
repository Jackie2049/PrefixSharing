"""verl 0.8.0 FSDP patch set.

Patch 目标：
1. FSDPEngineWithLMHead.forward_step → dense FSDP PrefixSharing forward helper

当前 patch set 是 FSDP 开源线的显式开发入口，需通过
``prefix_sharing.setup.install("verl080_fsdp")`` 安装；不要依赖默认兼容矩阵
自动选择，避免与 Megatron/MindSpeed patch set 混用。
"""

from prefix_sharing.setup.registry import PatchSpec

from .forward_step import patch_fsdp_forward_step


PATCH_SET: list[PatchSpec] = [
    PatchSpec(
        module_name="verl.workers.engine.fsdp.transformer_impl",
        target_getter=lambda mod: (
            getattr(mod, "FSDPEngineWithLMHead"),
            "forward_step",
        ),
        patch_factory=patch_fsdp_forward_step,
        description="FSDPEngineWithLMHead.forward_step → PrefixSharing dense FSDP helper",
    ),
]
