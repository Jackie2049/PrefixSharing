"""verl 0.8.0 FSDP patch set.

Patch 目标：
1. FSDPEngineWithLMHead.forward_step → dense FSDP PrefixSharing forward helper

当前 patch set 是 FSDP 开源线的默认入口，可通过兼容矩阵自动选择，也可通过
``prefix_sharing.setup.install("verl080_fsdp")`` 显式安装。
"""

from prefix_sharing.setup.registry import PatchSpec

from .forward_step import patch_fsdp_forward_step
from .attention import patch_transformers_attention


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
    PatchSpec(
        module_name="transformers",
        target_getter=lambda mod: (
            getattr(getattr(mod, "modeling_utils", None), "ALL_ATTENTION_FUNCTIONS", {}),
            "get_interface",
        ),
        patch_factory=patch_transformers_attention,
        description=(
            "ALL_ATTENTION_FUNCTIONS.get_interface → PrefixSharing-aware "
            "(HF attention KV store/load on Q-path kept tokens)"
        ),
        # transformers 在 Ray worker 启动早期就已加载，不设 eager 时 import hook 到 200 轮过期
        # 都等不到目标（worker 侧不发 stdlib import 事件）。改用 eager+import_module
        # 主动导入，确保注意力诊断 dump（attn_inputs/attn_outputs/expanded_kv）可用。
        eager=True,
    ),
]
