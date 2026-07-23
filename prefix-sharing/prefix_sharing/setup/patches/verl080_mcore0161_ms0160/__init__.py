"""verl 0.8.0.dev + megatron-core 0.16.1 + mindspeed 0.16.0（Qwen3.5 NPU 配套）

Patch 目标：
1. MegatronEngineWithLMHead.forward_step → 微批次重组 + runtime context 注入
2. Attention.forward                     → prefix-sharing attention 拦截
3. vocab_parallel_log_probs_from_logits  → 自动 logprob restore
4. no_padding_2_padding                  → PS 物理裁剪后修正序列长度
   (module-level + 所有 from...import 引用)
5. main_ppo.run_ppo                      → fixed rollout + synthetic prefix injection

所有业务逻辑由 integrations 层处理，本 patch set 只负责 thin wrapper 编排。
"""

from prefix_sharing.setup.registry import PatchSpec
from .forward_step import patch_verl_forward_step
from .attention import patch_megatron_attention
from .vocab_logprobs import patch_megatron_vocab
from .nopadding import patch_no_padding_2_padding
from .fit_hooks import create_run_ppo_patch

# no_padding_2_padding 被 4 个模块用 from...import 直接引用
_NOPADDING_PATCH_MODULES = [
    "verl.workers.utils.padding",
    "verl.workers.utils.losses",
    "verl.trainer.distillation.losses",
    "verl.trainer.ppo.ray_trainer",
]

PATCH_SET: list[PatchSpec] = [
    PatchSpec(
        module_name="verl.workers.engine.megatron.transformer_impl",
        target_getter=lambda mod: (
            getattr(mod, "MegatronEngineWithLMHead"),
            "forward_step",
        ),
        patch_factory=patch_verl_forward_step,
        description="MegatronEngineWithLMHead.forward_step → "
                    "micro-batch reorg + context (verl 0.8.0 engine)",
    ),
    PatchSpec(
        module_name="megatron.core.transformer.attention",
        target_getter=lambda mod: (getattr(mod, "Attention"), "forward"),
        patch_factory=patch_megatron_attention,
        description="Attention.forward → prefix-sharing intercept (mcore 0.16.1)",
    ),
    # verl080 算 logprob 的真正调用点在 transformer_impl 的 logits_processor
    # 闭包里（transformer_impl.py:932），该名字是模块加载时 from...import 绑定
    # 的局部引用。只 patch 源模块 verl.utils.megatron.tensor_parallel 无法命中
    # （setattr 源模块属性不会改 transformer_impl 的局部引用），必须直接 patch
    # transformer_impl 模块属性。grep 确认 verl080 里仅此一处调用该函数。
    #
    # 注意：不要同时 patch 源模块。restore 侧重算 prefix-last logp 时
    # （forward_step.py 内 from verl.utils.megatron.tensor_parallel import）
    # 需要拿原始函数；若源模块被 patch，重算会误入 patched_fn——传入 logits 仅
    # [1, V//tp]，而 index.provider_1d_pos 是全局 packed 偏移，切片为空会触发
    # vocab_logprobs.py 的 empty-slice RuntimeError。
    PatchSpec(
        module_name="verl.workers.engine.megatron.transformer_impl",
        target_getter=lambda mod: (mod, "vocab_parallel_log_probs_from_logits"),
        patch_factory=patch_megatron_vocab,
        description="vocab_parallel_log_probs → auto logprob restore (verl 0.8.0)",
    ),
    PatchSpec(
        module_name="verl.trainer.main_ppo",
        target_getter=lambda mod: (mod, "run_ppo"),
        patch_factory=create_run_ppo_patch,
        description="run_ppo → fixed rollout + synthetic prefix injection (after init_workers)",
    ),
] + [
    PatchSpec(
        module_name=mod_name,
        target_getter=lambda mod: (mod, "no_padding_2_padding"),
        patch_factory=patch_no_padding_2_padding,
        description=f"no_padding_2_padding in {mod_name} → PS trimming-aware",
    )
    for mod_name in _NOPADDING_PATCH_MODULES
]
