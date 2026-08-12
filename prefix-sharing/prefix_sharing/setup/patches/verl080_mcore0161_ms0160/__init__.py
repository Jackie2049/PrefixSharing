"""verl 0.8.0.dev + megatron-core 0.16.1 + mindspeed 0.16.0 (Qwen3.5 NPU companion)

Patch targets:
1. MegatronEngineWithLMHead.forward_step → micro-batch reorg + runtime context injection
2. Attention.forward                     → prefix-sharing attention intercept
3. vocab_parallel_log_probs_from_logits  → automatic logprob restore
4. no_padding_2_padding                  → sequence length correction after PS
   physical trimming (module-level + all from...import references)

All business logic is handled by the integrations layer; this patch set
only orchestrates thin wrappers.
"""

from prefix_sharing.setup.patch_installer import PatchSpec
from .forward_step import patch_verl_forward_step
from .attention import patch_megatron_attention
from .vocab_logprobs import patch_megatron_vocab
from .nopadding import patch_no_padding_2_padding

# no_padding_2_padding is directly referenced via from...import in 3 modules:
#   verl.workers.utils.padding           — original definition module
#   verl.workers.utils.losses            — called inside ppo_loss
#   verl.trainer.distillation.losses     — called inside distillation
#   verl.trainer.ppo.ray_trainer         — called on the trainer side
# from...import creates module-level attributes; setattr can update them.
# Each referencing module must be patched individually, otherwise its local
# reference still points to the original function.
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
        eager=True,  # verl Megatron engine is preloaded in __init__.py; import hooks cannot intercept
    ),
    PatchSpec(
        module_name="megatron.core.transformer.attention",
        target_getter=lambda mod: (getattr(mod, "Attention"), "forward"),
        patch_factory=patch_megatron_attention,
        description="Attention.forward → prefix-sharing intercept (mcore 0.16.1)",
        eager=True,  # megatron.core is already imported before engine initialization
    ),
    # The actual logprob call site in verl080 is inside the logits_processor
    # closure in transformer_impl (transformer_impl.py:932). That name is a
    # local reference bound at module load time via from...import. Patching
    # only the source module verl.utils.megatron.tensor_parallel would not
    # work (setattr on the source module does not change the local reference
    # in transformer_impl), so we must patch the transformer_impl module
    # attribute directly. grep confirms this is the only call site for this
    # function in verl080.
    #
    # Note: do NOT patch the source module at the same time. When restore
    # recomputes prefix-last logp (forward_step.py uses
    # ``from verl.utils.megatron.tensor_parallel import``), it needs the
    # original function; if the source module were patched, recomputation
    # would enter the patched_fn — the incoming logits would be only
    # [1, V//tp] while index.provider_1d_pos is a global packed offset,
    # producing an empty slice that triggers the RuntimeError in
    # vocab_logprobs.py.
    PatchSpec(
        module_name="verl.workers.engine.megatron.transformer_impl",
        target_getter=lambda mod: (mod, "vocab_parallel_log_probs_from_logits"),
        patch_factory=patch_megatron_vocab,
        description="vocab_parallel_log_probs → auto logprob restore (verl 0.8.0)",
        eager=True,  # Same as forward_step; module is already preloaded
    ),
] + [
    PatchSpec(
        module_name=mod_name,
        target_getter=lambda mod: (mod, "no_padding_2_padding"),
        patch_factory=patch_no_padding_2_padding,
        description=f"no_padding_2_padding in {mod_name} → PS trimming-aware",
        eager=True,  # These modules are already loaded by verl before PS hooks are installed
    )
    for mod_name in _NOPADDING_PATCH_MODULES
]