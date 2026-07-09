from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import pytest

from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError


@dataclass
class ModelConfig:
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    num_layers_per_virtual_pipeline_stage: Optional[int] = None
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False
    context_parallel_size: int = 1
    context_parallel_algo: Optional[str] = None
    override_transformer_config: Optional[object] = None
    dynamic_context_parallel: bool = False
    apply_rope_fusion: bool = False
    fused_single_qkv_rope: bool = False
    model_type: str = "text_only_causal_lm"
    use_remove_padding: bool = False
    ulysses_sequence_parallel_size: int = 1
    use_fused_kernels: bool = False


def test_disabled_config_does_not_validate_model_constraints():
    config = PrefixSharingConfig(enable_prefix_sharing=False)
    config.validate(ModelConfig(pipeline_model_parallel_size=8))


def test_enabled_config_accepts_phase_one_constraints():
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    config.validate(ModelConfig())


def test_enabled_config_accepts_verl_fsdp_integrate_mode():
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    config.validate(ModelConfig(), integrate_mode="verl_fsdp")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"ulysses_sequence_parallel_size": 2}, "ulysses_sequence_parallel_size=2"),
        ({"use_fused_kernels": True}, "use_fused_kernels=True"),
    ],
)
def test_verl_fsdp_config_rejects_unsupported_runtime_modes(kwargs, message):
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    with pytest.raises(PrefixSharingConfigError, match=message):
        config.validate(ModelConfig(**kwargs), integrate_mode="verl_fsdp")


@pytest.mark.parametrize("pp_size", [1, 2, 4, 8])
def test_enabled_config_accepts_physical_pipeline_parallel_sizes(pp_size):
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    config.validate(ModelConfig(pipeline_model_parallel_size=pp_size))


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_enabled_config_accepts_megatron_sequence_parallel_with_common_tp_sizes(tp_size):
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    config.validate(ModelConfig(tensor_model_parallel_size=tp_size, sequence_parallel=True))


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "y"])
def test_env_var_can_enable_prefix_sharing(monkeypatch, value):
    monkeypatch.setenv("ENABLE_PREFIX_SHARING", value)

    config = PrefixSharingConfig.from_raw(None)

    assert config.enable_prefix_sharing is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "n", ""])
def test_env_var_false_values_do_not_enable_prefix_sharing(monkeypatch, value):
    monkeypatch.setenv("ENABLE_PREFIX_SHARING", value)

    config = PrefixSharingConfig.from_raw(None)

    assert config.enable_prefix_sharing is False


def test_env_var_rejects_invalid_prefix_sharing_value(monkeypatch):
    monkeypatch.setenv("ENABLE_PREFIX_SHARING", "maybe")

    with pytest.raises(PrefixSharingConfigError, match="ENABLE_PREFIX_SHARING"):
        PrefixSharingConfig.from_raw(None)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("apply_rope_fusion", True, "apply_rope_fusion=True"),
        ("fused_single_qkv_rope", True, "fused_single_qkv_rope=True"),
        ("model_type", "vlm", "model_type"),
    ],
)
def test_enabled_config_rejects_unsupported_phase_one_constraints(field, value, message):
    model_config = ModelConfig()
    setattr(model_config, field, value)
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    with pytest.raises(PrefixSharingConfigError, match=message):
        config.validate(model_config)


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_enabled_config_accepts_static_kvallgather_context_parallel(cp_size):
    config = PrefixSharingConfig(enable_prefix_sharing=True)

    config.validate(
        ModelConfig(
            context_parallel_size=cp_size,
            context_parallel_algo="kvallgather_cp_algo",
            use_remove_padding=True,
        )
    )


def test_enabled_config_accepts_context_parallel_algo_from_override_transformer_config():
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    model_config = ModelConfig(
        context_parallel_size=2,
        use_remove_padding=True,
        override_transformer_config=SimpleNamespace(
            context_parallel_algo="kvallgather_cp_algo",
        ),
    )

    config.validate(model_config)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (
            {
                "context_parallel_size": 2,
                "context_parallel_algo": "ulysses_cp_algo",
                "use_remove_padding": True,
            },
            "context_parallel_algo",
        ),
        (
            {
                "context_parallel_size": 2,
                "context_parallel_algo": "kvallgather_cp_algo",
                "use_remove_padding": False,
            },
            "use_remove_padding=True",
        ),
        (
            {
                "context_parallel_size": 2,
                "context_parallel_algo": "kvallgather_cp_algo",
                "use_remove_padding": True,
                "dynamic_context_parallel": True,
            },
            "dynamic_context_parallel=True",
        ),
    ],
)
def test_enabled_config_rejects_unsupported_context_parallel_variants(kwargs, message):
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    with pytest.raises(PrefixSharingConfigError, match=message):
        config.validate(ModelConfig(**kwargs))


def test_validate_for_engine_accepts_static_kvallgather_context_parallel():
    config = PrefixSharingConfig(enable_prefix_sharing=True)

    config.validate_for_engine(
        use_remove_padding=True,
        context_parallel_size=2,
        context_parallel_algo="kvallgather_cp_algo",
        dynamic_context_parallel=False,
    )


def test_validate_for_engine_rejects_non_kvallgather_context_parallel():
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    with pytest.raises(PrefixSharingConfigError, match="context_parallel_algo"):
        config.validate_for_engine(
            use_remove_padding=True,
            context_parallel_size=2,
            context_parallel_algo="ring_cp_algo",
        )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("pipeline_model_parallel_size", 0, "pipeline_model_parallel_size"),
        ("virtual_pipeline_model_parallel_size", 2, "virtual_pipeline_model_parallel_size"),
        ("num_layers_per_virtual_pipeline_stage", 8, "num_layers_per_virtual_pipeline_stage"),
    ],
)
def test_enabled_config_rejects_unsupported_pipeline_parallel_variants(field, value, message):
    model_config = ModelConfig()
    setattr(model_config, field, value)
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    with pytest.raises(PrefixSharingConfigError, match=message):
        config.validate(model_config)


def test_enabled_config_rejects_non_phase_one_modes():
    with pytest.raises(PrefixSharingConfigError, match="detector"):
        PrefixSharingConfig(enable_prefix_sharing=True, detector="prompt").validate(ModelConfig())
    with pytest.raises(PrefixSharingConfigError, match="boundary_strategy"):
        PrefixSharingConfig(enable_prefix_sharing=True, boundary_strategy="restore_last_prefix_token").validate(ModelConfig())
    with pytest.raises(PrefixSharingConfigError, match="integrate_mode"):
        PrefixSharingConfig(enable_prefix_sharing=True).validate(ModelConfig(), integrate_mode="unknown")
