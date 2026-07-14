from dataclasses import dataclass
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


def test_env_backend_enables_flex_debug_path_when_backend_is_default(monkeypatch):
    monkeypatch.setenv("PREFIX_SHARING_BACKEND", "flex_attention")

    config = PrefixSharingConfig(enable_prefix_sharing=True)

    assert config.backend == "flex_attention"


@pytest.mark.parametrize("block_size", [64, 128, 256])
def test_flex_attention_config_accepts_supported_block_sizes(block_size):
    config = PrefixSharingConfig(
        enable_prefix_sharing=True,
        backend="flex_attention",
        flex_attention_block_size=block_size,
    )

    config.validate(ModelConfig(), integrate_mode="verl_fsdp")


def test_flex_attention_config_rejects_invalid_block_size():
    config = PrefixSharingConfig(
        enable_prefix_sharing=True,
        backend="flex_attention",
        flex_attention_block_size=96,
    )

    with pytest.raises(PrefixSharingConfigError, match="flex_attention_block_size"):
        config.validate(ModelConfig(), integrate_mode="verl_fsdp")


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("context_parallel_size", 2, "context_parallel_size=2"),
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
