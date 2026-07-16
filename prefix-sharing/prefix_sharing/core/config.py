"""Configuration and hard phase-1 constraint validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping


class PrefixSharingConfigError(ValueError):
    """Raised when prefix sharing is enabled under unsupported constraints."""


def _read_config_value(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _env_enables_prefix_sharing() -> bool:
    value = os.getenv("ENABLE_PREFIX_SHARING")
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "y"}:
        return True
    if normalized in {"0", "false", "no", "off", "n", ""}:
        return False
    raise PrefixSharingConfigError(
        "ENABLE_PREFIX_SHARING must be one of: 1/0, true/false, yes/no, on/off"
    )


def _to_plain_mapping(raw: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(raw):
            return dict(OmegaConf.to_container(raw, resolve=True))
    except ModuleNotFoundError:
        pass
    if isinstance(raw, Mapping):
        return dict(raw)
    if hasattr(raw, "__dict__"):
        return {
            key: value
            for key, value in vars(raw).items()
            if not key.startswith("_")
        }
    raise TypeError("prefix_sharing config must be a bool, mapping, or OmegaConf object")


@dataclass(frozen=True)
class PrefixSharingConfig:
    """User-facing phase-1 configuration.

    The defaults are deliberately conservative: enabling this feature without
    explicitly opting in should do nothing, and unsupported Megatron/verl paths
    should fail loudly instead of silently changing training semantics.
    """

    enable_prefix_sharing: bool = False
    detector: str = "trie"
    backend: str = "torch_ref"
    min_prefix_len: int = 1  # Prefixes shorter than this won't be cached (too short = not worth it)
    min_group_size: int = 2  # Groups smaller than this won't share (need 2+ samples to share)
    boundary_strategy: str = "prefix_last_restore"

    supported_cp_size: int = 1  # Context parallel size supported in phase 1 (1 = no CP)
    supported_rope_fusion: bool = False  # RoPE fusion kernel support (False = must disable)
    supported_fused_qkv_rope: bool = False  # Fused QKV+RoPE kernel support (False = must disable)

    validate_precision: bool = False
    integrate_mode: str = "verl_megatron_actor"
    model_type: str = "text_only_causal_lm"  # Model type identifier; phase 1 only supports text-only causal LM

    def __post_init__(self) -> None:
        _env_backend = os.getenv("PREFIX_SHARING_BACKEND")
        if _env_backend and self.backend == "torch_ref":
            object.__setattr__(self, "backend", _env_backend)

    @classmethod
    def from_raw(cls, raw: Any) -> "PrefixSharingConfig":
        """Build config from bool/mapping/object input and environment overrides.

        Environment variable ``ENABLE_PREFIX_SHARING`` is only consulted when
        the caller does not supply an explicit ``enable_prefix_sharing`` value
        (i.e. ``raw`` is ``None``, or a mapping that omits the key).  Explicit
        ``True`` / ``False`` always wins over the env var.
        """

        if raw is None:
            return cls(enable_prefix_sharing=_env_enables_prefix_sharing())
        if raw is False:
            return cls(enable_prefix_sharing=False)
        if raw is True:
            return cls(enable_prefix_sharing=True)
        values = _to_plain_mapping(raw)
        if "enable_prefix_sharing" not in values:
            values["enable_prefix_sharing"] = _env_enables_prefix_sharing()
        return cls(**values)

    def validate(self, model_config: Any | None = None, integrate_mode: str | None = None) -> None:
        """Validate phase-1 constraints against a model/config object.

        Args:
            model_config: Mapping or object with Megatron-like attributes.
            integrate_mode: Optional integration mode name.
        """

        if not self.enable_prefix_sharing:
            return
        if self.detector != "trie":
            raise PrefixSharingConfigError("phase 1 supports only detector='trie'")
        supported_backends = {"torch_ref", "flash_atten_gpu", "flash_atten_npu"}
        if self.backend not in supported_backends:
            raise PrefixSharingConfigError(
                f"backend='{self.backend}' is not supported. "
                f"Supported backends: {supported_backends}"
            )
        if self.boundary_strategy != "prefix_last_restore":
            raise PrefixSharingConfigError(
                "phase 1 currently implements only "
                "boundary_strategy='prefix_last_restore'; future strategies may include "
                "'boundary_token' and 'strict_suffix'"
            )
        if self.min_prefix_len < 1:
            raise PrefixSharingConfigError("min_prefix_len must be >= 1")
        if self.min_group_size < 2:
            raise PrefixSharingConfigError("min_group_size must be >= 2")

        active_mode = integrate_mode or self.integrate_mode
        if active_mode != "verl_megatron_actor":
            raise PrefixSharingConfigError(
                "phase 1 supports only integrate_mode='verl_megatron_actor'"
            )

        pp_size = _read_config_value(
            model_config,
            "pipeline_model_parallel_size",
            _read_config_value(model_config, "pipeline_parallel_size", 1),
        )
        virtual_pp_size = _read_config_value(model_config, "virtual_pipeline_model_parallel_size", None)
        num_layers_per_virtual_pipeline_stage = _read_config_value(
            model_config,
            "num_layers_per_virtual_pipeline_stage",
            None,
        )
        cp_size = _read_config_value(
            model_config,
            "context_parallel_size",
            self.supported_cp_size,
        )
        rope_fusion = _read_config_value(model_config, "apply_rope_fusion", False)
        fused_qkv_rope = _read_config_value(model_config, "fused_single_qkv_rope", False)
        model_type = _read_config_value(model_config, "model_type", "text_only_causal_lm")

        if int(pp_size) < 1:
            raise PrefixSharingConfigError(
                f"[Config Error] pipeline_model_parallel_size={pp_size} 不合法。"
                f"pipeline_model_parallel_size 必须 >= 1，"
                f"请修改为合法物理 PP 大小，或禁用 prefix sharing。"
            )
        if virtual_pp_size not in (None, 1):
            raise PrefixSharingConfigError(
                f"[Config Error] virtual_pipeline_model_parallel_size={virtual_pp_size} 不支持当前阶段。"
                f"当前仅支持物理 pipeline parallel，不支持 virtual pipeline parallel，"
                f"请关闭 virtual PP 或禁用 prefix sharing。"
            )
        if num_layers_per_virtual_pipeline_stage is not None:
            raise PrefixSharingConfigError(
                "[Config Error] num_layers_per_virtual_pipeline_stage 不支持当前阶段。"
                "当前仅支持物理 pipeline parallel，不支持 virtual pipeline parallel，"
                "请关闭 virtual PP 或禁用 prefix sharing。"
            )
        if cp_size != self.supported_cp_size:
            raise PrefixSharingConfigError(
                f"[Config Error] context_parallel_size={cp_size} 不支持当前阶段。"
                f"Phase 1 仅支持 context_parallel_size=1 (无上下文并行)，"
                f"请修改配置将 CP 大小设为 1，或禁用 prefix sharing。"
            )
        if not self.supported_rope_fusion and rope_fusion:
            raise PrefixSharingConfigError(
                "[Config Error] apply_rope_fusion=True 不支持当前阶段。"
                "Phase 1 要求关闭 rope fusion (apply_rope_fusion=False)，"
                "请修改配置或禁用 prefix sharing。"
            )
        if not self.supported_fused_qkv_rope and fused_qkv_rope:
            raise PrefixSharingConfigError(
                "[Config Error] fused_single_qkv_rope=True 不支持当前阶段。"
                "Phase 1 要求关闭 fused QKV rope (fused_single_qkv_rope=False)，"
                "请修改配置或禁用 prefix sharing。"
            )
        if self.model_type == "text_only_causal_lm" and model_type != "text_only_causal_lm":
            raise PrefixSharingConfigError(
                f"[Config Error] 当前模型类型 '{model_type}' 不支持当前阶段。"
                f"Phase 1 仅支持 model_type='text_only_causal_lm' (纯文本因果语言模型)，"
                f"请使用支持的模型类型或禁用 prefix sharing。"
            )

    def validate_for_engine(
        self,
        use_remove_padding: bool = True,
        integrate_mode: str = "verl_megatron_actor",
    ) -> None:
        """Validate phase-1 constraints for verl engine 架构（verl 0.8.0+）。

        与 validate() 不同，此方法从 engine_config 而非 model_config 读取配置。
        用于 setup/patches 中的 forward_step patch，此时只有 engine_config
        可用（self.engine_config），而非 Megatron TransformerConfig。
        """
        if not self.enable_prefix_sharing:
            return

        # 基础校验
        if self.detector != "trie":
            raise PrefixSharingConfigError("phase 1 supports only detector='trie'")
        if self.backend not in {"torch_ref", "flash_atten_gpu", "flash_atten_npu"}:
            raise PrefixSharingConfigError(
                f"backend='{self.backend}' is not supported. "
                f"Supported: torch_ref, flash_atten_gpu, flash_atten_npu"
            )
        if self.boundary_strategy != "prefix_last_restore":
            raise PrefixSharingConfigError(
                "phase 1 only supports boundary_strategy='prefix_last_restore'"
            )
        if self.min_prefix_len < 1:
            raise PrefixSharingConfigError("min_prefix_len must be >= 1")
        if self.min_group_size < 2:
            raise PrefixSharingConfigError("min_group_size must be >= 2")

        # THD packed layout 需要 use_remove_padding
        if not use_remove_padding:
            raise PrefixSharingConfigError(
                "[Config Error] Phase 1 THD 路径要求 use_remove_padding=True。"
                "BSHD 路径 (use_remove_padding=False) 尚未在当前 patch 中支持，"
                "请启用 use_remove_padding 或使用 BSHD 专用 patch set。"
            )
