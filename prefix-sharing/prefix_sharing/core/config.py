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
        """Validate constraints against model_config.

        Args:
            model_config: Model attributes.
            integrate_mode: Optional integration mode name.
        """

        if not self.enable_prefix_sharing:
            return

        # detector guard
        supported_detectors = {"trie"}
        if self.detector not in supported_detectors:
            raise PrefixSharingConfigError(
                f"detector='{self.detector}' is not supported. Supported detectors: {supported_detectors}"
            )

        # backend guard
        supported_backends = {"torch_ref", "flash_atten_gpu", "flash_atten_npu"}
        if self.backend not in supported_backends:
            raise PrefixSharingConfigError(
                f"backend='{self.backend}' is not supported. Supported backends: {supported_backends}"
            )

        # boundary_strategy guard
        supported_boundary_strategies = {"prefix_last_restore"}
        if self.boundary_strategy not in supported_boundary_strategies:
            raise PrefixSharingConfigError(
                f"boundary_strategy='{self.boundary_strategy}' is not supported. Supported strategies: {supported_boundary_strategies}"
            )

        # min_prefix_len guard
        if self.min_prefix_len < 1:
            raise PrefixSharingConfigError(f"min_prefix_len='{self.min_prefix_len}' is not supported. Supported min_prefix_len: >= 1")
        if self.min_group_size < 2:
            raise PrefixSharingConfigError(f"min_group_size='{self.min_group_size}' is not supported. Supported min_group_size: >= 2")

        # integrate_mode guard
        supported_integrate_modes = {"verl_megatron_actor", "verl_fsdp"}
        active_mode = integrate_mode or self.integrate_mode
        if active_mode not in supported_integrate_modes:
            raise PrefixSharingConfigError(
                f"integrate_mode='{active_mode}' is not supported. Supported integrate_modes: {supported_integrate_modes}"
            )

        # model_type guard
        supported_model_types = {"text_only_causal_lm"}
        model_type = _read_config_value(model_config, "model_type", "text_only_causal_lm")
        if self.model_type in supported_model_types and model_type not in supported_model_types:
            raise PrefixSharingConfigError(
                f"model_type='{model_type}' is not supported. Supported model_types: {supported_model_types}"
            )

        # verl_fsdp guard
        if active_mode == "verl_fsdp":
            ulysses_sp_size = _read_config_value(model_config, "ulysses_sequence_parallel_size", 1)
            use_fused_kernels = _read_config_value(model_config, "use_fused_kernels", False)
            if int(ulysses_sp_size) != 1:
                raise PrefixSharingConfigError(
                    f"ulysses_sequence_parallel_size={ulysses_sp_size} is not supported. "
                    "Supported ulysses_sequence_parallel_size: 1"
                )
            if use_fused_kernels:
                raise PrefixSharingConfigError(
                    f"use_fused_kernels={use_fused_kernels} is not supported. "
                    "Supported use_fused_kernels: False"
                )
            return

        # parallel strategy guards
        pp_size = _read_config_value(
            model_config,
            "pipeline_model_parallel_size",
            _read_config_value(model_config, "pipeline_parallel_size", 1),
        )
        if int(pp_size) < 1:
            raise PrefixSharingConfigError(
                f"pipeline_model_parallel_size={pp_size} is not supported. "
                "Supported pipeline_model_parallel_size: >= 1"
            )
        virtual_pp_size = _read_config_value(model_config, "virtual_pipeline_model_parallel_size", 1)
        if virtual_pp_size not in (None, 1):
            raise PrefixSharingConfigError(
                f"virtual_pipeline_model_parallel_size={virtual_pp_size} is not supported. "
                "Supported virtual_pipeline_model_parallel_size: 1"
            )
        num_layers_per_virtual_pipeline_stage = _read_config_value(
            model_config, "num_layers_per_virtual_pipeline_stage", None
        )
        if num_layers_per_virtual_pipeline_stage is not None:
            raise PrefixSharingConfigError(
                f"num_layers_per_virtual_pipeline_stage={num_layers_per_virtual_pipeline_stage} "
                "is not supported. Supported num_layers_per_virtual_pipeline_stage: None"
            )
        cp_size = _read_config_value(model_config, "context_parallel_size", self.supported_cp_size)
        if cp_size != self.supported_cp_size:
            raise PrefixSharingConfigError(
                f"context_parallel_size={cp_size} is not supported. "
                f"Supported context_parallel_size: {self.supported_cp_size}"
            )
        rope_fusion = _read_config_value(model_config, "apply_rope_fusion", False)
        if not self.supported_rope_fusion and rope_fusion:
            raise PrefixSharingConfigError(
                f"apply_rope_fusion={rope_fusion} is not supported. "
                "Supported apply_rope_fusion: False"
            )
        fused_qkv_rope = _read_config_value(model_config, "fused_single_qkv_rope", False)
        if not self.supported_fused_qkv_rope and fused_qkv_rope:
            raise PrefixSharingConfigError(
                f"fused_single_qkv_rope={fused_qkv_rope} is not supported. "
                "Supported fused_single_qkv_rope: False"
            )

    def validate_for_engine(
        self,
        use_remove_padding: bool = True,
        integrate_mode: str = "verl_megatron_actor",
    ) -> None:
        """Validate phase-1 constraints for the verl engine (verl 0.8.0+).

        Unlike ``validate()``, this reads constraints from engine-facing flags
        rather than Megatron ``model_config``. Used by setup/patches
        ``forward_step`` when only ``self.engine_config`` is available.
        """
        if not self.enable_prefix_sharing:
            return

        # Basic field checks.
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

        # THD packed layout requires use_remove_padding.
        if not use_remove_padding:
            raise PrefixSharingConfigError(
                "[Config Error] Phase 1 THD path requires use_remove_padding=True. "
                "BSHD (use_remove_padding=False) is not supported by the current "
                "patch set. Enable use_remove_padding or use a BSHD-specific patch set."
            )
