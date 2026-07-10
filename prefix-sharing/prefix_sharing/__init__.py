"""Prefix sharing phase-1 package.

The public API intentionally starts from framework-independent core pieces.
Integrations install patches around verl/Megatron, but the semantics live here.

Monkey-patch activation:

    Patch 在 ``import prefix_sharing`` 时自动安装，无需额外配置。
    版本组合不在 compat matrix 中时，patch 安装跳过并记录 warning。

    每个 micro-batch 是否执行 prefix sharing 逻辑由开关控制，两者平权，
    任意一个开启即可生效：

    * 配置文件 ``prefix_sharing_config.enable_prefix_sharing: true/false`` 优先；
    * 环境变量 ``ENABLE_PREFIX_SHARING=1/0`` 作为回退。
    * 都未配置时默认关闭。

    Patch 在关闭时无害——各 wrapper 检测到无开关或 context 时直接走原生路径。

    无需修改 verl/Megatron 源码。
"""


from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError
from prefix_sharing.core.prefix_detector import PrefixReuseSpec, TriePrefixDetector
from prefix_sharing.core.planner import PrefixLastRestoreSpec, PrefixSharingPlan, PrefixSharingPlanner


__all__ = [
    "PrefixSharingPlan",
    "PrefixSharingConfig",
    "PrefixSharingConfigError",
    "PrefixLastRestoreSpec",
    "PrefixReuseSpec",
    "PrefixSharingPlanner",
    "TriePrefixDetector",
]

# ── Monkey-patch auto-activation ──
# Patches are always installed on import. Each patch wrapper checks the switch
# (config or env) at runtime and falls through to the native path when disabled.
# This allows both ENABLE_PREFIX_SHARING env var and prefix_sharing_config yaml
# key to independently control the feature — no separate "install gate" needed.
#
# The setup.install() call is safe even when verl/Megatron are not present:
# if the detected versions match no compat matrix entry, it raises
# IncompatibleEnvironment which we catch and log as a warning (no patches
# applied, training proceeds normally without prefix-sharing).
_patch_handle = None  # module-level reference for introspection / rollback


def _auto_install_patches() -> None:
    """Install monkey patches on import.

    Patch set selection order (first match wins):

    1. ``PREFIX_SHARING_PATCHSET`` env var — explicit patch set id (e.g.
       ``verl080_fsdp``). Recommended for environments that install multiple
       backends (Megatron + FSDP) where auto-detection would be ambiguous.
       Aligns with docs/feature-fsdp.md §4.6.1 guidance.
    2. Compat matrix auto-detection — picks a patch set based on detected
       verl / megatron-core / mindspeed versions. Default path when no env
       var is set.

    Each patch wrapper checks the runtime switch (PrefixSharingConfig.from_raw,
    which respects both config file and env var) — when disabled, the wrapper
    passes through to the native path.

    Only runs once per process. Incompatible version combos are caught and
    logged without halting training.
    """
    global _patch_handle

    if _patch_handle is not None:
        print("[PS] Patches already installed, skipping.")
        return

    import os

    explicit_patch_set = os.getenv("PREFIX_SHARING_PATCHSET")
    try:
        from prefix_sharing.setup import install

        if explicit_patch_set:
            _patch_handle = install(explicit_patch_set)
            print(
                f"[PS] Auto-activation via PREFIX_SHARING_PATCHSET={explicit_patch_set}: "
                f"{_patch_handle.describe()}"
            )
        else:
            _patch_handle = install()
            print(f"[PS] Auto-activation succeeded: {_patch_handle.describe()}")
    except Exception as exc:
        # IncompatibleEnvironment or import errors — log and continue
        # Training proceeds normally without prefix-sharing patches.
        print(
            f"[PS] Auto-activation skipped: {exc}. "
            f"Training will proceed without prefix-sharing patches."
        )


_auto_install_patches()
