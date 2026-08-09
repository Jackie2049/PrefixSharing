"""Prefix sharing package.

The public API intentionally starts from framework-independent core pieces.
Integrations install patches around verl/Megatron, but the semantics live here.

Patches are installed automatically when ``import prefix_sharing`` is executed.
No modification of verl/Megatron source code is required.

If the detected version combination is not in the compatibility matrix, patch
installation is skipped and training proceeds normally without prefix sharing.

Whether prefix sharing runs for a given micro-batch is controlled by a runtime
switch. The preferred verl entry point is ``actor.use_prefix_grouper: true``
with ``prefix_grouper.mode: arbitrary_prefix``. Explicit
``prefix_sharing_config.enable_prefix_sharing: true`` remains a compatibility
entry point. ``ENABLE_PREFIX_SHARING=1`` is a development fallback when the
configuration does not set an explicit value.

If no switch is set, prefix sharing is disabled by default. Patches are
harmless when disabled: wrappers fall through to the native path when no switch
or context is active.
"""


import os

from prefix_sharing.core.config import PrefixSharingConfig, PrefixSharingConfigError
from prefix_sharing.core.prefix_detector import PrefixReuseSpec, TriePrefixDetector
from prefix_sharing.core.planner import PrefixLastRestoreSpec, PrefixSharingPlan, PrefixSharingPlanner
from . import setup


__all__ = [
    "PrefixSharingPlan",
    "PrefixSharingConfig",
    "PrefixSharingConfigError",
    "PrefixLastRestoreSpec",
    "PrefixReuseSpec",
    "PrefixSharingPlanner",
    "TriePrefixDetector",
]

_patch_handle = None  # Set by _install_patch_on_import(); used for introspection/rollback.


def _install_patch_on_import() -> None:
    """Install monkey patches on import.

    Patch set selection order:

    1. ``PREFIX_SHARING_PATCHSET`` env var — explicit patch set id(s), e.g.
       ``verl080_fsdp`` or ``verl080_fsdp,verl080_mcore0161_ms0160``. This is
       intended for debugging or narrowing patch scope.
    2. Compat matrix auto-detection — installs all patch sets matching the
       detected verl / megatron-core / mindspeed versions. Default path when no
       env var is set.

    Each patch wrapper checks the runtime switch (PrefixSharingConfig.from_raw,
    which respects both config file and env var) — when disabled, the wrapper
    passes through to the native path.

    ``setup.install()`` is safe even when verl/Megatron are not present:
    mismatched versions raise ``IncompatibleEnvironment``, which is caught and
    logged without interrupting training. Only runs once per process.
    """
    global _patch_handle

    if _patch_handle is not None:
        print("[PrefixSharing] Patches already installed, skip it.")
        return

    explicit_patch_set = os.getenv("PREFIX_SHARING_PATCHSET")

    try:
        if explicit_patch_set:
            _patch_handle = setup.install(explicit_patch_set)
        else:
            _patch_handle = setup.install()
        print(f"[PrefixSharing] Patch installation succeeded: {_patch_handle.describe()}")
    except Exception as exc:
        # IncompatibleEnvironment or import errors — log and continue
        # Training proceeds normally without PrefixSharing features.
        print(
            f"[PrefixSharing] Patch installation skipped: {exc}. "
            f"Training will proceed without PrefixSharing features."
        )

_install_patch_on_import()
