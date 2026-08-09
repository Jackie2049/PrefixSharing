"""prefix_sharing.setup.version_detector

Detect dependency versions in the current environment
e.g. verl, Megatron Core, and MindSpeed.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys


@dataclass(frozen=True)
class DetectedVersions:
    verl: str | None
    megatron_core: str | None
    mindspeed: str | None


def detect_versions() -> DetectedVersions:
    """Detect installed versions of dependencies.

    Lookup order: sys.modules → importlib.import_module → importlib.metadata.
    None means the library is not present in the current environment.
    """
    verl_ver = _detect_from_module("verl")
    mcore_ver = _detect_from_module("megatron.core")
    ms_ver = _detect_from_metadata("mindspeed") # MindSpeed has no __version__， read package metadata instead.

    print(
        f"[PrefixSharing] Detected versions: verl={verl_ver}, megatron_core={mcore_ver}, mindspeed={ms_ver}"
    )
    return DetectedVersions(
        verl=verl_ver,
        megatron_core=mcore_ver,
        mindspeed=ms_ver,
    )


def _detect_from_module(module_name: str, attr: str = "__version__") -> str | None:
    """Read a version attribute from a loaded or importable module."""
    if module_name in sys.modules:
        return getattr(sys.modules[module_name], attr, None)
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, attr, None)
    except ModuleNotFoundError:
        return None


def _detect_from_metadata(module_name: str, *, package: str | None = None) -> str | None:
    """Read version from package metadata after ensuring the module is importable.
    """
    package = package or module_name
    if module_name not in sys.modules:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            return None
    return _metadata_version(package)


def _metadata_version(package: str) -> str | None:
    """Read a package version from importlib.metadata."""
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return None
