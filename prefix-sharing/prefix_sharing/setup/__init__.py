"""module: prefix_sharing.setup

This module provides the version guard and runtime patch injection functionality.

Usage:
    import prefix_sharing
    handle = prefix_sharing.setup.install() # install prefix-sharing's patches for the current environment
    print(handle.describe())
    handle.disable() # rollback above patches
"""

from __future__ import annotations

import importlib
from prefix_sharing.setup.version_detector import detect_versions, DetectedVersions
from prefix_sharing.setup.compat_matrix import COMPAT_MATRIX, CompatEntry, IncompatibleEnvironment
from prefix_sharing.setup.patch_installer import (
    PatchHandle,
    PatchRegistry as PatchRegistry,
    PatchSpec,
    install_specs,
)


def check() -> DetectedVersions:
    """Detect versions and validate compatibility without installing patches.

    Returns: detected version info
    Raises: IncompatibleEnvironment — no matching patch set
    """
    versions = detect_versions()
    entries = _find_compat_entries(versions)
    if not entries:
        raise IncompatibleEnvironment(
            f"Incompatible version combination: verl={versions.verl}, "
            f"megatron_core={versions.megatron_core}, "
            f"mindspeed={versions.mindspeed}.\n"
            + _format_compat_matrix()
        )
    patch_set_ids = [entry.patch_set_id for entry in entries]
    print(
        f"[PS] Version check: verl={versions.verl}, megatron_core={versions.megatron_core}, "
        f"mindspeed={versions.mindspeed} → compatible (patch_sets={patch_set_ids})"
    )
    return versions


def install(patch_set_id: str | None = None) -> PatchHandle:
    """Install prefix-sharing patches.

    By default installs every patch set that matches the current environment.
    When ``patch_set_id`` is given, only the specified patch set is installed.

    Returns: PatchHandle — call describe() for details, disable() to roll back
    Raises: IncompatibleEnvironment — version combination is unsupported
    """
    patch_set_ids = _resolve_patch_set_ids(patch_set_id)
    if patch_set_id is not None:
        print(f"[PS] install() using explicit patch_sets={patch_set_ids}")

    patch_specs: list[PatchSpec] = []
    for patch_set in patch_set_ids:
        patch_specs.extend(_load_patch_set(patch_set))
    patch_specs = _dedupe_patch_specs(patch_specs)

    handle = install_specs(patch_specs)

    print(
        f"[PS] install() complete. {len(patch_specs)} patches active. patch_sets={patch_set_ids}"
    )
    return handle


def _resolve_patch_set_ids(
    patch_set_id: str | None,
    *,
    versions: DetectedVersions | None = None,
) -> list[str]:
    if patch_set_id is not None:
        values = [value.strip() for value in patch_set_id.split(",") if value.strip()]
        if not values:
            raise ValueError("patch_set_id must not be empty")
        return _dedupe(values)

    versions = versions or check()
    entries = _find_compat_entries(versions)
    if not entries:
        raise IncompatibleEnvironment(
            f"Incompatible version combination: verl={versions.verl}, "
            f"megatron_core={versions.megatron_core}, "
            f"mindspeed={versions.mindspeed}.\n"
            + _format_compat_matrix()
        )
    return _dedupe([entry.patch_set_id for entry in entries])


def _find_compat_entries(versions: DetectedVersions) -> list[CompatEntry]:
    return [entry for entry in COMPAT_MATRIX if entry.match(versions)]


def _find_compat_entry(versions: DetectedVersions) -> CompatEntry | None:
    entries = _find_compat_entries(versions)
    return entries[0] if entries else None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _dedupe_patch_specs(specs: list[PatchSpec]) -> list[PatchSpec]:
    seen: set[tuple[str, str]] = set()
    result: list[PatchSpec] = []
    for spec in specs:
        key = (spec.module_name, spec.description)
        if key in seen:
            continue
        seen.add(key)
        result.append(spec)
    return result


def _load_patch_set(patch_set_id: str) -> list[PatchSpec]:
    mod = importlib.import_module(
        f"prefix_sharing.setup.patches.{patch_set_id}"
    )
    return mod.PATCH_SET


def _format_compat_matrix() -> str:
    lines = ["Supported combinations:"]
    for e in COMPAT_MATRIX:
        parts = []
        if e.verl is not None:
            parts.append(f"verl={e.verl}")
        if e.megatron_core == "*":
            parts.append("megatron-core=*")
        else:
            parts.append(f"megatron-core={e.megatron_core}")
        if e.mindspeed is not None:
            parts.append(f"mindspeed={e.mindspeed}")
        lines.append(f"  combination {e.patch_set_id}: " + " + ".join(parts))
    return "\n".join(lines)
