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
from prefix_sharing.setup.version_detector import detect_dependency_versions, DependencyDetectedVersions
from prefix_sharing.setup.compat_matrix import COMPAT_MATRIX, CompatEntry, IncompatibleEnvironment
from prefix_sharing.setup.patch_installer import (
    PatchHandle,
    PatchRegistry as PatchRegistry,
    PatchSpec,
    install_specs,
)


def detect_and_validate_dependency_versions() -> DependencyDetectedVersions:
    """Detect dependency versions and validate them against the compatibility matrix.

    Returns: detected dependency versions
    Raises: IncompatibleEnvironment — no matching patch set
    """
    dependency_versions = detect_dependency_versions()
    compat_entries = _find_compat_entries(dependency_versions)
    if not compat_entries:
        raise IncompatibleEnvironment(
            f"Incompatible dependency versions: verl={dependency_versions.verl}, "
            f"megatron_core={dependency_versions.megatron_core}, "
            f"mindspeed={dependency_versions.mindspeed}.\n"
            + _show_compat_matrix()
        )
    patch_set_ids = [entry.patch_set_id for entry in compat_entries]
    print(
        f"[PrefixSharing] Version check: verl={dependency_versions.verl}, "
        f"megatron_core={dependency_versions.megatron_core}, "
        f"mindspeed={dependency_versions.mindspeed} → compatible (patch_sets={patch_set_ids})"
    )
    return dependency_versions


def install(patch_set_id: str | None = None) -> PatchHandle:
    """Install prefix-sharing patches.

    By default installs every patch set that matches the current environment.
    When ``patch_set_id`` is given, only the specified patch set is installed.

    Returns: PatchHandle — call describe() for details, disable() to roll back
    Raises: IncompatibleEnvironment — when version combination is unsupported
    """
    patch_set_ids = _resolve_patch_set_ids(patch_set_id)
    if patch_set_id is not None:
        print(f"[PrefixSharing] install() using explicit patch_sets={patch_set_ids}")

    patch_specs: list[PatchSpec] = []
    for patch_set in patch_set_ids:
        patch_specs.extend(_load_patch_set(patch_set))
    patch_specs = _dedupe_patch_specs(patch_specs)

    handle = install_specs(patch_specs)

    print(
        f"[PrefixSharing] install() complete. {len(patch_specs)} patches active. patch_sets={patch_set_ids}"
    )
    return handle


def _resolve_patch_set_ids(patch_set_id: str | None) -> list[str]:
    def _dedupe(patch_set_ids: list[str]) -> list[str]:
        return list(dict.fromkeys(patch_set_ids))

    if patch_set_id is not None:
        patch_set_ids = [
            part.strip() for part in patch_set_id.split(",") if part.strip()
        ]
        if not patch_set_ids:
            raise ValueError("patch_set_id must not be empty")
        return _dedupe(patch_set_ids)

    dependency_versions = detect_and_validate_dependency_versions()
    compat_entries = _find_compat_entries(dependency_versions)
    if not compat_entries:
        raise IncompatibleEnvironment(
            f"Incompatible version combination: verl={dependency_versions.verl}, "
            f"megatron_core={dependency_versions.megatron_core}, "
            f"mindspeed={dependency_versions.mindspeed}.\n"
            + _show_compat_matrix()
        )
    return _dedupe([entry.patch_set_id for entry in compat_entries])


def _find_compat_entries(
    dependency_versions: DependencyDetectedVersions,
) -> list[CompatEntry]:
    return [entry for entry in COMPAT_MATRIX if entry.match(dependency_versions)]


def _find_compat_entry(
    dependency_versions: DependencyDetectedVersions,
) -> CompatEntry | None:
    compat_entries = _find_compat_entries(dependency_versions)
    return compat_entries[0] if compat_entries else None


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


def _show_compat_matrix() -> str:
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
