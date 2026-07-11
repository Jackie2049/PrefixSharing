"""setup — 版本门卫 + 条件化运行时 patch 注入。

使用：
    import prefix_sharing
    handle = prefix_sharing.setup.install()
    print(handle.describe())
    handle.disable()
"""

from __future__ import annotations

import importlib
from prefix_sharing.setup.version_guard import detect_versions, DetectedVersions
from prefix_sharing.setup.compat_matrix import COMPAT_MATRIX, CompatEntry
from prefix_sharing.setup.registry import PatchSpec, PatchRegistry
from prefix_sharing.setup.logged_patch import PatchHandle


class IncompatibleEnvironment(RuntimeError):
    """版本组合不在兼容矩阵中。"""


def check() -> DetectedVersions:
    """仅探测版本并校验兼容性，不安装 patch。

    Returns: 探测到的版本信息
    Raises: IncompatibleEnvironment — 没有任何兼容 patch set
    """
    versions = detect_versions()
    entries = _find_compat_entries(versions)
    if not entries:
        raise IncompatibleEnvironment(
            f"不兼容的版本组合: verl={versions.verl}, "
            f"megatron_core={versions.megatron_core}, "
            f"mindspeed={versions.mindspeed}。\n"
            + _format_compat_matrix()
        )
    patch_set_ids = [entry.patch_set_id for entry in entries]
    print(
        f"[PS] Version check: verl={versions.verl}, megatron_core={versions.megatron_core}, "
        f"mindspeed={versions.mindspeed} → compatible (patch_sets={patch_set_ids})"
    )
    return versions


def install(patch_set_id: str | None = None) -> PatchHandle:
    """安装 prefix-sharing patch。

    默认安装当前环境所有匹配的 patch sets；显式传入 patch_set_id 时
    只安装指定 patch set。显式值支持逗号分隔，便于调试时限制 patch 范围。

    Returns: PatchHandle — 可调用 describe() 查看详情、disable() 回滚
    Raises: IncompatibleEnvironment — 版本组合不兼容
    """
    patch_set_ids = _resolve_patch_set_ids(patch_set_id)
    if patch_set_id is not None:
        print(f"[PS] install() using explicit patch_sets={patch_set_ids}")

    patch_specs: list[PatchSpec] = []
    for patch_set in patch_set_ids:
        patch_specs.extend(_load_patch_set(patch_set))
    patch_specs = _dedupe_patch_specs(patch_specs)

    handle = PatchRegistry.install_specs(patch_specs)

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
            f"不兼容的版本组合: verl={versions.verl}, "
            f"megatron_core={versions.megatron_core}, "
            f"mindspeed={versions.mindspeed}。\n"
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
    lines = ["支持的组合："]
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
        lines.append(f"  组合{e.patch_set_id}: " + " + ".join(parts))
    return "\n".join(lines)
