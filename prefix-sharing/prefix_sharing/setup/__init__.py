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
    Raises: IncompatibleEnvironment — 版本组合不兼容
    """
    versions = detect_versions()
    entry = _find_compat_entry(versions)
    if entry is None:
        raise IncompatibleEnvironment(
            f"不兼容的版本组合: verl={versions.verl}, "
            f"megatron_core={versions.megatron_core}, "
            f"mindspeed={versions.mindspeed}。\n"
            + _format_compat_matrix()
        )
    print(
        f"[PS] Version check: verl={versions.verl}, megatron_core={versions.megatron_core}, "
        f"mindspeed={versions.mindspeed} → compatible (patch_set={entry.patch_set_id})"
    )
    return versions


def install(patch_set_id: str | None = None) -> PatchHandle:
    """一键安装：版本探测 → 矩阵匹配 → 注册 patch → 应用 → 返回 handle。

    Args:
        patch_set_id: 显式指定 patch set。FSDP 开发线建议使用
            ``install("verl080_fsdp")``，避免在同时安装 Megatron/MindSpeed 的
            环境中被兼容矩阵自动选到 Megatron patch set。

    Returns: PatchHandle — 可调用 describe() 查看详情、disable() 回滚
    Raises: IncompatibleEnvironment — 版本组合不兼容
    """
    if patch_set_id is None:
        versions = check()
        entry = _find_compat_entry(versions)
        patch_set_id = entry.patch_set_id
    else:
        print(f"[PS] install() using explicit patch_set={patch_set_id}")
    patch_set = _load_patch_set(patch_set_id)

    for spec in patch_set:
        PatchRegistry.register(spec)

    handle = PatchRegistry.install_all()

    print(
        f"[PS] install() complete. {len(patch_set)} patches active. patch_set={patch_set_id}"
    )
    return handle


def _find_compat_entry(versions: DetectedVersions) -> CompatEntry | None:
    for entry in COMPAT_MATRIX:
        if entry.match(versions):
            return entry
    return None


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
        parts.append(f"megatron-core={e.megatron_core}")
        if e.mindspeed is not None:
            parts.append(f"mindspeed={e.mindspeed}")
        lines.append(f"  组合{e.patch_set_id}: " + " + ".join(parts))
    return "\n".join(lines)
