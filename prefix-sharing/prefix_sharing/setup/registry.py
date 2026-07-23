"""Patch 注册与调度：PatchSpec 注册 + LoggedPatchManager 安装 + import hook。

设计要点：
- 模块已加载且目标存在 → 立即 patch
- 模块已加载但目标不存在（模块正在 import 中）→ 加入 pending，稍后重试
- 模块未加载 → import hook 拦截，加载完成后 patch
- 同一模块多个 spec → import hook 批量处理
- import hook 完成后立即恢复原始 __import__
"""

from __future__ import annotations

import builtins
import sys
from dataclasses import dataclass
from typing import Callable

from prefix_sharing.setup.logged_patch import LoggedPatchManager, PatchHandle, PatchRecord


@dataclass
class PatchSpec:
    """一个待安装的 patch 规格。"""

    module_name: str          # 目标模块全限定名
    target_getter: Callable   # (module) → (target_obj, attr_name)
    patch_factory: Callable   # (original) → patched
    description: str = ""     # 人类可读描述


class PatchRegistry:
    """全局 patch 注册表，install_all() 时一次性应用所有已注册的 patch。"""

    _specs: list[PatchSpec] = []

    @classmethod
    def register(cls, spec: PatchSpec) -> None:
        cls._specs.append(spec)

    @classmethod
    def install_all(cls) -> PatchHandle:
        """应用所有已注册的 patch。"""
        shared_records: list[PatchRecord] = []
        mgr = LoggedPatchManager(shared_records)
        pending: list[PatchSpec] = []

        for spec in cls._specs:
            module = sys.modules.get(spec.module_name)
            if module is not None:
                try:
                    target_obj, attr_name = spec.target_getter(module)
                    original = getattr(target_obj, attr_name)
                    patched = spec.patch_factory(original)
                    mgr.patch_attr(target_obj, attr_name, patched)
                    print(
                        f"[PS] Immediately patched {spec.description} (module already loaded)"
                    )
                except (AttributeError, KeyError):
                    pending.append(spec)
                    print(
                        f"[PS] Target not yet defined in {spec.module_name}, "
                        f"deferring patch: {spec.description}"
                    )
            else:
                pending.append(spec)

        handle = PatchHandle(shared_records, specs=list(cls._specs))

        if pending:
            _activate_import_hook(pending, shared_records)

        return handle


_original_import = None


def _activate_import_hook(
    pending_specs: list[PatchSpec],
    shared_records: list[PatchRecord],
) -> None:
    """拦截 __import__，加载完成后批量 patch。"""
    global _original_import

    if _original_import is not None:
        print("[PS] Import hook already active, skipping re-activation")
        return

    # 同一模块可能有多条 spec，用 list 保存
    lookup: dict[str, list[PatchSpec]] = {}
    for spec in pending_specs:
        lookup.setdefault(spec.module_name, []).append(spec)

    _original_import = builtins.__import__

    def hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
        global _original_import
        module = _original_import(name, globals, locals, fromlist, level)

        if name in lookup:
            specs = lookup.pop(name)
            actual_module = sys.modules[name]

            for spec in specs:
                try:
                    target_obj, attr_name = spec.target_getter(actual_module)
                    original = getattr(target_obj, attr_name)
                    patched = spec.patch_factory(original)
                    setattr(target_obj, attr_name, patched)
                    shared_records.append(
                        PatchRecord(
                            target=target_obj,
                            attr_name=attr_name,
                            original=original,
                            replacement=patched,
                        )
                    )
                    print(
                        f"[PS] Auto-patched {spec.description} on import of {name}"
                    )
                except (AttributeError, KeyError):
                    print(
                        f"[PS] Could not resolve target for {spec.description} "
                        f"after import of {name}; skipping this patch."
                    )

            if not lookup:
                builtins.__import__ = _original_import
                _original_import = None
                print("[PS] All import hooks resolved, __import__ restored")

        return module

    builtins.__import__ = hooked_import
    print(
        f"[PS] Import hook activated for {len(lookup)} modules "
        f"({sum(len(v) for v in lookup.values())} specs): {list(lookup.keys())}"
    )
