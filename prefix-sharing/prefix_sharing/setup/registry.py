"""Patch 注册与调度：PatchSpec 注册 + LoggedPatchManager 安装 + import hook。

设计要点：
- 模块已加载且目标存在 → 立即 patch
- 模块已加载但目标不存在（模块正在 import 中）→ 加入 pending，稍后重试
- 模块未加载 → import hook 拦截，加载完成后 patch
- import hook 在所有 pending 处理完或连续 miss 达阈值后恢复原始 __import__
  （后者用于子进程场景，如 vLLM rollout worker 永不加载训练侧目标模块）
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
    eager: bool = False       # 为 True 时，install 阶段直接 importlib.import_module
                              # 强制加载目标模块并立即 patch，不走 import hook。
                              # 用于 verl FSDP 这类 lazy-load 模块：仅在 actor 实例化
                              # engine 时才被 import（远晚于任何 import-hook 窗口），
                              # 必须 eager 触发。


class PatchRegistry:
    """全局 patch 注册表，install_all() 时一次性应用所有已注册的 patch。"""

    _specs: list[PatchSpec] = []

    @classmethod
    def register(cls, spec: PatchSpec) -> None:
        key = _spec_key(spec)
        if any(_spec_key(existing) == key for existing in cls._specs):
            return
        cls._specs.append(spec)

    @classmethod
    def install_all(cls) -> PatchHandle:
        """应用所有已注册的 patch。"""
        return cls.install_specs(cls._specs)

    @classmethod
    def install_specs(cls, specs: list[PatchSpec]) -> PatchHandle:
        """应用给定 patch specs，不污染全局注册表。

        三种情况：
        1. 模块已加载且目标可解析 → 立即 patch
        2. 模块已加载但目标不可解析（模块正在 import 中）→ 加入 pending
        3. 模块未加载 → 加入 pending，由 import hook 在加载时 patch

        所有 pending 最终统一由 import hook 处理。
        import hook 在模块加载完成后才尝试解析目标，确保类定义已完成。
        """
        specs = _dedupe_specs(specs)
        shared_records: list[PatchRecord] = []
        mgr = LoggedPatchManager(shared_records)
        pending: list[PatchSpec] = []

        for spec in specs:
            module = sys.modules.get(spec.module_name)
            if module is None and spec.eager:
                # Lazy-load 目标模块（如 verl FSDP engine），立即 patch，避免依赖
                # import hook 在万级 import 中等不到目标。
                try:
                    import importlib

                    module = importlib.import_module(spec.module_name)
                    print(
                        f"[PS] Eager-imported {spec.module_name} for {spec.description}"
                    )
                except Exception as exc:
                    print(
                        f"[PS] Eager import of {spec.module_name} failed ({exc}); "
                        f"falling back to import hook for {spec.description}"
                    )
                    module = None
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
                    # 模块已加载但目标不存在——
                    # 可能是模块正在 import 中，类定义尚未完成。
                    # 加入 pending，等模块完全加载后再 patch。
                    pending.append(spec)
                    print(
                        f"[PS] Target not yet defined in {spec.module_name}, "
                        f"deferring patch: {spec.description}"
                    )
            else:
                pending.append(spec)

        handle = PatchHandle(shared_records, specs=list(specs))

        if pending:
            _activate_import_hook(pending, shared_records)

        return handle


def _spec_key(spec: PatchSpec) -> tuple[str, str]:
    return spec.module_name, spec.description


def _dedupe_specs(specs: list[PatchSpec]) -> list[PatchSpec]:
    seen: set[tuple[str, str]] = set()
    result: list[PatchSpec] = []
    for spec in specs:
        key = _spec_key(spec)
        if key in seen:
            continue
        seen.add(key)
        result.append(spec)
    return result


_original_import = None

# 连续未匹配 import 的阈值：超过此值后自动恢复 __import__。
# 用于子进程场景（如 vLLM rollout worker 经 VERL_USE_EXTERNAL_MODULES 导入
# prefix_sharing，但永远不会 import 训练侧的 FSDPEngineWithLMHead）——
# 此时 import hook 若不主动恢复，会永久劫持 builtins.__import__，
# 导致 torch.compile / CUDA graph 捕获报 "Graph break due to unsupported
# builtin builtins.__import__"。200 次覆盖常规 Python 启动的 import 量，
# 并留出足够余量确保目标模块若会被加载，一定在恢复前命中。
_IMPORT_HOOK_MISS_THRESHOLD = 200


def _activate_import_hook(
    pending_specs: list[PatchSpec],
    shared_records: list[PatchRecord],
) -> None:
    """对未加载或目标尚未定义的模块，临时拦截 __import__。

    模块加载完成后，尝试解析目标并 patch。如果目标仍然不存在
    （极端情况：模块被 import 但类在延迟定义），记录 warning 并跳过。

    两条恢复路径：
    1. 所有 pending 模块都被 import 且 patch 完成 → 立即恢复 ``__import__``；
    2. 连续 ``_IMPORT_HOOK_MISS_THRESHOLD`` 次 import 都未命中 pending 模块
       → 视为当前进程不会加载目标（典型场景：vLLM rollout 子进程），
       主动恢复 ``__import__`` 防止永久劫持破坏下游编译。
    """
    global _original_import

    if _original_import is not None:
        print("[PS] Import hook already active, skipping re-activation")
        return

    lookup = {spec.module_name: spec for spec in pending_specs}
    _original_import = builtins.__import__
    miss_count = [0]  # 闭包可变计数器

    def hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
        global _original_import
        real_import = _original_import
        if real_import is None:
            # 已超时恢复过 builtins.__import__：Python import 机器或第三方代码
            # 可能仍持有本闭包的陈旧引用，直接走当前（已恢复的）builtins.__import__，
            # 避免 NoneType not callable。
            return builtins.__import__(name, globals, locals, fromlist, level)
        module = real_import(name, globals, locals, fromlist, level)

        if name in lookup:
            miss_count[0] = 0
            spec = lookup.pop(name)
            # __import__ 在 fromlist 为空时返回顶层包而非子模块，
            # 必须从 sys.modules 取实际加载的模块对象。
            actual_module = sys.modules[name]

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
                # 模块已加载但目标仍未定义——
                # 这种情况极少发生，通常是模块结构异常。
                print(
                    f"[PS] Could not resolve target for {spec.description} "
                    f"after import of {name}; skipping this patch. "
                    f"The patch target may not exist in this module version."
                )

            if not lookup:
                builtins.__import__ = _original_import
                _original_import = None
                print("[PS] All import hooks resolved, __import__ restored")
        else:
            # 未命中：累计 miss 计数，超过阈值后主动恢复，避免子进程
            # 永不加载目标模块时 hook 永久劫持 builtins.__import__。
            miss_count[0] += 1
            if miss_count[0] == _IMPORT_HOOK_MISS_THRESHOLD and _original_import is not None:
                builtins.__import__ = _original_import
                _original_import = None
                print(
                    f"[PS] Import hook auto-restored after {_IMPORT_HOOK_MISS_THRESHOLD} "
                    f"consecutive unmatched imports; pending patches never applied: "
                    f"{list(lookup.keys())}"
                )

        return module

    builtins.__import__ = hooked_import
    print(
        f"[PS] Import hook activated for {len(lookup)} modules: {list(lookup.keys())}"
    )