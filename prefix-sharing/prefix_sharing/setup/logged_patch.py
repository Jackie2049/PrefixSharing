"""带日志的 monkey-patch manager — setup 生产 patch 入口专用。

提供：
- patch_attr 时 INFO 日志
- disable 时 INFO 日志（逐条打印恢复详情）
- describe() 方法返回人类可读 patch 清单（含已应用和待挂起状态）
- inspect_patch() 方法返回被替换函数的源码，供用户验证
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import TracebackType
from typing import Any


def _target_name(target: Any) -> str:
    """最佳的人类可读名称。"""
    if hasattr(target, "__module__") and hasattr(target, "__qualname__"):
        return f"{target.__module__}.{target.__qualname__}"
    if hasattr(target, "__name__"):
        return target.__name__
    return repr(target)


def _safe_signature(fn: Any) -> str:
    """尝试获取函数签名，失败时返回基本信息。"""
    try:
        sig = inspect.signature(fn)
        return f"{getattr(fn, '__qualname__', getattr(fn, '__name__', '?'))}{sig}"
    except (ValueError, TypeError):
        name = getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))
        return f"{name}(...)"


def _safe_source(fn: Any) -> str:
    """尝试获取函数源码，失败时返回签名。"""
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return f"(source unavailable) signature: {_safe_signature(fn)}"


@dataclass(frozen=True)
class PatchRecord:
    """一个已安装 patch 的记录，用于回滚和 describe。"""

    target: Any
    attr_name: str
    original: Any
    replacement: Any

    def describe(self) -> str:
        """一行人类可读描述：目标.属性: 原始 → 替换。"""
        orig = getattr(self.original, "__qualname__", repr(self.original))
        new = getattr(self.replacement, "__qualname__", repr(self.replacement))
        return f"{_target_name(self.target)}.{self.attr_name}: {orig} → {new}"


class PatchHandle:
    """patch 生命周期 handle：describe() 查看、inspect_patch() 验证、disable() 回滚。

    同时持有完整的 PatchSpec 列表和已应用的 PatchRecord 列表。
    PatchSpec 列表在 install() 时固定；PatchRecord 列表随 import hook 触发动态增长。
    describe() / inspect_patch() 对每个 spec 展示其当前状态（applied / pending）。
    """

    def __init__(
        self,
        records: list[PatchRecord],
        specs: list[Any] | None = None,
    ) -> None:
        self._records = records    # 共享可变列表，import hook 会追加
        self._specs = specs or []  # 完整 PatchSpec 列表，install() 时固定
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def _spec_to_record(self, module_name: str) -> PatchRecord | None:
        """根据 module_name 在 _records 中查找已应用的记录。"""
        for r in self._records:
            # PatchRecord.target 可能是类或模块，通过 module_name 匹配
            target_module = getattr(r.target, "__module__", None)
            if target_module == module_name:
                return r
            # target 是模块时，__module__ 属性不存在，用 __name__ 匹配
            target_name = getattr(r.target, "__name__", None)
            if target_name == module_name:
                return r
        return None

    def disable(self) -> None:
        """回滚所有已应用的 patch，逐条打印恢复日志。"""
        if not self._active:
            return
        for record in reversed(self._records):
            setattr(record.target, record.attr_name, record.original)
            print(
                f"[PS] Restored {_target_name(record.target)}.{record.attr_name} → "
                f"{getattr(record.original, '__qualname__', 'original')}"
            )
        self._active = False
        print(f"[PS] All {len(self._records)} patches reverted.")

    def describe(self) -> str:
        """返回所有 patch 的人类可读清单，区分已应用和待挂起。"""
        status_prefix = "ACTIVE" if self._active else "INACTIVE (rolled back)"
        lines = [f"PatchHandle ({status_prefix}, {len(self._specs)} patches):"]
        for i, spec in enumerate(self._specs, 1):
            record = self._spec_to_record(spec.module_name)
            if record:
                lines.append(f"  {i}. {record.describe()}  [applied]")
            else:
                lines.append(
                    f"  {i}. {spec.description}  "
                    f"[pending: awaiting import of {spec.module_name}]"
                )
        return "\n".join(lines)

    def inspect_patch(self, index: int | None = None) -> str:
        """查看被替换函数的源码，供用户验证 patch 内容。

        Args:
            index: patch 序号（1-based）。None 时返回所有 patch 的源码。

        Returns:
            源码字符串。已应用的 patch 显示替换函数源码；
            待挂起的 patch 显示 patch_factory 源码（即创建替换函数的工厂）。
        """
        if index is not None:
            specs = [self._specs[index - 1]]
            header = f"Patch #{index}:"
        else:
            specs = self._specs
            header = "All patches:"

        lines = [header]
        for i, spec in enumerate(specs, 1 if index is None else index):
            record = self._spec_to_record(spec.module_name)
            if record:
                lines.append(f"\n── {record.describe()} ── [applied]")
                lines.append(_safe_source(record.replacement))
            else:
                lines.append(
                    f"\n── {spec.description} ── "
                    f"[pending: import {spec.module_name} to activate]"
                )
                # 待挂起时显示 patch_factory 源码，让用户知道替换逻辑
                lines.append(_safe_source(spec.patch_factory))
            lines.append("")
        return "\n".join(lines)

    def __enter__(self) -> "PatchHandle":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.disable()


class LoggedPatchManager:
    """安装属性 patch 并写日志，供 setup patch registry 使用。"""

    def __init__(self, records: list[PatchRecord] | None = None) -> None:
        # 支持传入外部共享列表，使 import hook 和即时 patch 共用同一份记录
        self._records: list[PatchRecord] = records if records is not None else []

    def patch_attr(self, target: Any, attr_name: str, replacement: Any) -> None:
        """替换 target.attr_name 并记录 + 写日志。"""
        if not hasattr(target, attr_name):
            raise AttributeError(
                f"{_target_name(target)} has no attribute {attr_name!r}"
            )
        original = getattr(target, attr_name)
        if original is replacement:
            return  # idempotent
        setattr(target, attr_name, replacement)
        self._records.append(
            PatchRecord(
                target=target,
                attr_name=attr_name,
                original=original,
                replacement=replacement,
            )
        )
        print(
            f"[PS] Patched {_target_name(target)}.{attr_name}: "
            f"{getattr(original, '__qualname__', 'original')} → "
            f"{getattr(replacement, '__qualname__', 'replacement')}"
        )

    def handle(self) -> PatchHandle:
        """返回 PatchHandle，共享内部记录列表（不复制/不清空）。"""
        return PatchHandle(self._records)

    def rollback(self) -> None:
        self.handle().disable()
