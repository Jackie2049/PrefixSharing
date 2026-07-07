"""Small monkey-patch manager with rollback and idempotent handles."""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Any, Callable


@dataclass(frozen=True)
class _PatchRecord:
    target: Any
    attr_name: str
    original: Any
    replacement: Any


class PatchHandle:
    def __init__(self, records: list[_PatchRecord]) -> None:
        self._records = records
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def disable(self) -> None:
        if not self._active:
            return
        for record in reversed(self._records):
            if isinstance(record.target, dict):
                record.target[record.attr_name] = record.original
            else:
                setattr(record.target, record.attr_name, record.original)
        self._active = False

    def __enter__(self) -> "PatchHandle":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.disable()


class PatchManager:
    """Install attribute patches and roll back if any step fails."""

    def __init__(self) -> None:
        self._records: list[_PatchRecord] = []

    def patch_attr(
        self,
        target: Any,
        attr_name: str,
        replacement: Any,
        *,
        signature_check: Callable[[Any], None] | None = None,
    ) -> None:
        if not hasattr(target, attr_name):
            raise AttributeError(f"{target!r} has no attribute {attr_name!r}")
        original = getattr(target, attr_name)
        if original is replacement:
            return
        if signature_check is not None:
            signature_check(original)
        setattr(target, attr_name, replacement)
        self._records.append(
            _PatchRecord(
                target=target,
                attr_name=attr_name,
                original=original,
                replacement=replacement,
            )
        )

    def patch_item(
        self,
        target: dict[Any, Any],
        key: Any,
        replacement: Any,
        *,
        signature_check: Callable[[Any], None] | None = None,
    ) -> None:
        if key not in target:
            raise KeyError(key)
        original = target[key]
        if original is replacement:
            return
        if signature_check is not None:
            signature_check(original)
        target[key] = replacement
        self._records.append(
            _PatchRecord(
                target=target,
                attr_name=key,
                original=original,
                replacement=replacement,
            )
        )

    def handle(self) -> PatchHandle:
        records = list(self._records)
        self._records.clear()
        return PatchHandle(records)

    def rollback(self) -> None:
        handle = self.handle()
        handle.disable()
