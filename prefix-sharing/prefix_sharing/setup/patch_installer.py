"""prefix_sharing.setup.patch_installer

Install PrefixSharing's runtime monkey-patches for the current environment.

Installation rules:
- module loaded and target resolvable → patch immediately
- module loaded but target missing (import in progress) → pending, retry later (for lazy-loaded modules)
- module not loaded → import hook patches after load (for lazy-loaded modules)
- import hook restores builtins.__import__ when pending is empty, or after consecutive misses (for subprocesses that never load training-side modules)
"""

from __future__ import annotations

import builtins
import importlib
import inspect
import sys
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Callable


# ── Spec ──────────────────────────────────────────────────────────────────────


@dataclass
class PatchSpec:
    """One patch to install."""

    module_name: str
    target_getter: Callable | None = None  # (module) → (target_obj, attr_name)
    patch_factory: Callable | None = None  # (original) → patched
    installer: Callable | None = None  # (module, LoggedPatchManager) → None
    description: str = ""
    # When True, install_specs force-imports the module instead of waiting on the
    # import hook. Needed for verl FSDP engines that load only when the actor starts.
    eager: bool = False


# ── Records & apply primitives ─────────────────────────────────────────────────


def _target_name(target: Any) -> str:
    """Best-effort human-readable name for a patch target."""
    if hasattr(target, "__module__") and hasattr(target, "__qualname__"):
        return f"{target.__module__}.{target.__qualname__}"
    if hasattr(target, "__name__"):
        return target.__name__
    return repr(target)


def _safe_signature(fn: Any) -> str:
    """Return a signature string, or a fallback when inspect fails."""
    try:
        sig = inspect.signature(fn)
        return f"{getattr(fn, '__qualname__', getattr(fn, '__name__', '?'))}{sig}"
    except (ValueError, TypeError):
        name = getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))
        return f"{name}(...)"


def _safe_source(fn: Any) -> str:
    """Return source code, or the signature when source is unavailable."""
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return f"(source unavailable) signature: {_safe_signature(fn)}"


@dataclass(frozen=True)
class PatchRecord:
    """One applied patch, kept for rollback and describe()."""

    target: Any
    attr_name: str
    original: Any
    replacement: Any
    item_key: Any | None = None

    def describe(self) -> str:
        """One-line description: target.attr: original → replacement."""
        orig = getattr(self.original, "__qualname__", repr(self.original))
        new = getattr(self.replacement, "__qualname__", repr(self.replacement))
        if self.item_key is not None:
            target = f"{_target_name(self.target)}[{self.item_key!r}]"
        else:
            target = f"{_target_name(self.target)}.{self.attr_name}"
        return f"{target}: {orig} → {new}"


class LoggedPatchManager:
    """Apply attribute/item patches, log them, and retain records for rollback."""

    def __init__(self, records: list[PatchRecord] | None = None) -> None:
        # Shared list so immediate patches and import-hook patches share one ledger.
        self._records: list[PatchRecord] = records if records is not None else []

    def patch_attr(self, target: Any, attr_name: str, replacement: Any) -> None:
        """Replace ``target.attr_name``, record the original, and log."""
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

    def patch_item(self, mapping: Any, key: Any, replacement: Any) -> None:
        """Replace one mapping entry and retain enough state to restore it."""
        original = mapping[key]
        if original is replacement:
            return
        mapping[key] = replacement
        self._records.append(
            PatchRecord(
                target=mapping,
                attr_name="item",
                item_key=key,
                original=original,
                replacement=replacement,
            )
        )
        print(f"[PS] Patched attention registry entry {key!r}")

    def handle(self) -> PatchHandle:
        """Return a PatchHandle sharing the internal record list."""
        return PatchHandle(self._records)

    def rollback(self) -> None:
        self.handle().disable()


class PatchHandle:
    """Lifecycle handle: describe(), inspect_patch(), disable().

    Holds the full PatchSpec list (fixed at install) and the growing PatchRecord
    list (appended by the import hook). describe() / inspect_patch() show
    applied vs pending per spec.
    """

    def __init__(
        self,
        records: list[PatchRecord],
        specs: list[Any] | None = None,
    ) -> None:
        self._records = records  # shared mutable list; import hook appends
        self._specs = specs or []  # fixed PatchSpec list from install()
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def _spec_to_record(self, module_name: str) -> PatchRecord | None:
        """Find an applied record whose target belongs to ``module_name``."""
        for r in self._records:
            target_module = getattr(r.target, "__module__", None)
            if target_module == module_name:
                return r
            target_name = getattr(r.target, "__name__", None)
            if target_name == module_name:
                return r
        return None

    def disable(self) -> None:
        """Roll back all applied patches, logging each restore."""
        if not self._active:
            return
        for record in reversed(self._records):
            if record.item_key is None:
                setattr(record.target, record.attr_name, record.original)
            else:
                record.target[record.item_key] = record.original
            print(
                f"[PS] Restored {_target_name(record.target)}.{record.attr_name} → "
                f"{getattr(record.original, '__qualname__', 'original')}"
            )
        self._active = False
        print(f"[PS] All {len(self._records)} patches reverted.")

    def describe(self) -> str:
        """Human-readable list of patches, marking applied vs pending."""
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
        """Show replacement (or pending factory) source for verification.

        Args:
            index: 1-based patch index. None returns every patch.
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
                lines.append(_safe_source(spec.patch_factory))
            lines.append("")
        return "\n".join(lines)

    def __enter__(self) -> PatchHandle:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.disable()


# ── Installer / scheduling ─────────────────────────────────────────────────────


class PatchRegistry:
    """Install PatchSpecs (optional global register kept for compatibility)."""

    _specs: list[PatchSpec] = []

    @classmethod
    def register(cls, spec: PatchSpec) -> None:
        key = _spec_key(spec)
        if any(_spec_key(existing) == key for existing in cls._specs):
            return
        cls._specs.append(spec)

    @classmethod
    def install_all(cls) -> PatchHandle:
        """Apply every globally registered patch."""
        return cls.install_specs(cls._specs)

    @classmethod
    def install_specs(cls, specs: list[PatchSpec]) -> PatchHandle:
        """Apply the given specs without touching the global registry.

        Cases:
        1. Module loaded and target resolvable → patch now
        2. Module loaded but target missing → pending
        3. Module not loaded → pending; import hook patches on load
        """
        specs = _dedupe_specs(specs)
        shared_records: list[PatchRecord] = []
        mgr = LoggedPatchManager(shared_records)
        pending: list[PatchSpec] = []

        for spec in specs:
            module = sys.modules.get(spec.module_name)
            if module is None and spec.eager:
                # Force-load lazy modules (e.g. verl FSDP engine) instead of
                # waiting on the import hook among thousands of imports.
                try:
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
                    _apply_spec(spec, module, mgr)
                    print(
                        f"[PS] Immediately patched {spec.description} (module already loaded)"
                    )
                except (AttributeError, KeyError):
                    # Target may still be defining, or a @property may not have
                    # run yet. Eager specs get one re-import retry.
                    if spec.eager:
                        try:
                            module = importlib.import_module(spec.module_name)
                            _apply_spec(spec, module, mgr)
                            print(
                                f"[PS] Eager-retry patched {spec.description} "
                                f"(re-import to trigger @property)"
                            )
                            continue
                        except (AttributeError, KeyError, Exception):
                            pass
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


def _apply_spec(spec: PatchSpec, module: object, manager: LoggedPatchManager) -> None:
    if spec.installer is not None:
        spec.installer(module, manager)
        return
    if spec.target_getter is None or spec.patch_factory is None:
        raise AttributeError(
            f"PatchSpec {spec.description!r} has no installer or attribute patch"
        )
    target_obj, attr_name = spec.target_getter(module)
    original = getattr(target_obj, attr_name)
    manager.patch_attr(target_obj, attr_name, spec.patch_factory(original))


def _dedupe_specs(specs: list[PatchSpec]) -> list[PatchSpec]:
    """Keep first occurrence of each (module_name, description)."""
    deduped: dict[tuple[str, str], PatchSpec] = {}
    for spec in specs:
        deduped.setdefault(_spec_key(spec), spec)
    return list(deduped.values())


_original_import = None

# After this many consecutive unmatched imports, restore __import__.
# Covers subprocesses (e.g. vLLM rollout) that import prefix_sharing but never
# load training-side targets — otherwise a permanent __import__ hook breaks
# torch.compile / CUDA graph capture.
_IMPORT_HOOK_MISS_THRESHOLD = 200


def _activate_import_hook(
    pending_specs: list[PatchSpec],
    shared_records: list[PatchRecord],
) -> None:
    """Temporarily wrap ``__import__`` to patch modules as they load.

    Restore paths:
    1. All pending modules imported and patched → restore immediately
    2. ``_IMPORT_HOOK_MISS_THRESHOLD`` consecutive misses → restore to avoid
       permanently hijacking ``builtins.__import__``
    """
    global _original_import

    if _original_import is not None:
        print("[PS] Import hook already active, skipping re-activation")
        return

    lookup = {spec.module_name: spec for spec in pending_specs}
    _original_import = builtins.__import__
    miss_count = [0]

    def hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
        global _original_import
        real_import = _original_import
        if real_import is None:
            # Stale closure after timeout restore: use current builtins.__import__.
            return builtins.__import__(name, globals, locals, fromlist, level)
        module = real_import(name, globals, locals, fromlist, level)

        if name in lookup:
            miss_count[0] = 0
            spec = lookup.pop(name)
            # __import__ may return the top-level package when fromlist is empty;
            # take the real module from sys.modules.
            actual_module = sys.modules[name]

            try:
                _apply_spec(spec, actual_module, LoggedPatchManager(shared_records))
                print(
                    f"[PS] Auto-patched {spec.description} on import of {name}"
                )
            except (AttributeError, KeyError):
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
