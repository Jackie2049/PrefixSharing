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


# ── Public API ─────────────────────────────────────────────────────────────────


@dataclass
class PatchSpec:
    """One patch to install.

    Typical path: find ``module_name`` → ``target_getter`` picks (obj, attr) →
    ``patch_factory(old)`` builds the replacement → setattr monkey-patch.
    """

    # Which module to patch, e.g. "verl.workers.engine.fsdp.transformer_impl".
    module_name: str

    # Given the imported module, return which attribute to patch:
    # (owner, attr_name) format, e.g. (mod.FSDPEngineWithLMHead, "forward_step").
    target_getter: Callable | None = None

    # Build the new callable from the old one: new = patch_factory(old).
    patch_factory: Callable | None = None

    # Human-readable comments for this patch.
    description: str = ""

    # If True, import the module instantly instead of waiting on the import hook
    eager: bool = False


def install_specs(patch_specs: list[PatchSpec]) -> PatchHandle:
    """Apply the given PatchSpecs.

    Cases:
    1. Module loaded and target resolvable → patch now
    2. Module loaded but target missing → pending
    3. Module not loaded → pending; import hook patches on load
    """
    def _deduplicate(patch_specs: list[PatchSpec]) -> list[PatchSpec]:
        """Keep first occurrence of each (module_name, description)."""
        deduped: dict[tuple[str, str], PatchSpec] = {}
        for patch in patch_specs:
            deduped.setdefault((patch.module_name, patch.description), patch)
        return list(deduped.values())

    patch_specs = _deduplicate(patch_specs)
    patch_records: list[PatchRecord] = []
    patch_manager = LoggedPatchManager(patch_records)
    pending_patches: list[PatchSpec] = []

    for patch in patch_specs:
        module = sys.modules.get(patch.module_name)

        # target module is designed to be lazy-imported
        if module is None and patch.eager:
            # => force-load it instead of waiting on the import hook (e.g. verl FSDP engine).
            try:
                module = importlib.import_module(patch.module_name)
                print(
                    f"[PrefixSharing] Eager import {patch.module_name} for {patch.description}"
                )
            except Exception as exc:
                print(
                    f"[PrefixSharing] Eager import {patch.module_name} failed ({exc}); "
                    f"falling back to import hook for {patch.description}"
                )
                module = None

        # Case 1: target module is already loaded (by default or by the eager import above)
        if module is not None:
            # => apply monkey-patch immediately.
            try:
                _apply_patch(patch, module, patch_manager)
                print(
                    f"[PrefixSharing] Immediately patched {patch.description} (module already loaded)"
                )
            except (AttributeError, KeyError):
                # First apply failed (target attr missing). If eager, re-import this attr and retry patching;
                # otherwise defer to the import hook to patch it later.
                def _defer_patch() -> None:
                    pending_patches.append(patch)
                    print(
                        f"[PrefixSharing] Target not yet defined in {patch.module_name}, "
                        f"deferring patch: {patch.description}"
                    )

                if patch.eager:
                    try:
                        module = importlib.import_module(patch.module_name)
                        _apply_patch(patch, module, patch_manager)
                        print(
                            f"[PrefixSharing] Eager-retry patched {patch.description} "
                            f"(re-import to trigger @property)"
                        )
                    except (AttributeError, KeyError, Exception):
                        _defer_patch()
                else:
                    _defer_patch()
        # Case 2: target module is not yet loaded
        else:
            # => defer to the import hook to patch it later.
            pending_patches.append(patch)

    patch_handle = PatchHandle(patch_records, patch_specs=list(patch_specs))

    if pending_patches:
        # => activate the import hook to patch the pending modules later.
        _activate_import_hook(pending_patches, patch_records)

    return patch_handle


class PatchHandle:
    """Lifecycle handle: describe(), inspect_patch(), disable().

    Holds the full PatchSpec list (fixed at install) and the growing PatchRecord
    list (appended by the import hook). describe() / inspect_patch() show
    applied vs pending per patch.
    """

    def __init__(
        self,
        records: list[PatchRecord],
        patch_specs: list[Any] | None = None,
    ) -> None:
        self._records = records  # shared mutable list; import hook appends
        self._patch_specs = patch_specs or []  # fixed PatchSpec list from install()
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def _record_for_module(self, module_name: str) -> PatchRecord | None:
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
            setattr(record.target, record.attr_name, record.original)
            print(
                f"[PrefixSharing] Restored {_target_name(record.target)}.{record.attr_name} → "
                f"{getattr(record.original, '__qualname__', 'original')}"
            )
        self._active = False
        print(f"[PrefixSharing] All {len(self._records)} patches reverted.")

    def describe(self) -> str:
        """Human-readable list of patches, marking applied vs pending."""
        status_prefix = "ACTIVE" if self._active else "INACTIVE (rolled back)"
        lines = [f"PatchHandle ({status_prefix}, {len(self._patch_specs)} patches):"]
        for i, patch in enumerate(self._patch_specs, 1):
            record = self._record_for_module(patch.module_name)
            if record:
                lines.append(f"  {i}. {record.describe()}  [applied]")
            else:
                lines.append(
                    f"  {i}. {patch.description}  "
                    f"[pending: awaiting import of {patch.module_name}]"
                )
        return "\n".join(lines)

    def inspect_patch(self, index: int | None = None) -> str:
        """Show replacement (or pending factory) source for verification.

        Args:
            index: 1-based patch index. None returns every patch.
        """
        if index is not None:
            patch_specs = [self._patch_specs[index - 1]]
            header = f"Patch #{index}:"
        else:
            patch_specs = self._patch_specs
            header = "All patches:"

        lines = [header]
        for i, patch in enumerate(patch_specs, 1 if index is None else index):
            record = self._record_for_module(patch.module_name)
            if record:
                lines.append(f"\n── {record.describe()} ── [applied]")
                lines.append(_safe_source(record.replacement))
            else:
                lines.append(
                    f"\n── {patch.description} ── "
                    f"[pending: import {patch.module_name} to activate]"
                )
                lines.append(_safe_source(patch.patch_factory))
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


# ── Apply ledger ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PatchRecord:
    """One applied patch, kept for rollback and describe()."""

    target: Any
    attr_name: str
    original: Any
    replacement: Any

    def describe(self) -> str:
        """One-line description: target.attr: original → replacement."""
        orig = getattr(self.original, "__qualname__", repr(self.original))
        new = getattr(self.replacement, "__qualname__", repr(self.replacement))
        return f"{_target_name(self.target)}.{self.attr_name}: {orig} → {new}"


class LoggedPatchManager:
    """Apply attribute patches, log them, and append records to a shared ledger."""

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
            f"[PrefixSharing] Patched {_target_name(target)}.{attr_name}: "
            f"{getattr(original, '__qualname__', 'original')} → "
            f"{getattr(replacement, '__qualname__', 'replacement')}"
        )


# ── Internals ──────────────────────────────────────────────────────────────────


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


def _apply_patch(
    patch: PatchSpec, module: object, patch_manager: LoggedPatchManager
) -> None:
    if patch.target_getter is None or patch.patch_factory is None:
        raise AttributeError(
            f"PatchSpec {patch.description!r} needs target_getter and patch_factory"
        )
    target_obj, attr_name = patch.target_getter(module)
    original = getattr(target_obj, attr_name)
    patch_manager.patch_attr(target_obj, attr_name, patch.patch_factory(original))


_original_import = None

# After this many consecutive unmatched imports, restore __import__.
# Covers subprocesses (e.g. vLLM rollout) that import prefix_sharing but never
# load training-side targets — otherwise a permanent __import__ hook breaks
# torch.compile / CUDA graph capture.
_IMPORT_HOOK_MISS_THRESHOLD = 200


def _activate_import_hook(
    pending_patches: list[PatchSpec],
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
        print("[PrefixSharing] Import hook already active, skipping re-activation")
        return

    lookup = {patch.module_name: patch for patch in pending_patches}
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
            patch = lookup.pop(name)
            # __import__ may return the top-level package when fromlist is empty;
            # take the real module from sys.modules.
            actual_module = sys.modules[name]

            try:
                _apply_patch(patch, actual_module, LoggedPatchManager(shared_records))
                print(
                    f"[PrefixSharing] Auto-patched {patch.description} on import of {name}"
                )
            except (AttributeError, KeyError):
                print(
                    f"[PrefixSharing] Could not resolve target for {patch.description} "
                    f"after import of {name}; skipping this patch. "
                    f"The patch target may not exist in this module version."
                )

            if not lookup:
                builtins.__import__ = _original_import
                _original_import = None
                print("[PrefixSharing] All import hooks resolved, __import__ restored")
        else:
            miss_count[0] += 1
            if miss_count[0] == _IMPORT_HOOK_MISS_THRESHOLD and _original_import is not None:
                builtins.__import__ = _original_import
                _original_import = None
                print(
                    f"[PrefixSharing] Import hook auto-restored after {_IMPORT_HOOK_MISS_THRESHOLD} "
                    f"consecutive unmatched imports; pending patches never applied: "
                    f"{list(lookup.keys())}"
                )

        return module

    builtins.__import__ = hooked_import
    print(
        f"[PrefixSharing] Import hook activated for {len(lookup)} modules: {list(lookup.keys())}"
    )
