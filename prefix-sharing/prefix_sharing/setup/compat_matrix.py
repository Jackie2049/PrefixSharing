"""module: prefix_sharing.setup.compat_matrix

This module defines the compatibility matrix for dependency versions.
"""

from __future__ import annotations

from dataclasses import dataclass

from prefix_sharing.setup.version_detector import DetectedVersions


class IncompatibleEnvironment(RuntimeError):
    """Raised when the detected version combination is not in the compatibility matrix."""


@dataclass(frozen=True)
class CompatEntry:
    """For defining a compatibility rule with exact dependency versions.
    """

    verl: str | None
    megatron_core: str | None
    mindspeed: str | None
    patch_set_id: str
    notes: str = ""

    def match(self, versions: DetectedVersions) -> bool:
        """Return True if detected versions fully match this compatibility rule."""
        return (
            self._version_match(self.verl, versions.verl)
            and self._version_match(self.megatron_core, versions.megatron_core)
            and self._version_match(self.mindspeed, versions.mindspeed)
        )

    @staticmethod
    def _version_match(required: str | None, detected: str | None) -> bool:
        """Exact version match."""
        if required == "*":
            # library ignored: any detected value matches
            return True
        if required is None:
            # library not required: matches only when detected is None
            return detected is None
        return detected == required


# ── Compatibility matrix: supported version combinations ──
COMPAT_MATRIX: list[CompatEntry] = [
    # FSDP-first path. Megatron Core and MindSpeed presence does not change
    # the FSDP patch-set selection.
    CompatEntry(
        verl="0.8.0.dev",
        megatron_core="*",
        mindspeed="*",
        patch_set_id="verl080_fsdp",
        notes="verl 0.8.0 FSDP-first path; PrefixGrouper mode=arbitrary_prefix",
    ),
    # Advanced MCore path. Keep its exact version requirements because its
    # patch set relies on the corresponding verl/MindSpeed internals.
    CompatEntry(
        verl="0.8.0.dev",
        megatron_core="0.16.1",
        mindspeed="0.16.0",
        patch_set_id="verl080_mcore0161_ms0160",
        notes="verl 0.8.0 + Megatron Core 0.16.1 + MindSpeed 0.16.0",
    ),
]
