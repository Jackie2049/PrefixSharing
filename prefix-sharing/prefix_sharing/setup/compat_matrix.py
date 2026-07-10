"""版本兼容性矩阵：精确版本号匹配，不支持模糊范围。"""

from __future__ import annotations

from dataclasses import dataclass

from prefix_sharing.setup.version_guard import DetectedVersions


@dataclass(frozen=True)
class CompatEntry:
    """一条兼容性规则，使用精确版本号匹配。

    None 表示该库必须不存在；"*" 表示该库不关注。
    """

    verl: str | None
    megatron_core: str | None
    mindspeed: str | None
    patch_set_id: str
    notes: str = ""

    def match(self, versions: DetectedVersions) -> bool:
        """检查探测到的版本是否与本条规则完全匹配。"""
        # verl：规则要求 None（不需要 verl）时，探测到 None 才算匹配
        if not _version_match(self.verl, versions.verl):
            return False
        if not _version_match(self.megatron_core, versions.megatron_core):
            return False
        if not _version_match(self.mindspeed, versions.mindspeed):
            return False
        return True


def _version_match(required: str | None, detected: str | None) -> bool:
    """精确版本号匹配。

    - required 为 None：该库不需要，detected 为 None 时匹配
    - required 为 "*"：该库不关注，任意 detected 值都匹配
    - required 为字符串：detected 必须完全等于 required
    """
    if required == "*":
        return True
    if required is None:
        return detected is None
    return detected == required


# ── 兼容矩阵：支持以下版本组合 ──
COMPAT_MATRIX: list[CompatEntry] = [
    # 首推开源路线：verl 0.8.0 + FSDP。Megatron/MindSpeed 是否安装不影响
    # FSDP patchset 的默认选择；MCore 路线可显式 install("verl080_mcore0161_ms0160")。
    CompatEntry(
        verl="0.8.0.dev",
        megatron_core="*",
        mindspeed="*",
        patch_set_id="verl080_fsdp",
        notes="verl 0.8.0 FSDP first-class path; PrefixGrouper mode=arbitrary_prefix",
    ),
    # 组合一：verl 0.8.0 + Megatron Core 0.16.1 + MindSpeed 0.16.0（Qwen3.5 NPU 配套）
    CompatEntry(
        verl="0.8.0.dev",
        megatron_core="0.16.1",
        mindspeed="0.16.0",
        patch_set_id="verl080_mcore0161_ms0160",
        notes="verl 0.8.0 + Megatron core 0.16.1 + MindSpeed 0.16.0; "
              "Qwen3.5 NPU RL 官方推荐配套; no Parallel, no FA, no fused kernels, no MTP",
    ),
    # 组合二：纯 Megatron Core + MindSpeed（无 verl）
    CompatEntry(
        verl=None,
        megatron_core="0.12.0",
        mindspeed="0.12.0",
        patch_set_id="mcore012_ms012",
        notes="纯 Megatron + MindSpeed 训练; 无 verl; "
              "仅 patch Attention.forward",
    ),
]
