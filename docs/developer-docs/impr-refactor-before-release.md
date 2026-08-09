# 发布前重构遗留事项

## 1. setup：版本探测与兼容矩阵目前都是定制化字段，未来改为可扩展的键值形式

**状态**：搁置

**问题**：当前探测结果和兼容规则都把依赖名写死成固定字段，以后每加一个依赖都要改多处，扩展性差。

举例：

- **当前**：探测结果固定只有三个槽位（verl / megatron / mindspeed），兼容规则也是同样三个槽位一一比对。多一个没有、少一个也不行，规则里「不关心某库」只能靠特殊取值表达，不能省略字段。
- **扩展时的痛点**：若某条规则还要看 `transformers` 版本，至少得改探测结果结构、探测读取、兼容规则字段、匹配逻辑；错误提示和测试也要跟着改。
- **理想情况**：探测字典多写一行、规则映射多写一项即可；某条规则不关心的依赖直接不写。

**方向**：探测结果和兼容规则都改成「依赖名 → 版本约束」的映射；探测实现保持直观（按依赖显式读取版本），不要再套一层抽象配置对象。匹配语义不变（忽略 / 必须不存在 / 精确相等）。

**搁置原因**：改的是自动选型入口，本地缺少 MindSpeed 等真实版本组合，验证风险偏大。有完整环境后再做。

**相关代码（问题 ↔ 位置）**：

| 问题点 | 代码位置 |
|--------|----------|
| 探测结果三个固定字段 | `prefix-sharing/prefix_sharing/setup/version_detector.py`：`DependencyDetectedVersions` |
| 按固定字段逐个探测 | 同文件：`detect_dependency_versions()` |
| 兼容规则三个固定字段 | `prefix-sharing/prefix_sharing/setup/compat_matrix.py`：`CompatEntry` |
| 按固定字段做匹配 | 同文件：`CompatEntry.match()` |
| 错误提示/日志按固定字段拼装 | `prefix-sharing/prefix_sharing/setup/__init__.py`：`detect_and_validate_dependency_versions()`、`_show_compat_matrix()` |
| 测试按固定字段构造假版本 | `prefix-sharing/tests/unit_test/test_verl080_migration.py`、`tests/integrated_test/test_patch_integrations.py` |

---

## 2. setup：自动选型路径上存在重复的兼容性校验与死代码报错

**状态**：搁置（当前不改）

**问题**：解析「该装哪套 patch」时，会先做一轮依赖版本探测与兼容校验；通过之后又立刻再匹配一遍兼容矩阵，并再写了一段「不兼容则抛异常」。按正常调用顺序，不兼容早已在第一轮抛出，第二段报错走不到，属于重复逻辑里的死分支。

**方向**：校验只保留一处；自动选型在已通过校验后，只根据匹配结果取出 patch 包名即可，不要再重复判断、重复抛错。

**搁置原因**：行为上目前仍正确（真正报错的是第一处），属于可读性/冗余清理，优先级低于功能与选型相关改动。

**相关代码（问题 ↔ 位置）**：

文件：`prefix-sharing/prefix_sharing/setup/__init__.py`

| 问题点 | 代码位置 | 说明 |
|--------|----------|------|
| 第一轮探测 + 校验（有效报错） | `detect_and_validate_dependency_versions()` 内：`_find_compat_entries` 后 `raise IncompatibleEnvironment` | 不兼容时真正抛错的地方 |
| 自动选型入口再次匹配 | `_resolve_patch_set_ids()` 在调用上一函数之后，再次 `_find_compat_entries(...)` | 与第一轮重复扫描矩阵 |
| 死代码报错分支 | `_resolve_patch_set_ids()` 内第二次 `if not compat_entries: raise IncompatibleEnvironment` | 正常路径到不了；不兼容已在第一轮抛出 |
| 调用关系 | `install()` → `_resolve_patch_set_ids(None)` → 上述两段 | 仅「未显式指定 patch set」的自动选型路径会踩到 |
