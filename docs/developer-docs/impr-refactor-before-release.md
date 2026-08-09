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
| 错误提示/日志按固定字段拼装 | `prefix-sharing/prefix_sharing/setup/__init__.py`：`_match_compat_entries()`、`_show_compat_matrix()` |
| 测试按固定字段构造假版本 | `prefix-sharing/tests/unit_test/test_verl080_migration.py`、`tests/integrated_test/test_patch_integrations.py` |

---

## 2. setup：自动选型路径上存在重复的兼容性校验与死代码报错

**状态**：已处理

**处理**：自动选型拆成两步：`detect_dependency_versions()` → `_match_compat_entries(versions)`；匹配只做一次，空结果在 `_match_compat_entries` 抛错，不再二次扫描。
