# PrefixSharing 面向开源与合入 verl 的重构整改研究

本文档基于 `PrefixSharing_refactor` / `open-source_refactor` 分支当前代码，按“研究分析 → 方案设计 → 测试验证 → 开发计划 → 当前结论 → 遗留问题”的工作流组织。目标是提高当前实现的可读性、可维护性、可扩展性和社区可接受度，使后续开源到 GitHub 以及向 verl 提交接入补丁时，代码质量和接口设计能够经受社区 review。

当前阶段的核心约束：

- 对外尽量复用 verl 已有 PrefixGrouper 认知、配置入口和接入方式，让 arbitrary-prefix sharing 看起来像 PrefixGrouper 的扩展能力，而不是另一个并列的前缀复用特性。
- 对内保持 PrefixSharing 已验证的精度语义：One-Forward + KV Injection + Prefix-Last Restore，缓存 KV / activation 不 detach。
- 重构优先选择“改动小、收益明确、风险可控”的事项；涉及核心精度语义的变更必须先有测试保护。

## Chapter 1：研究分析

### 1.1 研究范围与边界

本轮研究已完成：

- 阅读当前 `prefix-sharing/prefix_sharing/` 的 core、backends、integrations、setup patch 主流程代码。
- 阅读 FSDP patch、Megatron patch、PrefixGrouper 风格配置兼容逻辑。
- 阅读现有测试目录，确认当前测试覆盖形态和缺口。
- 对照 `docs/developer-docs/feature-fsdp.md` 和 `docs/developer-docs/impr-perf.md` 中已有结论，避免重复提出已知性能优化事项。

本轮研究未做：

- 未运行全量测试；当前任务是问题识别和重构事项排序。
- 未修改业务代码；本文档作为后续重构 PR 的计划输入。
- 未重新审视上游 verl 最新主线。当前结论以本仓库 `dependency/verl_cdd9014f` 和当前分支代码为准。后续准备向 verl 提 PR 前，需要再对上游主线做一次接口核对。

### 1.2 当前代码结构概览

当前主包约 1.1 万行 Python，主要分布如下：

| 模块 | 当前职责 | 观察 |
|------|----------|------|
| `core/config.py` | 用户配置、环境变量、约束校验 | 同时承载内部 `prefix_sharing_config` 和 PrefixGrouper 风格入口的最终配置对象；校验逻辑偏 phase-1 / 内部实验口径。 |
| `core/prefix_detector.py` | Trie 检测 provider/reuser 复用关系 | 仍保留 `PrefixGroup` / `group_ids` 等 group 视图，运行时价值低。 |
| `core/planner.py` | 将检测结果转成 `PrefixSharingPlan` | 字段多、语义密度高，是当前 core 的中心对象；同时包含检测转抄视图、Q/KV layout、restore spec。 |
| `core/prefix_store.py` | 生命周期内的 attention KV / DeltaNet state store | 抽象方向合理，但当前 open-source 首批重点是 attention KV，DeltaNet 内容会扩大 review 面。 |
| `backends/torch_ref.py` | reference backend、KV expansion、debug attention、gated/deltanet reference | 文件较重，attention / gated / deltanet 混在一个类里；正式路径和 reference/debug 路径边界不够清晰。 |
| `backends/flash_atten_gpu.py` / `flash_atten_npu.py` | GPU/NPU FlashAttention backend | 作为正式性能路径存在，但依赖 `TorchReferenceBackend.build_kv()`。 |
| `integrations/context.py` | runtime context、store 生命周期、restore index、audit | 运行时对象职责清楚，但默认 audit `print()` 不适合开源默认路径。 |
| `integrations/verl_fsdp.py` | FSDP micro-batch 构建、attention runtime、2D restore | FSDP 侧核心入口，当前复用了 `verl_mcore.py` 的多个 helper，模块边界不够干净。 |
| `integrations/verl_mcore.py` | Megatron/MCore micro-batch 构建、配置读取、restore helper | 文件过长，仍包含 v070 叙述、调试 print、FSDP 复用 helper；后续社区 review 风险高。 |
| `setup/` | 版本检测、patch registry、patch set | 方向符合 monkey patch 接入，但 import hook、自动安装、版本矩阵、显式 patch set 的边界需要收敛。 |
| `setup/patches/verl080_fsdp/` | FSDP forward_step 和 HF attention patch | 接近目标形态，但当前是 PrefixSharing 独立 patch 口径，和 verl PrefixGrouper 原生入口仍有缝隙。 |
| `tools/` | 诊断 dump、精度对比、训练监控 | 工具价值高，但体量大、中文/内部诊断痕迹多，不宜默认暴露为开源主路径代码重点。 |

当前已有的正向基础：

- FSDP 和 Megatron 都已经接到 `PrefixSharingPlan + RuntimeContext + Backend` 这条主线。
- FSDP patch 已经复用 verl engine 的 `prepare_model_inputs()` / `prepare_model_outputs()`，不是完全绕开 verl。
- `read_ps_config_from_engine_config()` 已经支持 `use_prefix_grouper + prefix_grouper.mode`，具备向 PrefixGrouper 体系靠拢的基础。
- 性能分支已完成 no-sharing prefilter、`build_kv()` prealloc 等部分优化，说明代码不是纯原型状态。
- 测试覆盖已包含 detector、planner、store、layout、FSDP adapter、patch integration、FlashAttention backend 等层面。

### 1.3 与“像 PrefixGrouper 扩展版本”的目标差距

当前实现已经开始兼容 PrefixGrouper 入口：

```python
use_prefix_grouper = true
prefix_grouper.mode = "arbitrary_prefix"
```

但从 verl 用户和社区 reviewer 视角，仍有几类差距。

#### 1.3.1 用户入口仍显得像独立特性

README 当前 Quick Start 主要通过：

```bash
ENABLE_PREFIX_SHARING=1 bash examples/run_prefix_sharing.sh
```

这对内部调试方便，但对 verl 社区不理想。verl 已有 `actor.use_prefix_grouper=True` 认知，首批合入最好表现为：

```yaml
actor_rollout_ref:
  actor:
    use_prefix_grouper: true
    prefix_grouper:
      mode: arbitrary_prefix
```

`ENABLE_PREFIX_SHARING` 可以保留为开发/调试 fallback，但不应作为开源用户文档首选入口。

#### 1.3.2 `prefix_sharing_config` 仍是优先入口

`read_ps_config_from_engine_config()` 当前优先读取：

- `override_transformer_config.prefix_sharing_config`
- `engine_config.prefix_sharing_config`
- 然后才读取 `use_prefix_grouper + prefix_grouper.mode`

这对兼容内部历史配置有用，但从 verl 合入角度看，`prefix_sharing_config` 不应成为首批 PR 的公开主入口。更合理的定位：

- verl 对外主入口：`use_prefix_grouper + prefix_grouper.mode`
- PrefixSharing 独立包内部测试入口：`PrefixSharingConfig`
- `prefix_sharing_config`：仅作为临时兼容或实验入口，开源文档中不重点宣传

#### 1.3.3 patch 命名和安装路径仍偏 PrefixSharing 独立体系

FSDP patch set 需要显式：

```python
prefix_sharing.setup.install("verl080_fsdp")
```

这对本仓库可行，但进入 verl 后，社区更容易接受的是“已有 PrefixGrouper patch/forward helper 扩展 mode”，而不是新增一套独立 patch 体系。当前仓库仍可保留 monkey patch 作为外部包接入方式，但设计文档和代码命名需要表达：

```text
PrefixGrouper prompt_only mode -> existing PrefixGrouper package
PrefixGrouper arbitrary_prefix mode -> PrefixSharing runtime
```

而不是：

```text
PrefixGrouper feature vs PrefixSharing feature
```

#### 1.3.4 FSDP 与 Megatron 的对外优先级不够清晰

当前 README 开头仍强调 `verl + Megatron-LM RL pipeline`。但面向开源和第一波合入 verl，FSDP 路径应是更容易被社区复现和 review 的主线。Megatron/MindSpeed/NPU 可以作为后续 experimental 或 advanced backend。

建议后续文档和代码组织明确：

- 开源首推：verl FSDP + Transformers attention
- 已验证/内部扩展：verl Megatron/MCore
- 后续实验：NPU、MindSpeed、Megatron-Bridge、DeltaNet

### 1.4 Core 层代码质量问题

#### 1.4.1 `PrefixDetectionResult` 包含运行时价值低的 group 视图

当前 `PrefixDetectionResult` 包含：

```python
batch_size
reuse_specs
groups
group_ids
provider_index
prefix_lens
is_provider
```

其中 `groups` / `group_ids` 的主要用途是：

- detector 内构造兼容/debug 视图；
- planner 原样转抄到 `PrefixSharingPlan.group_ids`；
- `PrefixLastRestoreSpec.group_id` 保存但不参与关键 restore 逻辑；
- observability 用 `group_ids` 统计 `sharing_group_count`。

问题：

- group 相关字段不是当前 arbitrary-prefix runtime 的事实源。
- `sharing_group_count` 可以从 `reuse_specs` 中的 `(provider_idx_in_batch, prefix_len)` 唯一集合推导。
- `PrefixLastRestoreSpec.group_id` 当前没有精度语义价值。
- group 字段让 PrefixSharing 看起来像 PrefixGrouper 的 group 模型，但实际 planner 语义是 provider/reuser DAG，这会增加概念混淆。

建议：

- 第一批重构删除 `PrefixGroup`、`PrefixDetectionResult.groups`、`PrefixDetectionResult.group_ids`、`PrefixSharingPlan.group_ids`、`PrefixLastRestoreSpec.group_id`。
- 保留 `provider_index`、`prefix_lens`、`is_provider` 在 DetectionResult 和 Plan 中的短期重复，因为这些字段在 Trie 遍历时已自然产生，Plan/backend 又高频使用。为了“瘦身”而删除再重算没有必要。

优先级：P0。改动范围小，收益明确。

#### 1.4.2 `PrefixSharingPlan` 字段多，但不宜粗暴合并

`PrefixSharingPlan` 当前同时保存：

- 检测视图：`reuse_specs`、`provider_index`、`prefix_lens`、`is_provider`
- 执行 layout：`kept_lengths_q`、`expanded_lengths_kv`、`cu_seqlens_*`
- 位置语义：`q_position_offsets`、`kv_position_offsets`
- 裁剪范围：`input_keep_ranges`、`label_keep_ranges`、`loss_mask_keep_ranges`
- restore：`prefix_last_restore`

问题不是“字段多”本身，而是缺少字段分组和命名边界。Plan 是 backend/runtime 的核心契约，不能把这些字段都藏到 DetectionResult 里，也不能和 RuntimeState 合并。

建议：

- 保留 `PrefixSharingPlan` 作为 core 语义契约。
- 中期引入小型子结构只用于提升可读性，例如：
  - `PrefixReuseIndex`：`reuse_specs/provider_index/prefix_lens/is_provider`
  - `TrimLayout` 或 `KeptTokenLayout`：`kept_lengths_q/input_keep_ranges/q_position_offsets`
  - `RestorePlan`：`prefix_last_restore`
- 但第一批不要引入过多新类，优先删除 group 相关字段和清理注释。

优先级：P1。需要测试保护，不能和 group 删除混在一个大 PR。

#### 1.4.3 `is_provider` 命名存在语义歧义

当前 detector 中 `is_provider=False` 表示该行是 reuser；`is_provider=True` 表示该行不是 reuser。但这并不严格等价于“该行被别人复用”。一个 standalone row 也会是 `is_provider=True`。

这会影响可读性和 observability：

```python
provider_count=sum(prefix_sharing_plan.is_provider)
```

实际统计的是 non-reuser count，不是严格 provider count。

建议：

- 短期文档中明确 `is_provider` 当前语义是 “full-compute row / non-reuser row”。
- 中期重命名为 `is_full_compute_row` 或 `is_reuser` 反向字段。
- 如果需要真正 provider count，应从 `reuse_specs.provider_idx_in_batch` 去重统计。

优先级：P1。改名影响面较大，先记录，不和 P0 group 删除混做。

#### 1.4.4 PrefixGrouper group 模型与 PrefixSharing DAG 模型需要文档化

当前代码同时出现：

- PrefixGrouper 风格配置入口；
- `PrefixGroup` / group_ids；
- PrefixSharing provider/reuser DAG；
- chain reuse 逻辑。

这容易让 reviewer 误解：是不是要把 arbitrary-prefix 强行压成 PrefixGrouper `group_info`。

建议：

- 在 docs 中明确：对外沿用 PrefixGrouper feature 入口；对内 arbitrary-prefix 使用 provider/reuser DAG plan。
- 删除 `PrefixGroup` 这类容易混淆的内部结构。
- 在 `PrefixReuseSpec` docstring 中强调它是 arbitrary-prefix 的事实源。

优先级：P0。主要是代码和文档一致性问题。

### 1.5 Integration 层代码质量问题

#### 1.5.1 `verl_mcore.py` 过长且历史包袱明显

`integrations/verl_mcore.py` 约 950 行，当前包含：

- v070/v080 描述；
- Megatron runtime state；
- v070 actor micro-batch build；
- v080 engine micro-batch build；
- PrefixGrouper 风格配置读取；
- 2D restore；
- NestedTensor / plain THD trim helper；
- FSDP 复用的 helper。

问题：

- 面向开源 review 时，v070 叙述、PATH debug print、MCore/FSDP helper 混杂会显著降低可信度。
- FSDP 通过 `from prefix_sharing.integrations.verl_mcore import _trim_nested_batch` 等私有 helper 复用 MCore 代码，说明公共 batch/layout helper 没有抽出来。
- 文件名 `verl_mcore.py` 下承载 PrefixGrouper config 读取，也不符合职责。

建议：

第一批拆分方向：

```text
integrations/verl_config.py       # read_prefix_grouper_config / PrefixSharingConfig bridge
integrations/verl_batch.py        # NestedTensor / dense trim, kept_position_rows
integrations/runtime_state.py     # PrefixSharingRuntimeState
integrations/verl_mcore.py        # 只保留 MCore/Megatron 专属流程
integrations/verl_fsdp.py         # FSDP 专属流程
```

优先级：P0/P1。先抽 `verl_config.py` 和 `verl_batch.py`，收益大且能减少 FSDP 对 MCore 的反向依赖。

#### 1.5.2 FSDP adapter 接近目标，但 still too much “standalone helper”

`integrations/verl_fsdp.py` 已经有较完整的 FSDP path：

- `build_prefix_sharing_micro_batch_fsdp()`
- `PrefixSharingFSDPAttentionRuntime`
- `restore_prefix_sharing_outputs_2d()`
- `forward_prefix_sharing_fsdp_micro_batch()`

问题：

- `forward_prefix_sharing_fsdp_micro_batch()` 更像测试/fake engine helper，不一定是合入 verl 的主路径，应避免让 reviewer 误以为这是生产接入方式。
- `PrefixSharingFSDPAttentionRuntime.forward()` 直接忽略 `attn_func/attention_mask/kwargs`，对 HF attention 接口兼容性说明不足。
- dense `[B,L,H,D]` 与 packed `[1,T,H,D]` 两种路径混在同一个 runtime，缺少清晰的 input contract。
- FSDP restore 的 logits/log_probs/entropy/attention_output copy 语义复杂，需要更强测试和更小函数。

建议：

- 将 fake/local helper 标记为 test utility 或 internal fallback，生产接入主推 `setup/patches/verl080_fsdp/forward_step.py`。
- 把 FSDP runtime 拆成：
  - `pack_dense_qkv`
  - `run_packed_attention`
  - `scatter_output`
  - `restore_2d_outputs`
- 对 HF attention wrapper 只保留薄适配，业务语义留在 integration runtime。

优先级：P1。需要保持 FSDP 精度测试稳定。

#### 1.5.3 `PrefixSharingRuntimeState` 放在 `verl_mcore.py` 不合适

FSDP 也从 `verl_mcore.py` import `PrefixSharingRuntimeState`。这说明它已经不是 MCore 专属类型。

建议：

- 移到 `integrations/runtime_state.py` 或 `integrations/context.py` 附近。
- 字段保持：
  - `prefix_sharing_plan`
  - `attention_backend`
  - `packed_batch_layout`
  - `parallel_info`
  - optional `kept_position_ids`
  - optional `valid_indices`

优先级：P0。改动小，可读性收益高。

#### 1.5.4 PatchManager 与 setup/logged_patch 存在两套 patch 机制

当前同时有：

- `integrations/patch_manager.py`
- `integrations/megatron_attention.py`
- `setup/logged_patch.py`
- `setup/registry.py`

测试中仍覆盖 `PatchManager`、`MegatronAttentionIntegration`、`VerlMCoreIntegration`。但当前 open-source 目标显然更偏 `setup/patches/*` 这套统一 patch set。

问题：

- 两套 patch 机制会让 reviewer 质疑哪套是生产入口。
- `VerlFSDPIntegration.install()` 内仍使用 `PatchManager` patch `ALL_ATTENTION_FUNCTIONS`，但 setup patch set 也 patch 同一目标。
- 旧 integration 类更像历史原型接口。

建议：

- 明确生产入口只保留 `prefix_sharing.setup.install()` 和 patch set。
- 将 `integrations/patch_manager.py` / `megatron_attention.py` / `VerlMCoreIntegration` / `VerlFSDPIntegration` 标记为待删除或测试专用，优先评估是否还有真实调用。
- 如果没有真实调用，删除旧 patch manager 和相关测试，减少重复架构。

优先级：P0/P1。需先 `rg` 确认外部引用；若仅测试使用，可以尽快删除。

### 1.6 Setup / patch 层代码质量问题

#### 1.6.1 import side effect 太强

`prefix-sharing/prefix_sharing/__init__.py` 当前 import 后自动执行 `_auto_install_patches()`。这对内部快速试用方便，但开源包风险较高：

- 用户 `import prefix_sharing` 可能只是想使用 core API，却触发 patch 检测和 stdout 输出。
- 自动 patch 与 `PREFIX_SHARING_PATCHSET`、compat matrix、verl/Megatron import 状态耦合。
- 社区更倾向显式安装 patch，或由 verl 内部在明确配置开启时调用。

建议：

- 开源主路径改为显式：
  ```python
  import prefix_sharing
  prefix_sharing.setup.install("verl080_fsdp")
  ```
- 如果保留 auto install，应通过明确环境变量控制，例如 `PREFIX_SHARING_AUTO_INSTALL=1`，默认关闭。
- 对 verl 合入 PR，尽量由 verl 的 PrefixGrouper 路径显式调用，不依赖第三方包 import side effect。

优先级：P0。社区可接受度影响大。

#### 1.6.2 compat matrix 不覆盖 FSDP patch set

`compat_matrix.py` 当前主要匹配：

- `verl080_mcore0161_ms0160`
- `mcore012_ms012`

FSDP patch set 文档建议显式 `install("verl080_fsdp")`，避免自动选到 Megatron patch set。

问题：

- 对用户不自然。FSDP 是开源首推路径，却不能被 compat matrix 自然选择。
- 如果环境同时安装 Megatron/MindSpeed，默认选择 Megatron patch set，不符合“首批合入 FSDP”目标。

建议：

- 引入 patch target 参数，而不是单纯版本矩阵：
  ```python
  prefix_sharing.setup.install(target="fsdp")
  prefix_sharing.setup.install(target="megatron")
  ```
- 或从 PrefixGrouper mode / engine type 决定 patch set。
- 版本矩阵只做兼容性校验，不做唯一 patch set 决策。

优先级：P1。需要设计清楚，不建议匆忙改。

#### 1.6.3 import hook 复杂度高，需要开源化收敛

`setup/registry.py` 的 import hook 已处理 lazy module、miss threshold、eager import。工程上可用，但社区 review 会关注：

- 是否会影响全局 `builtins.__import__`；
- 是否会破坏 torch.compile / CUDA graph；
- 是否线程安全；
- 失败时是否可观测；
- 是否能 rollback。

建议：

- 首批开源文档明确默认使用 eager patch 目标，import hook 是 fallback。
- 对 FSDP patch set 尽量避免 import hook，使用明确 import + patch。
- 将 import hook 逻辑隔离并补充单测，避免隐藏全局副作用。

优先级：P1。

### 1.7 Backend / runtime 层代码质量问题

#### 1.7.1 TorchReferenceBackend 承载过多正式与实验语义

`TorchReferenceBackend` 当前同时包含：

- `apply_rope`
- attention KV `build_kv`
- debug/reference attention
- gated attention
- DeltaNet state reference

问题：

- 对首批 open-source / verl PR，DeltaNet 和 gated attention 会扩大 review 面。
- `build_kv()` 又被 GPU/NPU FlashAttention backend 复用，是正式路径核心，不只是 reference。
- 类名 `TorchReferenceBackend` 与其承担的正式 `build_kv()` 职责不完全一致。

建议：

- 将 KV expansion 抽成独立组件，例如 `AttentionKVBuilder` 或 backend shared helper。
- `TorchReferenceBackend.attention()` 保持 correctness/reference 用途。
- Gated/DeltaNet 放入 experimental 模块或后续分支，不进入首批 refactor 主 PR。

优先级：P1/P2。涉及 backend 测试较多，需分步做。

#### 1.7.2 Runtime context 默认 audit print 不适合开源主路径

`context.py` 在 context exit 时默认 `_log_prefix_sharing_audit(ctx)`，内部直接 `print()`。`verl_mcore.py` 和 `megatron_runtime.py` 也有大量热路径 print。

问题：

- 训练日志刷屏。
- 性能测试被 stdout I/O 干扰。
- 社区代码不接受默认 debug print。

建议：

- 引入 `logging.getLogger("prefix_sharing")`。
- 默认 warning/error，audit 需要显式 `PREFIX_SHARING_LOG_LEVEL=INFO` 或 config 开关。
- diagnostic dump 与 audit 分开：dump 是精度工具，audit 是运行统计。

优先级：P0。改动小，开源观感收益大。

#### 1.7.3 Diagnostic tools 与生产路径耦合偏多

FSDP attention patch、forward_step patch、Megatron runtime 中都有 `PREFIX_SHARING_DIAG_DUMP` 分支。诊断能力重要，但当前散落在生产 patch 内。

建议：

- 保留诊断能力，但集中到 `diagnostics` helper，例如：
  ```python
  diagnostics.enabled()
  diagnostics.dump_fsdp_attn_output(...)
  ```
- 生产 patch 只调用一个 helper，不直接 import dump 工具。

优先级：P1。

### 1.8 测试现状与缺口

当前测试目录覆盖面较广：

- unit：config、detector、planner、store、packed layout、runtime context、FSDP adapter、FlashAttention base。
- integrated：patch integration、verl080 restore e2e placeholder、optional GPU/NPU backend。
- system：phase1 core。

主要缺口：

1. `test_verl080_restore_e2e.py` 仍有 TODO placeholder，说明真实 engine e2e 验证还不闭环。
2. PrefixGrouper 风格配置入口已有测试，但还缺少“prompt_only 走原 PrefixGrouper / arbitrary_prefix 走 PrefixSharing”的端到端语义测试。
3. 删除 group 字段前，需要补充/调整 detector/planner 测试，明确 `reuse_specs + provider_index + prefix_lens` 是事实源。
4. auto install / explicit install / no side effect 的行为需要单测，否则修改 `__init__.py` 风险较高。
5. 日志门控需要测试默认不输出热路径 print。
6. FSDP patch 与 HF attention wrapper 需要更贴近真实 Qwen2.5/Qwen HF attention 的 fake fixture，而不只是 tiny model。

### 1.9 重构事项优先级排序

排序原则：

- P0：改动范围小、收益明确、能明显提升开源可读性/合入可接受度。
- P1：收益大但涉及多模块，需要测试保护。
- P2：重要但可延后，不应阻塞第一波开源整改。

#### P0-1：删除 group 相关冗余结构

范围：

- 删除 `PrefixGroup`
- 删除 `PrefixDetectionResult.groups`
- 删除 `PrefixDetectionResult.group_ids`
- 删除 `PrefixSharingPlan.group_ids`
- 删除 `PrefixLastRestoreSpec.group_id`
- observability 的 `sharing_group_count` 改为从 `reuse_specs` 推导

理由：

- group 相关字段未承载关键 runtime 语义；
- 容易混淆 PrefixGrouper group 模型和 PrefixSharing DAG 模型；
- 测试改动可控。

预期收益：

- core 概念更清晰；
- planner 字段减少；
- 为后续对外解释“PrefixGrouper 入口 + PrefixSharing arbitrary-prefix runtime”扫清概念噪音。

#### P0-2：将 `PrefixSharingRuntimeState` 移出 `verl_mcore.py`

范围：

- 新建 `integrations/runtime_state.py`
- 更新 FSDP/MCore/context/tests import

理由：

- RuntimeState 是 FSDP 和 MCore 共用类型，不属于 MCore。
- 当前 FSDP import MCore 私有定义，开源 review 观感差。

预期收益：

- integration 层边界立即改善；
- 为拆分 `verl_mcore.py` 做准备。

#### P0-3：抽出 verl 配置桥接逻辑

范围：

- 新建 `integrations/verl_config.py`
- 移出 `read_ps_config_from_engine_config()` 和 `_prefix_sharing_config_from_prefix_grouper()`
- 明确公开主入口是 PrefixGrouper 风格配置

理由：

- 配置桥接不是 MCore 专属；
- 这是“像 PrefixGrouper 扩展”的关键代码，应独立、短小、可测试。

预期收益：

- FSDP/MCore 共享同一配置入口；
- 首批 verl PR 更容易只 review 配置转换和 forward hook。

#### P0-4：默认关闭 import side-effect auto patch

范围：

- `prefix_sharing.__init__` 不再默认 `_auto_install_patches()`，或用 `PREFIX_SHARING_AUTO_INSTALL=1` gate。
- README 和开发文档改成显式 install。

理由：

- Python 包 import 产生 monkey patch 副作用是社区高风险点。
- verl 合入后应由 verl 配置路径显式触发，不应依赖第三方包 import 副作用。

预期收益：

- 开源包行为可预测；
- 降低社区 review 阻力。

#### P0-5：热路径 print 改为 logger 并默认关闭 audit

范围：

- `context.py`
- `verl_mcore.py`
- `megatron_runtime.py`
- `setup/*` 中非必要 stdout

理由：

- 默认训练不应刷屏；
- 直接 print 会污染性能测试和用户日志。

预期收益：

- 开源观感直接提升；
- 性能测试更干净。

#### P1-1：拆分 `verl_mcore.py` 的 batch/layout helper

范围：

- 新建 `integrations/verl_batch.py`
- 移出 NestedTensor / dense trim、sequence extraction、kept_position_rows
- FSDP 和 MCore 都依赖该公共 helper

理由：

- 解决 FSDP 反向 import MCore 私有 helper 的问题；
- 降低 `verl_mcore.py` 复杂度。

#### P1-2：收敛 patch 体系，只保留 setup patch set 作为生产入口

范围：

- 评估删除 `integrations/patch_manager.py`
- 评估删除 `integrations/megatron_attention.py`
- 删除或迁移相关测试
- 保留 `setup/logged_patch.py` / `setup/registry.py`

理由：

- 两套 patch 系统增加维护成本；
- 开源 reviewer 会质疑重复机制。

#### P1-3：FSDP runtime 函数化拆分

范围：

- 将 pack/run/scatter/restore 拆成小函数或小模块；
- 明确 fake helper 与真实 engine patch 的边界。

理由：

- 当前 FSDP 文件 500 行，逻辑密度高；
- restore 语义复杂，拆小后更容易测试。

#### P1-4：重新设计 setup patch set 选择

范围：

- `install(target="fsdp" | "megatron")`
- 或 `install(patch_set_id=...)` + 更明确文档
- compat matrix 不再单独决定 patch set

理由：

- FSDP 是首批开源主线，但当前需要显式 patch set 避免被 Megatron 版本矩阵抢走。

#### P1-5：整理 README 与用户文档

范围：

- README 首选 Qwen2.5-0.5B + verl080 统一依赖，不按模型区分依赖。
- Quick Start 改成 PrefixGrouper 风格配置。
- `ENABLE_PREFIX_SHARING` 降级为开发/调试入口。

理由：

- 用户心智要对齐 verl PrefixGrouper；
- 当前 README 容易让用户以为这是 Megatron-only 独立特性。

#### P2-1：Backend shared KV builder 抽象

范围：

- 从 `TorchReferenceBackend` 中拆出正式路径使用的 `build_kv`。
- TorchRef attention 保持 reference/debug。

理由：

- 当前 TorchRef 类名与正式 `build_kv` 职责不一致。

#### P2-2：DeltaNet / gated attention experimental 化

范围：

- 将 DeltaNet reference store/backend 文档标记为 experimental。
- 首批开源文档不作为主线能力宣传。

理由：

- 防止第一波 PR review 面过大。

#### P2-3：诊断工具模块化

范围：

- 将 `PREFIX_SHARING_DIAG_DUMP` 分支集中到 diagnostics helper。
- CLI 工具保持独立。

理由：

- 生产 patch 更短，诊断能力仍保留。

### 1.10 第一批建议 PR 切分

建议不要做一个“大重构 PR”。按以下顺序拆：

1. **PR-A：core 概念瘦身**
   - 删除 group 相关字段。
   - 更新 detector/planner/observability/tests。
   - 不改 runtime 语义。

2. **PR-B：integration 基础类型和配置桥接拆分**
   - 移出 `PrefixSharingRuntimeState`。
   - 新增 `verl_config.py`。
   - FSDP/MCore import 改为公共模块。

3. **PR-C：开源默认行为收敛**
   - 关闭默认 import auto patch 或加显式 gate。
   - print -> logger。
   - README 改成 PrefixGrouper 风格配置。

4. **PR-D：FSDP adapter 可读性整理**
   - 拆 pack/run/scatter/restore。
   - 明确 fake helper 与真实 engine patch。
   - 补 FSDP patch 测试。

5. **PR-E：patch 体系收敛**
   - 删除旧 `PatchManager` / integration class 体系，或明确 legacy。
   - 保留 setup patch set 作为唯一生产入口。

### 1.11 当前判断

当前 `open-source_perf` 分支已经接近“可向 PrefixGrouper 体系靠拢”的方向，但还不适合直接开源或提交给 verl 社区。主要阻塞不是算法，而是软件工程表达：

- 概念层面：PrefixGrouper group、PrefixSharing provider/reuser DAG、Plan/RuntimeState 混在一起，解释成本偏高。
- 接入层面：FSDP、MCore、旧 patch manager、setup patch set 多套入口并存。
- 用户层面：README 和默认开关仍像 PrefixSharing 独立特性，不像 PrefixGrouper 的 algorithm/mode 扩展。
- 工程层面：热路径 print、import side effect、诊断逻辑散落，会直接影响开源 review 观感。

第一阶段应优先做 P0 小步重构。完成后，代码会更接近以下形态：

```text
verl PrefixGrouper user entry
  -> mode=prompt_only: existing PrefixGrouper
  -> mode=arbitrary_prefix: PrefixSharing plan/runtime/backend

PrefixSharing core
  -> PrefixReuseSpec / PrefixDetectionResult
  -> PrefixSharingPlan
  -> PrefixSharingRuntimeState

Integrations
  -> verl_config.py
  -> verl_batch.py
  -> verl_fsdp.py
  -> verl_mcore.py

Setup
  -> explicit install / patch set
  -> no default intrusive side effect
```

## Chapter 2：方案设计

待补充。建议在完成 P0 项评审后，按 PR-A 到 PR-C 的顺序展开详细设计。

## Chapter 3：测试验证

待补充。需要覆盖 core 字段瘦身、配置入口兼容、import side-effect、日志门控、FSDP adapter 精度一致性、Megatron 既有路径回归。

## Chapter 4：开发计划

待补充。建议以小 PR 迭代，每个 PR 都保持测试可回归，避免把 core、integration、setup、文档重构混成一个不可 review 的大变更。

## Chapter 5：当前结论

待补充。Chapter 1 的当前结论是：优先做 P0 小步整改，不改变核心精度语义；对外入口向 PrefixGrouper mode 靠拢，对内保留 PrefixSharing provider/reuser DAG。

## Chapter 6：遗留问题

待补充。已知遗留方向包括：上游 verl 最新 PrefixGrouper 主线核对、FSDP 真实 engine e2e fixture、Megatron 路径是否作为首批开源能力、DeltaNet/gated attention 是否延后到 experimental。
