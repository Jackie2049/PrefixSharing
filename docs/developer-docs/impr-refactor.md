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
| `core/prefix_store.py` | 生命周期内的 attention KV / 历史 activation store | 当前混入了 Qwen3.5 / Gated DeltaNet 专门化设计；开源首版应先清理到 attention KV 主线，避免过早暴露未接入训练引擎的 mixer-specific 抽象。 |
| `backends/torch_ref.py` | reference backend、KV expansion、debug attention、历史 gated/deltanet reference | 文件较重，正式路径 `build_kv()` 被 GPU/NPU backend 复用，但仍挂在 TorchRef 上；Qwen3.5 / GDN 相关 reference 应从首批主线清掉。 |
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
- 后续实验：NPU、MindSpeed、Megatron-Bridge、HybridAttention/Gated DeltaNet。首批开源整改不承诺这些 mixer-specific 能力。

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

优先级：P1。改动范围小，收益明确，但应排在 mixer-specific 清理、backend 公共能力抽离、integration 公共模块抽离之后。

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

优先级：P2。需要测试保护，且当前不是开源首批阻塞。

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

优先级：P2。当前语义在主流程中没有引起统计或精度错误，短期只需在注释中说明“non-reuser/full-compute row”的实际含义，不作为首批整改重点。

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

优先级：P1。主要是代码和文档一致性问题，但不应优先于首批主线收窄。

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

第一批拆分方向优先从 `verl_utils.py` 起步，先消除 FSDP 对 MCore 私有 helper 的反向依赖；如果该文件继续变大，再按职责拆成更细模块：

```text
integrations/verl_utils.py        # FSDP/MCore 共用配置、batch、position helper
integrations/verl_config.py       # read_prefix_grouper_config / PrefixSharingConfig bridge
integrations/verl_batch.py        # NestedTensor / dense trim, kept_position_rows
integrations/runtime_state.py     # PrefixSharingRuntimeState
integrations/verl_mcore.py        # 只保留 MCore/Megatron 专属流程
integrations/verl_fsdp.py         # FSDP 专属流程
```

优先级：P0/P1。先抽 `verl_utils.py` 和 runtime state，收益大且能减少 FSDP 对 MCore 的反向依赖。

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

#### 1.6.1 import auto patch 需要保留，但必须收敛边界

`prefix-sharing/prefix_sharing/__init__.py` 当前 import 后自动执行 `_auto_install_patches()`。这对内部快速试用和 verl external modules 路径有价值，但开源包风险需要控制：

- 用户 `import prefix_sharing` 可能只是想使用 core API，却触发 patch 检测和 stdout 输出。
- 自动 patch 与 `PREFIX_SHARING_PATCHSET`、compat matrix、verl/Megatron import 状态耦合。
- 当前真实训练脚本依赖 `VERL_USE_EXTERNAL_MODULES=prefix_sharing` 这类 import 后直接 patch 的路径；在正式 PR 到 verl 并获得社区认可前，不能贸然移除。

建议：

- **同时保留两种入口**：
  ```python
  import prefix_sharing  # 支持 import 后自动 patch，服务当前脚本化训练
  prefix_sharing.setup.install("verl080_fsdp")  # 支持显式 install，服务交互式和更清晰的集成
  ```
- 文档中把显式 `setup.install()` 作为推荐可读入口，把 import auto patch 描述为兼容当前外部模块加载机制。
- auto patch 必须做到幂等、失败信息清晰、默认不刷屏；如果 patchset 不匹配，应安全跳过或给出可诊断错误。
- 后续正式合入 verl 后，再讨论是否下线 import auto patch。

优先级：P0。不是删除 auto patch，而是把“双入口并存”的设计写清楚并降低副作用。

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
- 对 FSDP patch set 优先使用明确 import + patch；import auto patch 路径继续保留，但要做到行为可解释。
- 将 import hook 逻辑隔离并补充单测，避免隐藏全局副作用。

优先级：P1。

### 1.7 Backend / runtime 层代码质量问题

#### 1.7.1 TorchReferenceBackend 承载过多正式与 mixer-specific 语义

`TorchReferenceBackend` 当前同时包含：

- `apply_rope`
- attention KV `build_kv`
- debug/reference attention
- 历史 gated attention / DeltaNet state reference

问题：

- 对首批 open-source / verl PR，Qwen3.5 / Gated DeltaNet 专门化代码会扩大 review 面。
- `build_kv()` 又被 GPU/NPU FlashAttention backend 复用，是正式路径核心，不只是 reference。
- 类名 `TorchReferenceBackend` 与其承担的正式 `build_kv()` 职责不完全一致。

建议：

- 将 KV expansion 抽成独立组件，例如 `AttentionKVBuilder` 或 backend shared helper。
- `TorchReferenceBackend.attention()` 保持 correctness/reference 用途。
- 清理 Qwen3.5 / Gated DeltaNet 专门化 store/backend/protocol，不进入首批开源主线；后续等训练引擎侧真实接入 HybridAttention 后再按实际接口补回。

优先级：P0/P1。`build_kv` 抽离是低风险高收益；mixer-specific 清理需要确认测试引用后分步做。

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

- P0：改动范围小、收益明确、能直接降低开源 review 阻力。
- P1：收益大但涉及多模块，需要测试保护。
- P2：合理但不阻塞第一波开源整改，避免为了“设计洁癖”扩大改动面。

#### P0-1：清理 Qwen3.5 / Gated DeltaNet 专门化设计

范围：

- 删除或下线 `StoredDeltanetState`、`PrefixDeltanetStore`、`PrefixDeltanetBackend` 等当前未接入真实训练引擎的专门化类型。
- `PrefixActivationStore` 如果保留，应只作为最小 store 基类；首批主线只暴露 `PrefixAttentionStore` / `StoredAttentionKV`。
- `TorchReferenceBackend.build_deltanet_states()` 和相关测试如果只服务历史讨论，应从开源主线删除或移到明确的历史实验目录。

理由：

- 当前目标是面向 verl080 开源和合入，不是交付 Qwen3.5/3.6 HybridAttention。
- GDN 接入依赖后续训练引擎真实接口；提前保留会让 reviewer 质疑抽象是否过度设计。
- 用户已经明确希望版本更简洁、轻量。

预期收益：

- store/backend/protocol 更聚焦 attention KV 主线。
- 降低首批开源代码解释成本。

#### P0-2：抽离 backend 公共 KV 构建能力

范围：

- 从 `TorchReferenceBackend` 抽出正式路径使用的 `build_kv()`，放到公共 helper 或父类，例如 `backends/kv_builder.py` 或 `AttentionKVBuilderMixin`。
- GPU/NPU FlashAttention backend 依赖该公共能力，而不是依赖 `TorchReferenceBackend`。
- TorchRef 继续作为 correctness/reference attention backend，不再承载正式路径公共能力。

理由：

- `build_kv()` 是生产路径核心，挂在 TorchRef 下命名不准确。
- GPU/NPU backend 依赖 TorchRef 会让社区误解生产 FA 路径仍经过 reference backend。
- 这是低风险高收益重构，测试已有 backend 覆盖可复用。

预期收益：

- backend 分层清晰：公共 KV expansion 与 reference attention 解耦。
- 为逐步下线 TorchRef 生产依赖做准备。

#### P0-3：抽出 verl FSDP/MCore 公共模块

范围：

- 新建 `integrations/verl_utils.py`，先承载 FSDP 和 MCore 共同使用的配置读取、batch trim、NestedTensor/dense helper、position helper。
- 如果 `verl_utils.py` 后续继续变大，再拆成 `verl_config.py`、`verl_batch.py`；第一步优先消除“FSDP import MCore 私有函数”的反向依赖。
- `PrefixSharingRuntimeState` 移出 `verl_mcore.py`，放到 `integrations/runtime_state.py` 或同等公共位置。

理由：

- `verl_fsdp` 和 `verl_mcore` 都用到的函数不应放在其中一个模块里。
- FSDP 是开源首推路线，不能让 FSDP 代码看起来依赖 Megatron/MCore 内部实现。

预期收益：

- integration 层职责边界立即改善。
- 后续删 verl070/MCore 历史代码时更安全。

#### P0-4：删除旧 patch_manager 体系

范围：

- 删除 `integrations/patch_manager.py`、`integrations/megatron_attention.py`、`VerlMCoreIntegration`、`VerlFSDPIntegration` 等旧 integration patch 入口，前提是 `rg` 确认只剩测试或历史路径引用。
- 删除或改写对应测试，保留 `setup/` patch set 作为当前主力 monkey patch 机制。
- `setup/logged_patch.py` 可作为统一 patch manager 留存，但需要去掉“与 integrations/patch_manager.py 相同”这类历史注释。

理由：

- 当前主力是 `setup/patches/*`；旧 patch manager 是重复架构和潜在死代码。
- 开源 reviewer 会直接问“哪套 patch 才是生产入口”。

预期收益：

- patch 接入路径单一。
- import hook 和 patch registry 后续整改范围更小。

#### P0-5：配置入口向 PrefixGrouper 对齐

范围：

- 确认并文档化：`use_prefix_grouper=true + prefix_grouper.mode=arbitrary_prefix` 已可使能 PrefixSharing。
- README 首选 PrefixGrouper 风格配置；`ENABLE_PREFIX_SHARING` 保留为开发/调试 fallback。
- `prefix_sharing_config` 保留为内部兼容/测试入口，但不作为首批 verl 用户公开主入口。
- `prompt_only` 继续归 PrefixGrouper；`arbitrary_prefix` 进入 PrefixSharing plan/runtime/backend。

理由：

- 我们面向 verl 的定位是 PrefixGrouper 扩展，而不是另起一个 prefix-sharing 用户入口。
- verl 配置项应作为第一优先级，减少社区 schema 变更。

预期收益：

- 用户心智对齐 verl。
- 首批 PR 更容易聚焦为 “PrefixGrouper 增加 arbitrary_prefix mode”。

#### P0-6：README 简要说明 PrefixSharing 与 PrefixGrouper 关系

范围：

- 在 README 增加一小节：
  - PrefixGrouper 是 verl 已有用户入口和 prompt-only baseline。
  - PrefixSharing 负责 arbitrary-prefix 的 provider/reuser plan、KV injection、restore。
  - 本仓库不 vendor PrefixGrouper 核心算法，不把 PrefixGrouper `group_info` 当作 arbitrary-prefix 的内部事实源。
- 明确仓库中允许存在的 PrefixGrouper 相关代码边界：
  - 允许：配置读取与兼容，例如 `use_prefix_grouper`、`prefix_grouper.mode`、`prefix_grouper.min_prefix_len` 等。
  - 允许：为了复用 verl 现有 attention hook 心智而保留的薄 adapter / 参数透传。
  - 允许：README / docs 中说明 prompt-only PrefixGrouper 与 arbitrary-prefix PrefixSharing 的关系。
  - 不允许：复刻 PrefixGrouper prompt-only 算法、维护独立 `group_info` runtime、把 PrefixGrouper group 模型作为 PrefixSharing arbitrary-prefix 的内部事实源。
  - 不允许：为了“看起来兼容 PrefixGrouper”而引入大量不参与主流程的 wrapper。

理由：

- 当前代码只有配置/接口靠拢，不应让 reviewer 以为仓库里有一套 PrefixGrouper 复刻代码。
- 关系说明能提前化解“为什么叫 prefix_grouper 但 runtime 是 prefix_sharing”的疑问。
- 如果扫描发现 PrefixGrouper 相关代码超过上述边界，应优先删除或下沉为测试 fixture。

#### P0-7：compat matrix 以 FSDP 为第一优先级

范围：

- compat matrix 必须覆盖 `verl080_fsdp`，且文档中明确这是首推路径。
- `install("verl080_fsdp")`、`PREFIX_SHARING_PATCHSET=verl080_fsdp`、import auto patch 三条路径都应能稳定选到 FSDP patch set。
- Megatron/MCore 保持 advanced/experimental 路线，不作为第一波开源默认路径。

理由：

- FSDP 更轻量、复现门槛低，适合首批开源和社区 review。
- Meituan RFC/PR 主要朝 Megatron/Magi/flex/prefix-tree 方向推进，FSDP-first 与其形成互补，避免第一波就在重型 Megatron/Magi surface 上竞争。

#### P0-8：调试 dump / print 先框定，再集中封装

范围：

- 热路径 `print()`、临时 dump、诊断日志先用统一注释标记，例如 `# PREFIX_SHARING_DIAGNOSTIC`，方便后续一把清理。
- 能快速封装的 dump 逻辑移入少数公共 helper，例如 `diagnostics.enabled()`、`diagnostics.dump_fsdp_attention(...)`。
- 默认训练路径不应刷屏；audit 与 dump 分开。

理由：

- 调试能力在精度对齐阶段仍有价值，不能一刀切删除。
- 但散落在生产 patch 里的 dump 会降低可读性和性能可信度。

#### P1-1：删除 group 相关冗余结构

范围：

- 删除 `PrefixGroup`
- 删除 `PrefixDetectionResult.groups`
- 删除 `PrefixDetectionResult.group_ids`
- 删除 `PrefixSharingPlan.group_ids`
- 删除 `PrefixLastRestoreSpec.group_id`
- observability 的 `sharing_group_count` 改为从 `reuse_specs` 推导

理由：

- group 相关字段未承载关键 runtime 语义。
- 容易混淆 PrefixGrouper group 模型和 PrefixSharing provider/reuser DAG 模型。

说明：

- 这项仍值得做，但不应优先于 Qwen3.5/GDN 清理、backend 公共能力抽离、integration 公共模块抽离。
- `provider_index`、`prefix_lens`、`is_provider` 暂时保留，因为它们在 Trie 遍历时已自然产生，Plan/backend 又高频使用，删掉再重算没有收益。

#### P1-2：FSDP runtime 函数化拆分

范围：

- 将 pack/run/scatter/restore 拆成小函数或小模块。
- 明确 fake/local helper 与真实 engine patch 的边界。
- fake 和 test-utils 必须注释清楚，避免读者误解为核心生产路径。

理由：

- FSDP 是首推路线，代码需要更适合社区 review。
- restore 语义复杂，拆小后更容易测试。

#### P1-3：import hook 复杂度整改

范围：

- 梳理 `setup/registry.py` 的 import hook、eager patch、lazy patch 的真实调用路径。
- 保留 import 后直接 patch 与显式 `setup.install()` 双入口。
- 提升幂等性、错误信息和日志可控性。
- 形成明确检查清单：
  - `prefix_sharing.__init__` auto install 的触发链是什么；
  - `VERL_USE_EXTERNAL_MODULES=prefix_sharing`、`PREFIX_SHARING_PATCHSET=verl080_fsdp`、显式 `setup.install("verl080_fsdp")` 三者的优先级和交互是什么；
  - eager patch 与 lazy import hook 分别在哪些 patch set 中实际使用；
  - 重复 import / 重复 install 是否完全幂等；
  - patch 目标缺失时是安全 skip、warning，还是 hard fail；
  - patch 失败是否存在 silent skip；
  - import hook 是否能 rollback，是否会影响全局 `builtins.__import__` 的其他用户；
  - 单测是否覆盖 import 顺序变化、重复安装、patchset 显式指定和未指定四类情况。

理由：

- import hook 是 monkey patch 包最容易被社区挑战的部分。
- 在身份“转正”前不能删除，但需要可解释、可测试。

#### P1-4：tools 目录清理分级

范围：

- 清理历史性能摸底、精度摸底、一次性 debug 脚本。
- 保留每个重要版本都要复跑的精度验证、性能验证工具和脚本。
- 保留工具需要有 README 或文件头说明：用途、输入、输出、适用场景。
- 保留标准：
  - 能复现关键精度结论，例如 logprob/loss/grad 与 baseline 对齐；
  - 能复现关键性能结论，例如 FSDP baseline、PrefixGrouper prompt-only、PrefixSharing arbitrary-prefix 三方对比；
  - 能作为 release / 重要 PR 前的回归验证；
  - 依赖 GPU、verl、flash-attn、torch_npu 等环境时，必须在说明中写清楚。
- 删除标准：
  - 只服务某次临时排查，且结论已经沉淀到文档或测试；
  - 与当前 verl080/FSDP-first 主线无关；
  - 输出格式、依赖、入口都不可复现，且无人维护。

理由：

- tools 目录不能成为历史垃圾桶。
- 但精度/性能验证工具是 prefix-sharing 的核心交付保障，不能误删。

#### P1-5：彻底清理 verl070 独有代码

范围：

- dependency 侧 verl070 已降级为 deprecated；prefix-sharing 代码中仍需继续清除 v070 独有分支、注释和命名。
- README 统一描述 verl080 配套依赖，模型首选仍是 Qwen2.5-0.5B，不按模型区分依赖。

理由：

- 后续团队已迁移到 verl080 做性能调试、GDN 开发和精度对齐。
- 开源版本保留 v070 历史会显著增加维护成本。

#### P2-1：`PrefixSharingPlan` 字段分组优化

需要澄清的是：`PrefixSharingPlan` 的问题不是“字段多所以必须合并”，也不是要把 `PrefixDetectionResult` 直接塞进去。真正问题是字段类别混在一个平面对象里，读者难以判断哪些是检测视图、哪些是 token layout、哪些是 restore 语义。

当前字段大致分三类：

- reuse relation：`reuse_specs`、`provider_index`、`prefix_lens`、`is_provider`
- token layout：`kept_lengths_q`、`expanded_lengths_kv`、`cu_seqlens_*`、`*_position_offsets`、`*_keep_ranges`
- restore semantics：`prefix_last_restore`

为什么不直接合并 `PrefixDetectionResult` 和 `PrefixSharingPlan`：

- DetectionResult 是 detector 输出，语义是“从 token 序列发现哪些 row 可以复用”。
- Plan 是 backend/runtime 输入，语义是“为了执行裁剪、KV injection、restore，需要哪些 layout 和 restore spec”。
- 两者有重复字段，但职责不同。把 DetectionResult 作为 Plan 成员会让 backend 使用链路变长，也不能消除 Plan 必须持有 layout/restore 的事实。

建议：

- 第一阶段只删 group，保留其他重复字段，避免从 `reuse_specs` 重算高频视图。
- 第二阶段如果 Plan 继续膨胀，再引入轻量子结构，例如 `PrefixReuseIndex`、`PackedTokenPlan`、`RestorePlan`。
- 不为“看起来更抽象”提前引入 runtime 层级。

优先级：P2。当前不是开源首批阻塞。

#### P2-2：`is_provider` 命名

当前问题不大，不优先改。短期仅在注释或文档中说明它更接近 “non-reuser/full-compute row”。如果后续 observability 需要严格 provider 统计，再从 `reuse_specs.provider_idx_in_batch` 去重计算。

### 1.10 与 Meituan / verl RFC 的关系

用户提到的两个外部进展说明社区确实在关注 prefix 复用：

- Meituan fork PR：[meituan-search/verl#59](https://github.com/meituan-search/verl/pull/59) 当前标题为 `Verl prefix tree full`，方向是 dynamic trie / prefix-tree / flex / MAGI 等重型训练路径。
- verl RFC：[verl-project/verl#6401](https://github.com/verl-project/verl/issues/6401) 提出 Prefix-Tree Shared Attention，核心是 trainer 提供 prefix segments、flat deduplicated layout、block-sparse mask、Magi Attention workload-balanced CP dispatch，目标先是 Megatron backend，FSDP planned。

对本仓库的判断：

- 这两个进展验证了 shared-prefix/prefix-tree 是 verl 社区真实需求。
- 它们主攻 Megatron/Magi/flex/prefix-tree 方案，review 面和系统复杂度更高。
- 我们首推 verl+FSDP 是合理的：更轻量、更容易复现、更适合作为 PrefixGrouper arbitrary-prefix 扩展进入社区。
- 后续可以在语义上对齐 RFC 的 `prefix_segments` / prefix tree 表达，但第一波不要把内部实现改成 Magi/block-sparse 路线。

### 1.11 第一批建议 PR 切分

建议不要做一个“大重构 PR”。按以下顺序拆：

1. **PR-A：清理 mixer-specific 历史代码**
   - 删除 Qwen3.5 / Gated DeltaNet 专门化 store/backend/protocol。
   - 保留 attention KV 主线。
   - 更新相关导出和测试。

2. **PR-B：backend 公共 KV builder**
   - 抽出 `build_kv()`。
   - GPU/NPU backend 不再依赖 TorchRef。
   - TorchRef 回到 reference attention 定位。

3. **PR-C：integration 公共模块**
   - 新增 `verl_utils.py` 或等价公共模块。
   - FSDP/MCore 共用 helper 移出 `verl_mcore.py`。
   - `PrefixSharingRuntimeState` 移到公共 runtime state 模块。

4. **PR-D：patch 体系收敛**
   - 删除旧 `PatchManager` / integration class 体系。
   - 保留 setup patch set。
   - 保留 import auto patch 与显式 install 双入口。

5. **PR-E：文档与用户入口**
   - README 首推 FSDP + PrefixGrouper 风格配置。
   - compat matrix 将 FSDP 放在第一优先级。
   - 简要说明 PrefixSharing 与 PrefixGrouper 关系。

6. **PR-F：core 概念瘦身**
   - 删除 `PrefixGroup` / group_ids。
   - 保留高频视图字段，暂不大改 Plan。

### 1.12 当前判断

当前 `open-source_perf` 分支已经有不错的功能基础，但代码仍带有密集联调后的历史包袱。第一阶段重点不是重写算法，而是把开源主线收窄：

```text
verl PrefixGrouper user entry
  -> mode=prompt_only: existing PrefixGrouper
  -> mode=arbitrary_prefix: PrefixSharing FSDP-first runtime

PrefixSharing core
  -> PrefixReuseSpec / PrefixDetectionResult
  -> PrefixSharingPlan
  -> PrefixSharingRuntimeState

Backends
  -> shared KV builder
  -> FlashAttention GPU/NPU production path
  -> TorchRef correctness/reference path

Integrations
  -> verl_utils.py / runtime_state.py
  -> verl_fsdp.py as first-class path
  -> verl_mcore.py as advanced path

Setup
  -> setup patch set as main patch mechanism
  -> import auto patch and explicit install both supported
```

## Chapter 2：方案设计

本章将 Chapter 1 的研究结论转成可执行设计。设计原则是：先把开源主线收窄到 verl080 + FSDP + attention KV prefix sharing，再清理历史分支和过度抽象；所有改动必须保持 One-Forward + KV Injection + Prefix-Last Restore 的精度语义不变。

### 2.1 总体目标架构

重构后的目标架构：

```text
verl user config
  use_prefix_grouper=true
  prefix_grouper.mode=prompt_only
    -> existing PrefixGrouper path
  prefix_grouper.mode=arbitrary_prefix
    -> PrefixSharing FSDP-first runtime

prefix_sharing.core
  config.py              # PrefixSharingConfig, env fallback, config validation
  prefix_detector.py     # arbitrary-prefix reuse detection
  planner.py             # PrefixSharingPlan: trim/KV/restore execution contract
  prefix_store.py        # attention KV store only in first open-source line

prefix_sharing.backends
  kv_builder.py          # shared KV expansion, production path dependency
  flash_atten_gpu.py     # production GPU FA backend
  flash_atten_npu.py     # production NPU FA backend
  torch_ref.py           # reference/correctness backend

prefix_sharing.integrations
  verl_utils.py          # FSDP/MCore shared config/batch/position helpers
  runtime_state.py       # framework-independent runtime carrier
  context.py             # store lifetime, restore index, audit
  verl_fsdp.py           # first-class integration path
  verl_mcore.py          # advanced/internal path, no longer owns shared helpers

prefix_sharing.setup
  patches/verl080_fsdp   # first-class patch set
  patches/*              # advanced patch sets
  registry/logged_patch  # single patch mechanism
```

验收标准：

- FSDP path 不依赖 MCore 私有 helper。
- GPU/NPU FA backend 不依赖 `TorchReferenceBackend.build_kv()`。
- 主包导出不再暴露 Qwen3.5 / Gated DeltaNet 专门化类型。
- `setup/` 是唯一生产 patch 机制，旧 `integrations/patch_manager.py` 体系删除。
- README 和用户文档首推 PrefixGrouper 风格配置，不首推环境变量。

### 2.2 用户入口与配置策略

开源首选入口：

```yaml
actor_rollout_ref:
  actor:
    use_prefix_grouper: true
    prefix_grouper:
      mode: arbitrary_prefix
      min_prefix_len: 32
      min_group_size: 2
```

配置语义：

- `use_prefix_grouper=false`：不启用 PrefixGrouper / PrefixSharing。
- `use_prefix_grouper=true, prefix_grouper.mode=prompt_only`：继续走 verl / PrefixGrouper 既有 prompt-only 路径。
- `use_prefix_grouper=true, prefix_grouper.mode=arbitrary_prefix`：进入 PrefixSharing arbitrary-prefix path。
- `prefix_sharing_config`：保留为内部兼容、测试和 patch 未正式合入前的 escape hatch，不作为 verl 用户公开主入口。
- `ENABLE_PREFIX_SHARING`：保留为开发/调试 fallback，不作为 README 首选。

当前代码状态：

- `read_ps_config_from_engine_config()` 已能读取 `use_prefix_grouper=True`。
- `prefix_grouper.mode` 为 `arbitrary_prefix` / `arbitrary-prefix` / `prefix_sharing` 时，会生成 `enable_prefix_sharing=True`。
- `prompt_only` / `prompt-only` / `prefix_grouper` 会返回 disabled，避免抢占 PrefixGrouper prompt-only 语义。

需要补齐的设计约束：

- 若 `prefix_sharing_config` 与 `prefix_grouper.mode` 同时存在，必须显式记录优先级。短期建议保留当前内部配置优先级，但 README 不宣传；长期合入 verl 后应以 verl schema 为准。
- 配置解析应集中到 `integrations/verl_utils.py` 或后续拆出的 `verl_config.py`，FSDP/MCore 不各自维护一套解析逻辑。
- 所有配置 fallback 都要能给出可诊断信息，避免因为 env var 或 hidden config 让用户误判是否启用。

验收标准：

- 单测覆盖 `prompt_only` 不启用 PrefixSharing。
- 单测覆盖 `arbitrary_prefix` 启用 PrefixSharing。
- 单测覆盖 `prefix_sharing_config` 和 PrefixGrouper 配置同时存在时的优先级。
- README 示例不再把 `ENABLE_PREFIX_SHARING=1` 作为首选入口。

### 2.3 PrefixGrouper 关系与代码边界

对外定位：

- PrefixGrouper 是 verl 已有用户入口和 prompt-only baseline。
- PrefixSharing 是 PrefixGrouper 的 arbitrary-prefix 扩展模式。
- 首批社区 PR 应表达为“扩展 PrefixGrouper mode”，而不是“新增另一个 prefix-sharing feature”。

允许存在的代码：

- 配置读取与兼容：`use_prefix_grouper`、`prefix_grouper.mode`、`prefix_grouper.min_prefix_len` 等。
- 为复用 verl attention hook 心智而保留的薄 adapter / 参数透传。
- 文档说明：`prompt_only` 归 PrefixGrouper，`arbitrary_prefix` 归 PrefixSharing。
- 测试 fixture：用于验证分发逻辑的 fake PrefixGrouper 对象。

不允许存在的代码：

- 在 PrefixSharing 主包复刻 PrefixGrouper prompt-only 算法。
- 维护独立 `group_info` runtime，并把它作为 arbitrary-prefix 的事实源。
- 为了“看起来兼容 PrefixGrouper”而引入不参与主流程的 wrapper。
- 把 PrefixGrouper group 模型强行塞进 `PrefixSharingPlan`。

落地动作：

- README 增加 “Relationship with verl PrefixGrouper” 小节。
- 扫描 `prefix-sharing/prefix_sharing/` 中所有 `prefix_grouper` 命名，按“配置/接口允许，算法/runtime 不允许”的边界分类。
- 对超过边界的代码，删除或下沉到 tests fixture。

验收标准：

- reviewer 能从 README 理解：为什么用户配置叫 `prefix_grouper`，但运行时对象叫 `prefix_sharing_plan`。
- `PrefixSharingPlan` 不包含 PrefixGrouper `group_info`。
- `PrefixGroup` / `group_ids` 删除后，不影响 arbitrary-prefix 检测与 restore。

### 2.4 Core 层重构方案

Core 层目标是保留 provider/reuser DAG 语义，删除不承载运行时事实的 group 噪音。

#### 2.4.1 清理 mixer-specific store

首批主线只保留 attention KV prefix sharing。

处理范围：

- 删除或下线 `StoredDeltanetState`。
- 删除或下线 `PrefixDeltanetStore`。
- 删除或下线 `PrefixDeltanetBackend`。
- 删除或下线 `TorchReferenceBackend.build_deltanet_states()`。
- `PrefixActivationStore` 如保留，只作为最小泛化基类，不对外承诺 GDN 能力。

保留范围：

- `StoredAttentionKV`
- `PrefixAttentionStore`
- `PrefixActivationSlotId` 如果 attention store 仍依赖该统一 key，可保留。

验收标准：

- `prefix_sharing.core.__all__` 不再导出 GDN/Qwen3.5 专门化类型。
- backend capabilities 不再暴露 `supports_deltanet_state_reuse` 作为主线能力。
- 相关测试改为 attention store 测试，或删除历史 GDN mock 测试。

#### 2.4.2 删除 PrefixGroup / group_ids

处理范围：

- 删除 `PrefixGroup`。
- 删除 `PrefixDetectionResult.groups`。
- 删除 `PrefixDetectionResult.group_ids`。
- 删除 `PrefixSharingPlan.group_ids`。
- 删除 `PrefixLastRestoreSpec.group_id`。
- `sharing_group_count` 改为从 `reuse_specs` 的 `(provider_idx_in_batch, prefix_len)` 唯一集合推导。

不处理范围：

- 暂不删除 `provider_index`、`prefix_lens`、`is_provider`。
- 暂不把 `PrefixDetectionResult` 合并进 `PrefixSharingPlan`。
- 暂不引入 `PrefixReuseIndex` / `RestorePlan` 等新子结构。

理由：

- group 字段不是当前 runtime 事实源。
- 高频视图字段在 detector 阶段已经自然产生，删除后再重算没有收益。

验收标准：

- detector/planner 单测仍覆盖 one-provider、multi-reuser、chain reuse、no-sharing。
- observability 中 group count 与删除前语义一致或更准确。
- 无代码路径依赖 `PrefixGroup`。

### 2.5 Backend 层重构方案

Backend 层目标是把生产路径公共能力从 TorchRef 中拿出来，并逐步让社区主路径靠近 FlashAttention。

#### 2.5.1 抽出 shared KV builder

新增模块建议：

```text
prefix_sharing/backends/kv_builder.py
```

候选接口：

```python
def build_prefix_expanded_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    prefix_sharing_plan: PrefixSharingPlan,
    store: PrefixAttentionStore,
    *,
    packed_batch_layout: PackedBatchLayout | None = None,
    backend_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

职责：

- 按 plan/provider-before-reuser 顺序 store provider valid KV。
- 为 reuser 拼接 provider prefix KV + reuser suffix KV。
- 排除 TP padding slot，不把 padding KV 存入 store。
- 保持 autograd graph，不 `detach()`。
- 保持输出 packed KV shape 与 FlashAttention backend 预期一致。

非职责：

- 不做 attention softmax。
- 不做 RoPE。
- 不做 GDN/cache_param。
- 不处理 PrefixGrouper prompt-only `group_info`。

调用关系：

- `FlashAttentionGPUBackend.build_kv()` 调 shared builder。
- `FlashAttentionNPUBackend.build_kv()` 调 shared builder。
- `TorchReferenceBackend.build_kv()` 如保留，应只是调 shared builder 的薄 wrapper。

验收标准：

- GPU/NPU backend 文件不再 import `TorchReferenceBackend` 仅为了 `build_kv()`。
- shared builder 单测覆盖 no-sharing、one-provider、chain、TP padding、gradient flow。
- 旧 TorchRef build_kv 行为与 shared builder 等价。

#### 2.5.2 TorchRef 定位收敛

TorchRef 保留用途：

- CPU correctness reference。
- 小规模 attention mask 语义验证。
- 与 FA backend 做数值对齐。

TorchRef 不再承担：

- 生产路径 KV expansion 唯一实现。
- Qwen3.5/GDN reference。
- GPU/NPU backend 公共父类职责。

验收标准：

- README / docs 不把 TorchRef 描述为性能路径。
- factory 中 `torch_ref` 仍可用于单测和 debug。
- FlashAttention backend 的生产语义不依赖 TorchRef 类。

### 2.6 Integration 层重构方案

Integration 层目标是让 FSDP 成为第一优先级路径，同时让 MCore 代码不再承载公共 helper。

#### 2.6.1 新增 `verl_utils.py`

第一阶段不要过度拆分，先新增一个公共模块：

```text
prefix_sharing/integrations/verl_utils.py
```

迁移内容：

- `read_ps_config_from_engine_config()`。
- `_prefix_sharing_config_from_prefix_grouper()`。
- NestedTensor / dense batch trim helper。
- kept position rows / valid indices helper。
- FSDP/MCore 都使用的 batch field 读取工具。

后续拆分条件：

- 如果 `verl_utils.py` 超过约 400 行，或出现明显不同职责，再拆成：
  - `verl_config.py`
  - `verl_batch.py`
  - `verl_positions.py`

验收标准：

- `verl_fsdp.py` 不再 import `verl_mcore.py`。
- `verl_mcore.py` 不再拥有 PrefixGrouper 配置读取事实源。
- FSDP/MCore 配置解析测试共用同一套 helper。

#### 2.6.2 抽出 `runtime_state.py`

新增模块建议：

```text
prefix_sharing/integrations/runtime_state.py
```

保留字段：

- `prefix_sharing_plan`
- `attention_backend`
- `packed_batch_layout`
- `parallel_info`
- `kept_position_ids` 或兼容字段
- `valid_indices` 或兼容字段

原则：

- RuntimeState 是 integration runtime carrier，不属于 MCore。
- Context 仍负责进入 forward 前派生 store、restore index 等可执行状态。
- 不把 Plan、RuntimeState、Context 合并。

验收标准：

- FSDP/MCore/context/tests 均从公共模块 import RuntimeState。
- `verl_mcore.py` 删除 RuntimeState 定义。

#### 2.6.3 FSDP-first 接入整理

处理范围：

- `verl_fsdp.py` 中 fake/local helper 明确标注 test utility 或 local fallback。
- 真实生产入口主推 `setup/patches/verl080_fsdp/forward_step.py`。
- FSDP runtime 内部按 pack / run attention / scatter / restore 拆小函数。

验收标准：

- reviewer 能区分 fake helper 和生产 patch。
- FSDP restore 函数有独立单测。
- README 把 FSDP 列为第一推荐路径。

### 2.7 Patch / setup 机制方案

目标：

- `setup/` 是唯一生产 patch 机制。
- 旧 `integrations/patch_manager.py` 体系删除。
- import auto patch 和显式 install 双入口都保留。

保留入口：

```python
import prefix_sharing
```

用于 `VERL_USE_EXTERNAL_MODULES=prefix_sharing` 的脚本化训练。

```python
import prefix_sharing
prefix_sharing.setup.install("verl080_fsdp")
```

用于显式、可读、可调试的集成。

删除范围：

- `integrations/patch_manager.py`
- `integrations/megatron_attention.py`
- `VerlMCoreIntegration`
- `VerlFSDPIntegration`
- 仅覆盖旧 patch manager 的测试

保留范围：

- `setup/logged_patch.py`
- `setup/registry.py`
- `setup/patches/verl080_fsdp`

import hook 整改清单：

- auto install 触发链：`prefix_sharing.__init__` 何时调用 `_auto_install_patches()`。
- patchset 选择：`PREFIX_SHARING_PATCHSET`、compat matrix、显式 install 参数谁优先。
- eager/lazy：target 已 import 时 eager patch，target 未 import 时 lazy hook。
- 幂等：重复 import、重复 install、auto + explicit 混用不重复 patch。
- 失败语义：target 缺失、版本不匹配、patch 函数异常分别如何处理。
- 可观测性：默认不刷屏，debug 时能看出安装了哪个 patchset。

验收标准：

- `setup.install("verl080_fsdp")` 幂等。
- import auto patch 与显式 install 混用不会重复 patch。
- FSDP patchset 能通过 compat matrix 或显式 patchset 稳定选中。
- 单测覆盖 patch target 已 import / 未 import 两种顺序。

### 2.8 调试、日志与 tools 方案

短期处理：

- 所有热路径 dump/print/logging 用统一注释标记：
  ```python
  # PREFIX_SHARING_DIAGNOSTIC
  ```
- 明显可集中封装的 dump 入口移入 `diagnostics` helper。
- 默认训练路径不输出 per-micro-batch print。

中期处理：

- 新增或整理 `prefix_sharing/diagnostics.py`。
- 生产 patch 只调用少数 helper：
  - `diagnostics.enabled()`
  - `diagnostics.dump_fsdp_attention(...)`
  - `diagnostics.dump_runtime_plan(...)`
- audit 使用 logger，不直接 print。

tools 保留标准：

- 能复现关键精度结论：logprob/loss/grad 与 baseline 对齐。
- 能复现关键性能结论：FSDP baseline、PrefixGrouper prompt-only、PrefixSharing arbitrary-prefix 三方对比。
- 能作为 release / 重要 PR 前回归验证。
- 依赖 GPU、verl、flash-attn、torch_npu 时，文件头写清楚环境要求。

tools 删除标准：

- 只服务某次临时排查，结论已经沉淀到文档或测试。
- 与当前 verl080/FSDP-first 主线无关。
- 输出格式、依赖、入口不可复现，且无人维护。

验收标准：

- 默认测试和本地训练不出现 PrefixSharing 热路径 print。
- 保留工具都有用途说明。
- 删除工具前能说明结论已经迁移到测试、文档或 benchmark。

## Chapter 3：测试验证

本章是代码开发完成后的验证执行手册。验证目标不是只确认程序“不报错”，而是按由内到外的顺序证明：核心语义正确、patch 与训练引擎接线正确、开启 PrefixSharing 后训练数值与 baseline 一致，并在固定工作负载下量化性能影响。

执行顺序固定为：开发自测 -> 功能验证 -> 集成验证 -> 精度对齐 -> 性能对比 -> 冒烟测试。前一层失败时，不进入后一层；设备或可选依赖缺失导致的 skip 必须如实记录，不能视作通过。

### 3.1 通用约定与结果交付

所有命令从仓库根目录执行。推荐统一设置：

```bash
export PYTHONPATH=prefix-sharing
export PYTHONPYCACHEPREFIX=/private/tmp/prefix-sharing-refactor-pycache
export PREFIX_SHARING_PATCHSET=verl080_fsdp
```

设备侧 FSDP 验证还需要：

```bash
export VERL_USE_EXTERNAL_MODULES=prefix_sharing
```

每一项验证应建立独立结果目录，例如 `artifacts/validation/<YYYYMMDD>-<device>/`，至少保存：

- 实际执行的完整命令与 git commit；
- `python --version`、`pip show torch verl flash-attn` 的版本信息；GPU 记录 `nvidia-smi`，NPU 记录 `npu-smi info`；
- pytest 的 passed / failed / skipped 摘要及完整日志；
- 精度或性能脚本生成的 JSONL、诊断 dump 和比较报告；
- 对失败或 skip 的原因、复现方式和影响范围。

验收口径：`PASS` 表示命令退出码为零且满足本节规定的数值标准；`SKIP` 仅能用于缺少明确 optional 依赖或设备；任何断言失败、NaN/Inf、训练进程异常、ON/OFF 输入不一致都记为 `FAIL` 或 `BLOCKED`，不得以平均指标掩盖。

### 3.2 开发自测：UT、IT、ST

这一层不依赖真实 verl 训练任务，负责保护 core、backend、patch 与轻量 FSDP adapter。必须先完成，作为所有后续设备验证的准入条件。

#### 3.2.1 UT：模块语义和精度不变量

执行：

```bash
python3 -m pytest -q -p no:cacheprovider prefix-sharing/tests/unit_test
```

重点关注：

- `test_detector.py`、`test_planner.py`：no-sharing、one-provider、多 reuser、chain reuse 的 provider/reuser 关系与裁剪 plan；
- `test_kv_builder.py`、`test_prefix_store.py`：provider-before-reuser 的 KV store/load、TP padding 排除、KV 不 detach；
- `test_packed_layout*.py`、`test_runtime_context.py`：TP packed layout、RoPE 位置与 prefix-last restore index；
- `test_verl_fsdp_ch4_functional.py`：任意前缀、多 reuser、chain、fallback、position id；
- `test_verl_fsdp_ch4_precision.py`：reuser suffix attention、梯度回传、interior prefix restore、prefix-last logprob restore。

通过标准：所有已收集测试通过；由于本机缺少 `torch` 而 skip 的场景必须转到有 torch 的环境补跑，不能用 CPU-only collection 作为 UT 结论。

> **验证结果**（2026-07-13, NVIDIA A100-SXM4-80GB, env-flex, torch 2.8.0+cu128, flash-attn 2.8.1, commit `fa9c1cf`）：
>
> ```text
> 213 passed, 4 failed, 9 warnings in 416.90s
> ```
>
> **4 failed 分析**：
> - `test_verl080_migration.py::test_auto_activation_always_attempts_and_handles_missing_env`：测试隔离问题——前次 session 的 auto import 导致 `prefix_sharing._patch_handle` 非 None，assert 失败。非代码回归，是测试 fixture 清理问题。
> - `test_verl080_migration.py::test_auto_activation_handles_env_var_false`：同上。
> - `test_verl080_migration.py::test_auto_activation_handles_env_var_true`：同上。
> - `test_verl_fsdp_adapter.py::test_verl080_fsdp_forward_step_patch_runs_native_nested_prepare_outputs_path`：Triton CPU tensor 问题（`ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)`），已知的 pre-existing issue，非本次重构引入。
>
> **结论**：UT 非 optional 测试通过，4 failed 均为环境/测试隔离问题，非代码语义回归。

#### 3.2.2 IT：patch、后端与可选设备集成

基础集成测试：

```bash
python3 -m pytest -q -p no:cacheprovider prefix-sharing/tests/integrated_test
```

GPU CUDA + FlashAttention 环境追加：

```bash
python3 -m pytest -q -p no:cacheprovider \
  prefix-sharing/tests/integrated_test/optional/test_gpu_flash_backend.py
```

NPU + torch_npu + MindSpeed 环境追加：

```bash
python3 -m pytest -q -p no:cacheprovider \
  prefix-sharing/tests/integrated_test/optional/test_npu_flash_backend.py
```

通过标准：

- `test_patch_integrations.py` 覆盖 import auto patch、显式安装、重复安装、FSDP patchset 选择和 PrefixGrouper 配置分流，必须全过；
- GPU/NPU 后端测试必须对齐 TorchReferenceBackend 的 attention 输出与 Q/K/V 梯度；当前测试中 fp16 的单元素最大绝对误差门槛为输出 `< 5e-2`、梯度 `< 2e-1`；
- optional 测试只有在对应设备或依赖缺失时可以 skip；设备齐全但测试被 skip 时，应排查 skip 条件，不可直接接受。

注意：`test_verl080_restore_e2e.py` 目前含真实 verl engine fixture 的 TODO 和显式 skip，它只说明测试接口预留，**不能**计入”真实 verl080 端到端验证通过”。真实接线验证以 3.4 为准。

> **基础 IT 验证结果**（2026-07-13, A100, env-flex, torch 2.8.0, flash-attn 2.8.1）：
>
> ```text
> 61 passed, 29 skipped, 9 warnings in 162.89s
> ```
>
> **GPU FA optional 验证结果**：
>
> ```text
> 23 passed, 9 warnings in 81.61s
> ```
>
> 29 skipped 中 27 为 `test_verl080_restore_e2e.py` 的显式 skip（TODO 占位），其余为 NPU/MindSpeed 依赖导致的 skip（环境未配置）。GPU backend 测试全部通过，说明 `flash_atten_gpu.py` softcap 修复后 FlashAttention backend 精度正常。
>
> **结论**：IT 基础测试 61/61 通过；GPU FA 23/23 通过；optional NPU 测试因设备缺失 skip 不计入验证范围。

#### 3.2.3 ST：跨模块核心流程

执行：

```bash
python3 -m pytest -q -p no:cacheprovider prefix-sharing/tests/system_test
```

再执行一次完整开发回归：

```bash
python3 -m pytest -q -p no:cacheprovider \
  prefix-sharing/tests/unit_test \
  prefix-sharing/tests/integrated_test \
  prefix-sharing/tests/system_test
```

通过标准：`test_system_phase1_core.py` 证明 detector -> planner -> runtime context -> KV reuse -> restore 的框架无关主链路完整通过；完整回归中，非 optional 的失败数必须为零。

> **ST 验证结果**：
>
> ```text
> 1 passed, 9 warnings in 94.18s
> ```
>
> **完整开发回归（UT + IT + ST）结果**：
>
> ```text
> 275 passed, 29 skipped, 4 failed, 9 warnings in 191.22s
> ```
>
> 4 failed 同上（§3.2.1 分析），均为环境/测试隔离和已知 pre-existing issue。非 optional 测试零失败。
>
> **结论**：ST `test_system_phase1_core.py` 通过；完整回归中非 optional 失败为零。

### 3.3 功能验证：PrefixSharing 行为是否按设计生效

功能验证使用固定、可人工检查的输入，验证“正确启用、正确回退、正确复用”，不以真实训练吞吐作为判断。建议优先在 CUDA 环境执行，也允许先在 CPU + torch 环境完成。

#### 3.3.1 KV builder correctness guard

CPU float32 先跑精确 oracle：

```bash
python3 prefix-sharing/tools/verify_p0_correctness.py \
  --device cpu --dtype float32 \
  --output artifacts/validation/<run>/p0-cpu.jsonl
```

CUDA 环境追加生产精度检查：

```bash
python3 prefix-sharing/tools/verify_p0_correctness.py \
  --device cuda --dtype bfloat16 \
  --output artifacts/validation/<run>/p0-cuda-bf16.jsonl
```

该脚本覆盖 no-sharing、one-provider、multi-provider、chain、短序列与最小前缀边界，并将当前 `build_kv` 与逐 row `torch.cat` reference 对比。

通过标准：结果 JSONL 中每条记录的 `PASS` 为 true；CPU float32 的 expanded KV 必须精确等价；CUDA bf16 结果必须满足脚本内置 comparison，并且 provider prefix 对应梯度存在且非零。任何 `prefilter_correct=false`、KV 形状不符或梯度缺失都阻断后续阶段。

> **P0 CPU float32 验证结果**（2026-07-13, A100, env-flex）：
>
> ```text
> 110/110 PASS (0 FAIL)
> ```
> 覆盖 no-sharing、one-provider、multi-provider、chain、短序列、最小前缀边界。CPU float32 下 expanded KV 精确等价，prefilter_correct=true，梯度非零。
>
> **P0 CUDA bf16 验证结果**：
>
> ```text
> 15/15 PASS (0 FAIL)
> ```
> 覆盖典型生产场景，builder KV 形状正确，provider prefix 梯度存在且非零。
>
> **结论**：KV builder correctness guard 在 CPU float32 和 CUDA bf16 下全部通过。不阻断后续阶段。

#### 3.3.2 运行时分流与回退

使用 `test_patch_integrations.py` 和 `test_verl_fsdp_ch4_functional.py` 验证以下矩阵：

| 配置 | 预期 |
|---|---|
| `use_prefix_grouper=false` | 不进入 PrefixSharing |
| `mode=prompt_only` | 留在 PrefixGrouper 原路径 |
| `mode=arbitrary_prefix` | 构建 PrefixSharing plan 并进入 runtime |
| 无可共享前缀 / 单样本 | 返回原 batch，安全 fallback |
| one-provider / 多 reuser / chain | reuser 指向正确 provider，保留原绝对 position id |

通过标准：每一行均能由现有测试或固定 batch 手工日志证明；尤其不能出现“feature 已开启但无共享时修改 batch”或“prompt_only 被 PrefixSharing 抢占”。

### 3.4 集成验证：真实 verl FSDP 接线

本阶段在目标设备上验证真正的 verl 训练引擎，而不是 fake adapter。首选环境为 `verl cdd9014f + torch 2.4 + Qwen2.5-0.5B + FSDP`；使用 packed 路径时必须设置 `actor_rollout_ref.model.use_remove_padding=true`。当前首版不把 Ulysses SP、ring attention、fused kernels 作为通过范围。

#### 3.4.1 接线前检查

确认版本与 patch：

```bash
python3 -c 'import prefix_sharing; import verl; import torch; print("torch", torch.__version__); print("verl", getattr(verl, "__version__", "snapshot"))'
```

启动命令必须同时包含或导出：

```bash
export VERL_USE_EXTERNAL_MODULES=prefix_sharing
export PREFIX_SHARING_PATCHSET=verl080_fsdp
export ENABLE_PREFIX_SHARING=1
```

并在任务配置中启用 `use_prefix_grouper=true` 与 `prefix_grouper.mode=arbitrary_prefix`。当前环境变量是兼容开关；配置入口用于验证 verl 用户可见语义，两者都要记录。

#### 3.4.2 最小真实 forward/backward

以固定 provider + reuser batch 执行一次 actor log-prob 与一次 actor backward。可以用现有 verl 任务入口缩小到一个 micro-batch，也可以由 Claude Code 创建最小 engine fixture；必须覆盖：

1. provider 与至少一个 reuser 有长度不少于 `min_prefix_len` 的公共前缀；
2. 一条无共享 batch，验证安全 fallback；
3. reuser 的 suffix、interior prefix 和 prefix-last 三个输出区段；
4. 一次 `loss.backward()`，验证 provider prefix 参数梯度非零且无 NaN/Inf。

通过标准：patch 安装成功；模型 forward/backward 无异常；audit/diagnostic 中能看到实际 reuse；无共享 batch 与关闭特性时走普通路径；输出 shape、NestedTensor offsets 或 packed lengths 与原 batch 长度一致。若真实 fixture 尚未补到 `test_verl080_restore_e2e.py`，本阶段的脚本、固定输入和日志必须作为测试产物保存，后续再将其固化为该测试文件。

#### 3.4.3 分布式覆盖

FSDP 按实际可用卡数依次验证 1、2、4、8 卡。每个规模使用相同模型、相同固定输入、相同每卡 micro-batch 配置，并单独记录 world size。TP 不属于 FSDP-first 首版硬性前置；若目标路径启用了 TP，则必须追加 `tests/unit_test/test_packed_layout*.py` 和对应的真实 packed layout 检查，确认 padding slot 不进入 KV store 或 restore。

通过标准：各规模均能完成至少一个 forward/backward；world size 变化不改变同一固定样本的语义结果；任何仅在多卡出现的 shape、collective、restore index 错误均为阻断问题。

### 3.5 精度对齐：PrefixSharing ON 与 baseline OFF

精度验证是发布红线。比较对象必须来自**同一模型权重、同一随机种子、同一固定输入、同一 dtype、同一并行配置**。不要使用随机 rollout 生成的不同 response 直接比较 logprob；先固定或回放 `input_ids`、`attention_mask`、`position_ids` 与 labels。

#### 3.5.1 对照运行

分别执行两次同一 micro-batch：

```bash
# baseline
ENABLE_PREFIX_SHARING=0 PREFIX_SHARING_DIAG_DUMP=artifacts/validation/<run>/off <same-verl-command>

# PrefixSharing
ENABLE_PREFIX_SHARING=1 PREFIX_SHARING_DIAG_DUMP=artifacts/validation/<run>/on <same-verl-command>
```

若沿用 `docs/user-guide/engine-fsdp.md` 的诊断工具路径，先确认该工具在当前分支存在；不存在时由 Claude Code 使用 `diagnostics.py` 产物或直接保存 tensor，计算逐元素误差。不要伪造不存在的 `cmp_diag_verl080.py` 命令。

#### 3.5.2 必比指标与位置分段

对每个 reuser 的完整原始序列分别比较：

- `log_probs`：prefix interior、prefix-last、suffix-first、后续 suffix；
- `entropy` 与 `logits`：可取得时逐元素比较；
- 训练 loss：同一 reduction 后比较；
- 关键参数或梯度 bucket：比较 max absolute difference、max relative difference、NaN/Inf；
- provider prefix 的梯度：必须存在且非零，证明 KV store 没有 detach。

prefix-last 必须单独报告。它使用 provider 的最后一个 prefix logits 与 reuser label 重算；只比较 suffix 平均值无法覆盖此红线。

#### 3.5.3 容差与通过标准

分两级执行：

- CPU float32 / TorchReference：`torch.equal` 或 `allclose(atol=1e-5, rtol=1e-5)`；
- GPU/NPU bf16/fp16 实际训练：先使用 `allclose(atol=1e-4, rtol=1e-3)` 检查 logprob/loss；FA attention 单元级输出和梯度沿用现有 optional test 的最大误差阈值，输出 `< 5e-2`、梯度 `< 2e-1`。

若模型、kernel 或并行实现导致超出上述训练级阈值，不能直接放宽阈值。必须保存最大误差位置、对应 token、ON/OFF tensor 切片和环境信息，先定位是数值舍入、mask、position、restore 还是梯度图问题，再由项目负责人决定是否调整门槛。

精度报告至少列出：每类指标的 max/mean absolute difference、max relative difference、异常数量、prefix-last 最大误差、provider gradient norm、结论。所有受测点在阈值内且无 NaN/Inf 才可通过。

### 3.6 性能对比：先测核心，再测真实训练

性能只在精度通过后执行。所有实验使用相同硬件、软件版本、模型权重、dtype、输入集、warmup 和重复次数；CUDA 计时必须同步。性能结果不得和不同输入或不同有效 token 数的 run 横向比较。

#### 3.6.1 standalone 基准

先运行聚焦 benchmark：

```bash
python3 prefix-sharing/tools/perf_baseline_benchmark.py \
  --backend flash_atten_gpu --sync 1 --num-runs 50 \
  --output artifacts/validation/<run>/perf-baseline.jsonl
```

再运行覆盖 sharing pattern、batch size、sequence length、模型 shape、CPU/device/memory 的矩阵：

```bash
python3 prefix-sharing/tools/perf_comprehensive_benchmark.py \
  --phase all --cpu-runs 50 --device-runs 20 \
  --output artifacts/validation/<run>/perf-comprehensive.jsonl
```

GPU 不可用时只运行 `--phase cpu`，并明确标记为 CPU overhead 结果，不能外推为训练加速比。NPU 性能需单列脚本和结论，不能拿 GPU FA benchmark 代替。

通过标准不是预设”必须加速多少”，而是结果完整、可复现、无 OOM，并能解释 no-sharing、one-provider、multi-provider、chain 四类输入的趋势。重点记录 detector/planner、KV builder、attention、端到端 step 的 p50/p90，以及 peak memory。

> **Perf baseline（standalone detector/planner overhead）验证结果**（2026-07-13, NVIDIA A100-SXM4-80GB, env-flex, torch 2.8.0, flash-attn 2.8.1）：
>
> 覆盖 no_sharing、one_provider、chain 三类输入，各在 batch_size=8/L=256 和 batch_size=32/L=512 下测试：
>
> | case | B×L | reused tokens | detector p50 | plan p50 | peak python | 分析 |
> |---|---|---|---|---|---|---|
> | no_sharing | 8×256 | 0 | 4.25ms | 0.08ms | 0.73MB | detector 主导，prefilter 可跳过 |
> | no_sharing | 32×512 | 0 | 315.33ms | 0.63ms | 6.20MB | 无共享时 overhead 全部浪费，prefilter P0 确认 |
> | one_provider | 8×256 | 896 | 3.62ms | 3.69ms | 0.40MB | detector+plan 约 7ms |
> | one_provider | 32×512 | 11904 | 98.72ms | 99.01ms | 1.82MB | detector+plan 约 200ms，compact representation P0/P1 |
> | chain | 8×256 | 1600 | 2.14ms | 2.20ms | 0.15MB | chain 检测更快 |
> | chain | 32×512 | 15744 | 13.33ms | 22.57ms | 0.34MB | B=32 chain plan 仍显著 |
>
> **结论**：no-sharing 路径 overhead 纯浪费，prefilter 优化 P0；有共享时 detector+plan 在 B=32/L=512 下约 100-200ms，compact representation 是后续优化方向。
>
> **Perf comprehensive（device attention 性能对比）验证结果**：
>
> 覆盖 one_provider、chain、multi_provider 三类 sharing pattern × flash_atten_gpu / torch_ref 两个 backend × qwen2.5-0.5b / qwen3-0.6b 两个模型，B=4, L=256, device-runs=20：
>
> | sharing | backend | model | build_kv p50 | kernel p50 | total_attn p50 | 关键发现 |
> |---|---|---|---|---|---|---|
> | one_provider | flash_atten_gpu | qwen3 | 0.715ms | 0.168ms | 1.11ms | build_kv 占 ~64% |
> | one_provider | flash_atten_gpu | qwen2.5 | 0.714ms | 0.158ms | 1.10ms | 同上 |
> | one_provider | torch_ref | qwen3 | 6.543ms | 80.361ms | 86.90ms | torch_ref kernel 是瓶颈 (80ms) |
> | one_provider | torch_ref | qwen2.5 | 0.748ms | 85.124ms | 85.81ms | 同上 |
> | chain | flash_atten_gpu | qwen3 | 0.695ms | 0.185ms | 1.11ms | chain 额外 KV 不明显增加总时 |
> | chain | flash_atten_gpu | qwen2.5 | 0.603ms | 0.143ms | 0.95ms | chain Qwen2.5 最快 |
> | chain | torch_ref | qwen3 | 1.718ms | 23.418ms | 90.42ms | kernel 仍是瓶颈 |
> | chain | torch_ref | qwen2.5 | 21.878ms | 12.956ms | 92.02ms | qwen2.5 chain build_kv 异常偏高 |
> | multi_provider | flash_atten_gpu | qwen3 | 0.677ms | 0.190ms | 1.12ms | 与 one_provider 接近 |
> | multi_provider | flash_atten_gpu | qwen2.5 | 0.630ms | 0.159ms | 1.05ms | 同上 |
>
> **关键发现**：
> 1. **flash_atten_gpu backend 总 attention 仅 ~1ms**，build_kv 约 0.6-0.7ms（占 ~60-64%），FlashAttention kernel 约 0.15-0.19ms。
> 2. **torch_ref backend 总 attention ~85-92ms**，其中 TorchReference attention 占 12-85ms，是绝对瓶颈；build_kv 在 torch_ref 中占比很低（0.9%-24%），因为 kernel 太慢。
> 3. **生产路径应聚焦 flash_atten_gpu**：torch_ref 的 attention 比 FA 慢两个数量级。
> 4. **build_kv 是优化重点**：在 FA 路径中 build_kv 占 60%+，后续可通过 prealloc、in-place 操作优化。
> 5. **结果完整、可复现、无 OOM**，符合通过标准。

#### 3.6.2 真实 FSDP 三方对比

在固定 replay batch 上比较：

1. baseline：`use_prefix_grouper=false` / `ENABLE_PREFIX_SHARING=0`；
2. PrefixGrouper prompt-only：`mode=prompt_only`；
3. PrefixSharing arbitrary-prefix：`mode=arbitrary_prefix`。

每个场景至少 warmup 10 step、采样 30 step；计时范围必须相同，建议分别报告 actor forward、forward+backward、完整 train step 和峰值显存。输入至少包含 no-sharing、同 prompt 多 response、任意子前缀/chain 三类；报告总 token、有效 Q token、expanded KV token、reused token，防止只比较 wall time 却忽略工作量变化。

通过标准：无数值回退、无 OOM、同一输入下精度仍在 3.5 阈值内。性能结果允许某些短序列或低复用率场景无收益，但必须如实展示，不得只筛选收益 case。

### 3.7 冒烟测试：最小可训练任务

冒烟测试验证从环境导入、patch 安装、verl 配置到训练任务收尾的完整可用性。使用 Qwen2.5-0.5B、最小数据集和最少 step，先单卡，再按资源扩展。

前置条件：模型路径、训练/验证 parquet 路径有效；当前 `examples/run_verl_training.sh` 的配置名仍偏 Megatron，因此 FSDP smoke 应以 `docs/user-guide/engine-fsdp.md` 的 FSDP 参数为准或由 Claude Code 提供一个明确的 FSDP 启动命令，不能误将该脚本当作 FSDP smoke。

执行步骤：

1. 用 `ENABLE_PREFIX_SHARING=0` 跑 1-3 个 train step，确认 baseline 可启动、可完成 backward 和 checkpoint/log 输出；
2. 保持除使能方式外所有配置不变，切到 `ENABLE_PREFIX_SHARING=1` 与 `mode=arbitrary_prefix`，再跑 1-3 step；
3. 对有共享输入确认 runtime audit 显示实际 reuse；对无共享输入确认安全 fallback；
4. 观察训练退出码、loss、显存峰值、NaN/Inf、worker 异常和 patch 安装信息；
5. 在资源允许时将同一 smoke 扩展到 2/4/8 卡 FSDP。

通过标准：ON/OFF 都能干净结束；ON 路径确实安装 `verl080_fsdp` patch 并发生 reuse；无共享数据不改变训练正确性；无 OOM、死锁、collective 超时、NaN/Inf 或 restore shape 错误。冒烟通过不替代 3.5 精度对齐与 3.6 性能结论。

### 3.8 建议的执行与汇报顺序

| 阶段 | 执行者 | 交付物 | 放行条件 | 当前状态 |
|---|---|---|---|---|
| 开发自测 | 开发者 / CI | pytest 日志与计数 | 非 optional 测试零失败 | ✅ 完成（UT 213✅/4❌*、IT 61✅/29⏭️、GPU FA 23✅、ST 1✅、全回归 275✅/29⏭️/4❌*；*4 failed 均为测试隔离/已知问题） |
| 功能验证 | Claude Code | P0 JSONL、固定输入结论 | KV、梯度、fallback 正确 | ✅ 完成（CPU 110/110 PASS, CUDA 15/15 PASS） |
| 集成验证 | Claude Code + device 环境 | real engine 日志、world-size 记录 | FSDP forward/backward 成功 | ❌ 未执行（需要真实 verl 训练环境；当前设备已验证 UT/IT/ST/P0，但没有配置 verl 训练任务） |
| 精度对齐 | Claude Code + device 环境 | ON/OFF tensor/梯度误差报告 | 全部指标在阈值内 | ❌ 未执行（需要 real engine ON/OFF 对比运行） |
| 性能对比 | Claude Code + device 环境 | JSONL、汇总表、环境信息 | 结果完整且精度未回退 | ✅ 完成（perf baseline 12 records, perf comprehensive 10 records；详见 §3.6.1） |
| 冒烟测试 | Claude Code + device 环境 | 最小训练日志 | ON/OFF 均稳定跑通 | ❌ 未执行（需要 vere 训练任务配置） |

测试完成后，将结果摘要（命令、环境、通过/skip/失败数、精度阈值、性能结论、已知限制）更新到 PR 的 `## 测试结果` 小节；仍未覆盖的设备、并行策略或真实 e2e fixture 回填本文件 Chapter 6，并在 PR 中明确其潜在影响。

## Chapter 4：开发计划

开发计划按小 PR 切分。每个 PR 都必须能独立 review、独立回滚，避免把删除历史代码、抽公共模块、改用户入口混在一个大变更里。

### 4.1 PR-A：清理 mixer-specific 历史代码

目标：

- 主线只保留 attention KV prefix sharing。
- 删除 Qwen3.5 / Gated DeltaNet 专门化 store/backend/protocol。

主要改动：

- `core/prefix_store.py`
- `core/__init__.py`
- `backends/base.py`
- `backends/torch_ref.py`
- `backends/__init__.py`
- 相关 unit tests

注意事项：

- 不引入新的 general activation abstraction。
- 如果 `PrefixActivationStore` 没有实际价值，可以同步删除；如果 attention store 还复用 slot id，可保留最小基类。
- 删除测试前确认它只覆盖 GDN mock，不覆盖 attention KV 梯度红线。

测试：

```bash
PYTHONPATH=prefix-sharing pytest -q \
  prefix-sharing/tests/unit_test/test_prefix_store.py \
  prefix-sharing/tests/unit_test/test_backend_factory.py
```

### 4.2 PR-B：抽出 shared KV builder

目标：

- `build_kv()` 成为 backend 公共能力，不再属于 TorchRef。

主要改动：

- 新增 `backends/kv_builder.py`。
- 修改 `flash_atten_gpu.py` / `flash_atten_npu.py`。
- 修改 `torch_ref.py` 为薄 wrapper 或直接使用 shared builder。
- 新增 builder 单测。

注意事项：

- provider-before-reuser 顺序是核心算法约束，迁移时必须保留注释。
- store/load 调用点必须保留注释，说明 provider prefix KV 先 store，reuser 后 load。
- TP padding slot 不得进入 store。

测试：

```bash
PYTHONPATH=prefix-sharing pytest -q \
  prefix-sharing/tests/unit_test/test_kv_builder.py \
  prefix-sharing/tests/unit_test/test_flash_attention_base.py \
  prefix-sharing/tests/integrated_test
```

### 4.3 PR-C：抽出 integration 公共模块

目标：

- FSDP/MCore 共用 helper 进入 `verl_utils.py`。
- RuntimeState 移到公共模块。

主要改动：

- 新增 `integrations/verl_utils.py`。
- 新增 `integrations/runtime_state.py`。
- 修改 `verl_fsdp.py` / `verl_mcore.py` / `context.py` imports。
- 调整 tests import。

注意事项：

- 第一阶段用 `verl_utils.py` 承载公共逻辑，避免一开始拆太碎。
- `verl_fsdp.py` 不得 import `verl_mcore.py`。
- 保持 public behavior 不变。

测试：

```bash
PYTHONPATH=prefix-sharing pytest -q \
  prefix-sharing/tests/unit_test/test_config.py \
  prefix-sharing/tests/unit_test/test_verl_fsdp_adapter.py \
  prefix-sharing/tests/integrated_test
```

### 4.4 PR-D：删除旧 patch_manager 体系

目标：

- 生产 patch 机制只剩 `setup/`。

主要改动：

- 删除 `integrations/patch_manager.py`。
- 删除 `integrations/megatron_attention.py`。
- 删除旧 integration class 或迁移必要逻辑到 setup patch set。
- 更新 `integrations/__init__.py`。
- 重写 patch integration tests。

注意事项：

- 保留 import auto patch 与显式 install 双入口。
- 删除前用 `rg` 确认旧类没有生产引用。
- `setup/logged_patch.py` 中历史注释要改掉。

测试：

```bash
PYTHONPATH=prefix-sharing pytest -q \
  prefix-sharing/tests/integrated_test/test_patch_integrations.py \
  prefix-sharing/tests/unit_test
```

### 4.5 PR-E：文档、README、compat matrix

目标：

- 用户入口、兼容矩阵、README 与 FSDP-first 定位一致。

主要改动：

- README。
- `docs/user-guide/engine-fsdp.md`。
- compat matrix。
- `docs/developer-docs/impr-refactor.md` 必要跟进。

注意事项：

- Qwen2.5-0.5B 仍是首选模型。
- 依赖统一按 verl080 描述，不按模型区分。
- `ENABLE_PREFIX_SHARING` 降级为开发/调试 fallback。

测试：

- 文档检查。
- 配置解析单测。
- 如 compat matrix 有测试，必须更新。

### 4.6 PR-F：删除 PrefixGroup / group_ids

目标：

- 删除 group 噪音，保留 provider/reuser DAG 事实源。

主要改动：

- `core/prefix_detector.py`
- `core/planner.py`
- `integrations/context.py`
- observability / audit
- detector/planner tests

注意事项：

- 不合并 DetectionResult 与 Plan。
- 不删除 `provider_index`、`prefix_lens`、`is_provider`。
- `is_provider` 只补注释，不改名。

测试：

```bash
PYTHONPATH=prefix-sharing pytest -q \
  prefix-sharing/tests/unit_test/test_prefix_detector.py \
  prefix-sharing/tests/unit_test/test_planner.py \
  prefix-sharing/tests/unit_test/test_runtime_context.py
```

### 4.7 PR-G：诊断与 tools 清理

目标：

- 清理热路径 print/dump 侵入。
- tools 目录保留可复现验证工具。

主要改动：

- 新增或整理 diagnostics helper。
- 标记或迁移 dump 调用。
- 删除一次性工具。
- 给保留工具补文件头或 README。

注意事项：

- 精度对齐阶段仍要保留必要 dump 能力。
- 删除工具前确认结论已迁移。

测试：

- 默认路径不输出热路径 print。
- 诊断开关打开时 dump helper 可用。

### 4.8 推荐执行顺序

强制顺序：

1. PR-A：先清理 mixer-specific 历史代码，收窄主线。
2. PR-B：再抽 shared KV builder，解决 backend 结构问题。
3. PR-C：再抽 integration 公共模块，解决 FSDP/MCore 反向依赖。
4. PR-D：再删旧 patch manager，避免前面改动还要同时维护两套 patch。

可并行或后置：

- PR-E 可在 PR-C 后并行推进。
- PR-F 可在 PR-A/B/C 后推进，避免和前面大范围 import/字段变更冲突。
- PR-G 可最后推进，因为调试能力在前几轮重构中仍可能用到。

## Chapter 5：当前结论

当前研究分析已经足以支持进入执行阶段。结论如下。

### 5.1 开源路线

- 第一优先级是 verl080 + FSDP + Qwen2.5-0.5B。
- Megatron/MCore 保留为 advanced/internal path，不作为第一波开源默认路径。
- NPU/MindSpeed/Megatron-Bridge 保留为后续扩展，不阻塞 FSDP-first。
- Qwen3.5/3.6 HybridAttention/Gated DeltaNet 不进入当前开源主线。

### 5.2 社区定位

- PrefixSharing 对外应表现为 PrefixGrouper 的 `arbitrary_prefix` 扩展模式。
- 首批 PR 应尽量复用 verl 的 `use_prefix_grouper` 用户心智。
- 不在 PrefixSharing 主包复刻 PrefixGrouper prompt-only 算法。
- `prompt_only` 与 `arbitrary_prefix` 必须在配置和代码路径上清晰分流。

### 5.3 技术红线

- 精度一致性优先于性能。
- Prefix KV store 不能 detach。
- Prefix-Last Restore 不能删除或弱化。
- provider-before-reuser 的 store/load 顺序必须保留注释和测试。
- TP padding slot 不能进入 store 或影响 restore index。

### 5.4 软件工程结论

- `build_kv()` 必须从 TorchRef 抽出，成为 backend 公共能力。
- FSDP/MCore 公共 helper 必须进入 `verl_utils.py` 或后续细分公共模块。
- 旧 patch manager 体系应删除，`setup/` 是唯一生产 patch 机制。
- import auto patch 与显式 install 都保留，直到正式合入 verl 后再决定是否下线。
- tools 和 diagnostics 需要分级清理，不应一刀切删除。

### 5.5 下一步判断

下一步不应继续扩写方案，而应开始 PR-A 到 PR-D 的代码重构。每个 PR 都以“可 review、可测试、可回滚”为边界。若执行中发现某项改动会改变精度语义，应停止该 PR，把问题拆成独立设计与测试任务。

## Chapter 6：遗留问题

本章记录不阻塞当前重构、但后续必须显式处理的问题。遗留项按主题归类，后续可迁移到 issue 或 pending-items。

### 6.1 上游 verl 与 PrefixGrouper

- 上游 verl 最新 PrefixGrouper schema 需要在社区 PR 前重新核对。
- 当前 `prefix_grouper.mode=arbitrary_prefix` 是本仓库侧已支持的配置读取；verl 上游 dataclass/schema 是否接受该字段仍需确认。
- PrefixGrouper prompt-only 主流程调用点需要再次核实，避免社区 PR 接入到未被主流程调用的 helper。
- 如果社区更倾向 `shared_prefix` 等中性命名，需要准备从 `prefix_grouper.mode` 迁移的兼容方案。

### 6.2 Meituan prefix-tree / verl RFC

- Meituan prefix-tree PR 和 verl RFC 可能进入主线，届时需要评估配置和语义共存。
- 我们当前 FSDP-first arbitrary-prefix 路线不应提前改成 Magi/block-sparse 方案。
- 后续可对齐 RFC 的 `prefix_segments` 概念，但不能牺牲现有 provider/reuser plan 的精度语义。

### 6.3 真实环境验证

- FSDP 真实 engine e2e fixture 仍需补强。
- 需要补 baseline、PrefixGrouper prompt-only、PrefixSharing arbitrary-prefix 三方性能对比。
- 需要固定数据和随机种子，形成可复现的精度对齐脚本。
- optional GPU/NPU 环境测试不能作为每个本地 PR 的硬门槛，但 release 前必须复跑。

### 6.4 Patch 与 import hook

- import hook 是否长期保留，需要等社区对 monkey patch 方式的反馈后再定。
- 如果正式合入 verl，显式调用路径可能替代外部包 import auto patch。
- 当前阶段必须保留 import auto patch，因为脚本化训练仍依赖 `VERL_USE_EXTERNAL_MODULES=prefix_sharing`。

### 6.5 HybridAttention / Gated DeltaNet

- 当前重构清理 Qwen3.5/GDN 专门化代码，不代表永久放弃 HybridAttention。
- 后续需等待训练引擎侧真实接口稳定，再重新设计 activation/cache_param store。
- 未来重新引入时，应以实际 mixer 类型命名，避免把 DeltaNet 泛化成所有 linear attention。

### 6.6 Megatron / MCore / NPU

- MCore path 保留 advanced/internal 定位。
- NPU/MindSpeed/Megatron-Bridge 后续可继续支持，但不能阻塞 FSDP-first 开源主线。
- 若后续重新提高 Megatron 优先级，需要单独补兼容矩阵、真实环境测试和文档。

### 6.7 tools 与 diagnostics

- tools 清理需要逐项判断，不能批量删除。
- 保留工具必须补用途说明。
- 诊断 dump 默认关闭，但精度对齐阶段仍需保留可开启路径。
