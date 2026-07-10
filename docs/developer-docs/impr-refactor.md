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

### 2.1 用户入口与配置策略

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

配置优先级建议：

1. verl 用户侧公开入口优先：`use_prefix_grouper + prefix_grouper.mode`。
2. `prefix_sharing_config` 保留为内部兼容、测试、patch 未正式合入前的 escape hatch。
3. `ENABLE_PREFIX_SHARING` 保留为开发/调试 fallback，不作为 README 首选。

`mode: arbitrary_prefix` 当前已经能触发 PrefixSharing：`read_ps_config_from_engine_config()` 会读取 `use_prefix_grouper=True`，并在 `prefix_grouper.mode` 为 `arbitrary_prefix` / `arbitrary-prefix` / `prefix_sharing` 时返回 `enable_prefix_sharing=True` 的配置；`prompt_only` / `prefix_grouper` 则返回 disabled。

PrefixGrouper 相关代码边界：

- 只把 PrefixGrouper 当作 verl 用户入口、配置命名和 prompt-only baseline。
- PrefixSharing arbitrary-prefix 不复用 PrefixGrouper `group_info` 作为内部 runtime 结构。
- 仓库中不应存在 PrefixGrouper prompt-only 算法复刻；如果需要测试 prompt-only 行为，应使用 fake fixture 或外部 PrefixGrouper 依赖，而不是在 PrefixSharing 主包里实现一份。
- 变量和文档命名可以保留 `prefix_grouper` 以兼容 verl 配置，但 runtime 对象应命名为 `prefix_sharing_*` 或更通用的 `shared_prefix_*`，避免误导读者。

### 2.2 Integration 分层

目标分层：

```text
integrations/
  verl_utils.py          # FSDP/MCore 共用配置、batch、position helper
  runtime_state.py       # PrefixSharingRuntimeState
  context.py             # PrefixSharingRuntimeContext
  verl_fsdp.py           # FSDP 专属逻辑
  verl_mcore.py          # Megatron/MCore 专属逻辑
```

原则：

- `verl_fsdp.py` 不 import `verl_mcore.py`。
- fake/local helpers 必须显式标注 test utility 或 local fallback。
- 真实生产入口优先在 `setup/patches/verl080_fsdp/` 中体现。

### 2.3 Backend 分层

目标分层：

```text
backends/
  base.py                # backend protocol/capabilities
  kv_builder.py           # shared build_kv / KV expansion
  flash_atten_gpu.py      # GPU FA production path
  flash_atten_npu.py      # NPU FA production path
  torch_ref.py            # correctness/reference path
```

原则：

- GPU/NPU backend 使用公共 KV builder，不依赖 TorchRef。
- TorchRef 保留用于单测、精度对齐和 CPU fallback。
- 首批主线只保留 attention KV；GDN/HybridAttention 等待真实训练引擎接口后重新设计。

### 2.4 Patch 机制

目标：

- `setup/` 是唯一生产 patch 机制。
- `integrations/patch_manager.py` 旧体系删除。
- 保留两种安装方式：
  - import 后自动 patch，服务 `VERL_USE_EXTERNAL_MODULES=prefix_sharing`。
  - 显式 `prefix_sharing.setup.install("verl080_fsdp")`，服务可读接入和交互式调试。

要求：

- patch 幂等。
- patchset 选择清晰，FSDP 第一优先级。
- import hook 行为有测试覆盖，失败路径可诊断。
- import hook 整改前必须先画清楚触发链：
  - import auto patch：`import prefix_sharing` 触发什么；
  - external modules：verl 如何 import `prefix_sharing`；
  - explicit install：用户手动调用 `prefix_sharing.setup.install(...)` 时如何避免重复 patch；
  - fallback：patch target 尚未 import 时是否走 lazy hook，target 已 import 时是否走 eager patch。

### 2.5 调试与工具策略

调试逻辑处理分两步：

1. 短期：所有热路径 dump/print/logging 加统一注释标记，避免后续漏清理。
2. 中期：集中到 `diagnostics` helper，生产 patch 只保留一行调用。

tools 目录处理原则：

- 一次性摸底脚本可以删。
- 版本级精度验证、性能验证脚本必须保留，并补充用途说明。
- 保留工具必须写清楚：命令入口、输入数据要求、输出结果含义、依赖环境、适合在哪类 PR 或 release 前复跑。
- 删除工具前应确认其结论已经迁移到测试、文档或仍保留的 benchmark/report 中。

## Chapter 3：测试验证

首批重构需要覆盖：

- Qwen3.5/GDN 清理后，公开导出、backend factory、store 单测仍通过。
- shared KV builder 与旧 `TorchReferenceBackend.build_kv()` 在 no-sharing、one-provider、chain、TP padding 场景输出一致。
- FSDP/MCore 公共 helper 抽离后，两条 integration 测试路径不再出现 FSDP import MCore 私有函数。
- `use_prefix_grouper=true + mode=arbitrary_prefix` 使能 PrefixSharing；`prompt_only` 不进入 PrefixSharing。
- import auto patch 和显式 `setup.install()` 都可用、幂等、不会重复 patch。
- 默认训练路径无热路径 print；诊断开关打开时 dump 路径仍可用。

建议回归命令：

```bash
PYTHONPATH=prefix-sharing pytest -q \
  prefix-sharing/tests/unit_test \
  prefix-sharing/tests/integrated_test \
  prefix-sharing/tests/system_test
```

文档或计划类改动可不跑测试，但提交说明必须明确。

## Chapter 4：开发计划

建议顺序：

1. 清理 Qwen3.5 / Gated DeltaNet 专门化代码。
2. 抽出 shared KV builder，修正 GPU/NPU backend 对 TorchRef 的依赖。
3. 抽出 `verl_utils.py` 和 runtime state 公共模块。
4. 删除旧 patch manager 体系。
5. 更新 README、compat matrix、PrefixSharing/PrefixGrouper 关系说明。
6. 再做 PrefixGroup / group_ids 删除和 Plan 字段说明。

每一步都应独立提交，避免把行为重构和大面积删除混成一个不可 review 的改动。

## Chapter 5：当前结论

本轮修正后的结论：

- 开源首推 verl080 + FSDP + Qwen2.5-0.5B 路线，依赖配置统一，不按模型区分依赖。
- PrefixSharing 对外应表现为 PrefixGrouper 的 `arbitrary_prefix` 扩展模式。
- 首批主线只聚焦 attention KV prefix sharing，清掉 Qwen3.5/GDN 专门化设计。
- `build_kv` 这类公共能力必须从 TorchRef 抽出来。
- FSDP/MCore 共用 helper 必须进入公共模块。
- import auto patch 和显式 install 都保留，直到正式合入 verl 后再决定是否下线。

## Chapter 6：遗留问题

- 上游 verl 最新 PrefixGrouper schema 与实际调用路径仍需在提交社区 PR 前再次核对。
- Meituan prefix-tree RFC/PR 后续若进入主线，需要评估我们的 FSDP arbitrary-prefix 路线如何与其配置和语义共存。
- FSDP 真实 engine e2e fixture 仍需补强。
- HybridAttention/Gated DeltaNet 等 mixer-specific 支持应等待训练引擎真实接口稳定后再重新设计。
- import hook 是否长期保留，需要等社区对 monkey patch 方式的反馈后再定。
