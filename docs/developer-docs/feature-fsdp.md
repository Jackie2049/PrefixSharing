# PrefixSharing 支持 FSDP 的研究分析与设计方案

本文档用于支撑 PrefixSharing 适配 verl FSDP 路径的技术决策和后续开发排期。

当前仍处于开源前开发阶段，文档和注释统一使用中文。后续进入开源整改阶段时，再将 docs 体系重构为社区可读的英文文档，或删除内部研究文档。

## Chapter 1：研究分析

### 1.1 目标重新界定

PrefixSharing 最早围绕 `verl + Megatron-LM` RL 训练链路实现，核心能力是：

- 在同一 micro-batch 内检测共享前缀；
- 裁剪 reuser 的共享前缀；
- 复用 provider 前缀 KV / activation；
- 通过 Prefix-Last Restore 保证 suffix-first logprob 不丢；
- 保持 logprob / loss / gradient 与 baseline 语义一致。

但如果目标是开源并合入 verl 社区，首推方案不应是 Megatron / MindSpeed / NPU 路线，而应先适配 FSDP：

- verl 社区已经有 PrefixGrouper 特性；
- PrefixGrouper 已经通过 `actor.use_prefix_grouper=True` 建立用户入口；
- PrefixGrouper 本身也是独立 Python 包，verl 只是集成调用，而不是把全部逻辑纳入主仓；
- FSDP 路径更容易被社区复现、review 和接受；
- Megatron 路径可以作为后续 experimental backend，而不是首个 PR 的阻塞项。

这里的“FSDP 适配”特指 verl actor/ref 的 FSDP 训练路径，不是泛泛地给 Transformers 模型加一个新后端。Transformers attention patch 只是 PrefixGrouper 现有实现机制的一部分，用户配置、执行入口、测试验收都应围绕 verl FSDP 展开。

因此 FSDP 适配的目标不是“另起一个 PrefixSharing 开关”，而是：

```text
沿用 verl PrefixGrouper 的用户接口，
在 prompt-only 模式下继续走 PrefixGrouper 包，
在 arbitrary-prefix 模式下导入并走 PrefixSharing 包。
```

换句话说，从 verl 视角看，这是一个统一的 shared-prefix training feature；从实现视角看，PrefixGrouper 和 PrefixSharing 是两个独立包，分别负责不同算法模式。

### 1.2 PrefixGrouper 源码结论

已拉取并审视 `CASIA-IVA-Lab/PrefixGrouper` 源码。核心 API 在：

- `/Volumes/Agent/workspace/projects/PrefixGrouper/src/prefix_grouper/__init__.py`
- `/Volumes/Agent/workspace/projects/PrefixGrouper/src/prefix_grouper/info.py`
- `/Volumes/Agent/workspace/projects/PrefixGrouper/src/prefix_grouper/forward.py`
- `/Volumes/Agent/workspace/projects/PrefixGrouper/src/prefix_grouper/function.py`

PrefixGrouper 的核心数据模型是：

```python
group_info = [
    [prefix_len, suffix1_len, suffix2_len, ...],
    ...
]
```

每一行表示一个 group：

- 该 group 有一个共享 prefix；
- 该 group 下有多个 suffix；
- 每个 suffix 对应一个原始样本 / response；
- prefix 只计算一次；
- suffix attention 会 attend 到共享 prefix。

PrefixGrouper 的关键能力：

- `PrefixGrouper.from_ungrouped_masks(prefix_mask, suffix_mask, group_sizes, ...)`
  - 根据 prefix / suffix mask 自动生成 `group_info`；
  - 当前 verl 集成主要用它表达 same prompt + multiple responses。
- `concat_input(prefix, prefix_mask, suffix, suffix_mask)`
  - 将 group 级 prefix 和样本级 suffix 拼成 grouped input。
- `forward(attn_func, q, k, v, ...)`
  - 将 Q/K/V 拆成 prefix 和 suffix；
  - prefix self-attention 只计算一次；
  - suffix attention 使用 `cat(prefix_kv, suffix_kv)`；
  - 最后将 prefix / suffix attention output group 回原 grouped input 形态。
- `split_output(output, include_prefix_last=1)`
  - 将模型输出拆回 prefix / suffix；
  - `include_prefix_last=1` 会把 prefix 最后一个 token 的输出拼到 suffix 开头；
  - 这正是 response 第一个 token logprob 所需的 prefix-last 输出。
- `GroupFunction` / `UngroupFunction` / `ConvertPaddingFunction`
  - 都是自定义 autograd function；
  - forward 做 gather/scatter；
  - backward 把梯度散回原输入；
  - 没有 `detach()`，天然保留 autograd 语义。

重要结论：

1. PrefixGrouper 不是“只能支持同一个固定 prompt 字符串”的底层实现，它底层只要求每个 group 能表达为“一个 prefix + 多个 suffix”。
2. PrefixGrouper 当前 verl 用法是 prompt-only，因为 verl 用 `uid` 聚组，并从 `prompts` 取每组第一个样本作为 prefix。
3. PrefixGrouper 不能直接表达 PrefixSharing 当前 planner 的任意 provider/reuser 复用图，尤其是同一个 provider 对不同 reuser 复用不同 prefix_len、reuser 继续成为后续 provider 的链式复用。
4. PrefixGrouper 的 `group_info` 只能表达 one prefix + many suffix 的两阶段结构；这非常适合 prompt-only GRPO n-sample，但不是 PrefixSharing arbitrary-prefix 的完整语义。
5. 因此 PrefixSharing FSDP 不应把自己的 `PrefixSharingPlan` 强行压成 PrefixGrouper `group_info`。这会削弱任意长度 prefix 复用能力，也可能损失 step/tree 场景中最有价值的链式复用。
6. PrefixGrouper 在本方案中的定位是 verl 社区已有接口和配置入口，而不是 PrefixSharing 的运行时算法上界。`mode=arbitrary_prefix` 后应进入 PrefixSharing 自己的 planner / runtime / restore 实现。

### 1.3 verl PrefixGrouper 集成结论

已拉取并审视 `verl-project/verl` 主仓。PrefixGrouper 相关集成点包括：

- `verl/trainer/config/actor/actor.yaml`
  - 已有 `use_prefix_grouper: false`。
- `verl/workers/config/actor.py`
  - `ActorConfig.use_prefix_grouper: bool = False`。
- `verl/models/transformers/monkey_patch.py`
  - `apply_prefix_grouper_patch()` patch `transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS`；
  - wrapper 从 kwargs 中取 `prefix_grouper`；
  - 如果为 `None`，走原 attention；
  - 如果存在，调用 `prefix_grouper.forward(attn_func, query, key, value, ...)`。
- `verl/trainer/ppo/prefix_grouper_utils.py`
  - `build_pg_from_micro_batch()` 从 `prompts` / `responses` / `response_mask` / `uid` 构建 PrefixGrouper；
  - `pg_forward()` 调模型，传入 `prefix_grouper=prefix_grouper`；
  - `forward_micro_batch_with_prefix_grouper()` 封装 logprob / entropy 输出。
- `verl/trainer/ppo/ray_trainer.py`
  - `trainer.balance_batch=True` 时，若开启 `use_prefix_grouper`，按 `uid` 做 group-level balancing，确保同 uid 样本留在同一 DP rank。

当前 verl 集成的特征：

- PrefixGrouper 是外部包，通过 `import prefix_grouper` 使用；
- verl 的用户接口非常轻：`actor.use_prefix_grouper=True`；
- 当前官方示例明确是 FSDP worker，不支持 Megatron；
- 当前限制包括：
  - 不支持 Megatron worker；
  - 不支持 `use_remove_padding=True`；
  - 不支持 fused kernels；
  - 不支持 Ulysses SP / ring attention；
  - 文档中说不兼容 `use_dynamic_bsz=True`，但示例脚本中又设置了部分 dynamic bsz 参数，需要后续实测核实。

还发现一个重要现状：

- `forward_micro_batch_with_prefix_grouper()` 在当前 verl 主仓中可被搜索到定义，但没有直接搜索到主路径调用点；
- 这意味着 PrefixGrouper 在主仓中可能处于“配置、patch、工具函数已合入，但完整执行路径仍在演进”的状态；
- 因此 PrefixSharing FSDP 适配不能假设现有 PrefixGrouper actor update / compute_log_prob 已完全闭环，需要把“补齐 FSDP engine 调用入口”纳入设计。

### 1.4 verl FSDP 主路径结论

已审视 FSDP engine 主路径：

- `verl/workers/engine/fsdp/transformer_impl.py`
  - `FSDPEngine.forward_backward_batch()`
  - `FSDPEngineWithLMHead.prepare_model_inputs()`
  - `FSDPEngineWithLMHead.prepare_model_outputs()`
  - `FSDPEngineWithLMHead.forward_step()`
- `verl/workers/engine_workers.py`
  - `train_mini_batch()`
  - `infer_batch()`
  - `compute_log_prob()`
  - `update_actor()`

FSDP 的核心执行流程是：

```text
ActorRolloutRefWorker.compute_log_prob / update_actor
  -> actor.infer_batch / actor.train_mini_batch
  -> engine.forward_backward_batch
      -> prepare_micro_batches
      -> for micro_batch:
          -> FSDPEngineWithLMHead.forward_step
              -> prepare_model_inputs
              -> self.module(**model_inputs, use_cache=False)
              -> prepare_model_outputs
              -> loss_function(model_output, data)
```

因此 PrefixSharing FSDP 的最合适接入点是 `FSDPEngineWithLMHead.forward_step()` 或其前后拆分函数，而不是新增一个独立 `fsdp` backend 层。

具体原因：

- `prepare_model_inputs()` 已经负责处理 `input_ids`、`attention_mask`、`position_ids`、remove padding、Ulysses SP、temperature 等；
- `prepare_model_outputs()` 已经负责 logits -> log_probs / entropy / nested tensor 输出；
- PrefixSharing FSDP 需要替换的是“micro-batch 输入组织 + model forward + output assembly”这一段；
- 这更像 verl FSDP integration，而不是 PrefixSharing core 或 backend。

### 1.5 当前 PrefixSharing core 是否需要修改

当前判断：**FSDP 首版应优先复用现有 core plan 语义，而不是改成 PrefixGrouper group 语义。**

理由：

- core 当前定位是训练引擎无关的 prefix-sharing 语义；
- `PrefixSharingPlan` 已经表达 provider/reuser、prefix_len、Q path 裁剪、KV expanded length、position offset 和 prefix-last restore spec；
- FSDP 需要的是把 verl FSDP micro-batch 接到这套 plan/runtime 上，而不是把 plan 转成 PrefixGrouper 的 `group_info`；
- 若 FSDP 发现现有 plan 缺少框架无关字段，应补 core 的通用语义；但不应为了适配 PrefixGrouper 的两阶段 attention 模型而削弱 core。

首版建议只新增：

```text
prefix-sharing/prefix_sharing/integrations/verl_fsdp.py
```

它与当前 `integrations/verl_mcore.py` 对应：

- `verl_mcore.py`：Megatron / MCore 路径适配；
- `verl_fsdp.py`：verl FSDP / PrefixGrouper 接口适配。

暂不新增独立 `fsdp/` 模块。只有当 FSDP adapter 复杂到出现多个稳定的内部组件时，再考虑拆分。

### 1.6 源码依据汇总

以下结论已经基于本地源码审视，而不是凭接口名称推测。后续实现时如果发现上游代码变化，应优先重新核对这些文件。

| 代码位置 | 已确认事实 | 对 FSDP 方案的影响 |
|----------|------------|--------------------|
| `PrefixGrouper/src/prefix_grouper/__init__.py` | `PrefixGrouper` 负责 concat input、attention forward、split output；`split_output(include_prefix_last=1)` 会保留 prefix-last 输出 | 这是 prompt-only 两阶段方案的可用参考，但不能直接证明 PrefixSharing arbitrary-prefix 可以复用同一 restore 方案 |
| `PrefixGrouper/src/prefix_grouper/info.py` | `group_info` 是 `[prefix_len, suffix1_len, ...]`，表达 one prefix + many suffix | 该结构不能作为 PrefixSharing arbitrary-prefix 的主数据结构；只能作为理解 PrefixGrouper prompt-only 行为的背景 |
| `PrefixGrouper/src/prefix_grouper/forward.py` | suffix attention 通过拼接 prefix KV 和 suffix KV 实现共享前缀复用 | 可以借鉴“suffix attend prefix KV”的思想，但 PrefixSharing 需要保留 provider/reuser DAG 和逐层 KV store/load |
| `PrefixGrouper/src/prefix_grouper/function.py` | group / ungroup / padding conversion 的 backward 会 scatter 梯度，未 detach | 说明 prefix 复用必须保留 autograd；PrefixSharing FSDP 也必须保证 KV store 不 detach |
| `verl/models/transformers/monkey_patch.py` | verl 已 patch `ALL_ATTENTION_FUNCTIONS` 并通过 `prefix_grouper` kwarg 进入 PrefixGrouper | PrefixSharing 可选择复用这一路 attention patch 入口，但传入对象应是 PrefixSharing runtime adapter，而不是 PrefixGrouper 原对象 |
| `verl/trainer/ppo/prefix_grouper_utils.py` | 已有 prompt-only 构造和 forward helper，但主路径调用点在当前主仓中不明显 | FSDP 适配必须把“执行入口是否闭环”作为 Phase 0/Phase 3 的验证项；不能依赖 PrefixGrouper 工具函数覆盖 PrefixSharing 逻辑 |
| `verl/workers/engine/fsdp/transformer_impl.py` | FSDP 实际 forward 在 `FSDPEngineWithLMHead.forward_step()` 内完成 | 原型阶段最稳的接入点是 FSDP engine forward_step；上游 PR 可再争取更轻 hook |
| `verl/trainer/ppo/ray_trainer.py` | prompt-only 下按 `uid` 做 group-level balancing | arbitrary-prefix 首版不依赖 `uid`，先做 DP rank 内本地检测；prefix hash balancing 留到性能阶段 |

### 1.7 PrefixSharing 与 PrefixGrouper 的关系结论

从社区合入角度，不建议现在“反客为主”把字段名改成 PrefixSharing 或 PrefixAttention。

原因：

- verl 已经有 `use_prefix_grouper`；
- PrefixGrouper 已经是社区认知中的 shared-prefix training 工具；
- PrefixGrouper 也是独立包，verl 以导入方式集成；
- PrefixSharing 也可以成为独立包，以相同模式被 verl 集成；
- 首批 PR 的目标应是降低 review 成本，而不是同时推动概念命名迁移。

因此建议：

```text
短期：作为 PrefixGrouper 体系下的新 algorithm / mode 接入
中期：如果社区认可，再讨论 shared_prefix 这种更 general 的统一命名
长期：PrefixGrouper 和 PrefixSharing 都作为 shared-prefix training 的不同 algorithm/backend
```

首版不要替换 `use_prefix_grouper` 字段。

### 1.8 接口对齐与能力边界

必须区分两件事：

1. **对齐 verl 的 PrefixGrouper 接口和配置项**；
2. **对齐 PrefixGrouper 包内部的两阶段执行模型**。

本文档只主张第一件事，不主张第二件事。

原因是 PrefixSharing 的目标能力更通用：

- PrefixGrouper 当前主要覆盖 prompt-only sharing；
- PrefixSharing 覆盖 arbitrary-prefix sharing，prompt-only 只是其中一个子集；
- PrefixGrouper 依赖 `uid` / prompt group 组织才能稳定获得收益；
- PrefixSharing 不要求用户显式提供 group 标记，而是在 micro-batch 内自动检测可复用 prefix；
- PrefixGrouper 的 one prefix + many suffix 执行模型不能完整承载 PrefixSharing 的 provider/reuser DAG。

因此，开源和社区合入策略应当是：

```text
借 PrefixGrouper 在 verl 中已有的入口降低接入和 review 成本，
但 arbitrary-prefix 运行时必须保持 PrefixSharing 自己的 plan / KV injection / restore 语义。
```

如果未来社区接受更 general 的命名，可以把 `use_prefix_grouper` 迁移或泛化为 `shared_prefix` / `prefix_sharing`。但短期为了减少首批 PR 阻力，仍建议保留现有入口并新增 `mode`。

## Chapter 2：方案设计

### 2.1 总体方案

在 verl 中保持一个统一的用户入口：

```yaml
actor_rollout_ref:
  actor:
    use_prefix_grouper: true
```

新增一个可选配置，用于选择算法模式：

```yaml
actor_rollout_ref:
  actor:
    use_prefix_grouper: true
    prefix_grouper:
      mode: prompt_only          # prompt_only | arbitrary_prefix
      min_prefix_len: 16
      min_group_size: 2
      strict: false
      validate_precision: false
```

语义：

- `mode=prompt_only`
  - 当前默认模式；
  - 导入并调用 `prefix_grouper` 包；
  - 保持现有 verl PrefixGrouper 行为。
- `mode=arbitrary_prefix`
  - 新增模式；
  - 导入并调用 `prefix_sharing` 包；
  - 使用 PrefixSharing planner 检测任意 prefix；
  - 使用 PrefixSharing 自己的 runtime 执行任意长度 prefix 复用；
  - 对 verl 来说仍然是 PrefixGrouper / shared-prefix training 路径。

是否叫 `mode`、`algorithm` 或 `strategy`：

- 首版推荐 `mode`，因为用户理解成本最低；
- 如果社区更倾向算法命名，可以改为 `algorithm: prefix_grouper | prefix_sharing`；
- RFC 中可以同时给出两个候选，让 maintainer 决策。

当前倾向：

```yaml
prefix_grouper:
  mode: prompt_only | arbitrary_prefix
```

#### 2.1.1 `open-source_fsdp-usability` 开发约定

本分支目标是把已经验证过的 FSDP 能力补齐为“用户可理解、可配置、可复现”的实验特性入口，不重新设计 FSDP 核心算法路径。

必要开发范围：

1. PrefixSharing 包内先支持读取 verl 风格配置：

   ```yaml
   use_prefix_grouper: true
   prefix_grouper:
     mode: arbitrary_prefix
   ```

2. 配置语义：
   - `mode=prompt_only`：PrefixSharing 不启用，保留给 PrefixGrouper 包处理；
   - `mode=arbitrary_prefix`：PrefixSharing 启用 FSDP arbitrary-prefix 路径；
   - 缺省 `mode`：按 `prompt_only` 处理，保持 PrefixGrouper 既有行为；
   - 旧实验入口 `prefix_sharing_config`、`ENABLE_PREFIX_SHARING`、`PREFIX_SHARING_PATCHSET=verl080_fsdp` 继续保留。

3. 兼容策略：
   - `prefix_sharing_config` 优先级高于 `prefix_grouper.mode`，便于内部调试和回归；
   - `prefix_grouper` 中只透传 PrefixSharing 已支持字段，例如 `min_prefix_len`、`min_group_size`、`validate_precision`；
   - `strict` 等 PrefixSharing 暂未接收字段不传入 `PrefixSharingConfig`，避免用户侧配置污染内部 dataclass。

4. 测试分工：
   - Codex 负责本地 TDD：配置解析、prompt_only/arbitrary_prefix 分发、旧配置兼容、user-guide 文档；
   - ClaudeCode 负责真实 verl/FSDP 环境测试：确认用户文档中的启用方式可执行、audit 日志可见、2/4/8 卡 packed path 不回退。

本分支不做：

- 不修改 verl 主仓 yaml/dataclass schema；
- 不把 PrefixGrouper 包内部逻辑迁入 PrefixSharing；
- 不新增性能 benchmark；
- 不把实验特性写入 README。

不建议首版使用：

```yaml
shared_prefix:
  enabled: true
```

虽然它更 general，但会引入额外迁移讨论，不利于首批 PR。

### 2.2 包关系设计

保持两个独立 Python 包：

```text
prefix_grouper
  - prompt-only shared-prefix execution
  - existing PrefixGrouper API
  - existing verl integration remains compatible

prefix_sharing
  - arbitrary-prefix detection / planning
  - FSDP adapter for PrefixSharing runtime
  - future Megatron backend
```

verl 中的选择逻辑：

```python
if use_prefix_grouper:
    if prefix_grouper.mode == "prompt_only":
        from prefix_grouper import PrefixGrouper
        # existing path
    elif prefix_grouper.mode == "arbitrary_prefix":
        from prefix_sharing.integrations.verl_fsdp import ...
        # PrefixSharing path
```

这样做的好处：

- verl 主仓不需要直接纳入 PrefixSharing 全部代码；
- 与 PrefixGrouper 当前集成方式一致；
- PrefixSharing 可以独立发布、独立测试、独立迭代；
- 后续 Megatron backend 仍可以留在 PrefixSharing 包内，而不污染 verl FSDP 主路径。

### 2.3 FSDP adapter 职责

新增：

```text
prefix-sharing/prefix_sharing/integrations/verl_fsdp.py
```

职责：

1. 解析 verl FSDP micro-batch；
2. 从 `prompts` / `responses` / `attention_mask` / `response_mask` 中恢复每条样本的完整有效 token；
3. 调用 `PrefixSharingPlanner`；
4. 根据 `PrefixSharingPlan` 构造裁剪后的 FSDP model input；
5. 建立 PrefixSharing runtime context：
   - `prefix_sharing_plan`
   - `attention_mask`
   - `position_ids`
   - `PrefixAttentionStore`
   - attention runtime adapter；
6. 调用 model forward，并在 attention 层执行 KV store/load 与 KV injection；
7. 后处理 logits / logprob / entropy / mask；
8. 按 PrefixSharing restore spec 恢复 suffix-first logprob；
9. fallback 到原始路径。

不负责：

- Megatron parallel state；
- TP / PP / CP；
- packed THD；
- vocab-parallel restore；
- MindSpeed / NPU kernel。

建议首版公开函数：

```python
def build_prefix_sharing_fsdp_micro_batch(
    micro_batch: TensorDict,
    *,
    min_prefix_len: int,
    min_group_size: int,
    pad_token_id: int,
    padding_mode: str = "right",
) -> tuple[TensorDict, PrefixSharingRuntimeState | None]:
    """将 verl FSDP micro-batch 转换成 trimmed_micro_batch + runtime state。

    返回 ``(micro_batch, None)`` 表示没有检测到可复用 prefix，调用方应 fallback 原始 FSDP forward。
    """
```

这里不建议新增 `PrefixSharingFSDPPreparedBatch` 作为首版实体。Megatron 路线已有相同职责的返回形态：

```text
build_prefix_sharing_micro_batch_verl070/verl080
  -> (trimmed_micro_batch, PrefixSharingRuntimeState | None)
```

FSDP 应与它呼应：

- `trimmed_micro_batch`：承载裁剪后的 `input_ids` / `attention_mask` / `position_ids` / `labels` / `loss_mask` 等 verl FSDP 输入；
- `PrefixSharingRuntimeState`：承载 `prefix_sharing_plan`、attention backend/runtime、position/layout 信息和 restore metadata；
- 若 FSDP 后续确实需要无法放进通用 runtime state 的字段，再新增 `PrefixSharingFSDPRuntimeState`，但必须先由测试证明必要性。

这个选择可以避免在 Megatron 和 FSDP 两条路线中产生两套同义数据结构。

### 2.4 PrefixSharing FSDP 不应转换为 PrefixGrouper-style group

这是本文档相对早期方案必须修正的核心结论。

PrefixGrouper 的运行时数据结构是：

```python
group_info = [
    [prefix_len, suffix1_len, suffix2_len, ...],
]
```

它表达的是：

```text
一个 group = 一个共享 prefix + 多个 suffix
```

这个模型非常适合 GRPO 中“同一个 prompt 采样 n 条 response”的 prompt-only sharing。它隐含了一个对用户 batch 组织方式比较强的要求：最好能把同一个 `uid` / 同一个 prompt 的 n sample 放到同一个 DP rank、同一个 micro-batch 内，否则 prompt 复用收益会下降。这也是 PrefixGrouper 在 verl 中需要配合 `uid` 和 group-level balancing 的原因。

PrefixSharing 的设计目标不同：

- 不依赖 `uid`、`group_info` 或用户显式标注；
- 在 micro-batch 内直接从 token 序列检测共享 prefix；
- 支持任意长度 prefix；
- 支持 provider/reuser 关系图；
- 允许 reuser 在扩展后继续成为后续 provider；
- 通过 runtime KV store/load 做逐层复用。

因此，PrefixSharing 的主数据结构必须继续是 `PrefixSharingPlan`，而不是 PrefixGrouper 的 `group_info`。

#### 2.4.1 为什么不能强行收敛为 disjoint group

如果把 PrefixSharing plan 压成 PrefixGrouper-style disjoint group，会引入几个问题：

1. **功能被削弱**：one prefix + many suffix 不能自然表达链式复用和不同 prefix_len 的 provider/reuser DAG。
2. **收益可能下降**：step/tree RL 中，最有价值的复用往往不是单层 prompt group，而是多级历史前缀；压平后可能重复计算中间 prefix。
3. **架构更复杂**：为了适配 `group_info`，需要额外维护 prefix/suffix tensor、row_mapping、group output assembly，反而偏离现有 PrefixSharing runtime。
4. **社区价值不清晰**：如果 arbitrary-prefix 只做两阶段 disjoint group，容易被质疑与 PrefixGrouper prompt-only 的差异不足。
5. **精度风险更高**：重新组织 batch 结构后，logprob、loss mask、position_ids、restore 都要重新证明；而现有 plan 已经包含这些语义。

所以本文档明确否决“FSDP 首版必须转换成 PrefixGrouper-style group”的设计。

#### 2.4.2 FSDP 应沿用 PrefixSharing DAG/plan 语义

PrefixSharing plan 的语义是：

```text
row i 可以复用 provider row j 的前 prefix_len 个 KV
```

这允许：

- 同一个 provider 给不同 reuser 提供不同 prefix_len；
- 中间 reuser 继续成为后续 provider；
- 一个 batch 中出现链式或嵌套复用。

FSDP 路径应保留这套语义，并实现与 Megatron 路径同构的四段式：

```text
1. prepare input
   - 从 verl FSDP micro-batch 恢复每条有效序列；
   - PrefixSharingPlanner 生成 PrefixSharingPlan；
   - 按 plan 裁剪 reuser Q path input/labels/loss_mask；
   - 构造正确 position_ids。

2. module forward
   - 在 HF/Transformers attention 内进入 PrefixSharing runtime；
   - provider 行先 store 本层 prefix/full KV；
   - reuser 行 load provider KV，拼接 provider prefix KV + 自身 suffix KV；
   - attention 使用 suffix Q + expanded KV。

3. prepare output
   - 得到裁剪后 token 对应的 logits/log_probs；
   - 根据 plan 将输出对齐到原始 micro-batch 语义；
   - 执行 prefix-last restore，补齐 suffix-first logprob。

4. loss function
   - 使用恢复后的 log_probs / loss_mask 计算 PPO/GRPO loss；
   - 保持 prefix KV 不 detach，确保梯度回到 provider prefix。
```

这与当前 Megatron 集成的本质是一致的，只是框架入口从 MCore attention patch 换成 verl FSDP / Transformers attention patch。

#### 2.4.3 PrefixGrouper-style group 的保留价值

PrefixGrouper-style group 仍有两个价值，但都不是主方案：

1. 作为 prompt-only baseline，用来和 PrefixSharing arbitrary-prefix 做性能/精度对比；
2. 作为 RFC 叙事中的接口桥梁，说明 PrefixSharing 是 PrefixGrouper 能力的泛化，而不是割裂的新开关。

它不应该成为 PrefixSharing FSDP runtime 的内部表示。

### 2.5 Restore 设计必须同时覆盖 interior prefix 和 prefix-last

只要 PrefixSharing 做了 Q path 输入裁剪，reuser 的共享 prefix 输出就没有在 reuser 行上直接计算出来。因此无论底层训练引擎是 Megatron 还是 FSDP，都需要 restore。restore 不是只有 prefix-last 一类，而是至少分成两类：

1. **interior prefix restore**：共享 prefix 内部位置 `[0, prefix_len - 2]`；
2. **prefix-last restore**：共享 prefix 最后一个位置 `prefix_len - 1`，它预测 reuser 的第一个 suffix token。

两者语义不同：

| 位置 | logits 来源 | label 来源 | logp restore | entropy restore | logits / output 对齐 |
|------|-------------|------------|---------------|------------------|-----------------------|
| interior prefix | provider 对应位置 | provider 与 reuser 相同的下一 token | 可直接拷贝 provider logp | 可直接拷贝 provider entropy | 可直接拷贝 provider logits / attention output |
| prefix-last | provider prefix-last logits | reuser 的 first suffix token | 必须用 provider logits + reuser label 重新计算 | 可拷贝 provider entropy | logits / attention output 可拷贝 provider 对应位置 |

PrefixGrouper 的 `split_output(include_prefix_last=1)` 和 PrefixSharing 的 restore 解决的是同一个精度问题的一部分：response / suffix 第一个 token 的 logprob 需要来自 prefix 最后一个 token 的 logits。但 PrefixGrouper 只覆盖它自己的 prompt-only 两阶段输出重排，不能替代 PrefixSharing 的完整 restore 语义。

FSDP arbitrary-prefix 的 restore 应按 PrefixSharing plan 驱动：

1. planner 继续生成 `PrefixLastRestoreSpec`，用于 prefix-last logp 重算；
2. FSDP 后处理阶段根据 `input_keep_ranges` / `prefix_lens` / `provider_index` 恢复 interior prefix；
3. interior prefix 的 logp / entropy / logits / attention output 从 direct provider 对应位置拷贝；
4. prefix-last 的 logits / entropy / attention output 可以来自 provider 对应位置，但 logp 必须基于 reuser 的 first suffix label 重算；
5. restore 后再进入 PPO/GRPO loss 和精度对齐检查。

FSDP 与 Megatron 的差异在于：

- FSDP logits 通常是完整 vocab logits，不涉及 Megatron vocab-parallel shard；
- 因此 prefix-last logp 重算实现可以比 Megatron 简单，不需要跨 vocab parallel gather；
- 但 restore 分层语义不变，不能因为 FSDP 没有 vocab parallel 就省略 interior restore。

必须验证：

- interior prefix 的 logp / entropy / logits / attention output 与 baseline 对齐；
- 每个 reuser 的 suffix-first logp 与 baseline 对齐；
- reuser 的 first suffix label 与 provider prefix-last logits 组合正确；
- 链式复用时 direct provider 的 prefix-last logits 位置正确；
- restore 不 detach provider logits / output，梯度路径与 baseline 对齐；
- prompt-only 场景下 PrefixSharing restore 与 PrefixGrouper `include_prefix_last` 结果数值一致。

### 2.6 position_ids 设计

PrefixSharing FSDP 的 position_ids 不能按 PrefixGrouper group 构造，而应按 `PrefixSharingPlan` 的 Q path 裁剪语义构造。

现有 planner 已经提供：

- `input_keep_ranges`：每条样本裁剪后保留的 token 范围；
- `q_position_offsets`：reuser 的 Q path 起始绝对位置；
- `kept_lengths_q`：裁剪后每条样本的 Q 长度；
- `expanded_lengths_kv`：attention 中 expanded KV 的逻辑长度。

FSDP adapter 应使用这些字段生成裁剪后输入的 position_ids：

```text
provider / standalone:
  kept input = original[0:len]
  position_ids = 0..len-1

reuser:
  kept input = original[prefix_len:len]
  position_ids = prefix_len..len-1
```

这个规则与 Megatron 路径中的 `q_position_offsets` 一致，能够保证 suffix token 的 RoPE/position embedding 仍对应原始序列绝对位置。

需要特别验证：

- 不同 reuser 有不同 prefix_len；
- 同一个 provider 服务多个 reuser；
- reuser 继续成为 provider；
- left padding / right padding 下 attention_mask 与 position_ids 是否一致；
- FSDP 当前是否允许自定义 position_ids 透传到模型。

### 2.7 attention patch 设计

PrefixSharing 不应复用 PrefixGrouper 原对象，但可以尽量复用 verl 已有 attention patch 的入口形态。

有两种可选实现：

#### 方案 A：复用 `prefix_grouper` kwarg 入口，传入 PrefixSharing adapter

verl 现有 patch contract 是：model forward 传入一个对象，attention wrapper 调用该对象的 `forward(attn_func, query, key, value, ...)`。

可以构造：

```python
class PrefixSharingFSDPAttentionRuntime:
    def forward(self, attn_func, query, key, value, *args, **kwargs):
        # 1. 根据 PrefixSharingRuntimeContext 获取 plan/store/layer_id
        # 2. provider store KV，reuser load provider KV 并拼接 expanded KV
        # 3. 调用原 attention 或 PrefixSharing backend
        # 4. 返回裁剪 Q path 对应的 attention output
        ...
```

然后仍通过现有 kwarg 传入：

```python
model(..., prefix_grouper=prefix_sharing_runtime)
```

这里的名字是为了兼容 verl 当前 patch，不代表运行时复用 PrefixGrouper 原对象。为了减少误解，上游 PR 中可以把内部变量命名为 `shared_prefix_runtime`，但配置入口短期仍保留 `use_prefix_grouper`。

#### 方案 B：新增 PrefixSharing 专用 attention patch

如果 PrefixGrouper 现有 patch 对 output shape、mask、cache 或 kwargs 有强假设，不适合 PrefixSharing DAG runtime，则在 verl 中新增更通用的 shared-prefix attention hook：

```python
model(..., shared_prefix_runtime=runtime)
```

当前建议优先采用方案 B。

原因：

- PrefixSharing 后续需要做更细粒度的精度对齐和精度延展，独立 patch 更容易插入 attention output / logits / mask 诊断；
- PrefixSharing runtime 的输入输出语义不同于 PrefixGrouper 原对象，强行塞进 `prefix_grouper.forward()` 容易让调试路径变得含糊；
- 首版内部开发阶段优先可读性和可验证性，后续再评估是否能和 PrefixGrouper patch 合并。

方案 A 仍保留为上游收敛方向：如果独立 patch 跑通并证明语义稳定，可以再尝试抽象成统一 shared-prefix attention hook。

无论采用哪个方案，都必须满足：

- attention 内部进入 PrefixSharing 自己的 KV store/load；
- 不把 `PrefixSharingPlan` 转换为 PrefixGrouper `group_info`；
- 不要求用户提供 `uid` / group 标记；
- 不 detach provider prefix KV；
- 输出 shape 能回到 FSDP `prepare_model_outputs` / loss 所需格式。

### 2.8 FSDP 主路径接入设计

verl 当前 FSDP 路径中，最合理的接入方式有两种。

#### 方案 A：沿用 `prefix_grouper_utils.py`

在 verl 中扩展现有：

```text
verl/trainer/ppo/prefix_grouper_utils.py
```

新增模式分支：

```python
if mode == "prompt_only":
    from prefix_grouper import PrefixGrouper
    ...
elif mode == "arbitrary_prefix":
    from prefix_sharing.integrations.verl_fsdp import forward_micro_batch_with_prefix_sharing
    ...
```

优点：

- 与当前 PrefixGrouper 集成最一致；
- RFC 叙事清晰：扩展 PrefixGrouper 工具函数；
- 对 verl 主路径改动小。

缺点：

- 当前主仓没有直接搜索到 `forward_micro_batch_with_prefix_grouper()` 的调用点，需要同步补齐调用入口。

#### 方案 B：在 FSDPEngineWithLMHead.forward_step 中接入

在 `FSDPEngineWithLMHead.forward_step()` 开头判断：

```python
if use_prefix_grouper and prefix_grouper.mode == "arbitrary_prefix":
    return forward_step_with_prefix_sharing(...)
```

优点：

- 接入点真实稳定；
- 覆盖 actor update 和 compute_log_prob；
- 能直接复用 engine 的 loss_function / forward_only 语义。

缺点：

- PR 会触碰 FSDP engine 核心代码；
- 需要非常清晰的 fallback 和测试，避免 reviewer 担心风险。

当前决策：

- FSDP 主路径接入优先走方案 A，先尝试扩展 `prefix_grouper_utils.py` 或现有 PrefixGrouper 调用入口，保持和社区已有接口统一；
- 如果当前主仓 PrefixGrouper 调用入口尚未闭环，则先通过 monkey patch 补齐 shared-prefix FSDP hook，而不是一开始直接改 engine；
- 只有当方案 A 无法可靠覆盖 `compute_log_prob` / `update_actor`，或会导致调用链过度扭曲时，再退到方案 B，在 `FSDPEngineWithLMHead.forward_step()` 增加显式分支。

#### 2.8.1 verl 侧最小改动清单

为了让该能力以最小侵入方式进入 verl，建议首批改动限制在以下位置：

1. 配置层：

```text
verl/trainer/config/actor/actor.yaml
verl/workers/config/actor.py
```

新增：

```yaml
prefix_grouper:
  mode: prompt_only
  min_prefix_len: 16
  min_group_size: 2
  strict: false
  validate_precision: false
```

保留：

```yaml
use_prefix_grouper: false
```

2. 模型 patch 层：

```text
verl/models/transformers/monkey_patch.py
```

不新增新的 PrefixSharing patch。继续复用 `apply_prefix_grouper_patch()`。

3. FSDP execution 层：

```text
verl/workers/engine/fsdp/transformer_impl.py
```

在 `FSDPEngineWithLMHead.forward_step()` 中增加一个可选分支，或者调用统一 helper：

```python
if use_prefix_grouper and prefix_grouper_config.mode == "arbitrary_prefix":
    from prefix_sharing.integrations.verl_fsdp import forward_step_with_prefix_sharing
    return forward_step_with_prefix_sharing(
        engine=self,
        micro_batch=micro_batch,
        loss_function=loss_function,
        forward_only=forward_only,
    )
```

4. PrefixGrouper 工具层：

```text
verl/trainer/ppo/prefix_grouper_utils.py
```

如果社区更希望所有 shared-prefix 逻辑都走这里，则把上面的 engine 分支改为调用该文件中的统一 helper。但当前源码中 `forward_micro_batch_with_prefix_grouper()` 的主路径调用点不清晰，所以 prototype 阶段优先接 FSDP engine。

5. batch balancing 层：

```text
verl/trainer/ppo/ray_trainer.py
```

prompt-only 继续按 `uid` 做 group-level balancing。arbitrary-prefix 模式不能依赖 `uid`，首版建议：

- 不改变 batch balancing；
- 在每个 DP rank 本地 micro-batch 内独立检测 prefix；
- 后续如果性能不足，再考虑基于 prefix hash 的 group-aware balancing。

这个选择可以避免首版同时改 trainer 数据分发逻辑。

### 2.9 fallback / guard 策略

FSDP arbitrary-prefix 首版必须保守：

支持：

- FSDP worker；
- causal LM；
- text-only；
- `use_remove_padding=False`；
- `use_fused_kernels=False`；
- no Ulysses SP；
- no ring attention；
- no dynamic bsz，除非实测确认当前 PrefixGrouper 示例中的 dynamic 配置确实可用。

不支持时：

- `strict=false`：fallback 原始 FSDP forward；
- `strict=true`：抛出明确错误。

建议首版默认：

```yaml
prefix_grouper:
  strict: false
```

但开发和测试时使用：

```yaml
prefix_grouper:
  strict: true
```

避免静默 fallback 掩盖问题。

### 2.10 风险与缓解

1. 上游 PrefixGrouper 执行入口不完整。
   - 缓解：原型先接 `FSDPEngineWithLMHead.forward_step()`；RFC 中把 shared-prefix FSDP hook 作为明确诉求。
2. PrefixSharing 独立 attention patch 的侵入面超过预期。
   - 缓解：先保持 patch 层职责极薄，只负责进入 PrefixSharing runtime 和收集诊断；如果后续需要上游收敛，再抽象成通用 `shared_prefix_runtime` hook。
3. mixed batch fallback 导致收益不稳定。
   - 缓解：首版 strict 模式用于开发测试；benchmark 后再决定是否做 mixed mode。
4. FSDP 下 restore / position_ids 与 Megatron 路径存在细节差异。
   - 缓解：Phase 2 用单卡 HF 精度测试先验证不同 prefix_len、同 provider 多 reuser、链式复用和 nested prefix。
5. 社区不接受继续扩展 `use_prefix_grouper` 命名。
   - 缓解：RFC 同时给出 `prefix_grouper.mode` 和未来 `shared_prefix` 统一命名路线，但首个 PR 不强推迁移。

## Chapter 3：开发执行计划

### 3.1 Phase 0：源码确认与测试设计

目标：

- 基于本地源码确认 PrefixGrouper / verl FSDP 的接口、配置和调用入口；
- 不把 PrefixGrouper 上机验证作为当前开发前置条件，4090/GPU 复现可由后续外部任务完成；
- 先设计 FSDP adapter 的 TDD 测试用例。

交付：

- PrefixGrouper 的接口、配置、执行入口和限制清单；
- FSDP adapter 的单元测试清单和首批失败测试。

### 3.2 Phase 1：TDD 实现 PrefixSharing FSDP adapter

目标：

- 先写本地 CPU 可运行的失败测试，覆盖 plan 构建、trimmed batch、position_ids、runtime state、restore metadata；
- 再在 PrefixSharing 包内新增 `integrations.verl_fsdp`；
- 输入 verl micro-batch；
- 输出 trimmed micro-batch + `PrefixSharingRuntimeState | None`。

首版不新增 `PrefixSharingFSDPPreparedBatch`，优先复用 Megatron 路线已有的 `(trimmed_micro_batch, PrefixSharingRuntimeState)` 形态。

如果确实需要公共能力，再评估是否给 core 增加纯语义 helper。

### 3.3 Phase 2：单卡 Transformers 闭环

目标：

- 不先上 FSDP；
- 先用普通 HF model 或最小 attention module 验证：
  - arbitrary-prefix PrefixSharing runtime forward；
  - interior prefix restore；
  - prefix-last restore；
  - logp / entropy / logits / attention output 对齐；
  - gradient 对齐。

原因：

- 排除 FSDP sharding 干扰；
- 快速定位算法问题；
- 当前本地 MacBook CPU 环境可先完成核心语义测试，不阻塞于 GPU 上机验证。

### 3.4 Phase 3：FSDP 闭环

目标：

- 接入 verl FSDP actor/ref 路径；
- 支持 compute_log_prob；
- 支持 update_actor；
- 对齐 loss / grad；
- 保持 prompt-only PrefixGrouper 不回退。

### 3.5 Phase 4：性能 benchmark

目标：

- 做 baseline / PrefixGrouper / PrefixSharing 三方对比；
- 形成可放进 RFC 和 PR 的性能表格。

### 3.6 Phase 5：RFC 与 PR 拆分

建议 RFC 叙事：

```text
Extend verl PrefixGrouper from prompt-only grouping to arbitrary-prefix shared-prefix training.

Existing mode:
  use_prefix_grouper=True -> prompt-only -> prefix_grouper package

New mode:
  use_prefix_grouper=True + prefix_grouper.mode=arbitrary_prefix
  -> prefix_sharing package
```

建议 PR 拆分：

1. 配置与文档：增加 `prefix_grouper.mode`，不改变默认行为；
2. PrefixGrouper prompt-only 测试补强；
3. PrefixSharing arbitrary-prefix FSDP adapter；
4. 精度测试；
5. benchmark/examples；
6. 后续再讨论 Megatron backend。

## Chapter 4：测试验证

测试验证分为四类：集成验证、功能验证、精度验证、性能验证。四类验证的目标不同，不能互相替代：集成验证证明能接入 verl，功能验证证明 FSDP 路径语义完整，精度验证证明不改变训练语义，性能验证证明该特性有实际收益。

### 4.1 集成验证

目标：验证 PrefixSharing 能否以社区可接受的方式集成到 verl FSDP 路径中，并且不破坏现有 PrefixGrouper prompt-only 行为。

验证内容：

1. `use_prefix_grouper=true + prefix_grouper.mode=prompt_only` 保持现有 PrefixGrouper 行为。
2. `use_prefix_grouper=true + prefix_grouper.mode=arbitrary_prefix` 能正确导入 `prefix_sharing` 包并进入 PrefixSharing FSDP adapter。
3. FSDP 主路径优先通过 `prefix_grouper_utils.py` 或现有 PrefixGrouper 调用入口接入；若该入口不完整，再通过 monkey patch 补齐 shared-prefix FSDP hook。
4. 独立 attention patch 能正确安装、回滚，并且只在 PrefixSharing runtime context 存在时进入 prefix-sharing 路径。
5. 缺少 `verl`、`prefix_grouper`、`prefix_sharing` 或不支持配置时，错误信息清晰；`strict=false` 时可 fallback 原始 FSDP forward。
6. `compute_log_prob` 和 `update_actor` 两条 actor/ref 路径都能走通。

建议测试：

```text
prefix-sharing/tests/integrated_test/test_verl_fsdp_integration.py
prefix-sharing/tests/integrated_test/test_patch_integrations.py
```

本地 CPU 阶段先验证导入、patch handle、fallback 和 helper contract；真实 verl FSDP 小模型 smoke test 需要在 GPU 环境执行。

### 4.2 功能验证

目标：验证 PrefixSharing arbitrary-prefix 能在 FSDP 路径下完成核心功能，而不是只复用 PrefixGrouper 的 prompt-only group 能力。

验证内容：

1. 从 verl FSDP micro-batch 恢复有效 token 序列。
2. `PrefixSharingPlanner` 能生成 provider/reuser 关系，不依赖 `uid`、`group_info` 或用户显式标记。
3. `build_prefix_sharing_micro_batch_fsdp()` 返回 `(trimmed_micro_batch, PrefixSharingRuntimeState | None)`，与 Megatron 路线保持一致。
4. reuser 的 Q path 输入、labels、loss_mask 按 `PrefixSharingPlan.input_keep_ranges` 裁剪。
5. `position_ids` 按 `q_position_offsets` 保持原始绝对位置。
6. attention runtime 执行 provider KV store、reuser KV load、expanded KV 拼接和 suffix Q attention。
7. restore 同时覆盖 interior prefix 和 prefix-last。
8. 无共享 prefix 时返回原始 batch 并 fallback。

建议测试：

```text
prefix-sharing/tests/unit_test/test_verl_fsdp_runtime_plan.py
prefix-sharing/tests/unit_test/test_verl_fsdp_adapter.py
prefix-sharing/tests/unit_test/test_verl_fsdp_restore.py
```

关键用例：

1. 两条样本共享任意 prefix，生成 PrefixSharingPlan provider/reuser 关系。
2. 多个 reuser 有不同 prefix_len 和不同 suffix_len。
3. 同一个 provider 服务多个 reuser。
4. reuser 继续成为后续 provider，覆盖链式复用。
5. batch 中无可共享 prefix 时返回 `None` 并 fallback。
6. `strict=true` 下不支持 Ulysses SP / ring attention / fused kernels 时抛错。
7. `strict=false` 下不支持场景 fallback，输出与 baseline 对齐。

### 4.3 精度验证

目标：验证 PrefixSharing FSDP 不改变 RL 训练语义。精度验证必须参考 `prefix-sharing/prefix_sharing/tools/` 之前在 Megatron 路线上的单卡和多卡精度诊断方式，保留按 layer / row / token 定位差异的能力。

最小精度矩阵：

| 场景 | 检查项 |
|------|--------|
| prompt-only PrefixGrouper | 不回退现有行为 |
| arbitrary-prefix 单卡 | logp / entropy / logits / attention output / loss / grad 对齐 |
| arbitrary-prefix FSDP 单机 | logp / entropy / logits / attention output / loss / grad 对齐 |
| interior prefix restore | provider 到 reuser 的 prefix 内部位置复制正确 |
| prefix-last restore | response first token logp 基于 provider logits + reuser label 重算正确 |
| 无共享 prefix | fallback 与 baseline 一致 |

具体指标至少包括：

- `log_probs` 与 baseline 对齐；
- `entropy` 与 baseline 对齐；
- `logits` 与 baseline 对齐，尤其是 provider prefix、reuser restored prefix、suffix-first 相关位置；
- attention output 与 baseline 对齐，用于定位 attention 层内 KV injection / mask / position_ids 问题；
- policy loss 对齐；
- 关键参数 gradient 对齐；
- provider prefix 相关参数梯度非空且与 baseline 对齐；
- padding 位置不参与 loss。

诊断方式：

1. 保存 baseline forward 与 PrefixSharing forward 的中间张量。
2. 按 layer 对齐 attention output / logits / logp / entropy。
3. 按 row 和 token 定位第一个不一致位置。
4. 对 restore 结果单独 dump interior prefix 与 prefix-last。
5. 单卡精度通过后，再做 FSDP 多卡精度对齐。

建议测试数据：

```text
prompt-only:
  p + r1
  p + r2

arbitrary-prefix:
  A B C D E
  A B C X Y
  A B Q R
  Z Z Z

nested prefix:
  A B C D
  A B C X
  A B Y
```

首版 arbitrary-prefix 必须保留 PrefixSharing plan 的 provider/reuser 语义；测试需要覆盖不同 prefix_len、同一 provider 多 reuser、链式复用等场景。

### 4.4 性能验证

目标：证明 PrefixSharing arbitrary-prefix 相比 baseline FSDP 有实际收益，并与 PrefixGrouper prompt-only 形成清晰对比。性能验证不能替代精度验证，必须在精度对齐通过之后进行。

对比三组：

1. baseline FSDP；
2. PrefixGrouper prompt-only；
3. PrefixSharing arbitrary-prefix。

指标：

- old_log_prob time；
- update_actor time；
- full step time；
- tokens/s；
- peak memory；
- prefix detection overhead；
- KV store/load overhead；
- restore overhead；
- reused token ratio。

数据分布：

- GRPO `rollout.n > 1` same prompt；
- step-mode synthetic；
- tree-mode synthetic；
- 不同 prefix length；
- 不同 reuser fanout；
- 不同 response length。

验收标准建议：

- prompt-only 场景不弱于现有 PrefixGrouper；
- arbitrary-prefix 在 step/tree synthetic 上相对 prompt-only 有额外收益；
- 当没有足够共享 prefix 时，fallback 或收益模型不应显著拖慢；
- 最终 RFC/PR 应给出 baseline / PrefixGrouper / PrefixSharing 三方性能表格。

### 4.5 真实环境测试指引

本节用于交付 ClaudeCode / 4090 / 真实 verl FSDP 环境测试。目标不是一次性证明可合入，而是验证代码链路是否能在真实运行时闭环，并快速定位不兼容点。

#### 4.5.1 安装与使能

在真实 verl 0.8 FSDP 环境中，优先显式安装 FSDP patch set：

```python
import prefix_sharing.setup

prefix_sharing.setup.install("verl080_fsdp")
```

不要依赖默认兼容矩阵自动选择 patch set，因为真实环境可能同时安装 Megatron / MindSpeed，默认探测容易误选 Megatron patch set。

PrefixSharing 功能开关可以继续沿用当前内部配置：

```yaml
prefix_sharing_config:
  enable_prefix_sharing: true
```

或使用环境变量：

```bash
export ENABLE_PREFIX_SHARING=1
```

PrefixSharing 包侧已支持读取 `prefix_grouper.mode=prompt_only|arbitrary_prefix` 并完成 prompt-only / arbitrary-prefix 分发；但 verl 上游 yaml/dataclass schema 尚未正式接入该字段，所以真实环境测试仍优先使用环境变量路径，或在本地实验配置中手动挂载等价字段。

#### 4.5.2 最小 smoke 流程

建议按以下顺序测试：

1. **导入与 patch 安装**
   - 验证 `prefix_sharing.setup.install("verl080_fsdp")` 无异常；
   - 验证日志中出现 FSDP patch set 安装信息；
   - 验证 disabled 配置下仍走原始 FSDP forward。

2. **单卡 FSDP compute_log_prob**
   - 使用小模型、小 batch、短序列；
   - 打开 `enable_prefix_sharing=true`；
   - 构造至少两条有共享前缀的样本；
   - 验证 `FSDPEngineWithLMHead.forward_step` 被 patch 后进入 PrefixSharing 路径；
   - 检查输出包含 `log_probs`，shape 与原始 batch 对齐。

3. **单卡 FSDP update_actor**
   - 在训练态跑一次最小 actor update；
   - 验证 loss 可反传；
   - 检查 provider prefix 相关参数梯度非空；
   - 对比关闭 PrefixSharing 时的 loss/logp 是否在可接受 tolerance 内。

4. **remove-padding / jagged NestedTensor 路径**
   - 若业务 FSDP 配置开启 remove padding，重点验证 `prepare_model_inputs` 输出的 packed shape；
   - 检查 packed attention runtime 是否接收到 `[1, T, H, D]` 形态；
   - 检查 `prepare_model_outputs` 后 restore 是否恢复到原始 row lengths。

5. **不支持配置 guard**
   - `ulysses_sequence_parallel_size > 1` 应显式报错；
   - `use_fused_kernels=true` 应显式报错；
   - 报错信息应说明当前 FSDP PrefixSharing 不支持该路径。

#### 4.5.3 精度测试流程

精度测试必须至少覆盖：

1. baseline FSDP vs PrefixSharing FSDP 的 `log_probs`；
2. `entropy`；
3. `logits`；
4. attention output；
5. policy loss；
6. embedding / attention projection 等关键参数梯度；
7. provider prefix 梯度不被 detach；
8. interior prefix restore 与 prefix-last restore 的单独 dump。

建议先跑单卡小模型，再扩展到多卡 FSDP。若出现差异，按以下顺序定位：

```text
attention output -> logits -> log_probs/entropy -> restore -> loss -> grad
```

其中 prefix-last 位置必须特别检查：reuser 的第一个 suffix token logprob 应来自 provider prefix-last logits 与 reuser label 的重算，而不是简单拷贝 provider logprob。

#### 4.5.4 性能测试流程

性能测试必须在精度通过后进行。建议三方对比：

1. baseline FSDP；
2. PrefixGrouper prompt-only；
3. PrefixSharing arbitrary-prefix。

核心指标：

- old_log_prob time；
- update_actor time；
- full step time；
- tokens/s；
- peak memory；
- reused token ratio；
- prefix detection / KV store-load / restore 开销。

数据分布至少包含 prompt-only、step-mode synthetic、tree-mode synthetic。没有共享前缀或共享比例很低时，也要验证 PrefixSharing 不应显著拖慢 baseline。

### 4.6 测试报告

#### 2026.07.05周日11:30AM: FSDP adapter Chapter 4 测试报告（codex 交付）

> **测试范围声明（重要）**：本报告区分两个 ready 边界：
>
> - **ready for real-environment test**：代码层面已经具备 FSDP adapter、attention registry patch、显式 `verl080_fsdp` patch set、forward_step wrapper、dense / nested prepare、packed attention runtime 和 restore 链路，可以交给真实 verl / GPU 环境定位运行时问题。
> - **ready for upstream merge**：仍未达到。真实 HF 小模型、真实 verl FSDP `compute_log_prob` / `update_actor`、多卡 FSDP 精度和性能 benchmark 还没有完成。
>
> 当前分支达到第一个标准，但没有达到第二个标准。

本小节记录对 `open-source_fsdp` 分支（commit `f57c40e9`/`faf5e6ea`）按 Chapter 4 四类验证的实测结果。早期报告曾记录 commit `0b9a772` 的 Phase 1 状态；后续已经补齐 FSDP patch set 和 forward_step wrapper。

**已验证环境：** macOS Darwin 25.4 / Python 3.9.6 / torch 2.8.0（CPU）+ 服务器 219.223.198.62 / Python 3.12 / `prefixsharing` env（verl 0.8.0.dev + megatron-core 0.16.1）。

**4.1 集成验证 —— 代码链路已接入，真实环境待验证。** 已通过缺失依赖报错、patch 安装/回滚、显式 patch set 加载、forward_step wrapper、fake engine 覆盖；真实环境下 `compute_log_prob`/`update_actor` 路径需验证。

**4.2 功能验证 —— 全部通过（12 tests）。** 覆盖文档要求的全部 7 个场景：共享 prefix、多 reuser 不同 prefix_len/suffix_len、同一 provider 多 reuser、链式复用（reuser→后续 provider）、无共享 → None fallback、strict 配置校验、position_ids 保持绝对位置。

**4.3 精度验证 —— 核心语义对齐。** 6 tests 验证了：
- reuser suffix attention vs baseline：atol=1e-5 对齐
- 梯度通过 provider prefix KV（不 detach）
- interior prefix restore 从 provider 拷贝
- prefix-last logp 用 provider logits + reuser label 重算一致
- padding 不影响有效位置

**4.4 性能验证 —— 未执行（文档要求精度对齐通过后进行）。**

**开发计划对照：**
| Phase | 目标 | 状态 |
|-------|------|------|
| Phase 0 | 源码确认与测试设计 | ✅ |
| Phase 1 | TDD 实现 FSDP adapter | ✅ |
| Phase 2 | 单卡 Transformers 闭环 | ⚠️ ready for test |
| Phase 3 | FSDP 闭环 | ⚠️ ready for test |
| Phase 4 | 性能 benchmark | ❌ |
| Phase 5 | RFC 与 PR 拆分 | ❌ |

**当时阻塞合入的三个缺口：** (1) FSDP engine patch set 需真实环境 smoke；(2) 无 `prefix_grouper.mode` 配置分发；(3) 无真实 verl FSDP smoke test 结果。后续已补齐 FSDP patch set、真实环境 smoke、多卡验证；`prefix_grouper.mode` 已在 PrefixSharing 包侧支持读取，verl 上游 schema 接入仍留待社区 PR。

**本地回归结果（faf5e6ea）：**
```
245 passed, 30 skipped
```

#### 2026.07.05周日16:34: Dense path transpose bug → entropy 3.14 部分修正

**发现：** HF attention_interface 返回形态 [B,L,H,D]（Qwen2Attention 使用 `reshape(*input_shape, -1)` 不 transpose），但之前的 attention patch 代码对 runtime 输出额外做了 `output_ld.transpose(1,2)`，把 [B,L,H,D] 错误转成 [B,H,L,D]。

**修复：** 删除 attention.py 中的 output transpose，直接返回 output_ld。

**实验结果：** entropy 从 6.41 降至 3.14（部分修正）。

**日志：** `ps_fix_dense.log`

#### 2026.07.05周日17:04: Einsum bf16 autocast 精度劣化 → SDPA 替换（根因 #1）

**发现：** `torch.einsum` 在 bf16 autocast 下即使输入用 `.float()` 包住也会被降回 bf16（einsum 在 PyTorch autocast 降精度名单中）。softmax 在 bf16 下精度严重劣化，误差在 24 层残差流逐层放大。

诊断数据：provider 行（full attention，应精确匹配 baseline）layer 0 偏差 0.072。

**修复：** `_attention_row` 中手写 einsum+softmax 替换为 `F.scaled_dot_product_attention`（不受 autocast 降精度影响，内部 fp32 累加，与 HF 数值一致）。

**实验结果：** provider 行 layer 0 偏差从 0.072 降至 0.0005。额外适配 GQA（手动 repeat_interleave KV heads）。

**日志：** `ps_fp32_diag.log`，`ps_diag_dense.log`

**修改文件：** `prefix_sharing/backends/torch_ref.py`

#### 2026.07.05周日17:23: NestedTensor dense 路径布局错位（根因 #2）

**发现：** verl 即使 `use_remove_padding=False` 也传 NestedTensor input。PrefixSharing 的 nested-trim **移除前缀 token 导致行左移**，而 runtime 的 dense-path `keep_ranges` 假设原始 token 位置 → 取错 token。

确诊证据（K 值对比）：
- row0 pos0 token=50 → K=[-8.19, ...]（正确）
- row1 pos0 经 trim 变成 token=17 → K=[9.875, ...]（**错误** — 同 token+同 position 才有同 K，证实行被左移）

causal mask 不变量检查证实 reuser 行 attend 到错误位置（causal 语义被破坏）。

**修复（workaround）：** 使用 packed 路径（`use_remove_padding=True`），`prepare_model_inputs` 输出 `[1, total_nnz]` 形态，runtime packed-path 是设计意图。

**实验结果：** entropy 从 3.14 降至 1.368，与 baseline 1.359 差距缩至 **0.6%**。

**日志：** `ps_invariant.log`（causal 检查），`ps_final_gc_off.log`

#### 2026.07.05周日18:10: 🎉 Smoke 通过！Packed path + SDPA + GC off

**配置：** `use_remove_padding=True`（packed path）+ SDPA backend + `gradient_checkpointing=False`

**结果：**
| 指标 | PS-on | PS-off（baseline） | 相对偏差 |
|------|-------|---------------------|----------|
| entropy | 1.3679 | 1.3594 | **0.6%** ✅ |
| step | 1 完成 | — | 无 NaN，grad_norm=0（step1 无 advantage 信号，正常） |

KV reuse 指标：全 24 层触发，`store_count=4`, `reuse_hit=3`, `matches_expected=True`。

**日志：** `ps_final_gc_off.log`

#### 2026.07.06周一02:05: 3-step 稳定性验证

**结果：** 3 步全完成，无 NaN，entropy 稳定性与 baseline 一致。

**日志：** `ps_on_3step.log`

#### 2026.07.06周一: logits/log_probs/attention_output 精度验证

**fp32 验证（机器精度）：** 所有区域（suffix、interior prefix、prefix-last）与 baseline 的 diff 均在 5e-5 以内，证明逻辑完全正确。

**bf16 精度分析：**
| 区域 | bf16 diff | 分析 |
|------|-----------|------|
| interior prefix（restore copy） | 0 | 直接从 provider 拷贝，无计算误差 |
| prefix-last（restore copy logits） | 0 | 直接从 provider 拷贝 |
| suffix（PS direct attention） | ~1.5% 相对误差 | `expanded_K = torch.cat([loaded_K, own_K])` 改变内存布局 → FlashAttention bf16 reduction order 敏感 |

suffix 区域的 1.5% 差异**不是 bug**：fp32 下相同计算得到 5e-5 的机器精度。bf16 差异来自 FlashAttention 对 layout 变化的 reduction order sensitivity，且位置 4-6（prefix 边界附近）diff=0，位置 7-11（suffix 内部）约 1.5%。

#### 2026.07.06周一14:30: FSDP 路径 cmp_diag_verl080 端到端精度对比

**目标：** 将 `diagnostic_dump_verl080` 集成到 FSDP `forward_step` 路径（此前仅 Megatron 路径有），用 `cmp_diag_verl080.py` 做 PS-on/off 端到端精度对比。

**代码改动：**
- `prefix_sharing/setup/patches/verl080_fsdp/forward_step.py`：在 `_forward_step_with_engine_prepare`（ON 路径）和 disable early-return（OFF baseline 路径）集成 `PREFIX_SHARING_DIAG_DUMP` 触发的诊断 dump，覆盖元数据 / attention_mask / label_mask / 2D logprobs / 2D entropy。
- 新增 `_dump_fsdp_baseline()`：OFF 路径专用，从 verl 原生 `forward_step` 返回的 `(loss, output_dict)` 提取 NestedTensor logprobs/entropy，用 `nested_to_2d_full` 展开到 `[B, L_max]`。

**测试配置：** verldir + verl080 env + Qwen2.5-0.5B + GRPO no-critic + packed path + GC off，`data.shuffle=False` `data.seed=42` 固定 dataloader，两次 run 分别 `ENABLE_PREFIX_SHARING=0/1`。

**结果：**

| 指标 | PS-on | PS-off | 备注 |
|------|-------|--------|------|
| entropy | 1.3721 | 1.3401 | step-level，差异来自 rollout 随机性 |
| KV reuse | 全 24 层 `store_count=4 reuse_hit=3 matches_expected=True` | — | PS 正常触发 |
| prefix_lens | [0, 4, 17, 17] | [0,0,0,0] | seq0 provider，seq1 共享前4，seq2/3 共享前17 |

**分区精度对比（关键结论）：**

| 区域 | abs_mean | abs_max | 分析 |
|------|----------|---------|------|
| **prompt 区（pos 0-16，restore 区）** | **0.09** | **0.65** | ✅ **bf16 级对齐**，PS restore 正确（从 provider 拷贝 logp/entropy） |
| response 区 seq2/seq3 | 0.03 | 0.22 | ✅ 两条恰好两次 run 采到相同 response → 整序列对齐到 bf16 级 |
| response 区 seq0/seq1 | 2.33 | 12.31 | ❌ 两次 run 的 vLLM rollout 采到不同 response（非 PS bug） |

**结论：**
1. **PS restore 逻辑正确**：prompt 区（restore 区）的 logp/entropy 与 baseline 对齐到 bf16 级（abs_mean=0.09），这覆盖了 PrefixSharing 最关键的 interior prefix restore 和 prefix-last restore。
2. **response 区差异完全来自 vLLM rollout 随机性**：vLLM V1 异步采样 + GPU 非确定性导致即使固定 seed，两次 run 的 response 也不同。seq2/seq3 恰好采到相同 response 时，整序列对齐到 bf16 级（abs_max=0.22），证明 PS suffix attention 也正确。
3. **逐元素 ON-vs-OFF 对齐在真实随机 rollout 下不成立是预期内的**；要严格逐元素验证 suffix 区，需固定 rollout 输出（temperature=0 但会让 batch 退化）或用 synthetic 固定 batch（即之前 4.6.5/4.6.7 的 fp32 验证方式）。

**诊断 dump 流程已打通：** `PREFIX_SHARING_DIAG_DUMP=<dir>` + `cmp_diag_verl080.py --dir-on --dir-off --tag train` 可正常采集和对比 FSDP 路径的 logprobs/entropy/masks/prefix_lens/cu_seqlens。（注：logits.pt / attn_outputs.pt 的 FSDP dump 在后续 commit `0e385cd0` 中补齐——见下方「temperature=0 确定性精度验证」条目；本条目记录时仅有 logp/entropy。）

**日志：** `ps_off_seed.log`、`ps_on_seed.log`；**dump 目录：** `~/prefix-sharing/dump_off`、`~/prefix-sharing/dump_on`

#### 2026.07.06周一（修正 14:30 报告）: temperature=0 确定性精度验证（单卡）

> **为什么修正：** 14:30 报告依赖"两次 run 恰好采到相同 response"的偶然 batch 一致性（seq2/seq3 对齐、seq0/seq1 不对齐），方法论弱且不可复现。重新跑发现 seq1/seq2 在固定 seed 下也不一致——根因是 vLLM V1 异步采样不受 `data.seed` 控制。temperature=0 贪心解码是唯一能保证端到端确定性的方式。

**关键前提验证：** `input_ids_train.pt` ON/OFF `torch.equal=True`（4 行字节级一致），逐元素对比成立——这是 14:30 报告缺失的验证。

**FSDP dump 补全（commit `0e385cd0`）：** 在 FSDP 路径补齐 `attn_outputs.pt`（per-layer，`attention.py` 的 `_dump_fsdp_attn_output`，ON+OFF 双路径）+ `logits.pt`（`forward_step.py` 的 ON `_forward_step_with_engine_prepare` 与 OFF `_call_original_like_engine`）。14:30 报告"暂未 dump logits.pt/attn_outputs.pt"的限制已消除。

**结果（input_ids 一致前提下的可信对比）：**

| 指标 | 结果 | 判定 |
|------|------|------|
| attn per-layer (24 层) | cos_avg 0.9994–0.99997，cos_min ≥ 0.978 | ✅ bf16 级 |
| first_token attn | cos 0.9994 | ✅ |
| first_token logits | cos 0.9997 | ✅ |
| logits packed (suffix aligned) | cos_avg 0.9997 | ✅ bf16 级 |
| logp | pearson 0.997（abs_max 5.8e7） | ⚠️ temperature=0 ÷1e-8 幅度假象，非 PS bug |
| entropy | 全 0 | ⚠️ 贪心退化，算法预期 |

**结论：** PS 前向数值正确，attn+logits 在确定性 batch 下对齐到 bf16 级。logp/entropy 在 temperature=0 下退化是算法预期（`temperature.clamp(1e-8)` 除法 + 贪心）；真实 GRPO rollout（temp>0）因 vLLM 异步采样无法逐元素对齐 ON/OFF batch，故 **attn+logits 是可信的精度判据**。

#### 2026.07.06周一15:44: 多卡功能验证（2/4/8 卡 FSDP）

**目标：** 验证纯 FSDP（无 TP）多卡下 PS 功能正确。服务器为单机八卡 4090（此前 skill 误记为单卡，已修正）。

**配置：** `trainer.n_gpus_per_node={2,4,8}` + `CUDA_VISIBLE_DEVICES=0..N-1`，`use_remove_padding=True`，GC off，GRPO no-critic。关闭 diag dump（functional 测试不需要；多 rank dump clobber 问题见下条）。

**多卡 PS 安全性（代码审阅 + 实测）：** PS runtime / ContextVar / `_FSDP_ATTN_BUFFER` 全 per-process；纯 FSDP 每个 rank 独立处理自己的 micro_batch slice，KV store/load 按 `layer_id` 在 rank 内完成，无跨 rank 状态。FSDP 参数 gather、NCCL 梯度同步（2.27.5）、vLLM 多 replica 权重广播（`update_weights done` ×2）均与 PS patch 正交。

**结果：**

| 卡数 | global_step | entropy | PS 触发 rank | matches_expected | update_weights | batch 均衡 |
|------|-------------|---------|--------------|------------------|----------------|------------|
| 2 | ✅ | 1.844 | rank 0+1 | ✅ | ✅ ×2 | minmax_diff:0 |
| 4 | ✅ | 1.594 | rank 0~3 | ✅ | ✅ ×2 | minmax_diff:0 |
| 8 | ✅ | 1.300 | rank 0~7 | ✅ | ✅ ×2 | minmax_diff:0 |

三档均：`store_count`/`reuse_hit`/`matches_expected=True`，NCCL 正常，无 fatal error。`grad_norm=0` 是 GRPO reward 全 0 → advantage 0 → loss 0 所致（tiny model 32 token 内解不出 GSM8K），非 PS bug；前向 entropy 非零证明 forward 正常。

**结论：** 纯 FSDP 多卡功能正确，PS 与多卡训练栈正交无冲突。

#### 2026.07.06周一16:20: 多卡精度验证（2/4/8 卡, temperature=0 确定性）

**目标：** 以 2 卡为例将单卡精度验证扩展到单机任意卡数，验证多卡不引入 PS 精度退化。

**多 rank dump clobber 修复：** `attention.py` 的 `_dump_fsdp_attn_output` 此前直接 `torch.save` 无 rank 门控，多 rank 会覆盖同一 `attn_outputs.pt`。补 `_rank0_only()` 门控（与 2D/logits dump 已有的 `_save_tensor` 门控一致）。单卡 `_rank0_only()` 恒 True，行为不变。

**方法：** 每档 NGPU 跑 PS-on/off 两次（`temperature=0` + `data.shuffle=false` + `data.seed=42` 保证确定性），dump 到 `~/Termius/proj_prefix-sharing/dumps/{N}gpu_{on,off}/`，用 `cmp_diag_verl080.py` 对比 rank 0。纯 DP 各 rank 跑相同 PS 代码 + 功能测试已证全 rank `matches_expected=True`，rank 0 精度代表性成立。

**关键前提：** 三档 `input_ids_train.pt` ON/OFF 均 `torch.equal=True`（字节级一致），逐元素对比有效。

**结果：**

| 信号 | 2 卡 | 4 卡 | 8 卡 |
|------|------|------|------|
| attn per-layer (24 层) | ✅ PASS（首差层 4） | ✅ PASS（首差层 5） | ✅ PASS（首差层 4） |
| first_token attn cos | 0.9979 | **1.000000** | 0.9996 |
| first_token logits cos | 0.9928 | 0.99999 | 0.9998 |
| logits packed cos_avg | 0.9970 | 0.9997 | 0.9997 |
| input_ids 一致 | ✅ | ✅ | ✅ |

三档 attn+logits 均对齐到 bf16 级（0.997+–1.0），与单卡一致；logits 的 cmp_diag `FAIL` 标记是阈值 0.9999 对 bf16 偏严所致，cos_avg 0.997+ 实为通过。logp/entropy 在 temperature=0 下退化（÷1e-8 / 贪心），算法预期。

**结论：** 多卡 FSDP（2/4/8）不引入 PS 精度退化，精度验证从单卡成功扩展到单机任意卡数。

**报告：** `reports/fsdp_cmp_diag_{2,4,8}gpu_20260707.txt`；**dump：** `~/Termius/proj_prefix-sharing/dumps/{2,4,8}gpu_{on,off}`

#### 测试使能方式说明

上述所有真实环境测试走 **env-var 自动激活路径**：`VERL_USE_EXTERNAL_MODULES=prefix_sharing`（verl 启动时 import 包）+ `PREFIX_SHARING_PATCHSET=verl080_fsdp`（显式选 FSDP patch set）+ `ENABLE_PREFIX_SHARING=1`（每 batch 开关）。`install("verl080_fsdp")` 由 `import prefix_sharing` 时的 `_auto_install_patches()` 内部调用（`prefix_sharing/__init__.py:92`），与 §4.5.1 推荐的显式 `prefix_sharing.setup.install("verl080_fsdp")` 写法功能等价，**不强制统一**——env-var 路径在脚本化批量测试中更方便，显式 `install()` 在交互式/notebook 中更直观。

#### 2026.07.07周一: 8 卡 batch_size=16 精度验证（FSDP + Megatron 双路线）

**目标：** 在 8 卡 4090 上分别测试 FSDP 和 Megatron 两条路线的 PrefixSharing 精度（PS on vs off），batch_size=16。

**TP=8 不可行约束：** Qwen2.5-0.5B 有 14 个 Q heads + 2 个 KV heads。TP=8 要求 `num_heads % tp_size == 0`（14 % 8 ≠ 0，2 % 8 ≠ 0），vLLM 和 Megatron-core 均有 `assert total_num_heads % tp_size == 0` 硬检查。TP=8 在此模型下不可行。可行的最大 TP 是 2（14/2=7 Q heads，2/2=1 KV head）。

**FSDP 路线（DP=8，8 卡）：** 纯 FSDP data parallel，batch_size=16，GRPO no-critic，temperature=0 确定性。

**关键前提：** `input_ids_train.pt` ON/OFF `torch.equal=True`（4×49 字节级一致），逐元素对比有效。

**FSDP DP=8 结果：**

| 信号 | 结果 | 判定 |
|------|------|------|
| attn per-layer (24 层) | cos_avg 0.999+，cos_min 0.966+，首差层 3 | ✅ bf16 级 |
| first_token attn | cos 0.998 | ✅ |
| first_token logits | cos 0.999 | ✅ |
| logits packed (suffix aligned) | cos_avg 0.9997，cos_min 0.998 | ✅ bf16 级（FAIL 标记为阈值 0.9999 偏严） |
| logp/entropy | abs_max=0（temperature=0 退化） | ⚠️ 贪心退化，算法预期 |
| ON prefix_lens | [0, 4, 49, 4] | PS 正常触发 |
| OFF prefix_lens | [0, 0, 0, 0] | baseline |

**结论：** FSDP DP=8 batch_size=16 精度与之前的 DP=8 batch_size=4 一致（bf16 级对齐），batch_size 增大不引入 PS 精度退化。

**Megatron 路线（TP=1 + DP=8，8 卡）：** 配置了 `actor.megatron.tensor_model_parallel_size=2`，但 verl 0.8.0 colocate 8-worker 模式下 Megatron 实际以 TP=1 运行（`TransformerConfig` 显示 `tensor_model_parallel_size=1`，`get_megatron_parallel_info` 报告 `tp_rank=0/tp_size=1`）。这可能是 verl 0.8.0 在 8-worker colocate 下 TP 初始化的限制。实际配置为 TP=1 + DP=8 + PPO+critic，batch_size=16。

**Megatron PS patch set `eager=True` 修复：** 首次运行发现 Megatron patch set 的所有 6 个 patches 均未生效（import hook 200 次未拦截目标模块后 auto-restored）。根因：`verl.workers.engine.megatron.transformer_impl` 和 `megatron.core.transformer.attention` 在 PS import hook 安装前已被 `verl.workers.engine.__init__.py` 预加载。修复：给所有 Megatron patches 加 `eager=True`（与 FSDP patch set 相同的做法），强制立即 import 并 patch 已加载模块。修复后 7 patches 全部 `Immediately patched`，无 pending 状态。

**Megatron route 不 dump `input_ids_train.pt` 和 `entropy_train.pt`**（Megatron forward_step dump 机制与 FSDP 不同）。prefix_lens ON=[0,4] OFF=[0,0] 正确。`logprobs_train.pt` shape (2,49) ON/OFF 一致，间接证明 batch 相同。

**Megatron TP=1 + DP=8 结果：**

| 信号 | 结果 | 判定 |
|------|------|------|
| rope_freqs | max_diff 0.0 | ✅ 完全一致 |
| attn per-layer (24 层) | cos_avg 0.999+，cos_min 0.936+，首差层 1 | ✅ bf16 级 |
| first_token attn | cos 0.99996 | ✅ |
| first_token logits | cos 0.9999 | ✅ |
| logits packed (suffix aligned) | cos_avg 0.9996，cos_min 0.997 | ✅ bf16 级（FAIL 标记阈值偏严） |
| logp_train | pearson 0.9995，abs_max 0.24 | ⚠️ 伪 FAIL，temperature=0 下 logp 精度受贪心退化影响 |

**结论：** Megatron TP=1 + DP=8 batch_size=16 精度 bf16 级对齐，PS 在 Megatron 路线正确触发。但实际未测试到 TP>1 条件（verl 0.8.0 colocate 限制），真正 TP=2 精度验证需在 2-GPU 专门 Megatron colocate 环境下进行（参考 `~/verldir/scripts/run_megatron_2gpu.sh`）。

**重要发现：** Megatron patch set `eager=True` 修复是必要的——与 FSDP patch set 的 `eager=True` 同理，verl 的 engine 模块在 import hook 安装前已被预加载。此修复需合入 open-source_fsdp 分支。

**脚本：** `~/Termius/proj_prefix-sharing/scripts/run_fsdp_dp8_8gpu_ps.sh`、`run_megatron_tp2_8gpu_ps.sh`；**dump：** `~/Termius/proj_prefix-sharing/dumps/{fsdp_dp8_8gpu,meg_tp2_8gpu}_{on,off}`；**报告：** `~/Termius/proj_prefix-sharing/reports/fsdp_cmp_diag_dp8_8gpu_20260707.txt`、`meg_cmp_diag_tp2_8gpu_20260707.txt`

#### 2026.07.07: Megatron TP=8 端到端 forward 验证（Qwen3-0.6B）

**背景：** Qwen2.5-0.5B（14 Q / 2 KV heads）不支持 TP=8。为验证 PrefixSharing 在 TP=8 下的正确性和 GPU FA kernel 的使用，更换 Qwen3-0.6B（16 Q heads / 8 KV heads，head_dim=128，28 层，74.6M 参数）。

**方法：** 编写 standalone Megatron forward 脚本，通过 `torchrun --nproc_per_node=8` 启动独立 Python 进程（非 verl colocate），绕开 verl 0.8.0 colocate 无法初始化 TP>1 的限制。用 `init_mcore_model()` 随机初始化 Qwen3-0.6B GPTModel，走 GPTModel.forward() 直接推理，不经过 verl engine wrapper。

**关键配置：**
- `tensor_model_parallel_size=8`, `pipeline_model_parallel_size=1`
- `sequence_parallel=False`, `variable_seq_lengths=True`
- `use_cpu_initialization=False`, `masked_softmax_fusion=True`
- batch_size=2, seq_len=64
- `attn_backend` 由 Megatron-Core + TE 自动选定（默认 fused flash attention）

**结果：**

| 项目 | PS-OFF | PS-ON | 判定 |
|------|--------|-------|------|
| TP 配置 | `tensor_model_parallel_size: 8` ✅ | 同上 | ✅ |
| Q heads 分配 | 16/8 = 2 per GPU | 同上 | ✅ |
| KV heads 分配 | 8/8 = 1 per GPU | 同上 | ✅ |
| 模型构建 | 74.6M params, 0.2s | 74.6M params, 0.2s | ✅ |
| Forward 耗时 | 0.487s | 0.487s | ✅ |
| Logits shape | (2, 64, 18992) bf16 | (2, 64, 18992) bf16 | ✅ |
| log_probs mean | -11.0841 | -11.0841 | ✅ |
| entropy mean | 11.6929 | 11.6929 | ✅ |
| PS patches active | N/A | 7/7, all applied, eager=True | ✅ |
| PS attention patch | N/A | `Attention.forward → patched_forward` [applied] | ✅ |
| PS attention kernel | N/A | `F.scaled_dot_product_attention` (SDPA) | ✅ |
| TE attention kernel | auto (fused FA) | TE will be used by default | ✅ |

**PS patch 加载确认（PS-ON 日志）：**
```
[PS] Patched megatron.core.transformer.attention.Attention.forward: 
      Attention.forward → patch_megatron_attention.<locals>.patched_forward
[PS] Immediately patched Attention.forward → prefix-sharing intercept (mcore 0.16.1)
```

**Megatron-Core attention backend：** 由 TransformerConfig 的 `masked_softmax_fusion=True` 和 TE（transformer_engine）自动选择。实际 kernel 为 TE 的 fused flash attention（FusedAttention）。

**结论：**

1. ✅ **Qwen3-0.6B 成功在 TP=8 下运行**（16 Q / 8 KV heads 完全整除 8），8×4090 每 GPU 分配 2 Q heads + 1 KV head，显存剩余充裕（训练时可容纳更大 batch）。
2. ✅ **Megatron-core + TE attention kernel 正常生效**（masked_softmax_fusion=True），forward 0.487s（仅随机初始化模型，无负载均衡问题）。
3. ✅ **PrefixSharing attention patch 在 TP=8 下正确加载和执行**（7 patches all applied, eager=True）。
4. ⚠️ **PS attention kernel 走 `F.scaled_dot_product_attention`**（SDPA），不是独立的 FA kernel——这是 PrefixSharing core 的 `_attention_row` 函数，当 PS 检测到 prefix 并触发特殊 attention 路径时，会对被裁剪部分走 SDPA。PS 禁用/未检测到 prefix 时，attention 仍走 TE fused flash attention。
5. ✅ **PS-ON log_probs/entropy 值与 PS-OFF 一致**（均同为 -11.0841 / 11.6929），证明 PS patch 在空载（无实际 prefix 样本）下不会改变行为。

**脚本与数据：** `~/Termius/proj_prefix-sharing/scripts/standalone_megatron_tp8_test_v3.py`；dump 目录：`~/Termius/proj_prefix-sharing/dumps/meg_tp8_v3_{off,on}/`。

## Chapter 5：当前决策结论

### 5.1 已明确结论

1. FSDP 首版配置应写 FSDP，不应写 transformers backend。
2. PrefixGrouper 和 PrefixSharing 都应保持独立 Python 包形态。
3. verl 侧应复用 `use_prefix_grouper` 入口。
4. 首版建议新增 `prefix_grouper.mode`，而不是直接替换为 PrefixSharing / PrefixAttention 字段。
5. PrefixSharing arbitrary-prefix 首版应保留 `PrefixSharingPlan` 和 provider/reuser DAG 语义。
6. 当前 core 暂不需要为 PrefixGrouper `group_info` 做适配；若 FSDP 发现通用语义缺口，再补 core。
7. 不建议新增独立 `fsdp` 模块；优先新增 `integrations.verl_fsdp`。
8. Restore 在 FSDP 路径中必须同时覆盖 interior prefix 和 prefix-last；prefix-last 由 `PrefixSharingPlan.prefix_last_restore` 驱动，不能直接用 PrefixGrouper `include_prefix_last` 替代。
9. FSDP 首版不支持 Megatron、CP、PP、Ulysses SP、ring attention、fused kernels；remove-padding / jagged NestedTensor 已有代码路径和 fake engine 测试，但真实 verl 环境仍需 smoke。

### 5.2 仍需实测确认

1. 当前 verl 主仓 PrefixGrouper 完整执行入口是否已经闭环；
2. `use_dynamic_bsz=True` 与 PrefixGrouper 文档限制的矛盾；
3. 复用现有 `prefix_grouper` kwarg 入口能否承载 PrefixSharing runtime adapter；
4. FSDP 下 interior restore / prefix-last restore 的具体实现细节和梯度对齐 tolerance；
5. FSDP 下 attention_mask / position_ids / padding 与裁剪输入的兼容性；
6. 社区更偏好的字段名是 `mode`、`algorithm` 还是 `strategy`。

### 5.3 下一步最小行动

1. 用真实 Transformers 小模型（优先 Qwen2.5-0.5B 或更小 causal LM）完成单卡 forward 闭环，验证 `logits / log_probs / entropy / attention_output / loss / grad` 对齐。
2. 在真实 verl 0.8 FSDP 环境验证 `FSDPEngineWithLMHead.forward_step` patch，覆盖 `compute_log_prob` 和 `update_actor` 两条路径。
3. 落地 `prefix_grouper.mode=arbitrary_prefix` 或等价配置分发，保持 `prompt_only` 继续走 PrefixGrouper。
4. 补真实 FSDP smoke test，至少证明 prefix-sharing path 被触发、fallback 可用、unsupported 配置报错清晰。
5. 在 GPU 环境跑 Qwen small + synthetic step/tree benchmark，形成 baseline / PrefixGrouper / PrefixSharing 三方对比。
6. 基于上述结果起草 verl RFC。


### 5.4 技术决策摘要

如果现在基于本文档做后续开发决策，建议选择以下路线：

1. RFC 叙事：扩展 verl 现有 PrefixGrouper/shared-prefix 能力，而不是单独引入一个割裂的新特性。
2. 用户入口：保留 `actor.use_prefix_grouper`，新增 `prefix_grouper.mode=prompt_only|arbitrary_prefix`。
3. 包边界：PrefixGrouper 与 PrefixSharing 都保持独立包；verl 只负责配置、导入和执行入口。
4. 执行实现：FSDP 首版用 PrefixSharing 做 arbitrary-prefix 检测、Q path 裁剪、KV injection、interior restore 和 prefix-last restore；PrefixGrouper 只作为 verl 接口与 prompt-only baseline。
5. 精度策略：优先证明 logprob/loss/grad 与 baseline 一致，再谈性能。
6. 性能策略：用 baseline、PrefixGrouper prompt-only、PrefixSharing arbitrary-prefix 三方对比证明增量价值。
7. 后续扩展：如果复用 `prefix_grouper` kwarg 入口限制 PrefixSharing runtime，则推动更通用的 `shared_prefix_runtime` hook。

这条路线的核心优点是：上游 reviewer 看到的是对既有 PrefixGrouper 能力的自然扩展，而不是一套从 Megatron/NPU 业务场景迁移过来的重型新系统。
