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

### 2.10 精度验证方案

必须先做精度，再做性能。

最小精度矩阵：

| 场景 | 检查项 |
|------|--------|
| prompt-only PrefixGrouper | 不回退现有行为 |
| arbitrary-prefix 单卡 | logprob / loss / grad 对齐 |
| arbitrary-prefix FSDP | logprob / loss / grad 对齐 |
| prefix-last restore | response first token logprob 对齐 |
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

精度诊断应参考 `prefix-sharing/prefix_sharing/tools/` 下已有工具思路：先保存 baseline 与 prefix-sharing 的中间张量，再按 layer / token / row 定位差异。FSDP 首版不要求复用 Megatron 诊断工具代码，但应保持同等粒度的对齐能力。

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

### 2.11 性能验证方案

性能 benchmark 必须和 PrefixGrouper 对齐，而不是只和 baseline 对齐。

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
- 当没有足够共享 prefix 时，fallback 或收益模型不应显著拖慢。

### 2.12 首版验收边界

FSDP 首版完成的判定标准如下。后续开发排期可以直接按这些条目拆任务。

必须完成：

1. `mode=prompt_only` 行为不变，现有 PrefixGrouper 示例和测试不回退。
2. `mode=arbitrary_prefix` 能在单卡 HF causal LM 上完成 PrefixSharing runtime forward、logprob 计算和 backward。
3. arbitrary-prefix 的 restored prefix logp / entropy / logits / attention output 与 baseline 对齐，覆盖 interior prefix restore 和 prefix-last restore。
4. arbitrary-prefix 的 loss 与关键参数 gradient 和 baseline 对齐，且 prefix 相关梯度没有被 detach。
5. FSDP actor `compute_log_prob` 和 `update_actor` 至少在小模型上跑通。
6. 不支持场景有明确 guard 或 fallback，不允许静默产生错位输出。
7. 形成 baseline / PrefixGrouper prompt-only / PrefixSharing arbitrary-prefix 三方性能数据。

首版明确不做：

1. Megatron / MindSpeed / NPU 路径合入 verl 社区主 PR。
2. CP、PP、TP/SP packed THD 适配。
3. Ulysses SP、ring attention、remove padding、fused kernels。
4. 跨 DP rank 的 prefix hash balancing。

### 2.13 风险与缓解

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

### 3.7 建议测试清单

PrefixSharing 包内测试：

```text
prefix-sharing/tests/unit_test/test_verl_fsdp_runtime_plan.py
prefix-sharing/tests/unit_test/test_verl_fsdp_logprob.py
prefix-sharing/tests/integrated_test/test_verl_fsdp_adapter.py
```

关键用例：

1. 两条样本共享任意 prefix，生成 PrefixSharingPlan provider/reuser 关系。
2. 多个 reuser 有不同 prefix_len 和不同 suffix_len。
3. batch 中无可共享 prefix 时返回 `None` 并 fallback。
4. response first token logprob 与 baseline 对齐。
5. provider prefix 相关参数梯度与 baseline 对齐。
6. `strict=true` 下不支持 remove padding / Ulysses SP / ring attention 时抛错。
7. `strict=false` 下不支持场景 fallback，输出与 baseline 对齐。

verl 集成测试：

1. `mode=prompt_only` 保持现有 PrefixGrouper 行为。
2. `mode=arbitrary_prefix` 在 FSDP compute_log_prob 路径跑通。
3. `mode=arbitrary_prefix` 在 FSDP update_actor 路径跑通。
4. 小模型单进程和多进程 FSDP 都能通过 logprob/loss smoke test。

建议本地命令按实际仓库测试框架调整，最低要求是先跑 PrefixSharing 包内单测，再跑 verl FSDP 小模型 smoke test。

## Chapter 4：当前决策结论

### 4.1 已明确结论

1. FSDP 首版配置应写 FSDP，不应写 transformers backend。
2. PrefixGrouper 和 PrefixSharing 都应保持独立 Python 包形态。
3. verl 侧应复用 `use_prefix_grouper` 入口。
4. 首版建议新增 `prefix_grouper.mode`，而不是直接替换为 PrefixSharing / PrefixAttention 字段。
5. PrefixSharing arbitrary-prefix 首版应保留 `PrefixSharingPlan` 和 provider/reuser DAG 语义。
6. 当前 core 暂不需要为 PrefixGrouper `group_info` 做适配；若 FSDP 发现通用语义缺口，再补 core。
7. 不建议新增独立 `fsdp` 模块；优先新增 `integrations.verl_fsdp`。
8. Restore 在 FSDP 路径中必须同时覆盖 interior prefix 和 prefix-last；prefix-last 由 `PrefixSharingPlan.prefix_last_restore` 驱动，不能直接用 PrefixGrouper `include_prefix_last` 替代。
9. FSDP 首版不支持 Megatron、CP、PP、Ulysses SP、ring attention、fused kernels、remove padding。

### 4.2 仍需实测确认

1. 当前 verl 主仓 PrefixGrouper 完整执行入口是否已经闭环；
2. `use_dynamic_bsz=True` 与 PrefixGrouper 文档限制的矛盾；
3. 复用现有 `prefix_grouper` kwarg 入口能否承载 PrefixSharing runtime adapter；
4. FSDP 下 interior restore / prefix-last restore 的具体实现细节和梯度对齐 tolerance；
5. FSDP 下 attention_mask / position_ids / padding 与裁剪输入的兼容性；
6. 社区更偏好的字段名是 `mode`、`algorithm` 还是 `strategy`。

### 4.3 下一步最小行动

1. 写一个单卡 HF model 的 arbitrary-prefix PrefixSharing runtime forward 原型；
2. 证明 PrefixSharing interior restore 与 prefix-last restore 下 logp / entropy / logits / attention output 对齐；
3. 实现 `integrations.verl_fsdp` 的最小 batch adapter；
4. 接入 verl FSDP `forward_step` 或现有 `prefix_grouper_utils.py`；
5. 补 logprob / loss / grad 精度测试；
6. 跑一个 Qwen small + synthetic step/tree benchmark；
7. 基于上述结果起草 verl RFC。


### 4.4 技术决策摘要

如果现在基于本文档做后续开发决策，建议选择以下路线：

1. RFC 叙事：扩展 verl 现有 PrefixGrouper/shared-prefix 能力，而不是单独引入一个割裂的新特性。
2. 用户入口：保留 `actor.use_prefix_grouper`，新增 `prefix_grouper.mode=prompt_only|arbitrary_prefix`。
3. 包边界：PrefixGrouper 与 PrefixSharing 都保持独立包；verl 只负责配置、导入和执行入口。
4. 执行实现：FSDP 首版用 PrefixSharing 做 arbitrary-prefix 检测、Q path 裁剪、KV injection、interior restore 和 prefix-last restore；PrefixGrouper 只作为 verl 接口与 prompt-only baseline。
5. 精度策略：优先证明 logprob/loss/grad 与 baseline 一致，再谈性能。
6. 性能策略：用 baseline、PrefixGrouper prompt-only、PrefixSharing arbitrary-prefix 三方对比证明增量价值。
7. 后续扩展：如果复用 `prefix_grouper` kwarg 入口限制 PrefixSharing runtime，则推动更通用的 `shared_prefix_runtime` hook。

这条路线的核心优点是：上游 reviewer 看到的是对既有 PrefixGrouper 能力的自然扩展，而不是一套从 Megatron/NPU 业务场景迁移过来的重型新系统。
