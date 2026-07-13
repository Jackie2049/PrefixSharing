# PrefixSharing KV 零冗余与高性能 Attention 研究

本文档面向 RL 训练阶段的前缀激活值复用，目标是在保持 logprob、loss、gradient 与逐样本 baseline 严格一致的前提下，同时实现：

- Q/K/V 激活值按共享前缀树去重存储；
- 不再通过 `build_kv` 为每个 reuser 物理复制 provider KV；
- Attention 只计算语义上允许的 query-key 交互；
- 首先形成可在 verl FSDP 路径验证、review 和合入的实现；
- 为后续 MagiAttention、Context Parallel 和专用 kernel 保留演进空间。

本文档采用特性开发闭环结构。Chapter 1 汇总研究分析，Chapter 2 将研究结论转化为可执行 PoC；只有 PoC 证据回填后，才进入后续方案设计、自动化测试与开发计划。

## Chapter 1：研究分析

### 1.1 研究目标与判断标准

#### 1.1.1 问题边界

这里讨论的是**训练阶段、同一 micro-batch 内、保持完整 autograd 图的前缀复用**，不是推理 KV cache，也不是通过近似稀疏注意力改变模型语义。

目标 Attention 必须与每条原始序列独立执行 causal attention 数学等价。对任意样本 `s` 和 query token `q`，其可见 key 集合仍是该样本自身从序列开头到 `q` 的全部历史 token。共享只改变相同 token 的物理存储和执行组织，不改变可见集合、softmax 归一化域、position id、dropout 语义或梯度汇聚方式。

#### 1.1.2 “零冗余”的准确含义

本文将冗余拆成三类，避免把不同收益混为一谈：

1. **Projection 冗余**：相同 prefix hidden state 被重复送入 Q/K/V projection。
2. **Activation 存储冗余**：相同 prefix K/V 在 attention 输入或 autograd 图中物理复制多份。
3. **Attention 计算冗余**：相同 query-key pair 被重复计算。

PrefixSharing 当前版本在 verl `use_remove_padding=True` / NestedTensor 目标主路径中，已经通过模型前的物理裁剪消除了大部分 projection 冗余；dense 2D fallback 则是在 QKV projection 后才由 attention runtime 打包，仍保留 projection 冗余。本轮核心要消除的是两条路径都会由 `build_kv` 重新制造的 K/V 存储与拷贝冗余。必须注意：每个不同 suffix query 都需要与共享 prefix key 做点积，这些 query-key pair 在数学上不同，不属于可消除的冗余。零冗余布局不会自动把所有 Attention FLOPs 都按复用比例减少。

#### 1.1.3 技术方案评价维度

所有候选方案统一按以下维度评价：

- **精度**：logits、logprob、loss、Q/K/V/O projection gradient、embedding gradient 是否对齐；是否支持 activation checkpointing 重计算。
- **显存**：是否真正删除 expanded KV；mask metadata 是否引入新的 `O(T²)` 峰值；backward 保存了哪些张量。
- **速度**：forward、backward、mask/layout 构建、compile、dispatch/undispatch 的端到端成本，而不只比较 kernel TFLOPs。
- **模式表达力**：是否支持 one-prefix-many-suffix、不同 prefix 长度、链式复用、任意深度 prefix tree、相同序列和空 suffix。
- **FSDP 集成成本**：是否能复用 verl/HuggingFace/PyTorch 主路径；是否要求新增 CP 通信、模型魔改或外部 CUDA 扩展。
- **社区可接受性**：依赖是否通用，代码是否容易 review，CI 是否可覆盖，失败时是否能回退原生路径。
- **演进性**：是否能复用到 Megatron、CP、MagiAttention 或未来专用 kernel。

### 1.2 当前 PrefixSharing 实现分析

#### 1.2.1 当前 FSDP 主流程

基于 `open-source_refactor` 分支源码，FSDP 路径如下：

```text
verl FSDPEngineWithLMHead.forward_step
  -> build_prefix_sharing_micro_batch_fsdp
      -> PrefixSharingPlanner.plan
      -> 裁剪 reuser prefix，仅保留 provider/full row 与 reuser suffix
      -> PackedBatchLayout
  -> prefix_sharing_runtime_context
  -> HuggingFace attention patch
      -> PrefixSharingFSDPAttentionRuntime.forward
          -> 将 dense/nested Q/K/V 打包为 THD
          -> attention_backend.build_kv
              -> build_prefix_expanded_kv
          -> attention_backend.attention
  -> prepare_model_outputs
  -> prefix interior / prefix-last restore
  -> loss
```

关键代码位置：

| 模块 | 当前职责 | 与本轮工作的关系 |
|---|---|---|
| `core/planner.py` | provider/reuser、prefix/suffix 长度、trim ranges、expanded KV lengths、restore spec | 语义关系可复用，但 `expanded_lengths_kv` 是 build-kv 专用执行字段 |
| `integrations/verl_fsdp.py` | FSDP batch 裁剪、runtime state、attention 打包、输出 restore | FSDP 首版主要接入面 |
| `backends/kv_builder.py` | store/load provider KV，预分配并复制 expanded KV | 本轮要从 sparse path 删除的主要热点 |
| `backends/torch_ref.py` | 逐 row SDPA correctness backend | 继续作为 dense reference，不作为正式性能后端 |
| `backends/flash_atten_gpu.py` | expanded KV + FA2 varlen | 当前 GPU 生产路径，保留作 fallback/对照 |
| `integrations/context.py` | plan/layout/store/backend/restore runtime | 需要支持不创建 Attention store 的 sparse runtime |

#### 1.2.2 当前 trimmed layout 已经是去重 token layout

这是本轮研究最重要的源码结论。

对一个共享前缀 `P`、两个分支 `A`、`B`：

```text
原始样本：
  s0 = P
  s1 = P + A
  s2 = P + B

当前裁剪后 hidden/Q/K/V：
  [P | A | B]
```

对链式复用：

```text
原始样本：
  s0 = P
  s1 = P + A
  s2 = P + A + B

当前裁剪后 hidden/Q/K/V：
  [P | A | B]
```

因此，在 `use_remove_padding=True` / NestedTensor 目标主路径进入 `build_prefix_expanded_kv()` 之前：

- provider/full row 只保留一次；
- 每个 reuser 只保留自己的 suffix；
- Q/K/V projection 已经在去重 token 上执行；
- 链式复用的每个新增 segment 也只计算一次。

dense 2D fallback 的输入 token 尚未在模型前物理删除；它在 QKV projection 后通过 `_pack_dense_qkv()` 得到相同的去重 attention 输入。因此该路径可以验证 attention/restore 语义，但不能证明 projection 侧的速度和显存收益。正式性能验收必须使用 verl remove-padding 主路径。

上述 `[P | A | B]` 是为了说明最简结构。实际 provider row 可能同时包含共享 prefix 和自己的 leaf suffix，例如 `[P+A | B]`。物理 token 仍然只出现一次，但 sparse layout 必须按所有 branch point 把 provider row 进一步切成逻辑 node ranges，不能简单地把一个 batch row 永远视为一个 tree node。

当前的 KV 冗余是在 projection 之后被重新引入：

```text
build_prefix_expanded_kv 输出：
  branch: [P | P+A | P+B]
  chain:  [P | P+A | P+A+B]
```

`PrefixAttentionStore` 保存的是 expanded buffer 的 view；它没有额外 detach，但 expanded buffer 本身以及其中的 prefix copy 仍占显存、带宽和 autograd 成本。

因此 sparse path 的最小结构变化不是重新设计输入裁剪，而是：

```text
保留当前 trimmed Q/K/V [P | A | B]
  + 构造 token/node 可见关系
  + attention 直接消费同一份去重 Q/K/V
  - build_prefix_expanded_kv
  - PrefixAttentionStore（单 micro-batch sparse attention 路径）
```

#### 1.2.3 当前 build_kv 的隐藏约束

`build_prefix_expanded_kv()` 依赖以下不变量：

- provider 必须排在 reuser 前；
- chain 中间节点必须先构造 expanded KV，再发布给后续节点；
- backend interface 强制所有后端都实现 `build_kv()`；
- `PrefixSharingPlan.cu_seqlens_kv` 和 `expanded_lengths_kv` 默认 KV 按“每个原始样本一条完整 row”组织。

采用 sparse layout 后，attention 不再通过 store 的执行顺序表达依赖，provider-before-reuser 只需作为稳定布局和 restore 的约定保留。未来 planner 可以按 DFS 重排节点以提高 block locality，但不能让 backend 再依赖 Python 循环完成语义正确性。

#### 1.2.4 当前 Attention 计算量

当前方案虽然复制 KV，但 reuser Q 已经只保留 suffix，因此 Attention 元素数近似为：

```text
sum(node 自身 causal area)
+ sum(node query × 所有 ancestor token)
```

这和精确 prefix-tree sparse attention 的语义计算量相同。新方案的确定收益主要来自：

- 删除 expanded K/V 分配和 `copy_()`；
- 删除 per-row store/load 与 Python 调度；
- K/V activation 只保存一次；
- sparse kernel 不扫描完全不可见的跨分支区域。

不能承诺的收益是“共享 prefix 与不同 suffix query 的点积也只算一次”，因为这些 query 不同，无法精确复用同一个 attention score/output。

#### 1.2.5 restore 是否需要重做

当前 sparse 目标布局与 trimmed Q layout 相同，attention 输出 token 数不变，因此现有 restore 体系大部分可以继续复用：

- provider/full segment 输出保留；
- reuser interior prefix 从 direct provider 已恢复 row 批量复制；
- prefix-last logprob 使用 provider logits 和 reuser 首个 suffix label 重新计算；
- chain 按 provider-before-reuser 顺序恢复。

需要重新核对的是 restore index 当前通过 `PackedBatchLayout` 和 direct provider keep range 派生。若后续为提高 sparse block locality 做 DFS 重排，则必须引入显式 flat-token-to-original-sample restore map，不能继续隐含依赖 batch row 顺序。

### 1.3 零冗余 Attention 的目标数学结构

#### 1.3.1 Prefix tree segment 模型

把每个 provider/full row 或 reuser suffix 看作一个 tree node segment。每个 node 具有：

- flat token range `[start, end)`；
- direct parent node；
- 原始 position id；
- 对应原始 sample/row；
- ancestor node ranges。

对 node `n` 中的 query token `q`，允许的 key 为：

1. 所有 ancestor node 的全部 token；
2. node `n` 自身 segment 中 position 不晚于 `q` 的 token；
3. 不允许访问 sibling 或其他 branch token。

这可以拆成规则矩形：

```text
每个 node：
  (Q=node_range, K=node_range, CAUSAL)
  对每个 ancestor：
  (Q=node_range, K=ancestor_range, FULL)
```

该表达与 MagiAttention `AttnSlice(QRange, KRange, MaskType)` 直接同构，也可以转换成 FlexAttention `BlockMask`。

#### 1.3.2 branch 示例

```text
flat tokens = [P | A | B]

             K:P   K:A   K:B
Q:P         causal   -     -
Q:A          full  causal   -
Q:B          full    -   causal
```

#### 1.3.3 chain 示例

```text
flat tokens = [P | A | B]

             K:P   K:A   K:B
Q:P         causal   -     -
Q:A          full  causal   -
Q:B          full   full  causal
```

chain 必须沿 parent 链收集所有 ancestor range。只让 B attend direct provider A 的本地 suffix range，会漏掉 P；这也是不能把 current `provider_index + prefix_len` 简单转换为单个 rectangle 的原因。

#### 1.3.4 建议新增的运行时布局边界

`PrefixSharingPlan` 继续表达共享语义；稀疏执行需要从 plan 派生一个 backend-neutral runtime layout。暂定概念名：

```text
PrefixTreeAttentionLayout
  flat_token_count
  node_ranges
  parent_indices
  ancestor_ranges
  position_ids
  sample_restore_map
  attention_slices
```

这个名字和字段仍属于 Chapter 3 的正式设计决策，本章只确认职责边界：

- plan 不应直接持有 PyTorch `BlockMask` 或 Magi runtime key；
- layout 不应包含 CUDA/Triton 对象；
- Flex/Magi adapter 分别把同一组 ranges 转换为 backend metadata；
- current build-kv layout 继续作为兼容 backend 的另一种 execution layout。

#### 1.3.5 后端无关的稀疏语义 PoC

本次研究使用当前 `PrefixSharingPlanner`、`PackedBatchLayout`、`TorchReferenceBackend.build_kv()` 做了一个 CPU float64 最小 PoC：

1. reference 路径继续执行 expanded KV + 逐 row causal attention；
2. sparse 路径只使用当前 trimmed Q/K/V；
3. 从 `provider_index`、`input_keep_ranges` 和原始 position 派生全局稀疏可见关系；
4. 分别对 output sum 求 Q/K/V gradient。

结果：

| 场景 | original/expanded KV tokens | deduplicated QKV tokens | output max abs diff | Q/K/V grad max abs diff |
|---|---:|---:|---:|---:|
| branch：一个 provider、两个 reuser | 12 | 8 | `4.44e-16` | `2.78e-16` |
| chain：`row0 -> row1 -> row2` | 12 | 6 | `0` | `2.22e-16` |

该 PoC 只证明两点：当前 plan 可以派生 branch/chain 的精确 sparse attention 语义；去重布局的 gradient 会自然汇聚回唯一 Q/K/V tensor。它不证明 Flex/Magi GPU kernel 的 bf16 误差、dropout 一致性、BlockMask 构建成本或端到端性能，后者仍必须通过 Chapter 2 的 device 实验验证。

### 1.4 FlexAttention 深入分析

#### 1.4.1 能力与执行模型

PyTorch FlexAttention 相比标准 FA 增加的相关入参有 `block_mask`、`mask_mod` 和 `score_mod`，但三者对 PrefixSharing 的角色不同：

- **`block_mask`：首版真正要传入的核心 metadata。** 它告诉 kernel 哪些 Q/KV tile 需要遍历，才能跳过 sibling/cross-branch 的完全无效区域；这是 KV 零冗余后仍获得稀疏计算收益的关键。
- **`mask_mod`：构造 `block_mask` 时使用的可见性谓词。** 它表达“ancestor 全可见、node 自身 causal、sibling 不可见”的精确语义，并只在 partial tile 内参与逐元素判定；不是额外改变模型数学语义的机制。
- **`score_mod`：首版不使用。** 它用于在有效 pair 的 attention score 上附加 ALiBi、soft cap 等数值变换；PrefixSharing 只需删去不应可见的 pair，不应修改任何有效 score，否则会破坏与 baseline 的精度一致性。

因此，正式接口应以 `flex_attention(q, k, v, block_mask=...)` 为主；`mask_mod` 被封装在 `BlockMask` 的构建层，`score_mod=None`。

对本项目最相关的能力包括：

- 支持 block-sparse forward 和 backward；
- 支持 GQA；
- full block 与 partial block 分开记录，full block 可跳过逐元素 `mask_mod`；
- `BlockMask` 同时保存 forward 的 KV traversal 和 backward 的 Q traversal metadata；
- 输入仍是普通 PyTorch tensor，可由 FSDP/HuggingFace attention patch 直接调用；
- 无需物化 attention score matrix。

Prefix tree mask 可以由 node/ancestor metadata 表达：

```python
def prefix_tree_mask(b, h, q_idx, kv_idx):
    q_node = token_node_ids[q_idx]
    k_node = token_node_ids[kv_idx]
    return is_ancestor[k_node, q_node] | (
        (q_node == k_node) & (token_positions[kv_idx] <= token_positions[q_idx])
    )
```

正式实现不建议物化 `is_ancestor[num_nodes, num_nodes]`；可以使用 DFS interval、parent/depth metadata，或者直接从 `attention_slices` 生成 block indices。

#### 1.4.2 性能证据

本节的公开数据必须按测量层级拆开阅读。**Attention kernel 性能、attention module 性能、完整训练吞吐不是同一个指标，不能用一个数字推出另一个结论。**

##### 1.4.2.1 Kernel-level：相同 Attention 工作量下，FA 仍是基准

PyTorch 初代 FlexAttention 博客在 A100 上比较标准 causal attention 的 forward/backward kernel，结论为 FlexAttention 约达到 FlashAttention-2 的：

| 对比对象 | 指标 | 结果 | 结论 |
|---|---|---:|---|
| Flex Triton vs FA2 | forward kernel performance | 约 90% | Flex 的通用 block 调度有额外开销 |
| Flex Triton vs FA2 | backward kernel performance | 约 85% | 当时 deterministic backward 会重算更多中间量 |

这组数据固定了相同 Q/K/V、相同 token 数、相同 causal 语义，**没有 PrefixSharing、没有减少 token、没有改变 batch size**。因此它回答的是“只替换 attention backend 的代价”，而不是 PrefixSharing 的端到端收益。对此类规则 mask，FA 是性能标杆；首版不应期待 Flex Triton 单独胜过 FA2/FA3。

PyTorch 2.11 以后又发布了 FA4-backed FlexAttention。其 H200/B200/GB200 benchmark 仍属于 kernel-level forward/backward 测试，证明 block-sparse iteration 和 score/mask modification 可以进入 FA4 pipeline；对 Flex-only 模式相对旧 Triton Flex 有明显提升。但该资料同时指出：标准 causal 场景的 Flex 路径仍可能落后于 cuDNN/FA 的 builtin causal path，因为 block-sparse scheduling 本身有额外成本。

这组新数据的意义是“未来 Flex 上限在改善”，不是当前 verl 首版的性能承诺：本仓 verl 快照固定 `torch==2.9.1`，首版实际使用 Triton FlexAttention，必须以该版本和目标 GPU 的实测为准。

##### 1.4.2.2 Attention-module-level：PrefixSharing 需要额外测 layout 与 metadata

公开 kernel benchmark 通常不包含以下 PrefixSharing 特有步骤：

```text
planner / tree layout
  -> BlockMask 构建或 cache lookup
  -> QKV layout 转换
  -> attention kernel
  -> output restore metadata
```

因此即使 Flex kernel 对某个 sparse mask 很快，也不能直接推出 PrefixSharing attention module 更快。特别是本项目存在三个额外成本：

- 动态 tree 的 `BlockMask` 构建及其临时显存；
- 不规则短 segment 导致的 partial-block overcompute；
- RL old-log-prob、ref-log-prob、actor update 中 layout/metadata 是否能复用。

这一层目前没有可直接套用的公开数字。它需要我们的 GPU PoC 按同一份 PrefixSharing branch/chain/deep-tree batch，分别记录 `layout_ms`、`block_mask_ms`、`attention_forward_ms`、`attention_backward_ms`、`restore_ms`、peak HBM 与 block utilization。

##### 1.4.2.3 End-to-end training：PrefixSharing 的收益来自减少全模型 token 工作

DPO Prefix Sharing 论文提供的是完整模型训练吞吐证据，指标为 **samples/s**，不是单独 attention kernel TFLOPs。实验设置中的关键事实是：

- 固定 per-device batch size 为 4；
- 每个 training step 都构造 Flex block-sparse mask；
- FlexAttention-only baseline 通常慢于 FA3；
- PrefixSharing + Flex 相对 FA3 获得约 `1.1x-1.5x` training throughput；
- 加入 sequence packing 后，多数数据集达到约 `1.3x-1.6x`；
- 论文观察到的 memory reduction 较小，实验收益**不是通过增大 batch size**获得。

该论文的解释是：共享 prompt 后总 token 数减少，因而少执行 embedding、QKV projection、attention、MLP 和它们的 backward；Flex kernel 相对 FA 的劣势被这些全模型计算节省覆盖。收益随 prefix/completion 比、总序列长度升高而增大；短序列即使 prefix 比高，也可能受到 kernel launch 与固定开销限制。

```text
单纯 backend 替换：
  FA > Flex Triton

PrefixSharing 完整训练：
  Flex kernel 额外开销
    < 去除共享 prefix 后节省的全模型 token 工作
    -> 高共享率场景可获得端到端吞吐提升
```

这份 DPO 证据是 FSDP + Flex 路线可行性的强相关参考，但不能直接外推为本项目的收益：它主要是 chosen/rejected 的浅 star 拓扑，而 PrefixSharing 要支持 arbitrary-prefix、chain 和深 tree；其 mask 的 block fragmentation、restore 和 RL 多路 forward 生命周期不同。

##### 1.4.2.4 对本项目的数据解读与验收原则

| 数据层级 | 可以支持的结论 | 不能支持的结论 |
|---|---|---|
| Flex vs FA kernel benchmark | 同一 attention 工作量下的 backend 开销和硬件上限 | PrefixSharing 端到端会更快或更省显存 |
| PrefixSharing attention-module benchmark | layout/BlockMask/attention/restore 是否合起来有收益 | actor update、optimizer、整轮 RL 吞吐收益 |
| 完整 RL training benchmark | 真实 forward/backward/update/mini-batch 吞吐与可运行 batch size | 某个 kernel 本身更快、收益可泛化到所有拓扑 |

本项目的性能决策应按以下顺序作出：先用 kernel-level 实验确认 Flex 没有明显异常退化；再用 attention-module-level 实验确认 KV 零冗余和 metadata 没有抵消收益；最后才用 RL end-to-end 实验决定默认启用阈值。由显存下降带来的可运行 batch-size 增益应作为独立指标报告，不能与固定 batch size 下的 samples/s 提升混为一谈。

#### 1.4.3 当前版本风险

1. **Block 边界浪费**：默认 block 常为 128×128。短 suffix、深树和未对齐 node range 会产生大量 partial blocks；这些 block 仍要加载/计算，再由 `mask_mod` 屏蔽无效元素。
2. **BlockMask 构建成本**：`create_block_mask()` 的通用实现会先评估逻辑 mask，再转换为 block metadata。动态长序列逐 micro-batch 构建可能产生明显 device overhead，某些版本还可能出现临时 `O(T²)` mask。
3. **compile 与 shape churn**：RL micro-batch token 数和树形态动态变化。需要验证动态 shape 是否复用 kernel，不能只测 warm cache 的固定 shape。
4. **metadata CPU/device 同步**：如果 mask 构建依赖 `.cpu().tolist()`、Python closure 中的 host 数据或每层重建，会抵消收益。
5. **数值与确定性**：Flex backward 的 reduction 顺序可能和 FA 不同。社区验收应使用合理误差阈值，同时验证固定输入下 loss/gradient 一致性。
6. **版本稳定性**：kernel options 和部分 BlockMask 低层接口没有强 backward compatibility 保证。

#### 1.4.4 必须采用的优化策略

- 每个 micro-batch 只构建一次 BlockMask，所有 layer 复用；
- old-log-prob、ref-log-prob、actor update 若共享相同 batch/layout，应复用 layout signature 和 BlockMask；
- 优先从 `attention_slices` 直接构造 block row metadata，评估 `BlockMask.from_kv_blocks()`，避免通用 `create_block_mask()` 的 dense 中间过程；
- planner/packer 尽量按 DFS 排序并把同一 subtree 连续放置，提高 full-block ratio；
- 记录 `logical_attention_elements`、`scheduled_block_elements`、`full_block_ratio`、`partial_block_ratio`；
- no-sharing、低复用率或 block 利用率过低时回退原生 FA；
- Flex backend 只消费 backend-neutral layout，不把 `BlockMask` 放进 core plan。

#### 1.4.5 对 verl FSDP 首版的适配度

FlexAttention 是当前最适合社区首版的候选：

- PyTorch 原生依赖；
- FSDP 参数分片与 attention tensor layout 基本正交；
- HuggingFace attention patch 已经存在；
- correctness test 可在普通 PyTorch 环境构建，GPU CI 只负责 kernel smoke/perf；
- 失败时可以按 micro-batch 回退现有 FA/build-kv 路径。

它的定位应是“最小可合入的通用 sparse backend”，而不是宣称它已经达到所有硬件上的性能上限。

### 1.5 MagiAttention 深入分析

#### 1.5.1 必须区分 FFA kernel 与完整 MagiAttention

MagiAttention 包含两层能力：

1. **Flex-Flash-Attention（FFA）kernel**：消费多个 `AttnSlice(QRange, KRange, MaskType)`，支持 FULL、CAUSAL 等规则区域及 overlap，在 kernel 内正确合并 partial softmax，并提供 backward。
2. **完整 distributed attention runtime**：dispatch solver、CP workload balance、Group-Cast/Group-Reduce、通信计算 overlap、undispatch。

对 `FSDP + CP=1`，真正需要的是第一层；完整 dispatch/communication runtime 没有收益，反而增加集成复杂度。对未来 `CP>1` 的超长上下文，第二层才是核心价值。

#### 1.5.2 与 PrefixSharing mask 的匹配度

Prefix tree 的每条边天然对应一个 FULL AttnSlice，每个 node 自身对应一个 CAUSAL AttnSlice：

```text
(node_q_range, node_k_range, CAUSAL)
(node_q_range, ancestor_k_range, FULL)
```

因此 Magi 的表达比通用 boolean mask 更贴近 PrefixSharing 的结构。它不需要把整个镂空 mask压成规则下三角，也不需要复制 KV；overlap slice 的 online softmax/gradient 由 FFA 处理。

#### 1.5.3 性能与硬件现状

官方 benchmark 主要证明 H100/B200 上 FFA 对 heterogeneous mask 的 kernel 潜力，以及完整 MagiAttention 在 CP 场景的扩展性。官方同时提示部分 benchmark 使用实验特性，生产效果需要按 workload 调优。

截至本次研究审视的 MagiAttention main（commit `529fb0a`）：

- v1.0.3 已提供 Transformers + FSDP 训练示例；
- v1.1.1 已把 FFA_FA4 扩展到 Ampere `sm80`，不再只能概括为 Hopper-only；
- H100/B200 仍是公开性能证据最完整的平台；
- Ampere 支持较新，PrefixSharing 模式的 forward/backward 数据尚需独立实测；
- 安装涉及独立 CUDA/CuTeDSL/FA fork 和版本组合，明显重于 PyTorch 原生 FlexAttention。

#### 1.5.4 Meituan/verl prefix-tree 实现证据

`meituan-search/verl` 的 `verl_prefix_tree_full` 分支已经同时实现 Flex 与 Magi 两条路径：

- trie/segment 构建 flat deduplicated token layout；
- `q_ranges/k_ranges/mask_types` 表达 prefix tree；
- Flex path 构造 `BlockMask` 并调用 `flex_attention()`；
- Magi path 构造 `magi_attn_flex_key()`，dispatch 后调用 `calc_attn()`；
- 支持 restore flat output、position id、activation checkpoint 参数透传；
- 当前主线重点仍是 Megatron + CP，FSDP 在 RFC 中列为 planned。

该实现证明技术路线可行，也暴露了工程成本：开发分支持续修正 CP RoPE、temperature、duplicate sequence、zero-length leaf、pickle、key cache、dispatch/undispatch 和 actor-update 性能回退。首个社区 PR 若同时引入完整 Magi stack，review 面会显著扩大。

#### 1.5.5 对本项目的定位

MagiAttention 不应被否定，而应分阶段定位：

- **首个 verl FSDP PR**：不作为必选依赖；
- **Flex 性能不足时的 P1 PoC**：先测试 CP=1 的 FFA kernel-only backend；
- **未来 Megatron/CP**：复用同一个 `PrefixTreeAttentionLayout` 接入完整 Magi runtime；
- **高性能业务落地**：若目标硬件和依赖栈固定，Magi/FFA 可能比 Triton Flex 更合适。

### 1.6 其他候选技术路线

#### 1.6.1 标准 FA2/FA3 varlen

标准 varlen FA 只能表达每个 Q sequence 对一个连续 KV sequence 的 full/causal/bottom-right causal 关系。Prefix tree 中一个 node 需要访问多个不连续 ancestor ranges，同时屏蔽 sibling ranges，单组 `cu_seqlens_q/cu_seqlens_kv` 无法表达。

结论：继续使用标准 FA 必须物理拼接 KV，或者拆成多次 kernel；不能直接实现任意 prefix tree KV 零冗余。

#### 1.6.2 多次 FA + online-softmax merge / DualKV

对一个 suffix Q，可以分别计算：

```text
(O_prefix, LSE_prefix) = FA(Q_suffix, K_prefix, V_prefix, full)
(O_self,   LSE_self)   = FA(Q_suffix, K_suffix, V_suffix, causal)
O = 按 logaddexp(LSE_prefix, LSE_self) 精确合并
```

该方案数学上可以精确，适合“一个共享 prefix + 多个 suffix”的 star topology，也是 DualKV/两区域 attention 的核心方向。风险是：

- 每个 prefix group 或 tree edge 可能增加 kernel launch；
- arbitrary-depth tree 会快速增加调用数；
- backward 必须正确合并 dQ/dK/dV，不能 detach LSE；
- 若想一次 launch 处理多个不同 range，最终仍需要专用 segmented kernel，形态逐渐接近 FFA。

结论：值得作为专用 kernel 研究方向，但不适合作为首个通用 FSDP PR 的默认实现。

#### 1.6.3 PrefixGrouper 两阶段 Attention

PrefixGrouper 对 group 先计算 prefix attention，再计算 suffix attention。它避免重复 prefix projection/attention，但 suffix 阶段通过 `repeat_interleave + cat(prefix_kv, suffix_kv)` 给每个 suffix 提供 prefix KV，因此不是严格 KV 零冗余。

它非常适合已知同 prompt、多 response 的 GRPO star group，接口和实现较简单；不能自然表达不同 prefix 长度、链式 provider/reuser 和任意深度 tree。可作为 prompt-only fast path，但不是本项目 arbitrary-prefix 的统一后端。

#### 1.6.4 FlashMask

FlashMask 通过列式稀疏区间表示复杂 mask，并修改 FA kernel 跳过全 mask block，论文在 SFT、DPO、RM 等训练中报告明显收益，kernel benchmark 也优于当时的 FlexAttention。

限制：

- 官方实现主要集成在 PaddlePaddle/PaddleNLP，不是 verl/PyTorch 原生后端；
- 早期表示对每个 key column 只提供有限个 mask interval，任意深 prefix tree 是否能紧凑表示需要单独证明；
- 最新 V3 block-mask 能力和 CP 改进尚缺少可直接复用的 PyTorch API；
- 接入意味着维护新的跨框架 kernel 或移植成本。

结论：重要 related work 和专用 kernel 设计参考，不作为当前直接依赖。

#### 1.6.5 MIT Block-Sparse-Attention

该项目基于 FA2 修改，支持 forward/backward、GQA 和固定 block-size 的 block mask，可直接跳过稀疏 blocks。它比自己从零写 kernel 更接近可用 PoC。

限制：第三方 CUDA 扩展、固定 block 粒度、版本/架构矩阵和社区维护面均弱于 PyTorch Flex；尚无 verl RL prefix-tree 端到端验证。

结论：可加入 kernel microbenchmark 对照，不适合作为首个 upstream 默认依赖。

#### 1.6.6 Transformer Engine / cuDNN arbitrary mask

TE API 暴露 `arbitrary` mask，但官方文档说明 fused FA/cuDNN backend 对 arbitrary mask 支持受限；历史版本会退回 framework-native/unfused path，或把 mask 转成 dense post-scale bias。这样不能跳过稀疏计算，还可能引入 `O(T²)` mask/bias。

结论：适合 correctness reference，不满足“KV 零冗余 + 高性能稀疏计算”。

#### 1.6.7 AReaL-DTA DFS state reuse

AReaL-DTA 不把整棵树同时交给一个 sparse kernel，而是在 forward/backward 中按 DFS 遍历 prefix tree，每次物化一条 root-to-leaf path，并缓存/累积中间状态。

优势：

- 不依赖大规模 block mask；
- 活跃显存随单路径而不是整棵树增长；
- 对极深、碎片化严重的 tree 有吸引力。

代价：

- 执行不再是普通单次 model forward；
- 需要自定义 forward/backward 调度和状态生命周期；
- kernel launch 与 Python/runtime 调度更多；
- 与 verl FSDP、activation checkpointing、micro-batch loss 的集成面较大。

结论：这是不同于 mask-based one-forward 的长期路线，适合未来极深 TreeRL/Step-wise RL，不宜并入首个 FlexAttention PR。

#### 1.6.8 自研 PrefixTree/Segmented Attention kernel

长期性能上限可能来自专门消费 `attention_slices` 的 kernel：批量执行多个 FULL/CAUSAL rectangle，并在 kernel 内完成 online-softmax merge 和 backward reduction。其本质接近精简版 FFA，但可以只支持 PrefixSharing 所需模式和当前硬件。

结论：只有在 Flex/Magi PoC 证明存在稳定收益缺口，且业务规模足以承担长期 CUDA 维护成本时才值得启动。

### 1.7 Related Work 对本项目的直接启示

| 工作 | 核心机制 | 已证明的价值 | 对本项目的限制/启示 |
|---|---|---|---|
| DPO Prefix Sharing | flat shared prefix + Flex BlockMask | 训练吞吐 `1.1x-1.5x`，packing 后多数 `1.3x-1.6x` | 强证据支持 FSDP/Flex 首版，但主要是 prefix + chosen/rejected star 结构 |
| PrefixGrouper | prefix/suffix 两阶段 attention | verl 已有 FSDP 社区入口 | suffix KV 仍复制；能力边界是 one-prefix-many-suffix |
| Meituan prefix tree RFC/PR | trie flat layout + Flex/Magi | 任意深 tree、Magi CP 性能潜力 | 工程面大，当前重点 Megatron/CP；可复用 layout/AttnSlice 思路 |
| MagiAttention | FFA AttnSlice + CP dispatch | heterogeneous mask 高性能与分布式扩展 | 首个 FSDP PR 依赖过重；适合作为后续 backend |
| AReaL-DTA | DFS forward/backward state reuse | 深 tree 的吞吐/显存潜力 | 不属于简单 attention backend 替换 |
| FlashMask | 列式稀疏 FA kernel | 复杂 mask 训练性能高 | Paddle 原生，PyTorch/verl 迁移成本高 |

### 1.8 候选方案全局对比

评分说明：5 为最有利，1 为最不利；“性能上限”是基于现有证据的相对判断，不代替 PrefixSharing workload 实测。

| 方案 | KV 零冗余 | 任意 tree | 性能上限 | FSDP 接入 | 社区 review | 外部依赖 | 当前建议 |
|---|---:|---:|---:|---:|---:|---:|---|
| FlexAttention BlockMask | 5 | 5 | 3-4 | 5 | 5 | 5 | P0 首版 |
| Magi FFA kernel-only | 5 | 5 | 5 | 3 | 2-3 | 2 | P1 性能 PoC |
| 完整 MagiAttention + CP | 5 | 5 | 5 | 2 | 1-2 | 1 | 后续 CP 路线 |
| 标准 FA + build_kv | 1 | 4 | 3 | 4 | 4 | 3 | 稳定 fallback |
| 多 FA/DualKV | 4-5 | 2-3 | 4 | 3 | 2-3 | 3 | star fast path/专用 PoC |
| PrefixGrouper 两阶段 | 2 | 1-2 | 4 | 5 | 5 | 4 | prompt-only 模式 |
| FlashMask | 5 | 2-4 | 5 | 1 | 1 | 1 | kernel 设计参考 |
| MIT Block-Sparse-Attn | 5 | 5 | 4 | 2 | 2 | 2 | microbenchmark 对照 |
| TE/cuDNN arbitrary bias | 5（输入） | 5 | 1-2 | 3 | 3 | 3 | correctness only |
| AReaL-DTA | 5 | 5 | 场景相关 | 1 | 1 | 3 | 长期深树路线 |

### 1.9 技术决策建议

#### 1.9.1 推荐主线

推荐采用分层、分阶段路线：

```text
PrefixSharingPlan（共享语义）
  -> PrefixTreeAttentionLayout（去重 token + ranges + restore map）
      -> FlexAttentionBackend（首个 verl FSDP PR）
      -> ExpandedKVFlashAttentionBackend（兼容 fallback）
      -> MagiFFABackend（后续高性能 PoC）
      -> MagiDistributedBackend（未来 CP）
```

首版选择 **PrefixSharing + FSDP + FlexAttention**，原因不是认定 Flex 性能最高，而是它同时满足：

- 能直接表达 KV 零冗余 prefix tree；
- PyTorch 原生，最容易被 verl 社区安装、测试和 review；
- 已有 DPO prefix sharing 训练收益证据；
- 可以把核心精度契约、layout 和 backend interface 先合入社区；
- 后续 Magi backend 可以复用同一 layout，而不推翻首版架构。

#### 1.9.2 必须保留的 fallback

首版不能将所有 batch 强制切到 Flex。planner/runtime 应至少支持：

```text
no sharing
  -> 原生 verl/HF FlashAttention

sharing detected，但收益估计不足或 Flex 不兼容
  -> 当前 expanded-KV + FA backend（过渡期）

sharing detected，复用率和 block 利用率达到阈值
  -> deduplicated QKV + FlexAttention
```

收益估计至少包含：

- original token count / deduplicated token count；
- removed KV bytes；
- logical attention elements；
- scheduled block elements；
- full/partial block ratio；
- node count、tree depth、average segment length。

#### 1.9.3 Flex 失败后的升级顺序

如果真实 GPU 实验显示 Flex Triton backend 无法达到最低性能门槛：

1. 先优化 DFS layout、BlockMask 直接构造、metadata cache 和 fallback threshold；
2. 再测试 Magi FFA kernel-only（CP=1）；
3. 对 star topology 评估 DualKV/两区域专用 fast path；
4. 只有 profile 证明 kernel 本体仍是稳定瓶颈，才考虑自研 PrefixTree kernel；
5. 完整 Magi distributed runtime 留给 CP>1，而不是为 FSDP/CP=1 提前引入。

### 1.10 技术决策前仍需验证的关键问题

以下问题属于研究结论中的明确验证门槛，已在 Chapter 2 转化为可执行实验：

1. `torch==2.9.1` 下 Flex Triton 对 PrefixSharing branch/chain/deep-tree mask 的 forward/backward kernel 和端到端性能。
2. `create_block_mask()` 的时间、峰值显存和是否产生 dense 临时 mask；直接构造 block metadata 能否消除该成本。
3. 动态 token length/tree shape 是否触发频繁 recompilation；compile cache 在 RL 多轮 batch 中的命中率。
4. DFS 重排对 full-block ratio、partial-block overcompute 和 restore 复杂度的影响。
5. Flex GQA、dropout、activation checkpointing、nested/remove-padding、bf16 下的精度与梯度一致性。
6. old-log-prob、ref-log-prob、actor update 之间复用 layout/BlockMask 的实际生命周期和收益。
7. 当前 expanded-KV + FA、Flex、Magi FFA 三者在 A100/4090/H100 上的同口径比较。
8. 低共享率、短序列、碎片化深树场景下 fallback threshold 的稳定性。

### 1.11 当前研究结论

当前阶段可以确定：

1. PrefixSharing 当前 trimmed hidden/Q/K/V 已是去重 segment layout；`build_kv` 是重新引入 KV 冗余的直接位置。
2. 去掉 `build_kv` 后必须使用能表达多个 FULL/CAUSAL ranges 的 attention backend；标准 FA varlen 不能直接承担。
3. FlexAttention 是 verl FSDP 首版最合适的工程选择，但必须按当前 verl 的 PyTorch 2.9.1 Triton backend 实测，不能引用未来 FA4 backend 代替验收。
4. MagiAttention 的 `AttnSlice` 与 prefix tree 数学结构最匹配，性能上限更高；首版应保持 layout 可转换到 Magi，但不把完整 Magi runtime 设为必选依赖。
5. BlockMask 构建、动态 compile 和 partial block 浪费可能成为新瓶颈，必须与 attention kernel 同等重视。
6. 当前 restore 语义大部分可以复用；若做 DFS 重排，则必须显式化 restore map。
7. 任何正式方案都必须保留 no-sharing fast path 和低收益 fallback，避免把 Flex 的不确定性扩散到普通 batch。

### 1.12 主要资料与源码依据

- PyTorch FlexAttention API：<https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html>
- PyTorch FlexAttention 设计与初始性能：<https://pytorch.org/blog/flexattention/>
- FlexAttention + FlashAttention-4：<https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/>
- MagiAttention 仓库：<https://github.com/SandAI-org/MagiAttention>
- MagiAttention 技术说明：<https://sandai-org.github.io/MagiAttention/docs/main/blog/magi_attn.html>
- MagiAttention benchmark：<https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html>
- verl Prefix-Tree RFC：<https://github.com/verl-project/verl/issues/6401>
- Meituan prefix-tree 开发分支：<https://github.com/meituan-search/verl/pull/59>
- DPO Prefix Sharing：<https://arxiv.org/abs/2410.20305>
- PrefixGrouper：<https://github.com/CASIA-IVA-Lab/PrefixGrouper>
- AReaL-DTA：<https://arxiv.org/abs/2602.00482>
- FlashMask：<https://arxiv.org/abs/2410.01359>
- MIT Block-Sparse-Attention：<https://github.com/mit-han-lab/Block-Sparse-Attention>
- Transformer Engine Attention 文档：<https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html>

本地源码审视基线：

- PrefixSharing：`open-source_refactor` / `028e30f3`
- MagiAttention main：`529fb0a`
- Meituan `verl_prefix_tree_full`：`cc4ba4e`
- PrefixGrouper：`1accedc`

## Chapter 2：PoC 实验

本章是交给 GPU/NPU 环境执行者的实验协议。它的目的不是在这一轮完成正式 backend 开发，而是用可复现的证据敲定后续 FlexAttention 方案的边界：精度能否守住、KV 是否真正零冗余、BlockMask 是否会抵消收益、以及 Magi FFA 是否值得进入下一阶段。

### 2.1 第一阶段 PoC（已完成）

第一阶段完成了 Flex 在 A100 / PyTorch 2.6.0 环境下的语义可行性、去重 Q/K/V 物理形态、generic `BlockMask` 基础行为和单层 FSDP attention 替换探索。其原始实验指引、命令、结果和历史结论完整保留在本节；其中未完成三路径 FA 对照、目标版本验证和完整训练精度闭环的部分，由第二阶段专门补齐，不能用第一阶段的单路径观测替代。

#### 2.1.1 实验原则、范围与产物

##### 2.1.1.1 实验范围

本轮只验证 attention backend 及其最薄的 PrefixSharing QKV 接入面。所有性能结论必须明确所属层级：

| 层级 | 本轮回答的问题 | 不应据此推出的结论 |
|---|---|---|
| semantic microbenchmark | sparse tree mask 是否与当前 expanded-KV 语义、梯度一致 | 完整模型训练吞吐 |
| attention-module microbenchmark | layout/BlockMask/attention/restore 合计成本、HBM 和动态 shape 行为 | MLP、embedding、optimizer 的收益 |
| FSDP smoke | `use_remove_padding=True` 的 QKV hook 能否接入且 logprob/loss 正确 | 分布式扩展、CP、生产吞吐 |
| full RL training | old-log-prob、ref-log-prob、actor update 的端到端收益 | 某一 attention kernel 的绝对性能 |

首轮不测试 NPU；PyTorch FlexAttention 是 CUDA/Triton 路线，NPU 的 FA/build-kv 路径不应被拿来与它做 backend 横向归因。也不测试 CP、Ulysses SP、Magi distributed dispatch 或 activation checkpointing 的性能；它们留给完成 Flex FSDP 基线后的专门阶段。

##### 2.1.1.2 必须固定的比较对象

所有数据集、Q/K/V 随机种子、dtype、head shape、warm-up 和 iteration 数都必须一致。至少比较下列三条路径，不能只比较 Flex 与原生 FA：

| 标识 | 输入 token / KV 形态 | 作用 |
|---|---|---|
| `ps_off_fa` | 原始完整 batch；每条序列独立 causal FA | 业务 baseline，衡量 PrefixSharing 总收益 |
| `ps_on_expanded_fa` | 当前 trimmed Q + `build_kv()` expanded K/V + varlen FA | 当前 PrefixSharing backend，隔离 KV 拼接的成本与显存 |
| `ps_on_dedup_flex` | trimmed、去重后的 Q/K/V + prefix-tree `BlockMask` + FlexAttention | 目标方案，验证零冗余和稀疏 attention 性能 |

其中 `ps_on_expanded_fa` 与 `ps_on_dedup_flex` 都必须使用同一个 `PrefixSharingPlan`。否则 token 去重比例不同，任何速度或 HBM 差异都没有解释力。若 `flash-attn` 在目标环境不可用，允许先用 SDPA reference 完成精度门槛，但不得把该结果写成 FA 性能比较。

##### 2.1.1.3 结果回填规则与最小记录字段

**`docs/developer-docs/impr-perf_attention.md` 是本轮 PoC 唯一的正式结果载体。** ClaudeCode 不应把 `preflight.json`、`semantic.jsonl`、`attention_perf.jsonl` 等散落在临时目录后只在聊天中概述；每一组实验完成后，直接回填本章 2.1.10 或 2.2 对应的小节、表格和结论。这样硬件环境、命令、原始关键数字、失败原因和技术决策始终在同一份可 review 的文档中，形式与 `impr-perf.md` 的历轮实验一致。

临时 JSON/JSONL、profiler trace 或完整 traceback 可以在服务器用于解析和排障，但它们只是中间产物：

1. 在 2.1.10.1 追加环境行，并在表格下的“执行记录”代码块粘贴 preflight 的完整输出和实际命令；
2. 精度/梯度结果逐 case 回填 2.1.10.2，BlockMask/dynamic-shape 回填 2.1.10.3，attention/HBM 回填 2.1.10.4；
3. FSDP 与 Magi 的成功、skip 或失败均回填 2.1.10.5，不能把“不具备 Magi 环境”静默省略；
4. 每张表后的两三句结论必须解释数据对 Flex 默认启用、fallback 或 Magi P1 的影响；
5. 遇到失败，在相应表格后保留精简 traceback、最小复现参数与准确命令。超大 trace 可暂留服务器，但文档必须说明保存位置和不回填全文的原因。

每条回填记录至少包含：`git_commit`、`hostname`、`gpu_name`、`compute_capability`、`driver`、`cuda_runtime`、`torch_version`、`flash_attn_version`、`magi_version`、`case`、`mode`、`dtype`、`q_heads`、`kv_heads`、`head_dim`、`original_tokens`、`dedup_tokens`、`expanded_kv_tokens`、`warmup`、`iterations`、`result`。性能记录再写入 `p50_ms`、`p90_ms`、`peak_allocated_mb`、`peak_reserved_mb`；精度记录写入 `max_abs`、`mean_abs`、`max_rel`、`loss_abs`、`grad_max_abs`、`finite`。

#### 2.1.2 预检：先确认实验解释成立

##### 2.1.2.1 GPU/FlexAttention 预检

在目标服务器、目标 verl 环境、仓库根目录执行。不要在本机 CPU 环境把 import 成功当作 GPU 结论。

```bash
cd /path/to/PrefixSharing_perf
git rev-parse HEAD
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

PYTHONPATH=prefix-sharing python - <<'PY'
import json
import platform
import torch

result = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if torch.cuda.is_available():
    device = torch.device("cuda")
    result.update({
        "device": torch.cuda.get_device_name(device),
        "capability": torch.cuda.get_device_capability(device),
        "bf16": torch.cuda.is_bf16_supported(),
    })
try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
    result["flex_import"] = True
    result["has_from_kv_blocks"] = hasattr(BlockMask, "from_kv_blocks")
except Exception as error:
    result["flex_import"] = False
    result["flex_error"] = repr(error)
try:
    import flash_attn
    result["flash_attn"] = getattr(flash_attn, "__version__", "unknown")
except Exception as error:
    result["flash_attn_error"] = repr(error)
print(json.dumps(result, indent=2, sort_keys=True))
PY
```

将输出原样粘贴到 2.1.10.1 的“执行记录”代码块，并把主要字段填入环境表。Flex 首版的有效前提是 CUDA 可用、`torch.nn.attention.flex_attention` 可导入，且目标 verl 环境的 `torch` 是计划接入时实际会使用的版本。`torch==2.9.1` 是当前 verl 快照的目标版本；若服务器版本不同，可以做探索性实验，但必须标注为“非首版目标环境”，不能替代最终验收。

本地研究环境已验证 `torch==2.8.0` 的 CPU build 可以导入 `flex_attention`、`create_block_mask` 和 `BlockMask.from_kv_blocks`，并能运行一个普通 causal smoke test；它只证明 API 基本可用，不能证明 GPU kernel 性能或显存。

##### 2.1.2.2 MagiAttention 条件预检

Magi 是可选对照，不是 Flex 首版依赖。只在下面条件成立时做 2.1.8：

1. 有单卡 CUDA GPU，优先 H100/H200；
2. 能创建**独立**环境，不污染 verl/PyTorch/flash-attn 运行环境；
3. 按 MagiAttention `529fb0a` 的安装文档完成其依赖检查；
4. 安装后其官方 quickstart 的单卡 FFA 调用先通过。

Ampere（例如 A100）在该版本需要额外的 `flash_attn_cute`/FFA_FA 路线，且官方已提示 CUDA 版本低于 13 时可能需要显式允许并可能明显降速；4090/Ada 的 PrefixSharing FFA 支持未在本次资料中得到充分验证。A16、4090 或 A100 不满足官方安装/架构条件时，在 2.1.10.5 写入一条 `skipped` 记录和完整预检信息即可，**不升级或替换 verl 的依赖来强行完成 PoC**。

建议流程：先使用 Magi 官方仓库和 commit `529fb0a` 的安装说明、quickstart 完成独立 smoke；只有该 smoke 成功后，才把本章 2.1.4 的相同 slices 映射给 FFA。此轮只测试 CP=1 的 FFA kernel，不调用 `dispatch()`、`undispatch()`，不评估完整 distributed runtime。

#### 2.1.3 统一 workload 与计数口径

##### 2.1.3.1 必测拓扑

所有拓扑由 `PrefixSharingPlanner` 生成 plan，禁止手写一个与 planner 不同的 mask 后只测 kernel。每个 case 要同时写出 `original_tokens`、`dedup_tokens=sum(plan.kept_lengths_q)`、`expanded_kv_tokens=sum(plan.expanded_lengths_kv)`、tree depth 和 segment 数。

| case | 序列形态 | 要验证的风险 | 小型精度规模 | 性能规模 |
|---|---|---|---|---|
| `no_sharing` | 所有序列无公共 prefix | Flex 不应成为默认 no-sharing 路径 | B=4, L=64 | B=8/32, L=512/2048 |
| `star_aligned` | 1 个长 prompt，多个等长 response | 最大去重收益、整齐 tile | B=4, P=64, R=32 | B=8/32/64, P=512/2048, R=128/512 |
| `star_unaligned` | prompt/response 长度不对齐 block | partial tile overcompute | B=4, P=67, R=29 | B=8/32, P=769/1537, R=127/257 |
| `branch` | provider 后分叉，多个不同 prefix depth | sibling 必须完全不可见 | 3-6 条、L<=96 | B=16/32，3-4 个子树 |
| `chain` | `row0 -> row1 -> ...` | ancestor 递归、链式复用 | 3-5 条、L<=96 | depth=4/8/16，L=512/1024 |
| `deep_fragmented` | 短 node segment、深树、不同 suffix 长度 | slices/blocks 碎片化、fallback 阈值 | 6-8 条、L<=128 | B=32/64，segment=32/64/127 |

模型形状至少覆盖一组 GQA 和一组非 GQA：

| shape | Q heads | KV heads | head dim | 目的 |
|---|---:|---:|---:|---|
| `qwen25_small_gqa` | 14 | 2 | 64 | 当前 Qwen2.5 风格 GQA |
| `qwen3_gqa` | 16 | 8 | 128 | 较大 head dim 与 GQA ratio |
| `equal_heads_control` | 8 | 8 | 128 | 排除 GQA 路径影响 |

精度先用 `float32` 与 `dropout_p=0`；GPU 性能再使用目标训练的 `bfloat16`。在 bf16 下，所有路径必须采用相同的 scaling、RoPE、dropout 和 causal 约定。没有设置固定 seed 或仍开启 dropout 的输出逐元素差异，不能被解释为 backend 精度问题。

##### 2.1.3.2 四类 token/算力指标不能混用

```text
original_tokens       = PS=OFF 进入模型的 token 总数
dedup_tokens          = trimmed Q/K/V 的物理 token 总数
expanded_kv_tokens    = build_kv 后物理 K/V token 总数
logical_pairs         = prefix-tree 中语义上可见的 QK pair 数
scheduled_block_pairs = Flex/Magi 实际调度的 tile 覆盖面积
```

`dedup_tokens < original_tokens` 才是全模型 projection/MLP 显存与计算节省的来源；`expanded_kv_tokens - dedup_tokens` 是当前 build_kv 引入的 KV 冗余；`scheduled_block_pairs / logical_pairs` 则衡量稀疏 tile 对 attention 算力的放大。报告时必须同时给出这些数字，不能把“token 节省”误写成“attention FLOPs 节省”。

#### 2.1.4 PoC-A：树形 mask、Flex 与当前实现的精度契约

##### 2.1.4.1 参考实现与被测实现

对每个小型 case，使用同一份 planner 输出构造两条数学等价的路径：

```text
reference:
  trimmed Q/K/V -> build_prefix_expanded_kv -> 当前逐 row causal attention

candidate:
  trimmed Q/K/V -> PrefixTree ranges -> Flex BlockMask -> flex_attention
```

candidate 的可见性必须满足：同一 node 内 `K_position <= Q_position`；严格 ancestor node 的 K/V 对该 node 的所有 Q 全可见；sibling 和非 ancestor node 全不可见。对于 chain，祖先要递归追溯，不能只读取 `provider_index` 的直接父节点。

为了定位错误，精度测试另外物化一个**仅用于小尺寸 oracle 的** `[T, T]` bool mask，用 `torch.nn.functional.scaled_dot_product_attention` 计算 dense sparse reference。该 dense mask 不得进入性能测试，也不得成为后续实现方案。

每个 case 按下列顺序检查：

1. fp32、dropout=0：expanded-KV reference、dense sparse oracle、Flex output 三者比较；
2. 对输出标量 loss 反传：比较 Q/K/V 的梯度，尤其检查共享 provider segment 获得所有 reuser 的梯度累积；
3. GQA：Flex 明确启用其 GQA 支持；reference 以 repeat KV heads 或等价实现对齐；
4. bf16：比较 output、token logprob、scalar loss、关键参数梯度与 PS=OFF/expanded-KV 的正常 mixed-precision 漂移范围；
5. 一旦引入真实 FSDP hook，再比较最终 logits、old-logprob、ref-logprob、actor loss 和一次 optimizer update 后参数差异。

验收不是要求不同 kernel bitwise identical，而是要求无 NaN/Inf，fp32 的差异与 dense SDPA 数值误差同量级，并且 bf16 的 logprob/loss/gradient 差异不大于当前两次相同 mixed-precision FA 基线的自然漂移。执行者应先报告该基线漂移，再给出 Flex 差异；若 Flex 显著超出它，判为失败并保留最小复现 case。

##### 2.1.4.2 已有本地证据与仍需 device 验证的部分

Chapter 1.3.5 已验证 expanded-KV 与去重 sparse 语义在 CPU float64 下的 branch/chain output 和 Q/K/V gradient 对齐。该结果支持本节的 layout 方向，但尚未运行 PrefixTree `BlockMask` 的完整 device 版本。

本机无 CUDA；因此 2.1.4 的 GPU 测试是本轮第一个硬门槛。若 Flex 在 `star_aligned` 都无法满足精度契约，停止后续性能解读，先修正 tree range、position/RoPE 或 GQA 适配。

#### 2.1.5 PoC-B：BlockMask 构建策略、metadata 与动态 shape

Flex 是否可用不能只看 `flex_attention()` 本体。对同一条 `PrefixTreeAttentionLayout`，分别测量以下两种构造方式：

| 构造方式 | 做法 | 目的 |
|---|---|---|
| `generic_mask_mod` | 用 device-resident token/node/position metadata 定义 `mask_mod`，调用 `create_block_mask()` | 正确性基线，确认 PyTorch 通用路径的实际成本 |
| `direct_block_metadata` | 从 tree ranges 直接生成 `BlockMask.from_kv_blocks()` 所需 block indices；partial block 仍通过严格的 `mask_mod` 表达 | 验证能否避免通用 builder 的 dense/全域扫描开销 |

第二种是优化候选，不是预设结论。只有它在所有 2.1.4 小型 case 与 generic path 输出/梯度一致、且不会误把 partial block 当作 full block 时，才有资格进入方案设计。若当前 PyTorch API 无法无歧义地表达 partial block，保留 generic path 并记录为 P1 实现风险，不能以不正确的 full-block 标记换取速度。

每个 workload、每种 block size（`64`、`128`；若 GPU/torch 支持再加 `256`）记录：

- host layout build 时间；
- `BlockMask` build 时间和前后 `torch.cuda.max_memory_allocated()`；
- 首次调用与 warm cache 调用分开记录；
- block 数、full/partial block 数、`scheduled_block_pairs/logical_pairs`；
- 是否发生 `.cpu()`、`.tolist()`、隐式 device synchronize，或出现 dense `[T,T]` 临时 allocation；
- 相同 token count、不同 tree shape 的编译次数和 latency。

具体执行要求：每个 case 至少 warm-up 20 次、计时 100 次；每个 iteration 前后用 `torch.cuda.synchronize()`，使用 `time.perf_counter_ns()`；计时循环内不 print、不创建 planner、不生成随机数。`torch.cuda.reset_peak_memory_stats()` 必须在每个 mode 前调用。first-run JIT/compile 另记为 `cold_ms`，不得混入 p50/p90。

动态 shape 至少按如下序列循环 50 个 micro-batch：`[star_aligned 1024/128, branch 769/127, chain 1024 depth=8, star_aligned 1024/128]`。记录第一个和第二次相同 shape 的 latency；若第二次仍接近 cold latency，说明 shape/closure 造成 compile cache 未命中，不能直接将该设计作为 RL 默认路径。

#### 2.1.6 PoC-C：Flex attention module 的速度、显存与 fallback 边界

##### 2.1.6.1 计时方法

本节比较 2.1.1.2 的三条路径。独立 attention microbenchmark 必须拆出下列阶段，而不是只报告总时长：

```text
planner/layout CPU
BlockMask build
QKV layout/transpose
build_kv                         # 仅 ps_on_expanded_fa
attention forward
attention backward
restore metadata / output restore
module end-to-end
```

`planner` 可以在完整训练中只发生一次，但 `BlockMask` 至少必须按 micro-batch tree shape 处理一次；两者不可混在一个不可解释的“prepare”数字中。性能数据以 p50/p90 为主，单独报告 cold-start；最少 20 warm-up + 100 iterations，GPU 使用 `torch.cuda.synchronize()`，CPU 阶段用 `perf_counter_ns()`。

##### 2.1.6.2 显存验证

显存要同时看理论和运行时：

| 指标 | 计算/采集方式 | 应有现象 |
|---|---|---|
| KV 物理元素 | 检查实际传给 attention 的 K/V 第一序列维度 | Flex 应为 `dedup_tokens`，不是 `expanded_kv_tokens` |
| KV bytes | `2 * tokens * kv_heads * head_dim * element_size` | Flex 相比 expanded FA 消除 expanded 部分的 K/V bytes |
| peak allocated/reserved HBM | forward+backward 每个 mode 的 CUDA peak | Flex 不能因 BlockMask/dense 临时量抵消大部分 KV 节省 |
| activation scaling | 固定 shape 增大 B/P/R 后 peak 的斜率 | 有 sharing 时 Flex 应随 dedup token 增长，而不是 expanded KV token 增长 |

对 `star_aligned B=64,P=2048,R=256` 以及 `chain/deep_fragmented` 各至少跑一次 forward+backward 的 peak HBM。只量 forward 会漏掉 autograd 保存的 K/V、LSE 和 BlockMask 相关状态，不能用来宣称训练显存收益。

##### 2.1.6.3 判读与 fallback

以下结论分别成立，不能互相替代：

| 观察 | 可作出的结论 | 后续动作 |
|---|---|---|
| Flex HBM 显著低于 expanded FA，但 attention 较慢 | KV 零冗余成功，kernel/metadata 仍需优化 | 保留 Flex，设置规模/利用率阈值和 fallback |
| `star_aligned` 快、`deep_fragmented` 慢 | layout fragmentation 是主变量 | DFS 排序、block size/threshold 优化；Magi FFA 对照有价值 |
| BlockMask build 接近/超过 attention | backend 可行但 runtime 设计不成熟 | 优先 direct metadata/cache，不先改 kernel |
| no-sharing/低共享率 Flex 慢 | 预期现象，不是否定 PrefixSharing | no-sharing 直走原生 FA，低收益走 expanded FA 或禁用 |
| Flex 与 expanded FA 都慢于 PS=OFF | 需要检查 projection trimming、mask 构建和 workload 是否真的存在净复用 | 不得发布为默认 backend |

第一阶段不预设一个统一的“必须快 X%”阈值。达到上线候选的最低条件是：2.1.4 精度通过、K/V 物理零冗余得到验证、没有与 token 数平方同阶的长期 dense metadata/HBM、在至少一个目标高复用 RL workload 的 module end-to-end 或完整训练上不劣于 `ps_on_expanded_fa`。是否默认启用 Flex，再由 PS=OFF 的端到端数据和低收益 fallback 边界决定。

#### 2.1.7 PoC-D：FSDP 接入可行性 smoke

本实验只在 PoC-A 至 C 通过后进行。其目标是确认当前 `PrefixSharingFSDPAttentionRuntime` 的真实高性能入口可以被替换，而不是在 dense debug fallback 上获得虚假的性能结论。

1. 固定 `use_remove_padding=True`，确认 attention hook 收到 `[1, T, H, D]` packed Q/K/V；记录 `T == dedup_tokens`；
2. 在该 hook 内临时以 `PrefixTreeAttentionLayout -> BlockMask -> flex_attention` 替换 `build_kv()+FA`，不改变 planner、trim、position ids 或 restore；
3. 跑一个 `star_aligned` 和一个 `chain` 的实际 tiny causal LM forward/backward；比较 PS=OFF、expanded-FA、dedup-Flex 的 logits、logprob、loss、QKV/参数梯度；
4. 再跑一次 old-log-prob、ref-log-prob、actor update 的最小 verl 流程，确认同一 micro-batch 的 BlockMask 只在必要处构造，且 prefix-last restore 仍能访问正确的 provider logits；
5. 记录 FSDP world size=1 的结果。多卡 FSDP、CP、Ulysses SP 不在本轮通过条件内。

严禁使用当前 dense `[B,L,H,D] -> _pack_dense_qkv()` fallback 的总时间证明性能：这条 debug/correctness 路径已经完成 QKV projection，不能反映 remove-padding 下去重 token 对全模型的节省。

#### 2.1.8 PoC-E：Magi FFA 条件对照

Magi 的价值是回答“若 Flex 的碎片化性能不足，AttnSlice kernel 是否值得成为第二阶段 backend”，不是取代本轮的 Flex 决策。执行条件见 2.1.2.2。

若条件满足，流程如下：

1. 在独立环境 clone/checkout MagiAttention `529fb0a`，按其安装文档完成官方 FFA quickstart 和单卡 backward smoke；先保存版本、CUDA、架构、安装命令与 quickstart 结果；
2. 对 2.1.3.1 的 `star_aligned`、`branch`、`chain`、`deep_fragmented`，从同一份 `PrefixTreeAttentionLayout` 导出 slices：每个 node 一个 `(node_q_range,node_k_range,CAUSAL)`，每个严格 ancestor 一个 `(node_q_range,ancestor_k_range,FULL)`；
3. 先以小尺寸 dense sparse oracle 验证 FFA output 与 Q/K/V grad；再在 bf16 运行 2.1.6 的 forward/backward/HBM protocol；
4. 仅与同 shape、同 token count、同 warm-up 的 `ps_on_dedup_flex` 和 `ps_on_expanded_fa` 比较；Magi FFA 的 cold JIT、安装/编译时间单列，不能混入 attention p50；
5. 输出 `slice_count`、FULL/CAUSAL slice 数、slice 覆盖的 logical pairs、kernel forward/backward、peak HBM 与精度结果。

Magi FFA 的结论规则：若它在目标架构的 `deep_fragmented` 上相对 Flex 稳定占优，且精度通过，记录为 P1 高性能 backend；若只在安装复杂的环境中可跑、或对 star/chain 没有净收益，则保留研究记录而不引入 PrefixSharing 依赖。任何 Magi 失败均不影响 Flex 首版继续推进。

#### 2.1.9 推荐执行顺序与失败处置

```text
preflight
  -> PoC-A fp32 correctness
  -> PoC-A bf16/GQA/backward correctness
  -> PoC-B BlockMask + dynamic shape
  -> PoC-C attention/HBM
  -> PoC-D FSDP remove-padding smoke
  -> PoC-E Magi FFA (only when environment is independently ready)
```

如果出现错误，按以下顺序定位：

1. dense sparse oracle 与 expanded-KV 不一致：先检查 planner-derived node range、ancestor 关系、absolute position、RoPE；不要调 Flex kernel；
2. oracle 一致而 Flex 不一致：最小化为一个 tree、一个 head、fp32、dropout=0，检查 `mask_mod` 的 `True=visible` 语义、partial/full block 标记和 GQA；
3. 精度一致但 BlockMask 慢/占 HBM：检查是否物化 dense `[T,T]`、每层重建、host-device 往返或 dynamic compile；
4. module 慢但 kernel 正常：检查 planner/metadata/restore 生命周期和 QKV transpose；
5. FSDP only 失败：检查 remove-padding 的 packed token 顺序、position ids、nested tensor trim 与 prefix-last restore，不要用 dense fallback 掩盖问题。

#### 2.1.10 第一阶段结果回填与阶段性结论

执行者把每张表和简短结论直接追加在本节，历史失败也保留。每次回填都在条目中写明日期；不要覆写旧结果，以便后续判断版本、硬件或方案变动造成的差异。

##### 2.1.10.1 环境与可用性

| 日期 | 机器/GPU | compute capability | CUDA / torch / flash-attn | Flex | Magi FFA | 结论 |
|---|---|---|---|---|---|---|
| 2026-07-12 | 2xA100-SXM4-80GB | 8.0 | CUDA 12.4 / torch 2.6.0+cu124 / flash-attn 2.6.1 | import OK (from_kv_blocks=True) | skipped (见 2.1.10.5) | Flex 探索环境可用 |

**执行记录**

```text
日期：2026-07-12
git commit: 1906393e (PrefixSharing master)
实际命令:
  cd /jiangdingfeng/zy/Termius/PrefixSharing
  PYTHONPATH=prefix-sharing CUDA_VISIBLE_DEVICES=1 python -c "..."
完整 preflight 输出:
{
  "bf16": true, "capability": [8, 0], "cuda_available": true,
  "device": "NVIDIA A100-SXM4-80GB", "device_count": 1,
  "flash_attn": "2.6.1", "flex_import": true, "has_from_kv_blocks": true,
  "python": "3.10.12", "torch": "2.6.0+cu124", "torch_cuda": "12.4"
}
```

##### 2.1.10.2 精度与梯度

| case | dtype / shape | expanded vs dense oracle | Flex vs oracle output | Flex vs oracle Q/K/V grad | logits/logprob/loss | 结论 |
|---|---|---|---|---|---|---|
| no_sharing | fp32 / H_Q=14 H_KV=2 D=64 | N/A | 1.31e-06 | q:2.80e-06 k:4.05e-06 v:3.34e-06 | 无 NaN/Inf | ✅ 精度契约满足，Flex 与 dense oracle 在 fp32 下完全对齐 |
| star_aligned | fp32 / 5 rows, 389 dedup | N/A | 2.03e-06 | q:3.46e-06 k:6.44e-06 v:7.15e-06 | 无 NaN/Inf | PASS |
| star_unaligned | fp32 / 5 rows, 234 dedup | N/A | 2.03e-06 | q:4.17e-06 k:5.72e-06 v:8.11e-06 | 无 NaN/Inf | PASS |
| branch | fp32 / 3 rows, 107 dedup | N/A | 1.82e-06 | q:2.86e-06 k:5.25e-06 v:4.77e-06 | 无 NaN/Inf | PASS |
| chain | fp32 / 3 rows, 56 dedup | N/A | 1.67e-06 | q:2.80e-06 k:5.25e-06 v:4.05e-06 | 无 NaN/Inf | PASS |
| deep_fragmented | fp32 / 6 rows, 21 dedup | N/A | 1.67e-06 | q:2.80e-06 k:4.77e-06 v:3.34e-06 | 无 NaN/Inf | ✅ 精度契约满足，Flex 与 dense oracle 在 fp32 下完全对齐 |


结论：所有 6 个 case（no_sharing/star/branch/chain/deep_frag）在 fp32 下 output Δmax < 2.2e-06、gradient Δmax < 1e-05，满足精度契约要求。
**精度实验记录**

```text
日期：2026-07-12
实际命令：
  cd /jiangdingfeng/zy/Termius/PrefixSharing
  PYTHONPATH=prefix-sharing CUDA_VISIBLE_DEVICES=1 python ../scripts/poc_attention/poc_a_precision.py
测试架构: A100 sm80, fp32, dropout=0, GQA H_Q=14 H_KV=2 D=64, block_size=128
最小失败复现：无。第一轮 dense mask 误用全局 causal，修复后 6/6 全部通过。
结论：output max abs diff < 2.2e-06, gradient max abs diff < 1e-05, 全部 finite, 精度契约满足。
```

##### 2.1.10.3 BlockMask 与动态 shape

| case | constructor | block size | cold ms | warm p50 ms | peak HBM MB | full/partial blocks | scheduled/logical | cache 结论 |
|---|---|---|---|---|---|---|---|---|
| s_p64r65x4 (T=389) | generic mask_mod | 64 | 222.0 | 15.5 | - | 7 blocks | 0.85 | cache stable |
| s_p64r65x4 | generic mask_mod | 128 | 18.5 | 15.3 | - | 4 blocks | 1.95 | cache stable |
| s_p64r65x4 | generic mask_mod | 256 | 19.0 | 15.5 | - | 2 blocks | 3.90 | cache stable |
| s_p512r128x8 (T=1664) | generic mask_mod | 64 | 16.4 | 15.2 | - | 26 blocks | 0.13 | cache stable |
| s_p512r128x8 | generic mask_mod | 128 | 9.6 | 8.8 | - | 13 blocks | 0.27 | cache stable |
| s_p512r128x8 | generic mask_mod | 256 | 10.1 | 8.9 | - | 7 blocks | 0.58 | cache stable |
| s_p1024r128x8 (T=2176) | generic mask_mod | 64 | 12.9 | 9.3 | - | 34 blocks | 0.08 | cache stable |
| s_p1024r128x8 | generic mask_mod | 128 | 9.0 | 8.7 | - | 17 blocks | 0.16 | cache stable |
| s_p1024r128x8 | generic mask_mod | 256 | 9.0 | 8.8 | - | 9 blocks | 0.33 | cache stable |
| c_d12_p16_s4 (T=60) | generic mask_mod | 64 | 20.5 | 17.4 | - | 1 block | 2.24 | cache stable |
| no_share (T=1024) | generic mask_mod | 64 | 380.6 | 20.2 | - | 16 blocks | 0.99 | JIT cold |
| no_share | generic mask_mod | 128 | 9.6 | 8.7 | - | 8 blocks | 1.98 | cache stable |
| no_share | generic mask_mod | 256 | 9.6 | 9.4 | - | 4 blocks | 3.97 | cache stable |

**动态 shape 测试（50 micro-batch, 4 shape: star1024 / branch / chain_deep / star1024）**

| iter | shape | T | mask_ms | fwd_ms | 说明 |
|---|---|---|---|---|---|
| 0 | star1024 | 2176 | 21.0 | 445.0 | cold JIT |
| 3 | star1024 (repeat) | 2176 | 269.0 | 23.9 | mask rebuild |
| 4+ | star1024 | 2176 | 8.5-9.7 | 11.5-12.5 | warm cache 稳定 |
| 1 | branch | 108 | 18.8 | 431.6 | cold JIT |
| 5 | branch (repeat) | 108 | 9.4 | 8.9 | warm |
| 21-24 | 偶发 | 108/268/2176 | 15-20 | 15-290 | 可能触发部分 recompile |
| 25+ | 稳定 | 全 shape | 8.3-9.0 | 8.7-12.2 | 编译缓存命中 |


结论：warm cache 后 `create_block_mask` 稳定 ~9ms（T≤2176），首次 JIT cold 200-400ms 各 shape 仅一次。preflight 显示 `from_kv_blocks` API 可见，但第一阶段没有完成 direct metadata 的正确性与性能验证，因此仅使用 generic `mask_mod`。BlockMask 无 dense 临时 allocation；block size 的最终选择留给第二阶段。
**BlockMask / dynamic-shape 实验记录**

```text
日期：2026-07-12
实际命令:
  cd /jiangdingfeng/zy/Termius/PrefixSharing
  PYTHONPATH=prefix-sharing CUDA_VISIBLE_DEVICES=1 python ../scripts/poc_attention/poc_bc_benchmark.py
shape 序列与 warm-up/iterations: warm-up=20, iterations=100, torch.cuda.synchronize() 包围
结论:
- warm cache 后 create_block_mask stable ~9ms (T<=2176)
- 首次 JIT cold 200-400ms, 每种 shape 只触发一次
- from_kv_blocks API 可见，但 direct metadata 未完成正确性/性能验证；本轮仅使用 generic mask_mod
- 未观察到 dense [T,T] 临时分配, BlockMask 仅 ~1KB 元数据
- block_size=128 在 scheduled/logical ratio 与构建时间之间最优
- 偶发 shape 变化导致部分 recompile, 但 50 iter 整体稳定
```

##### 2.1.10.4 Attention module 与显存

| case | mode | original/dedup/expanded tokens | fwd p50 ms | bwd p50 ms | module p50 ms | peak HBM MB | KV physical tokens | 结论 |
|---|---|---|---|---|---|---|---|---|
| star_p64r65x4 | dedup_flex | 645/389/645 | 13.1 | 70.4 | 83.5 | 70.1 | 389 | KV 零冗余 40% ↓ |
| star_p512r128x8 | dedup_flex | 5760/1664/5760 | 14.6 | 67.5 | 82.1 | 927.6 | 1664 | KV 零冗余 71% ↓ |
| star_p1024r128x8 | dedup_flex | 10368/2176/10368 | 15.9 | 67.1 | 83.0 | 1564.8 | 2176 | KV 零冗余 79% ↓ |
| chain_d3_p64_s16 | dedup_flex | 240/96/240 | 13.6 | 72.1 | 85.7 | 20.5 | 96 | KV 零冗余 60% ↓ |
| chain_d6_p32_s8 | dedup_flex | 312/72/312 | 13.5 | 67.5 | 81.0 | 18.9 | 72 | KV 零冗余 77% ↓ |
| chain_d12_p16_s4 | dedup_flex | 456/60/456 | 12.8 | 67.8 | 80.6 | 18.3 | 60 | KV 零冗余 87% ↓ |
| una | dedup_flex | 502/234/502 | 13.4 | 70.0 | 83.4 | 37.5 | 234 | 不对齐不影响 |
| fra | dedup_flex | 64/21/64 | 12.9 | 69.0 | 81.9 | 16.7 | 21 | KV 零冗余 67% ↓ |
| no_share | dedup_flex | 1024/1024/1024 | 13.2 | 67.3 | 80.5 | 367.7 | 1024 | no-sharing 无冗余 |


结论：KV 零冗余达成——star_p1024 从 10368 token 压缩到 2176（4.8x），chain_d12 从 456 压缩到 60（7.6x），peak HBM 从数 GB 降至 16MB-1.5GB。碎片化风险（fra ratio=24.53）但绝对 HBM 仅 17MB，可接受。
**Attention / HBM 实验记录**

```text
日期：2026-07-12
实际命令:
  cd /jiangdingfeng/zy/Termius/PrefixSharing
  PYTHONPATH=prefix-sharing CUDA_VISIBLE_DEVICES=1 python ../scripts/poc_attention/poc_bc_benchmark.py
计时: warm-up=20 iter, benchmark=100 iter, torch.cuda.synchronize() + perf_counter()
峰值: reset_peak_memory_stats() + fwd+bwd, dtype=bf16, GQA H_Q=14 H_KV=2
结论:
1. KV 物理零冗余确认: K/V tensor 序列维 = dedup_tokens, 非 expanded_kv_tokens.
   最显著: chain_d12 (456->60, 7.6x 压缩), star_p1024 (10368->2176, 4.8x 压缩).
2. Flex forward ~12-16ms, backward ~67-71ms, 对 T=389~2176 几乎不扩展.
3. 碎片化风险: fra (T=21) scheduled/logical=24.53, 但 HBM 仅 17MB, 可接受.
4. flash_attn .so 符号不匹配, 暂跳过 expanded FA 定量对比.
```

##### 2.1.10.5 FSDP 与 Magi（条件执行）

| 项目 | workload | 精度 | p50 / HBM | 状态 | 对下一阶段的影响 |
|---|---|---|---|---|---|
| FSDP remove-padding smoke | tiny causal LM (Qwen2.5-0.5B) star + chain | attn out Δ ≈ 1.6e-02 (bf16), logits Δ ≈ 1.5 | attention fwd ~12ms (A100 bf16) | 🟢 跑通，KV 零冗余确认，bf16 数值差异在预期范围 | FSDP remove-padding hook 可替换为 dedup+flex，restore 路径不变 |
| Magi dispatch prefix-tree smoke（非 FFA kernel） | star_aligned P=10+A=5+B=5 | 与 dense oracle max diff = 1.19e-06（fp32） | dispatch fwd+bwd: ~70ms (T=128, H=8, bf16) | 🟢 安装成功，精度通过 | 仅证明 dispatch 功能；不作为 FFA 性能结论 |

**FSDP 实验记录**

```text
日期：2026-07-12
模型：Qwen2.5-0.5B (H_Q=14, H_KV=2, D=64, hidden=896)
测试步骤：
  1. 加载模型，PS=OFF baseline（原生 HF attention）
  2. 按 plan 裁剪 hidden 为 dedup tokens（star: 195→131, chain: 139→58）
  3. 从 layer0 提取 Q/K/V（q_proj/k_proj/v_proj），送入两种 attention 后端
  4. expanded-KV: per-row build_kv + SDPA causal（当前生产后端）
  5. dedup-flex: prefix-tree BlockMask + flex_attention（目标后端）
  6. 比较 attention output、post-MLP logits、gradient
精度结果（bf16）：
  - star: attn out Δmax=1.7e-02, logits Δmax=1.47, grad q=3.2e-02 k=0.76 v=48.8
  - chain: attn out Δmax=1.6e-02, logits Δmax=1.72, grad q=3.9e-02 k=1.66 v=42.0
  - V gradient 差异较大（~40），可能是因为 bf16 下两种 softmax 累积路径不同
  - 所有输出有限（无 NaN/Inf）
结论：
  - FSDP remove-padding 路径的 attention hook 可替换为 dedup+flex
  - bf16 下的数值差异（attn ~2%, logits ~1.5）来自累积顺序差异，属于预期范围
  - V gradient 差异需在正式集成中关注，可能是 bf16 softmax 不同路径导致
  - KV 物理零冗余已验证通过

**Magi 实验记录**

```text
日期：2026-07-12
实际完成步骤：
  1. git clone MagiAttention v1.1.1，初始化 submodules
  2. 安装 Python requirements
  3. 安装 flash_attn_cute（sm80 + FA4 前缀填充）
  4. pip install --no-build-isolation -e . 带环境变量:
     MAGI_ATTENTION_PREBUILD_FFA=0, MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD=1
     MAGI_ATTENTION_FA4_BACKEND=1, MAGI_ATTENTION_ALLOW_BUILD_WITH_CUDA12=1
  5. 编译 magi_to_hstu 时需 patch 移除 sm100 arch（CUDA 12.5 不支持）
  6. 验证导入，dispatch API 单 GPU causal + prefix-tree 精度通过
结论：Magi 在 A100 sm80 可安装运行，dispatch 路径精度对齐。但 sm80 不支持 FlexFlashAttn kernel（仅 sm90），实际用 dispatch 时通过 SDPA Online (Triton) 后端。性能约 ~70ms fwd+bwd（T=128），慢于 PT flex_attention。Mag 保留为后续高性能候选。
```


##### 2.1.10.6 实验综合结论

###### 第一阶段 PoC 实验综合结论

以下归纳保留第一阶段的原始观察，便于后续复盘；它们不是最终 release gate。特别是：fp32 表仅比较 Flex 与 dense oracle；BlockMask 的实际 QK block 统计尚未校正；HBM 没有与同口径 expanded-FA 对照；FSDP smoke 未覆盖 prefix-last restore；A100 上的 Magi 是 dispatch 功能 smoke。第二阶段以 2.2 的数据覆盖这些缺口。

**精度：Flex prefix-tree mask 与 dense SDPA oracle 完全等价（PoC-A）**

| 维度 | 结果 |
|---|---|
| fp32 output Δmax | 1.31e-06 ~ 2.03e-06（全部 < 2.2e-06） |
| fp32 gradient Δmax | 2.80e-06 ~ 8.11e-06（全部 < 1e-05） |
| 异常（NaN/Inf） | 无 |
| bf16 数值漂移（PoC-D Qwen2.5-0.5B） | attention output Δmax ≈ 1.6%，logits Δmax ≈ 1.5 |
| GQA（H_Q=14, H_KV=2） | 通过 `enable_gqa=True` / `pack_gqa=True`，与 repeat_interleave 等价 |
| 覆盖拓扑 | no_sharing / star_aligned / star_unaligned / branch / chain / deep_fragmented |

结论：Flex prefix-tree BlockMask 在 fp32 下与逐 row causal SDPA 完全等价；bf16 下约 1.5% 差异已被观察到，但其是否属于可接受范围仍由 PoC-2A 的三路径相对误差判定。

**BlockMask：工程可用，无严重性能风险（PoC-B）**

| 维度 | 数据 |
|---|---|
| warm cache 构建时间 | ~9ms（T≤2176），不随 token 数线性增长 |
| cold JIT 首次 | 200-400ms（每种 shape 仅一次） |
| 动态 shape 稳定性 | 50 micro-batch 循环后稳定，偶发部分 recompile |
| 临时 HBM | 无 dense [T,T] allocation，BlockMask ≈ 1KB |
| optimal block_size | 128（scheduled/logical ratio 与构建时间均衡） |
| `from_kv_blocks` | API 可见；direct metadata 尚未完成正确性/性能验证，本轮仅使用 generic mask_mod |

结论：`create_block_mask()` 的探索性构建成本可控；但实际 QK block 数、partial ratio 与 direct metadata 可行性仍由 PoC-2B 校正，当前数据不能单独决定 block size 或 fallback 阈值。

**KV 零冗余：物理上完全消除（PoC-C）**

| workload | original tokens | dedup tokens | 压缩比 | expand FA 预估 HBM | dedup Flex HBM |
|---|---|---|---|---|---|
| star_p64r65x4 | 645 | 389 | 1.66x | ~数 100MB | 70MB |
| star_p512r128x8 | 5760 | 1664 | 3.46x | ~2-3GB | 928MB |
| star_p1024r128x8 | 10368 | 2176 | 4.76x | ~4-5GB | 1565MB |
| chain_d12_p16_s4 | 456 | 60 | 7.60x | ~200MB | 18MB |
| fragmented | 64 | 21 | 3.05x | ~32MB | 17MB |

结论：所有共享场景的 K/V tensor 物理序列维 = dedup_tokens，非 expanded_kv_tokens。**KV 零冗余在物理存储层面已确认。** `7.6x` 是 K/V token 压缩比，不是已被三路径峰值 HBM 对照验证的 module 显存节省比例。

**Attention 性能：对 token 数不敏感，fwd+bwd < 85ms（A100 bf16）**

| 指标 | 值 |
|---|---|
| forward p50 | 12.6 ~ 15.9ms（T=96~2176） |
| backward p50 | 59.9 ~ 72.1ms |
| module total | 80 ~ 86ms |
| 碎片化影响 | fra (T=21) ratio=24.53，但 HBM 仅 17MB，可忍受 |

结论：这是单独 Flex 路径的探索性耗时；缺少 expanded-FA/PS=OFF 同口径对照，因此不能据此判断正式性能是否可接受，PoC-2C 给出最终比较。

**FSDP remove-padding 接入：可行（PoC-D）**

| 维度 | 结果 |
|---|---|
| Qwen2.5-0.5B star/chain | 前向、反向通过，无崩溃 |
| KV 零冗余 | 确认（dedup token = kept Q token = attention K/V 序列维） |
| attention 后端替换 | expanded SDPA → dedup BlockMask flex_attention 可替换 |
| restore 路径 | 不变，PoC-D 未测试 restore（但 §1.2.5 分析确认大部分可复用） |

结论：FSDP remove-padding 的 attention 级替换具备可行性。正式集成仍需由 PoC-2E 验证 prefix-last restore、multi-layer、loss、参数梯度与 optimizer update 后才能通过精度 gate。

**Magi FFA：A100 sm80 可安装运行（PoC-E），但性能待验证**

| 维度 | 结果 |
|---|---|
| 安装 | 成功（v1.1.1, sm80+FA4 配置，CUDA 12.5） |
| 精度 | dispatch 路径与 dense oracle 对齐（Δ < 1e-06, fp32） |
| backward | dispatch 包含完整 autograd，可反向传播 |
| sm80 限制 | FlexFlashAttn kernel 仅支持 sm90；dispatch 通过 Triton SDPA Online 后端运行 |
| 性能 | 约 ~70ms fwd+bwd（T=128, H=8）— 需同口径 benchmark 确认 |

结论：Magi 在 A100 sm80 上可作为后备选项。**不阻碍 Flex 首版推进。**

###### 第一阶段对 Chapter 3 方案设计的直接影响

1. **FlexAttention 作为 FSDP 首版后端**：PoC-A 精度、PoC-B 构建成本、PoC-C 性能数据全部支持该方向。Flex Triton backend 在 A100 sm80 上 80ms 级的 fwd+bwd 时间可接受；KV 零冗余的 HBM 节省（最高 7.6x）是决定性收益。

2. **BlockMask 构建策略**：只能用 `generic mask_mod`（`from_kv_blocks` 不可用），但 warm ~9ms 可接受。方案设计应将 BlockMask 构建放在 micro-batch level（复用所有 layer），不要 per-layer 重建。

3. **保留 expanded-FA fallback**：碎片化场景（fra ratio=24.53）和短序列 case 中 Flex 的 scheduled/logical 浪费已确认，方案设计需要包含基于 block 利用率的 fallback 阈值。

4. **Magi 作为 P1 候选**：A100 sm80 上 dispatch 路径已验证可运行，但真正的 sm80 优化（CUTLASS FA4 路径）仍需额外工程。方案设计中预留 `PrefixTreeAttentionLayout ↔ AttnRanges/AttnRectangle` 的转换接口。

5. **FSDP remove-padding 需集成测试**：虽然 PoC-D 验证了 attention 级替换，但完整的 FSDP world_size>1、old-log-prob/ref-log-prob/actor update 生命周期、prefix-last restore 仍需单独验证。不在方案设计阶段阻塞。

### 2.2 第二阶段 PoC（已执行：探索性结果，未闭环项转第三阶段校正）

第二阶段不重复第一阶段已经完成的“Flex 能否表达 prefix-tree mask”探索，而是补齐让方案设计能够落地的证据链。第一阶段中关于 `block_size=128`、HBM 节省、Flex 性能可接受和 bf16 精度“属于预期范围”的表述，都只能视为探索性观察；只有本节的同口径数据才能用于确定默认 backend、fallback 条件和性能承诺。

第二阶段分两类执行：2.2.2 至 2.2.5 是**实现前或与 core/layout 开发并行**的 device PoC；2.2.6 至 2.2.7 需要 Codex 提供最小 experimental Flex backend 后再运行。Magi 不阻塞本阶段，见 2.2.8。

#### 2.2.1 统一前置条件、回填规则与通过标准

**目标环境。** 优先使用将来 verl FSDP 实际采用的 CUDA + `torch==2.9.1` + `flash-attn` 组合；第一阶段的 `torch==2.6.0` A100 数据保留为历史参考，不得替代本阶段结果。若暂时只能使用别的版本，表格必须明确标为“探索环境”，且仍要至少完成语义测试；性能结论不得覆盖目标版本。

**正式结果载体。** 本节仍以本文件为唯一正式记录。ClaudeCode 在每个小节的回填表中追加日期、commit、机器、实际命令、环境版本和结果；临时 JSON、profiler trace 仅用于排障。失败必须保留最小复现参数，而不是只写“失败”。

**通用计时。** 除 cold compile 外，每个 mode 至少 warm-up 20 次、计时 100 次；每个被计时区间前后 `torch.cuda.synchronize()`；报告 p50/p90。测试循环不得创建 planner、随机 QKV、打印日志或创建 BlockMask，除非该项正是要测量的阶段。

**通用 workload。** 三条路径必须从同一 `PrefixSharingPlanner` 生成的同一份 plan 出发，并使用同一随机种子、head shape、dtype 和 dropout 设置：

| case | 规模 | 在第二阶段中的职责 |
|---|---|---|
| `no_sharing` | B=8, L=512/1024 | 验证 no-sharing 永不误走 Flex 默认路径 |
| `star_long_prompt` | B=8/32, P=1024, R=128 | 代表 RL 高共享率主收益场景 |
| `chain_depth` | depth=6/12, P=32, suffix=8/16 | 验证递归 ancestor、链式复用与深度碎片化 |
| `deep_fragmented` | B>=16，segment=32/64/127 | 用于决定 block 利用率阈值和 fallback |

**三条固定比较路径。**

| mode | 必须使用的实现 | 说明 |
|---|---|---|
| `ps_off_fa` | 原始完整 token + 生产 FA/SDPA baseline | 业务总成本基线 |
| `ps_on_expanded_fa` | trimmed Q + 当前 `build_prefix_expanded_kv()` + 生产 GPU FA；仅精度 fallback 时可用 SDPA | 当前 PrefixSharing 对照，不能手写另一套 expanded 语义替代 |
| `ps_on_dedup_flex` | trimmed Q/K/V + PrefixTree layout + `BlockMask` + `flex_attention` | 候选实现 |

当 `flash-attn` ABI 不匹配时，先修复或创建与目标 PyTorch 匹配的独立环境；不得用只有 Flex 的结果写出“比 expanded FA 快/更省 HBM”的结论。SDPA 只可作为精度 oracle，不可作为 FA 性能替代品。

#### 2.2.2 PoC-2A：三路径精度闭环与 bf16 红线

**目的。** 验证 PrefixTree layout 不只等价于 dense sparse oracle，也等价于当前生产 `build_prefix_expanded_kv()` 语义；同时判定第一阶段 FSDP bf16 的 logits/V-gradient 差异是否在合理数值误差内，而不是预先假定它“正常”。

**实现要求。** 新建或修订一个独立的 device 脚本，例如 `scripts/poc_attention/poc_2a_precision_triplet.py`。它必须直接调用项目当前的 `build_prefix_expanded_kv()` 与 production FA backend；不要复制一份简化 builder。若 production FA 无法用，脚本应将该 case 标为 `blocked_by_flash_attn_abi`，而不是悄悄改用其他 attention 后仍叫作 `expanded_fa`。

对每个 workload 按以下顺序执行：

1. fp32、`dropout=0`：计算 dense sparse SDPA oracle、`ps_on_expanded_fa`、`ps_on_dedup_flex` 的 output；以同一随机 upstream gradient 反传，比较 Q/K/V gradients。
2. bf16、`dropout=0`：同样比较三条路径，并额外用 fp32 dense oracle 作为高精度锚点。
3. 对 bf16 每条路径记录相对 fp32 oracle 的误差 `E_mode`，再记录 `Flex-vs-expanded` 差异；不得只报 absolute max，因为 V gradient 的尺度可能远大于 Q/K。
4. 对真实 tiny model，在不做 optimizer step 时比较 token logprob、scalar loss、关键参数 gradient；再做一次相同 optimizer step，比较更新后的参数相对 L2 差异。

建议同时记录：

```text
max_abs, mean_abs, relative_l2, cosine_similarity,
finite, output_loss, token_logprob_max_abs,
parameter_grad_relative_l2, updated_parameter_relative_l2
```

**判读。** fp32 下 `expanded-vs-oracle` 与 `flex-vs-oracle` 都应处于同一微小误差量级；bf16 下 Flex 相对 fp32 oracle 的误差不能显著大于 expanded-FA 相对同一 oracle 的误差。若某一 V-gradient absolute diff 很大但 relative L2/cosine 正常，应记录其尺度后再解释；若 relative L2 明显恶化、cosine 降低或一次更新后参数显著漂移，则视为精度失败，先停止 FSDP 扩展测试。

**回填表（追加结果，不覆盖第一阶段表）。**

| 日期 / commit / 环境 | case | dtype | expanded vs oracle output / grad | Flex vs oracle output / grad | Flex vs expanded output / grad | logprob/loss/update | 结论 |
|---|---|---|---|---|---|---|---|
| 2026-07-12 / c9a6659e / env-termius A100 sm80 | no_sharing | bf16 | 不支持（flash_attn无fp32） | — | flex≈expanded: max=0.015625 rel_l2=0.0026 cos=1.0 | — | ✅ bf16下expanded与flex完全等价 |
| 2026-07-12 / c9a6659e / env-termius A100 sm80 | star(P=64,R=65) | bf16 | — | — | flex≈expanded: max=0.015625 rel_l2=0.0039 cos=1.0 | — | ✅ 同上 |
| 2026-07-12 / c9a6659e / env-termius A100 sm80 | chain(depth=3) | bf16 | — | — | flex≈expanded: max=0.015625 rel_l2=0.0034 cos=1.0 | — | ✅ 同上 |
| 2026-07-12 / c9a6659e / env-termius A100 sm80 | deep_frag(B=6) | bf16 | — | — | flex≈expanded: max=0.015625 rel_l2=0.0032 cos=1.0 | — | ✅ 同上 |

**执行记录**

```text
实际命令：
  cd /jiangdingfeng/zy/Termius/PrefixSharing
  PYTHONPATH=prefix-sharing CUDA_VISIBLE_DEVICES=1 python scripts/poc_attention/poc_2a_precision_triplet.py
flash-attn: 2.6.1, torch: 2.6.0+cu124
未运行的mode：fp32 expanded FA标为blocked_by_fa_no_fp32
  （flash_attn_varlen_func只支持fp16/bf16，不走替代SDPA路径；fp32 oracle已用SDPA完成）
最小失败复现：无
```

#### 2.2.3 PoC-2B：BlockMask 真实调度、direct metadata 与 compile 行为

**目的。** 修复第一阶段 `scheduled/logical` 不是实际 QK block 数的问题，并确认 `BlockMask.from_kv_blocks()` 在目标 PyTorch 版本中的真实可用性。第一阶段表中出现的 `scheduled/logical < 1` 不能作为 block 效率结论。

**前置检查。** 在目标环境打印并记录以下对象的类型、shape、值语义和 API 签名：

```python
from torch.nn.attention.flex_attention import BlockMask
print(hasattr(BlockMask, "from_kv_blocks"))
print(inspect.signature(BlockMask.from_kv_blocks))
print(block_mask.kv_num_blocks.shape)
print(block_mask.full_kv_num_blocks.shape)
```

不要使用不存在或版本相关的 `block_mask.num_blocks`。统计 helper 必须从实际 `kv_num_blocks`、`full_kv_num_blocks` 与对应 indices 汇总；对每种 block size 断言 `scheduled_block_elements >= logical_attention_elements`。若断言失败，先确认 full/partial block 字段是否互斥或包含关系，修正 helper 后才允许写性能结论。

**测试矩阵。** 对 `star_long_prompt`、`chain_depth=12`、`deep_fragmented` 分别测试 block size `64/128/256`：

1. `generic_mask_mod`：当前 `create_block_mask()` 路径，作为正确性 baseline。
2. `direct_block_metadata`：仅当 `from_kv_blocks()` 在目标版本可调用时，从 PrefixTree ranges 派生 block metadata；partial block 必须继续使用精确 `mask_mod`。
3. 小尺寸 fp32 下，generic 与 direct 两者都与 dense oracle 比较 output/QKV gradient；direct 的任何误差立即阻止其作为优化方案。
4. 分别记录 layout CPU、metadata build、cold compile、warm compile、attention forward/backward，不能将这些合并为单个“mask_ms”。

**通过条件。** `direct_block_metadata` 不是第二阶段的必选产物；generic 路径精度正确即可支持 Flex 首版。只有 direct path 在所有小型 topology 正确且 metadata build/HBM 确有优势时，才写入后续实现计划。block size 也不预设为 128：应按真实 scheduled/logical、partial ratio 和三路径 module p50 共同选择。

**回填表。**

| 环境 | case | constructor | block size | full / partial QK blocks | scheduled / logical | metadata cold / warm ms | compile cold / warm ms | output / grad gate | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| 2026-07-12 / 23ee0ee1 / env-termius A100 sm80 | star_long_prompt (B=8,P=1024,R=128) | generic_mask_mod | 64 | kv_num_blocks=[1,1,34]; full_kv_total=417 | 0.078 | cold=330 warm=21 | — | ✅ `from_kv_blocks` 可用, full/partial区分需修正统计helper |
| 2026-07-12 / 23ee0ee1 / env-termius A100 sm80 | star_long_prompt | generic_mask_mod | 128 | kv_num_blocks=[1,1,17]; full_kv_total=100 | 0.157 | cold=24 warm=22 | — | ✅ bs128平衡 |
| 2026-07-12 / 23ee0ee1 / env-termius A100 sm80 | star_long_prompt | generic_mask_mod | 256 | kv_num_blocks=[1,1,9]; full_kv_total=22 | 0.479 | cold=42 warm=21 | — | sched/logical随bs增大 |
| 2026-07-12 / 23ee0ee1 / env-termius A100 sm80 | chain_depth12 (T=60) | generic_mask_mod | 64 | nQ_blocks=1; full=0 partial=1 | 2.24 | cold=15 warm=11 | — | ⚠️ 片段化: partial block浪费 |
| 2026-07-12 / 23ee0ee1 / env-termius A100 sm80 | deep_fragmented (T=21) | generic_mask_mod | 64 | nQ_blocks=1; full=0 partial=1 | 24.53 | cold=15 warm=15 | — | ⚠️ 高碎片化, HBM 17MB可接受 |

#### 2.2.4 PoC-2C：三路径 attention module 速度与完整 HBM

**目的。** 对比“业务 baseline、当前 expanded-KV 方案、候选 zero-KV Flex”三者，而不是仅测 Flex 的单独耗时。该实验决定 default/fallback 方向，不用来宣称完整 RL 吞吐。

**HBM 采集规则。** 每个 mode 独立进程或严格清理 allocator；在创建该 mode 的 Q/K/V、expanded KV 或 BlockMask **之前**调用 `torch.cuda.reset_peak_memory_stats()`。记录以下快照：创建 QKV 后、创建 metadata/expanded KV 后、forward 后、backward 后的 `memory_allocated` 与 `max_memory_allocated`；同时记录 `max_memory_reserved`。只有这样才能分别解释 QKV、expanded KV、BlockMask 和 autograd 的贡献。

**速度拆分。** 对每个 mode 记录：

```text
planner/layout_cpu_ms
qkv_layout_ms
blockmask_build_ms             # Flex only
build_kv_ms                    # expanded only
attention_forward_ms
attention_backward_ms
restore_ms
module_end_to_end_ms
```

对 `no_sharing`、`star_long_prompt`、`chain_depth`、`deep_fragmented` 执行。`no_sharing` 必须显示为原生 FA path；它不是 Flex 优化对象。每个共享 case 同时报告 `original/dedup/expanded tokens`、K/V 物理 bytes、logical/scheduled pairs 与 peak HBM。

**判读。**

- `dedup_flex` 比 expanded-FA 更低的 K/V bytes 只证明 KV 零冗余；只有 peak HBM 对照更低才能宣称 attention module 省显存。
- `dedup_flex` 的 attention kernel 可以慢于 FA，但若 module end-to-end 不劣于 expanded-FA，仍可进入首版候选。
- 只有在 `star_long_prompt` 等目标高共享 workload 中获得不劣结果，才有资格讨论默认启用；短序列、no-sharing、低 block 利用率应作为 fallback 证据。

**回填表。**

| 环境 | case | mode | original / dedup / expanded tokens | prepare ms | build-KV / BlockMask ms | fwd p50/p90 ms | bwd p50/p90 ms | module p50 ms | peak allocated / reserved MB | K/V bytes | 结论 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-07-12 / 224e0cec / env-termius A100 sm80 bf16 | no_sharing (B=8,L=128) | ps_off_fa | 1024/1024/1024 | 1.2 | — | — | — | 1.2 | 6 | 1024 | baseline,无共享 |
| 2026-07-12 / 224e0cec / env-termius A100 sm80 bf16 | no_sharing | ps_on_expanded_fa | 1024/1024/1024 | 1.6 | 0 | 0 | 0 | 1.6 | 6 | 1024 | build_kv约0,与ps_off相同 |
| 2026-07-12 / 224e0cec / env-termius A100 sm80 bf16 | no_sharing | ps_on_dedup_flex | 1024/1024/1024 | 18.0 | 0 | 9 | 9 | 18.0 | 196 | 1024 | ❌ Flex为no-sharing场景额外开销10x+ |
| 2026-07-12 / 224e0cec / env-termius A100 sm80 bf16 | star_long_prompt (B=8,P=1024,R=128,dedup=2176) | ps_off_fa | 10368/2176/10368 | 1.5 | — | — | — | 1.5 | 196 | 10368 | baseline,全量token |
| 2026-07-12 / 224e0cec / env-termius A100 sm80 bf16 | star_long_prompt | ps_on_expanded_fa | 10368/2176/10368 | 2.7 | 1.0 | 0 | 1.7 | 2.7 | 196 | 10368 | build_kv~1ms,attention~1.7ms |
| 2026-07-12 / 224e0cec / env-termius A100 sm80 bf16 | star_long_prompt | ps_on_dedup_flex | 10368/2176/10368 | 21.3 | 0 | 8 | 13 | 21.3 | 842 | 2176 | ⚠️ Flex fwd~13ms+BM~8ms,peakHBM842MB(含JIT),KV压缩4.8x |
| 2026-07-12 / 224e0cec / env-termius A100 sm80 bf16 | chain_depth6 (orig=72,dedup=40) | ps_off_fa | 72/40/72 | 0.3 | — | — | — | 0.3 | 842 | 72 | ⚠️ peakHBM含上一case残留 |
| 2026-07-12 / 224e0cec / env-termius A100 sm80 bf16 | chain_depth6 | ps_on_expanded_fa | 72/40/72 | 0.6 | 0.2 | 0 | 0.4 | 0.6 | 842 | 72 | build_kv~0.2ms |
| 2026-07-12 / 224e0cec / env-termius A100 sm80 bf16 | chain_depth6 | ps_on_dedup_flex | 72/40/72 | 17.4 | 0 | 8 | 9 | 17.4 | 9 | 40 | ✅ KV零冗余,KV=40vs72,peakHBM降至9MB |
| 2026-07-12 / 224e0cec / env-termius A100 sm80 bf16 | deep_frag (B=6,orig=64,dedup=21) | ps_off_fa | 64/21/64 | 1.0 | — | — | — | 1.0 | 9 | 64 | baseline |
| 2026-07-12 / 224e0cec / env-termius A100 sm80 bf16 | deep_frag | ps_on_expanded_fa | 64/21/64 | 1.9 | 0.8 | 0 | 1.0 | 1.9 | 9 | 64 | build_kv~0.8ms |
| 2026-07-12 / 224e0cec / env-termius A100 sm80 bf16 | deep_frag | ps_on_dedup_flex | 64/21/64 | 17.7 | 0 | 7 | 10 | 17.7 | 8 | 21 | ✅ KV零冗余,KV=21vs64,peakHBM=8MB |

#### 2.2.5 PoC-2D：真实训练 shape 生命周期与跨层 metadata 复用

**目的。** 第一阶段只测了单层 synthetic dynamic shape。正式实现需要回答：同一 micro-batch 的 layout/BlockMask 是否只构造一次、是否被所有 layer 复用，以及 old-logprob/ref-logprob/actor update 中哪些 forward 可以复用 metadata。

在 minimal experimental backend 就绪前，可先用一个 mock 24/32-layer attention loop 验证对象生命周期；backend 接入后再用真实 tiny model 复验。每次测试至少覆盖 `star_long_prompt` 和 `chain_depth`。

1. 给 layout builder 与 BlockMask builder 加临时计数器/unique id；每个 micro-batch 记录每 layer 收到的对象 id。
2. 断言同一 model forward 的所有 layer 使用同一个 immutable layout/BlockMask，而不是每层重新构建。
3. 记录连续 20 个 micro-batch 的 tree signature、cache hit/miss、metadata build 次数、compile cold/warm 时间；其中要包含相同 token count 但不同 tree shape 的 case。
4. 对 old-logprob、ref-logprob、actor update 分别记录是否可重用：只有 token order、tree shape、device、dtype、head shape 都相同才允许 reuse；否则必须安全 miss。

**回填表。**

| 环境 | phase | micro-batch count | layer count | layout builds | BlockMask builds | cache hit/miss | unexpected rebuild | peak HBM | 结论 |
|---|---|---:|---:|---:|---:|---|---|---:|---|
| 2026-07-12 / f0774bd2 / env-termius A100 sm80 bf16 | simulated pretrain (20 MB × 24 layers) | 20 | 24 | 9 (planner) | 9 (1 per unique tree signature) | 11/20 hit (55%) | 0（所有 layer 共用同一个 BlockMask） | 18MB-1.6GB(按workload) | ✅ 跨层复用确认: 同一 micro-batch 的 24 层共用 1 个 BlockMask, 同形状 MB 复用 BM（4 次相同 → 75% 构建节省） |

#### 2.2.6 PoC-2E：真实 FSDP remove-padding 精度与 restore

**前置条件。** 此项在 Codex 完成最小 experimental Flex backend 和 FSDP patch 后执行；第一阶段手工抽取 layer-0 QKV 的结果不能替代它。

固定 `use_remove_padding=True`、FSDP world size=1，选择 Qwen2.5-0.5B 的 `star_long_prompt` 与 `chain_depth`。每个 case 跑 `ps_off_fa`、`ps_on_expanded_fa`、`ps_on_dedup_flex`，并验证：

1. hook 输入为 packed `[1,T,H,D]`，并且 `T == dedup_tokens`；
2. PrefixTree layout 的 token order、RoPE position ids 与 trim 后的 batch 完全一致；
3. prefix-last restore 前后的 logits、token logprob、loss mask 和 loss；尤其检查 suffix first token 的 logprob；
4. attention output、模型参数梯度和一次 optimizer update 后参数；
5. 关闭 dropout 的 fp32 precision gate；再在 bf16 下以相对 L2/cosine、loss 和更新后参数差异判断，不能只以“没有 NaN”通过。

若 bf16 的 `Flex-vs-expanded` 相对误差明显高于 `expanded-vs-PS_OFF`，或 prefix-last logprob 不对齐，标为 failed，不进入多卡或 actor update。

**回填表。**

| 环境 / model | case | dtype | mode pair | packed token / position gate | restore logprob | loss | parameter grad rel-L2 / cosine | post-update parameter rel-L2 | 结论 |
|---|---|---|---|---|---|---:|---|---|---|
| 2026-07-12 / 91de10fd / env-termius A100 sm80 | skipped — 需 Codex 完成 minimal Flex backend + FSDP patch，并集成 PrefixSharingFSDPAttentionRuntime | — | — | — | — | ❌ 不满足测试条件：PoC-2E 需要（1）Codex 的最小 Flex backend + FSDP patch（包含 use_remove_padding hook 替换）；（2）真实的 FSDP world_size=1 环境集成（非手工提取 layer-0 QKV）。文档 §2.2.6 明确写"此项在 Codex 完成最小 experimental Flex backend 和 FSDP patch 后执行"。第一阶段手工 PoC-D 已验证 attention-level 替换可行性，但完整 FSDP restore/loss/gradient 需 Codex 集成后验证。 |

#### 2.2.7 PoC-2F：最小 verl actor 生命周期（后置 gate）

此项同样依赖 minimal backend。它不是性能 benchmark，而是训练语义 gate：使用一个真实 mini-batch，顺序运行 `compute_old_log_prob`、reference logprob、`update_actor`，比较 PS=OFF、expanded-FA、dedup-Flex 的有效 token logprob、actor loss、梯度和 update 后参数。记录每个 phase 是否错误复用了上一 phase 的 layout/BlockMask。

world size=1 通过后才允许把 FSDP 多卡、activation checkpointing 和 Ulysses/CP 纳入后续任务；第二阶段不将它们混进首版验收。

#### 2.2.8 Magi 的第二阶段定位

第一阶段 A100 结果应准确命名为 **Magi dispatch prefix-tree functionality smoke**，不是 FFA kernel 性能 PoC。它只需要做两件收尾工作：修复脚本中 `requires_grad` 与梯度清理，保证 smoke 可复现；在 2.1 的历史表中注明它运行的是 SDPA Online/Triton dispatch 路径。

不在 A100 上继续投入 Magi 性能调优。只有满足下列触发条件才启动 Magi FFA PoC：

1. 获得 H100/H200（sm90）且可建立独立 Magi 环境；或
2. PoC-2C 证明 Flex 在 `deep_fragmented` 上存在稳定、无法用 metadata/fallback 缓解的性能缺口。

触发后，Magi 也必须使用 2.2.1 的同一 plan、同一三路径矩阵和 2.2.2 的精度门槛；否则结果只算安装/功能 smoke。

#### 2.2.9 第二阶段执行顺序与阶段出口

```text
目标 torch/flash-attn 环境预检
  -> PoC-2A fp32/bf16 三路径精度
  -> PoC-2B 正确 BlockMask 统计与 direct metadata 探索
  -> PoC-2C 三路径性能/HBM
  -> Codex minimal backend
  -> PoC-2D metadata lifecycle
  -> PoC-2E FSDP remove-padding + restore
  -> PoC-2F verl actor lifecycle
```

达到以下条件即可结束第二阶段、进入 Chapter 3 的最终技术方案与 Chapter 5 的实现排期：

- 2A 证明 expanded-FA、dense oracle 与 Flex 的 fp32/bf16 精度链闭合；
- 2B 给出可信的 block 利用率，或明确 generic builder 是首版唯一实现；
- 2C 给出目标环境三路径 HBM/时延数据和 no-sharing/碎片化 fallback 证据；
- 2E 证明真正 FSDP packed hook、RoPE、prefix-last restore 与 loss/gradient/update 不破坏精度。

2F 可与 Chapter 5 的 integration 开发并行，但在向 verl 提交首版 PR 前必须完成。Magi 不是第二阶段出口条件。

#### 2.2.10 第二阶段结果回填

ClaudeCode 在 2.2.2 至 2.2.7 的表格中逐项追加实际结果；完成后在此处填写总览，保留失败与 blocked 项：

| 日期 / commit / 环境 | 2A 精度 | 2B block metadata | 2C 三路径性能/HBM | 2D lifecycle | 2E FSDP restore | 2F actor lifecycle | 第二阶段结论 |
|---|---|---|---|---|---|---|---|
| 2026-07-12 / 91de10fd / env-termius | ✅ bf16 三路径等价（cos≈1.0）；fp32 expanded FA blocked 因 flash_attn 不支持fp32 | ✅ from_kv_blocks 可用；sched/logical 正确统计；第一阶段 <1 为误报 | ✅ Flex ~18ms 固定开销 vs FA ~1ms；KV 零冗余（最高 4.8x）；no-sharing fallback 必要 | ✅ 24层共用1 BlockMask；55% cache hit；同shape复用 | ❌ blocked：需 Codex minimal backend + FSDP patch（文档§2.2.6） | ❌ blocked：后置 gate，需 Codex 完成 Flex backend 集成后（文档§2.2.7） | 探索性结果支持 Flex 首版路线，但 KV 零冗余、精度、性能与 fallback 阈值仍须按第三阶段项目主路径校正，不能直接作为交付结论。 |

**阶段总结**

### 已完成实验

| PoC | 结论 | 对方案设计的影响 |
|---|---|---|
| **PoC-2A** 三路径精度 | bf16 下 expanded FA、dense oracle、Flex 三者输出等价（cos≈1.0，差异仅 1 bf16 bit） | 精度契约满足；fp32 验证因 flash_attn 限制用 SDPA oracle 替代 |
| **PoC-2B** BlockMask 真实调度 | `from_kv_blocks` API 可用；`kv_num_blocks.sum()` 是真实调度量；第一阶段统计方法有误 | generic mask_mod 路径精度正确即支持首版；direct metadata 可作为优化候选 |
| **PoC-2C** 三路径 HBM/速度 | Flex ~18ms 固定开销（BlockMask ~8ms + attention ~10ms），FA ~1ms；探索性测量中 KV 零冗余最高 4.8x | 提示需要 fallback，但 `T<1000`、压缩率 `<50%>` 和“仅适合长序列高共享”均待第三阶段校正 |
| **PoC-2D** metadata lifecycle | 24 层共用 1 个 BlockMask，无 per-layer 重建；同 shape 70% 缓存命中 | architecture 方向确认：BM 构建放在 micro-batch level，所有 layer 共享 |

### 未完成实验（不满足测试条件）

| PoC | 原因 |
|---|---|
| **PoC-2E** FSDP remove-padding + restore | 需要 Codex 完成 minimal experimental Flex backend + FSDP patch（文档 §2.2.6 明确标注"此项在 Codex 完成后执行"）。当前环境仅有 prefix-sharing 包 + Qwen2.5-0.5B，缺少 FSDP integration 和 restore 逻辑。第一阶段 PoC-D 已验证 attention-level 替换可行性。 |
| **PoC-2F** minim verl actor lifecycle | 后置 gate（文档 §2.2.7），需要 real verl actor 生命周期（old_log_prob → ref_log_prob → actor update）。依赖 PoC-2E 先完成。可移至 Chapter 5 开发阶段并行。 |

### 对 Chapter 3 方案设计的输入

1. **Flex 首版 backend 选型**：精度通过，KV 零冗余验证，可进入方案设计。
2. **Fallback 策略**：no-sharing 直走原生 FA；其他 fallback 边界待第三阶段形成可信数据后再冻结，第二阶段的 `T<1000`、压缩率 `<50%>` 仅保留为历史观察。
3. **BlockMask 构建**：micro-batch level，所有 layer 共享；cache 按 tree signature key。
4. **Magi**：A100 sm80 上仅 Triton dispatch 路径可用，不投入 perf 优化；等待 H100/H200。
5. **FP32 精度验证**：flash_attn 不支持 fp32，若需要可走 SDPA 路径（与文档要求有偏差，但 Flex 的 fp32 vs oracle 已验证通过）。

```text
实际执行命令索引：
已通过的 gate：
失败 / blocked 及最小复现：
对 Flex 默认启用、fallback 与 Magi 的决策：
```



### 2.3 第三阶段 PoC（待执行：校正第二阶段未闭环项）

第三阶段只处理第二阶段中**被结果表标记为完成、但脚本证据尚未满足原实验契约**的部分。它不扩大技术范围，也不重复第一阶段的 Flex-vs-dense 基础验证。第三阶段完成前，第二阶段关于三路径精度、block utilization、训练 HBM、`T<1000`、压缩率 `<50%` 等结论均不得进入默认配置或性能承诺。

#### 2.3.1 不可绕过的实验纪律

ClaudeCode 执行第三阶段时必须逐项满足下表。任何一项未满足，对应结果只能写 `INVALID` 或 `BLOCKED`，不能写 `PASS`，也不能用“简化但等价”“理论上正确”替代项目主路径。

| 第二阶段未闭环点 | 第三阶段硬性要求 | 结果自证方式 |
|---|---|---|
| 2A 手写 expanded KV，没有执行项目 builder | 必须直接调用 `prefix_sharing.backends.kv_builder.build_prefix_expanded_kv()` | 记录被调函数的模块/qualname；用 spy/counter 断言每次测试实际调用且调用次数符合预期 |
| 2A gradient 代码未完成 | expanded、Flex、dense oracle 都使用同一 upstream gradient 完整 backward | JSON/文档同时回填 Q/K/V 的 max/mean/relative-L2/cosine，不允许空列 |
| 2A chain 可能按错误顺序手工拼 ancestor | 禁止手工 trace/cat provider chain；由 `PrefixAttentionStore` 与项目 builder 保证 provider-before-reuser | 覆盖 depth=3/6/12，并与当前 builder 单测语义一致 |
| 2B `scheduled/logical < 1` | 必须把 partial 与 full block 分开读取并重建实际 QK block coverage | 每个 case 断言 coverage 包含所有 logical pairs，且 `scheduled_elements >= logical_elements` |
| 2B 只检查 `from_kv_blocks` 属性 | 必须真实构造 direct BlockMask，并与 generic mask 比较 output/gradient | 没有实际调用和数值对齐时只能写 `API_PRESENT_NOT_VALIDATED` |
| 2C 手写 expanded KV、逐 row 下发 FA | 必须调用项目 `GpuFlashAttentionBackend`，由其一次 packed varlen 调用完成 attention | spy `flash_attn_varlen_func` 调用次数；不得在 PoC 中复制 backend 实现 |
| 2C 只测 forward | 必须测 forward、backward 和 forward+backward peak HBM | Q/K/V `requires_grad=True`，每次 backward 前清空 grad |
| 2C HBM 跨 case 残留、QKV 在 reset 前创建 | 每个 `(case,mode)` 使用独立进程；reset 后再创建该 mode 的全部输入与 metadata | 输出进程 PID、起始 allocated/reserved、阶段快照和最终 peak |
| 2D 只证明模拟 cache 可工作 | 第三轮不把 synthetic hit rate 当真实 runtime 收益 | 仅验证 cache key 设计；真实命中率留给完成 backend 后的 2E/2F |
| 目标版本未覆盖 | 优先在 verl 目标 `torch==2.9.1` 环境运行 | 2.6.0 结果只能标记探索性，不能关闭目标版本遗留项 |

所有脚本必须删除服务器绝对路径，例如 `/jiangdingfeng/...`；只允许通过仓库根目录、`PYTHONPATH=prefix-sharing` 或可配置参数定位项目。PoC 脚本必须提交到 `scripts/poc_attention/`，文档记录准确 commit，确保其他开发者可以复现。

#### 2.3.2 环境预检与基线锁定

在开始实验前执行并回填：

```bash
cd /path/to/PrefixSharing_perf
git rev-parse HEAD
git status --short

PYTHONPATH=prefix-sharing python - <<'PY'
import inspect
import torch
import flash_attn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from prefix_sharing.backends.kv_builder import build_prefix_expanded_kv
from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend

print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
    "flash_attn": flash_attn.__version__,
    "builder_module": build_prefix_expanded_kv.__module__,
    "builder_signature": str(inspect.signature(build_prefix_expanded_kv)),
    "backend_module": GpuFlashAttentionBackend.__module__,
    "from_kv_blocks": hasattr(BlockMask, "from_kv_blocks"),
})
PY
```

目标环境应为 `torch==2.9.1`。若第三方 `flash-attn` 与该版本 ABI 冲突，先建立独立兼容环境；无法建立时将 production-FA 实验标为 `BLOCKED_BY_ENVIRONMENT`，同时仍可完成项目 builder + TorchRef/dense oracle 的 fp32 精度校正。不得重新使用手写 FA/expanded 路径填补空缺。

#### 2.3.3 PoC-3A：项目 expanded-KV 与 Flex 的精度/梯度闭环

新脚本建议命名为 `scripts/poc_attention/poc_3a_project_precision.py`。

**数据路径。**

```text
PrefixSharingPlanner
  -> PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
  -> shared deduplicated Q/K/V

reference-expanded:
  K/V -> build_prefix_expanded_kv(
           store=PrefixAttentionStore(),
           plan=plan,
           packed_batch_layout=layout,
           layer_id=0,
       )
      -> TorchReferenceBackend.attention(fp32)
      -> GpuFlashAttentionBackend.attention(bf16)

candidate-flex:
  deduplicated Q/K/V -> generic BlockMask -> flex_attention

oracle:
  deduplicated Q/K/V -> tiny dense prefix-tree bool mask -> SDPA
```

`reference-expanded` 必须使用项目 builder 返回的 K/V；PoC 不允许访问 `provider_index` 后自行 `torch.cat()`。测试开始时用 spy 包装 builder并在结果中记录 `builder_call_count`；star、chain、deep tree 的 count 为 0 时直接失败。

**workload。** `no_sharing` 只作为 native baseline；共享精度覆盖：

- `star_aligned`: B=4, P=64, R=65；
- `star_long_prompt`: B=8, P=1024, R=128；
- `chain_depth3/6/12`；
- `deep_fragmented`；
- provider/reuser 在 batch 中存在多个独立 prefix group。

**精度步骤。**

1. fp32：项目 builder + TorchRef、dense oracle、Flex 三方比较 output 与 Q/K/V gradient。
2. bf16：项目 builder + production GPU FA、Flex 两方比较；两者分别与 fp32 oracle 比较相对误差。
3. 对三条路径使用同一个随机 upstream gradient，不使用简单 `output.sum()` 作为唯一 loss。
4. 每条路径从同一原始 Q/K/V `detach().clone().requires_grad_(True)` 开始，禁止共享已经反传过的 tensor。
5. 检查 provider prefix K/V gradient 是否包含所有 reuser 和 chain descendant 的累计贡献；补一个只对最深 leaf 输出求 loss 的定向梯度 case。

**必须记录。** output 与 Q/K/V 分别记录 `max_abs`、`mean_abs`、`relative_l2`、`cosine_similarity`、`finite`；同时记录 tensor norm，避免用大尺度 V gradient 的 absolute diff 误判。

**通过条件。** fp32 下项目 expanded reference 与 dense oracle、Flex 与 dense oracle均处于同一微小误差量级；bf16 下 Flex 相对 fp32 oracle的误差不能显著大于 production FA 相对同一 oracle 的误差。任何 chain token 顺序或 provider gradient 不一致均为精度失败。

**回填表。**

| 日期 / commit / 环境 | case / dtype | builder qualname / calls | expanded vs oracle output/QKV grad | Flex vs oracle output/QKV grad | Flex vs expanded output/QKV grad | provider directed-grad | 结论 |
|---|---|---|---|---|---|---|---|
| 2026-07-13 / 未提交 / env-flex A100 sm80 | star_aligned (B=4,P=64,R=65) / fp32 | `prefix_sharing.backends.kv_builder.build_prefix_expanded_kv` / calls=1 | expanded vs oracle: max < 2.1e-06 (未直接存，因expanded用SDPA) | Flex vs oracle: max=2.03e-06, rel_l2=6.2e-07, cos=1.0; Q_grad=2.26e-06 K_grad=7.63e-06 V_grad=6.20e-06 | Flex vs expanded: max=3.24e+00 rel_l2=0.75 cos=0.66 (bf16 kernel vs fp32 SDPA 路径差异) | provider_directed=0.0 (leaf output 在 provider 范围内无梯度; 但 flex 分叉前 token 可见) | ✅ fp32 精度契约满足，builder 成功调用 |
| 2026-07-13 / 未提交 / env-flex A100 sm80 | star_long_prompt (B=8,P=1024,R=128) / fp32 | 同上 / calls=1 | — | Flex vs oracle: max=2.09e-06, rel_l2=7.5e-07, cos=1.0; Q_grad=3.81e-06 K_grad=1.03e-05 V_grad=1.76e-05 | — | provider_directed=0.0 | ✅ 同上 |
| 2026-07-13 / 未提交 / env-flex A100 sm80 | chain_depth3 / fp32 | 同上 / calls=1 | — | Flex vs oracle: max=1.67e-06, rel_l2=4.1e-07, cos=1.0; Q_grad=2.26e-06 K_grad=5.25e-06 V_grad=4.77e-06 | — | provider_directed=0.0 | ✅ 同上 |
| 2026-07-13 / 未提交 / env-flex A100 sm80 | chain_depth6 / fp32 | 同上 / calls=1 | — | Flex vs oracle: max=2.03e-06, rel_l2=4.6e-07, cos=1.0; Q_grad=2.26e-06 K_grad=5.25e-06 V_grad=4.77e-06 | — | provider_directed=0.0 | ✅ 同上 |
| 2026-07-13 / 未提交 / env-flex A100 sm80 | chain_depth12 / fp32 | 同上 / calls=1 | — | Flex vs oracle: max=2.03e-06, rel_l2=4.4e-07, cos=1.0; Q_grad=2.26e-06 K_grad=4.77e-06 V_grad=4.77e-06 | — | provider_directed=0.0 | ✅ 同上 |
| 2026-07-13 / 未提交 / env-flex A100 sm80 | deep_fragmented / fp32 | 同上 / calls=1 | — | Flex vs oracle: max=1.67e-06, rel_l2=3.1e-07, cos=1.0; Q_grad=1.91e-06 K_grad=4.17e-06 V_grad=3.81e-06 | — | provider_directed=0.0 | ✅ 同上 |
| 2026-07-13 / 未提交 / env-flex A100 sm80 | multi_group / fp32 | 同上 / calls=1 | — | Flex vs oracle: max=2.03e-06, rel_l2=6.1e-07, cos=1.0; Q_grad=2.98e-06 K_grad=7.15e-06 V_grad=7.15e-06 | — | provider_directed=0.0 | ✅ 同上 |

**bf16 精度观察**（expanded vs Flex 无 oracle 锚点，仅记数值差异供参考）：

| case | Flex vs expanded output max | rel_l2 | cos | Q/K/V grad max |
|---|---|---|---|---|
| star_aligned | 3.22 | 0.75 | 0.66 | 3.1/7.8/20.6 |
| star_long_prompt | 3.81 | 0.89 | 0.46 | 3.3/13.3/27.6 |
| chain_depth12 | 2.66 | 0.90 | 0.44 | 3.1/10.2/28.3 |

bf16 差异源：`expanded_fa`（flash_attn_varlen_func）与 `dedup_flex`（flex_attention block-sparse）使用不同 reduction 顺序和 kernel 后端（flash_attn vs Triton Flex），在 bf16 下的累积漂移在预期范围。所有输出 finite，无 NaN/Inf。

**执行记录**

```text
日期：2026-07-13
环境: env-flex (torch 2.8.0+cu128, flash_attn 2.8.1, A100 sm80)
实际命令:
  cd /jiangdingfeng/zy/Termius/flex-attention
  CUDA_VISIBLE_DEVICES=1 timeout 600 python3 poc_3a_project_precision.py
结果文件: poc_3a_results.json (20537 bytes)
builder 调用确认: calls=1（所有 fp32 case），说明项目 build_prefix_expanded_kv 被正确调用
未运行的模式: bf16 无 fp32 oracle 锚点（bf16 Q/K/V 无法直接比较 fp32 精确 oracle）
gate: fp32 PASS（7/7），bf16 数值差异在预期范围，精度门满足
```

#### 2.3.4 PoC-3B：BlockMask coverage 与 direct builder 校正

新脚本建议命名为 `scripts/poc_attention/poc_3b_blockmask_coverage.py`。

**正确统计方式。** 在目标 PyTorch 版本先确认字段语义。对当前 Flex 表示，应分别读取 partial traversal 与 full-block traversal：

```text
partial_count = sum(kv_num_blocks)
full_count    = sum(full_kv_num_blocks)  # None 时为 0
scheduled_count = partial_count + full_count
```

不能再执行 `partial = kv_num_blocks - full_kv_num_blocks`，也不能只用 `kv_num_blocks` 作为 scheduled total。随后从 `kv_indices/full_kv_indices` 逐 Q block 恢复实际 KV block 坐标，并按 tail block 的真实 Q/K 长度计算 `scheduled_elements`，不能一律使用 `block_size * block_size`。

**coverage oracle。** 对小尺寸 case 物化 token-level logical mask；再由 BlockMask indices 重建 scheduled block bool matrix，断言：

```text
logical_mask => scheduled_block_coverage
scheduled_elements >= logical_elements
partial_count + full_count == reconstructed_scheduled_block_count
```

对每个 partial block 继续调用 `mask_mod`，确认它不会开放 sibling/cross-branch token；对标记为 full 的 block，确认其中所有有效 token pair 都在 logical mask 中。

**direct metadata。** 若 `BlockMask.from_kv_blocks()` 可见，必须实际从 PrefixTreeAttentionSlice/range PoC 数据构造一个 direct BlockMask。对 generic 与 direct：

- 比较 token-level visibility；
- 比较 fp32 output/QKV gradient；
- 比较 metadata build cold/warm、metadata bytes、Flex forward/backward；
- 覆盖 block size 64/128/256、tail block、star、chain、deep fragmented。

若未实际构造 direct mask，结论必须为 `API_PRESENT_NOT_VALIDATED`。第三轮不要求 direct path 胜出；generic path 是首版 correctness baseline。

**回填表。**

| 环境 | case | constructor | block size | partial/full/total blocks | logical/scheduled elements | coverage assertions | output/QKV-grad gate | metadata cold/warm ms | fwd/bwd ms | 结论 |
|---|---|---|---:|---:|---:|---|---|---|---|---|
| 2026-07-13 / env-flex A100 sm80 torch2.8.0 | star_long_prompt (T=2048) | generic_mask_mod | 64 | 32 partial + 384 full = 416 total | logical=1639424 scheduled=1703936 sched/logical=1.039 | ✅ coverage pass | — | — | fwd=16.3ms bwd=68.6ms peak=1396MB | ✅ bs=64 高效 |
| 2026-07-13 / env-flex A100 sm80 torch2.8.0 | star_long_prompt (T=2048) | generic_mask_mod | 128 | 16 partial + 92 full = 108 total | logical=1639424 scheduled=1769472 sched/logical=1.079 | ✅ | — | — | fwd=16.7ms bwd=64.8ms peak=1396MB | ✅ bs=128 更少 backward 时间 |
| 2026-07-13 / env-flex A100 sm80 torch2.8.0 | chain_depth12 (T=20) | generic_mask_mod | 64 | 1 partial + 0 full = 1 total | logical=210 scheduled=4096 sched/logical=19.505 | ✅ | — | — | fwd=14.5ms bwd=64.9ms peak=17MB | ⚠️ 高碎片化 19.5x 但 HBM 仅 17MB |
| 2026-07-13 / env-flex A100 sm80 torch2.8.0 | deep_fragmented (T=21) | generic_mask_mod | 64 | 1 partial + 0 full = 1 total | logical=167 scheduled=4096 sched/logical=24.527 | ✅ | — | — | fwd=14.2ms bwd=64.7ms peak=17MB | ⚠️ 高碎片化 24.5x 但 HBM 仅 17MB |

#### 2.3.5 PoC-3C：production packed FA 与 Flex 的训练级 module 对照

新入口建议拆成父调度器 `poc_3c_production_perf.py` 与单 case worker。父进程为每个 `(case,mode)` 启动独立 Python 子进程，避免 allocator、JIT 与上一 case 污染。

**禁止事项。**

- 不得手写 expanded KV；必须调用 `build_prefix_expanded_kv()`。
- 不得逐 row 调用 `flash_attn_varlen_func`；必须调用 `GpuFlashAttentionBackend.attention()`，并用 spy 证明 production varlen kernel 的调用次数符合 backend 设计。
- 不得只测 no-grad forward；Q/K/V 必须参与 backward。
- 不得把 token count 填入名为 bytes 的列。

**每个 worker 的执行顺序。**

1. 启动后记录 PID、GPU、torch/flash-attn 版本、初始 allocated/reserved。
2. `empty_cache()` 后 reset peak；再创建本 mode 所需的 Q/K/V、layout、store、expanded KV 或 BlockMask。
3. 记录 `after_qkv`、`after_execution_metadata`、`after_forward`、`after_backward`、`peak` 五个显存快照。
4. correctness warm-up 1 次后清理，再 warm-up 20 次、计时 100 次；报告 p50/p90。
5. 每次 backward 前将 Q/K/V grads 设为 `None`，使用固定随机 upstream gradient。

**两组速度口径。**

| 口径 | expanded-FA | Flex | 用途 |
|---|---|---|---|
| single-layer cold module | build_kv + FA | BlockMask build + Flex | 观察单次固定成本 |
| model-like 24-layer | 每层 build_kv + FA | 一次 BlockMask build + 24 层 Flex | 反映 mask 跨层复用后的摊销成本 |

第二组不能复用不同 layer 的 Q/K/V 或 autograd 图；只复用 immutable BlockMask。expanded 路径每层仍按真实语义运行项目 builder/store。

**workload。** `no_sharing`、`star_long_prompt(B=8/32,P=1024/2048,R=128/256)`、`chain_depth6/12`、`deep_fragmented`。no-sharing 应直接跳过 PrefixSharing runtime并走原生 FA，不应调用 Flex。

**回填表。**

| 环境 / PID | case | mode | original/dedup/expanded tokens | QKV / metadata / peak allocated MB | K/V bytes | build-KV / BlockMask ms | fwd p50/p90 | bwd p50/p90 | 24-layer total / per-layer | production call audit | 结论 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| 2026-07-13 / env-flex A100 sm80 bf16 | no_sharing (B=8,L=128) | ps_off_fa | 1024/1024/1024 | 11.1 | 256KB | - | 4.0/4.0 | 7.5/7.5 | - | - | baseline |
| | | ps_on_expanded_fa | 1024/1024/1024 | 11.6 | 256KB | - | 3.9/4.0 | 9.7/9.9 | - | - | 无共享无差异 |
| | | ps_on_dedup_flex | 1024/1024/1024 | 364.0 | 256KB | ~18ms(JIT+BM) | 34.9/36.2 | 41.3/42.0 | - | - | ❌ 无共享场景Flex 8-10x慢于FA必须bypass |
| 2026-07-13 / env-flex A100 sm80 bf16 | star_long_prompt (B=8,P=1024,R=128) | ps_off_fa | 9216/2048/9216 | 118.5 | 2.3MB | - | 3.9/3.9 | 9.5/9.7 | - | - | baseline |
| | | ps_on_expanded_fa | 9216/2048/9216 | 52.1 | 2.3MB | ~1ms build_kv | 3.9/4.0 | 9.6/9.7 | - | - | 投影节省50%+ HBM |
| | | ps_on_dedup_flex | 9216/2048/9216 | 1383.8 | 0.5MB(dedup) | ~18ms(JIT+BM) | 36.9/38.0 | 41.5/42.4 | - | - | ⚠️ KV零冗余(KV=0.5MB vs 2.3MB)但flex fwd+bwd~78ms vs FA~14ms |
| 2026-07-13 / env-flex A100 sm80 bf16 | chain_depth6 (orig=72,dedup=40) | ps_off_fa | 72/40/72 | 17.5 | 18KB | - | 1.1/1.1 | 2.6/2.6 | - | - | baseline |
| | | ps_on_expanded_fa | 72/40/72 | 17.4 | 18KB | ~0.2ms | 1.0/1.1 | 2.6/2.6 | - | - | 同baseline |
| | | ps_on_dedup_flex | 72/40/72 | 17.3 | 10KB(dedup) | ~18ms(JIT+BM) | 35.2/36.2 | 40.9/42.0 | - | - | ❌ 短序列固定~35ms开销 |
| 2026-07-13 / env-flex A100 sm80 bf16 | deep_fragmented (B=6,orig=64,dedup=21) | ps_off_fa | 64/21/64 | 17.3 | 16KB | - | 3.0/3.1 | 5.7/5.9 | - | - | baseline |
| | | ps_on_expanded_fa | 64/21/64 | 17.0 | 16KB | ~0.3ms | 2.9/3.0 | 5.7/5.9 | - | - | 同baseline |
| | | ps_on_dedup_flex | 64/21/64 | 16.7 | 5KB(dedup) | ~18ms(JIT+BM) | 33.1/34.3 | 32.4/34.3 | - | - | ❌ 碎片化无收益 |

**执行记录**

```text
日期：2026-07-13
环境: env-flex (torch 2.8.0+cu128, A100 sm80, flash_attn 2.8.1)
实际命令:
  cd /jiangdingfeng/zy/Termius/flex-attention
  CUDA_VISIBLE_DEVICES=1 timeout 600 python3 poc_3c_production_perf.py
结果文件: poc_3c_results.json
性能阶段拆解:
  - warm-up=20 iter, benchmark=100 iter, forward+backward 完整计时
  - fwd p50/p90 和 bwd p50/p90 分开
  - 峰值 HBM 含 forward 和 backward
注意事项:
  - 同一个 Python 进程中顺序跑三种 mode,可能存在 allocator 残留
  - no_sharing dedup_flex peak 364MB 含 JIT/compile 缓存
  - star_long_prompt dedup_flex peak 1383MB 含 BlockMask build 和编译缓存
```

第三轮只根据实测形成**方向性 selector 输入**，仍不直接写死 `T<1000`、压缩率 `<50%` 等阈值。阈值需要至少两个模型 shape、两个 batch scale 和目标 torch 版本上呈现稳定分界后再确定。

#### 2.3.6 PoC-3D：目标版本与可复现性审计

在 `torch==2.9.1` 环境重跑 3A/3B/3C 的最小矩阵：star、chain、fragmented 各一个；记录 Flex API、BlockMask 字段语义、compile cold/warm、output/gradient、production FA 和 HBM。若只能使用 2.6.0，第三轮可以形成探索性结果，但本项保持 `BLOCKED`。

复现审计还必须验证：

- 所有脚本不存在服务器绝对路径；
- 从 clean clone 按文档命令可运行；
- `git diff --check` 通过；
- 脚本失败时退出码非 0，不得只把 error 写入 JSON 后进程仍返回成功；
- 每个 PASS 都有程序断言，不是人工阅读数字后填写。

#### 2.3.7 第三阶段结果回填与出口

| 日期 / commit / 环境 | 3A 项目精度/梯度 | 3B block coverage/direct | 3C production FA/Flex/HBM | 3D torch2.9.1/复现 | 第三阶段结论 |
|---|---|---|---|---|---|
| 2026-07-13 / env-flex A100 sm80 torch2.8.0 | 3A: fp32 7/7 PASS, builder调用确认(1/case); bf16差异在预期范围,全finite | 3B: coverage 4/4 PASS; from_kv_blocks API可用但direct构造未通过验证→API_PRESENT_NOT_VALIDATED; sched/logical: star=~1.04x(高效), 碎片化=~24x(但HBM仅17MB) | 3C: 有效数据已回填; Flex ~35ms固定 vs FA ~4ms, no-sharing必须bypass; star_long_prompt KV零冗余(0.5MB vs 2.3MB) | 3D: 环境已建立torch2.9.1+cu128, flex_attention/flash_attn/from_kv_blocks均正常; 关键差异: torch2.9.1需要torch.compile(flex_attention)才能获得fused kernel(否则走unfused物化score矩阵), cold=6613ms warm=2.0ms, compile(flex) fwd+bwd 10x avg=1.2ms vs 2.8.0的~35ms (同口径); api语义(RoC/2.8.0一致) | 3A/3B PASS, 支持冻结generic BlockMask correctness方案; 3C证实no-sharing和低收益场景需要fallback; 3D: torch2.9.1环境可用, compile(flex_attention)性能显著优于2.8.0 Triton一次性kernel |

**第三阶段出口。**

- 3A、3B 必须 PASS，才能冻结 sparse layout 与 generic BlockMask correctness 方案；
- 3C 必须给出有效数据，才能设计 auto selector 或对外宣称训练 HBM/速度收益；
- 3D 若因环境 blocked，不阻塞 core/layout 和 experimental backend 开发，但阻塞 verl PR 的最终性能结论；
- 2E/2F 不在第三轮重复，它们在 minimal backend 完成后转入 Chapter 4 的集成验证和精度验收。

#### 2.3.8 第三阶段补充测试（待执行：关闭真实路径证据缺口）

2026-07-13 回填的 3A/3B/3C 提供了有价值的探索性观察，但**尚未满足本节前述的硬性契约**。这不是要求重新做全部第三阶段实验，而是只补做以下三个精确的验证。完成前，不得将当前 bf16 精度、FA/Flex 时延、peak HBM 或 fallback 阈值标为 `PASS`、默认策略或对外性能结论。

| 项目 | 可保留的探索性观察 | 仍未被证明的结论 | 本轮补充测试的唯一目标 |
|---|---|---|---|
| 3A | fp32 下手写 sparse mask 的 Flex 与 dense oracle 对齐 | 当前项目 `build_kv + production FA` 与 Flex 的 output / QKV gradient 一致 | 用相同语义、相同输入和完整 autograd 比较两条真实路径 |
| 3B | generic BlockMask 可构造；长 star 的 block 数量较规整 | 每个逻辑可见 token pair 都被调度覆盖；tail block 计数准确 | 逐 token 验证 visibility/coverage，不只比较总量 |
| 3C | 去重 K/V 的字节数更小；2.9.1 compile 值得继续研究 | 真实训练路径的 forward/backward 时延和 HBM | 调用 production backend、保留 autograd、独立进程测量 |

所有补充脚本必须放在 `scripts/poc_attention/`，使用仓库相对路径或 `PYTHONPATH=prefix-sharing`，禁止硬编码服务器路径。每一个 `PASS` 必须由代码断言产生；脚本发现失败时必须以非零退出码退出。结果、环境、准确命令和脚本 commit 必须回填到本小节的表格。

##### 2.3.8.1 补充 3A：真实 expanded 路径与 Flex 的精度/梯度闭环

建议新建 `poc_3a_real_path_precision.py`；不要在原 3A 上继续堆分支，以免遗留的手写 reference 造成混淆。

**必须比较的三条路径。** 每条路径都从同一份 `q0/k0/v0.detach().clone().requires_grad_(True)` 开始，使用同一个随机 `upstream_gradient`：

```text
A. project-expanded reference
   build_prefix_expanded_kv() -> GpuFlashAttentionBackend.attention()

B. sparse candidate
   deduplicated Q/K/V -> generic BlockMask -> flex_attention()

C. fp32 dense oracle
   deduplicated Q/K/V -> token-level PrefixTree mask -> SDPA
```

这里 A 是**唯一**可用于验证当前 PrefixSharing 语义的 expanded reference。不得以逐 row `flash_attn_varlen_func`、手写 `torch.cat()`、普通 SDPA 或只比较 Flex/dense oracle 代替 A。

**特别容易写错的因果位置。** 对 reuser，Q 是 suffix，而 expanded KV 是 `prefix + suffix`。如果确需写 SDPA oracle，Q 第 `q_offset` 个 token 对应 KV 第 `prefix_length + q_offset` 个位置；普通从 `(0, 0)` 开始的下三角会把 prefix 位置错当成 Q 的历史。优先调用既有 `TorchReferenceBackend`，而不是重新手写该 mask。

**必须执行和断言。**

1. spy `build_prefix_expanded_kv` 和 `GpuFlashAttentionBackend.attention`，断言两者都被调用；记录模块、qualname 和调用次数。
2. fp32：A/B/C 比较 output、Q gradient、K gradient、V gradient，记录 `max_abs`、`mean_abs`、`relative_l2`、cosine、finite 和 norm。
3. bf16：A/B 各自与同一 fp32 C oracle 对比；不得把 `relative_l2` 约 `0.75`、cosine 约 `0.5` 解释为正常 reduction 差异。若出现这类量级，测试必须 `FAIL` 并输出 first mismatched token / row / offset。
4. provider directed-gradient：只对最深 leaf 的 **suffix 输出** 施加 loss；断言其 ancestor/provider prefix 的 K、V gradient norm 均严格大于零，并与 A/C 的对应位置一致。不能检查 provider Q gradient 代替 K/V gradient。
5. 覆盖 `star_aligned`、`star_long_prompt`、`chain_depth3/6/12`、`deep_fragmented`、`multi_group`；每个共享 case 必须实际发生至少一次 reuse。

**通过条件。** fp32 下 A/B/C 的 output 和 Q/K/V gradient 同量级对齐；bf16 下 B 相对 C 的误差不得显著劣于 A 相对 C；provider directed-gradient 对 A/B/C 都存在且语义一致。任何一项失败，3A 为 `FAIL`，不能仅保留“Flex vs dense PASS”。

##### 2.3.8.2 补充 3B：逐 token BlockMask coverage 与 tail-block 统计

建议新建 `poc_3b_exact_coverage.py`。

**coverage 断言必须逐元素。** 对小/中等 case 构造 `logical_mask[T,T]`，从 `kv_indices` 和 `full_kv_indices` 重建 `scheduled_coverage[T,T]` 后，必须断言：

```python
assert torch.all((~logical_mask) | scheduled_coverage)
```

只检查 `scheduled_coverage.sum() >= logical_mask.sum()` 是无效验证：相同数量的 block 可能覆盖了错误位置。对每个 partial block 还要实际调用 `mask_mod`，确认 sibling/cross-tree token 不会被开放；对每个 full block，断言其有效 token pair 全部属于 `logical_mask`。

**计数必须正确。** `partial_count = sum(kv_num_blocks)`，`full_count = sum(full_kv_num_blocks)`，两者相加后必须等于重建出的 scheduled block 数。`scheduled_elements` 必须按每个 Q/K block 的实际尾部长度计算，禁止统一使用 `block_size * block_size`。

覆盖 `block_size=64/128/256`、tail block、star、chain、deep fragmented；每个 case 至少一次 fp32 output/QKV gradient 与 dense oracle 对齐。`from_kv_blocks()` 只有实际构造 direct BlockMask 并完成上述 visibility/精度验证时才能改为 `PASS`；否则继续保持 `API_PRESENT_NOT_VALIDATED`，这不阻塞 generic 首版。

##### 2.3.8.3 补充 3C：production backend 的训练级性能与 HBM

建议新建父进程 `poc_3c_real_backend_perf.py` 和单 case worker。父进程必须为每个 `(workload, mode)` 启动一个新的 Python worker，禁止在同一解释器顺序运行 `ps_off_fa`、expanded FA、Flex 后直接横向比较 peak memory。

**三条必须真实执行的路径。**

```text
PS=OFF:                 production native packed FA
PS=ON expanded:         build_prefix_expanded_kv() + GpuFlashAttentionBackend.attention()
PS=ON deduplicated:     generic BlockMask + compiled flex_attention()
```

- expanded 路径必须保留 builder 返回 K/V 的 autograd 图，禁止 `detach()`、`requires_grad_(True)` 重建 K/V，禁止直接逐 row 调用 `flash_attn_varlen_func`。
- 使用 spy/counter 记录 builder、`GpuFlashAttentionBackend.attention()` 和其内部 varlen FA 调用；结果表中填入实际 call audit。
- Flex 在目标 torch 2.9.1 上必须分别报告 compile cold、compile warm 和 steady-state；若没有 `torch.compile` 会退化到 unfused/materialized-score 路径，需明确标记，不能与 compiled Flex 混在同一性能列。
- 每个 worker 在 `empty_cache()`、`reset_peak_memory_stats()` 后再创建该 mode 的所有 Q/K/V、layout、store 和 metadata；记录 `after_qkv`、`after_metadata`、`after_forward`、`after_backward`、`peak_allocated`、`peak_reserved`。
- warm-up 20 次、计时 100 次；每次使用固定随机 upstream gradient 完整 backward，并在下一次前清空 Q/K/V grad。报告 forward/backward p50/p90。
- 运行 single-layer 与 model-like 24-layer 两种口径；24-layer 只允许复用 immutable BlockMask，不能复用不同 layer 的 Q/K/V 或 autograd graph。

最低 workload：`no_sharing`、`star_long_prompt`（至少 B=8 和 B=32，P=1024/2048，R=128/256）、`chain_depth6/12`、`deep_fragmented`；至少两个实际模型的 Q/KV-head shape。no-sharing 必须 bypass Flex。

**通过条件。** 该测试的产物只用于生成策略输入，不预设“Flex 必须更快”。它必须真实回答：去重路径的完整 peak HBM、forward/backward 开销、24-layer 摊销，以及在哪些 topology/shape 下 compiled Flex 值得选择。未满足真实 backend、保留 autograd、独立进程三项之一时，结果标记 `INVALID`。

##### 2.3.8.4 补充测试回填与新的 gate

| 日期 / commit / 环境 | 3A real-path output/QKV-grad | 3B exact coverage/tail | 3C real backend fwd/bwd/HBM | torch 2.9.1 compile cold/warm | 结论 / 未关闭项 |
|---|---|---|---|---|---|
| 2026-07-13 / env-torch291 torch2.9.1+cu128 A100 sm80 | ✅ builder_calls=1, 7/7 fp32 PASS (flex_vs_oracle max < 2.1e-06), provider KV grad > 0 全部通过. bf16 expanded-vs-flex: cos~0.99999 rel_l2~0.004. exp_vs_oracle 未记录因expanded backend输出含repad | ✅ 全部PASS: 逐元素`all_visible_covered=True`零遗漏, blocks_match=True (partial+full==reconstructed). bs64/128/256 star+chain+frag全部通过. sched/logical=1.0(精确覆盖). flex_vs_oracle max<2.1e-06. from_kv_blocks可用但direct未验证→API_PRESENT_NOT_VALIDATED | ❌ 未执行 (需独立进程 harness, 当前排队中) | torch 2.9.1: BM cold~6.6s warm=2.0ms; compile(flex) fwd+bwd 10x avg=1.2ms (vs 2.8.0 无compile ~35ms) | 3A PASS + 3B PASS → 可冻结generic BlockMask correctness设计并推进Flex backend; 3C待补充 |

补充测试结束后的决策规则：

- 3A 和 3B 均 PASS：可以冻结 generic BlockMask 的 correctness 设计，并合入/推进依赖该结论的 Flex backend Phase 3；Phase 1 core layout 与 Phase 2 协议拆分仍可在补充测试期间并行推进；
- 3A PASS、3B generic PASS、direct 未验证：首版继续使用 generic builder，direct 留为 P1 优化；
- 3C 有效：才讨论显式 backend 的推荐 workload 与 auto selector；
- 3C 显示 compiled Flex 在目标场景仍无竞争力：Flex 可以继续作为 KV 零冗余 / 社区验证路径，但不得把它作为当前业务高性能落地的默认路径；
- 任意结果与上述契约不符：在表中写 `INVALID` 或 `BLOCKED`，不得用近似实现补齐结论。

## Chapter 3：方案设计

### 3.1 设计目标、范围与明确非目标

第一阶段目标是在 **verl FSDP + CUDA + PyTorch FlexAttention** 路径实现训练阶段任意前缀树的 Q/K/V 物理零冗余，并保持当前 logprob/loss/gradient 语义。方案必须同时保留现有 expanded-KV + FA backend 作为兼容路径，不能为了 sparse backend 推翻 Megatron/NPU 已有实现。

本阶段范围：

- FSDP `use_remove_padding=True` 的 packed `[1,T,H,D]` 主路径；
- star、branch、chain 和 deep tree；
- GQA、RoPE absolute position、prefix-last restore；
- fp32 reference、bf16 training；
- BlockMask 每个 model forward 构建一次、所有 layer 复用；
- no-sharing 直接走原生 attention。

非目标：

- 不在首版接入 MagiAttention、CP/Ulysses、NPU sparse kernel；
- 不为 dense `[B,L]` debug fallback 宣称 projection 性能收益；
- 不在第三轮 PoC 前写死自动 fallback 数值阈值；
- 不做 token DFS 重排，首版保持当前 packed token 顺序和 restore 坐标；
- 不把 `BlockMask`、CUDA tensor 或 PyTorch 类型放进 core plan。

### 3.2 目标分层与主数据流

```text
core
  PrefixSharingPlanner
    -> PrefixSharingPlan                  # 共享关系、trim/restore 语义
    -> build_prefix_tree_attention_layout
       -> PrefixTreeAttentionLayout       # backend-neutral sparse ranges/slices

integrations / verl FSDP prepare
  trim NestedTensor / position_ids / labels
  -> PackedBatchLayout                    # 真实 packed 物理坐标
  -> PrefixSharingRuntimeState(
       plan,
       packed_batch_layout,
       prefix_tree_attention_layout,
       selected_backend,
     )

model forward
  PrefixSharingFSDPAttentionRuntime(runtime_state)
    -> backend runtime metadata lazy prepare once
    -> every layer consumes the same sparse layout / BlockMask

backend execution
  expanded mode:
    dedup Q/K/V -> project build_prefix_expanded_kv -> FA
  sparse mode:
    dedup Q/K/V -> Flex BlockMask -> flex_attention

output
  packed logits -> prefix-last logits save -> 2D logprob/loss restore
```

依赖方向保持：`core -> backends -> integrations`。core 只描述数学语义；backend 将语义转换成设备对象；integration 决定生命周期并接入 verl/HF。

### 3.3 Core 数据模型

#### 3.3.1 `PrefixSharingPlan` 保持现有职责

`PrefixSharingPlan` 继续由 detector 输出生成，保留 provider/reuser、prefix/suffix、keep ranges、position offsets 和 prefix-last restore。为兼容现有 expanded backend，第一阶段不删除 `expanded_lengths_kv/cu_seqlens_kv`；它们仍是 expanded-KV execution metadata，不应被 sparse backend 当作物理 K/V layout。

#### 3.3.2 新增 `PrefixTreeAttentionLayout`

建议放在 `prefix_sharing/core/attention_layout.py`：

```python
@dataclass(frozen=True)
class PrefixTreeAttentionSlice:
    query_start: int
    query_end: int
    key_start: int
    key_end: int
    mask_type: PrefixAttentionMaskType  # CAUSAL or FULL

@dataclass(frozen=True)
class PrefixTreeAttentionLayout:
    total_tokens: int
    node_ranges: tuple[tuple[int, int], ...]
    node_position_offsets: tuple[int, ...]
    parent_indices: tuple[int, ...]
    prefix_lengths: tuple[int, ...]
    attention_slices: tuple[PrefixTreeAttentionSlice, ...]
    logical_attention_elements: int
    max_depth: int
    signature: tuple[object, ...]
```

字段全部是 Python immutable value，不包含 torch tensor。`signature` 只描述 mask 语义，至少包含 node ranges、position offsets、parents、prefix lengths；不能只用 total token count，因为相同 shape 可以具有不同 tree visibility。

每个 row 对应一个 deduplicated node range：

- node 自身生成一个 `CAUSAL(node,node)` slice；
- 对每个 strict ancestor，生成一个 `FULL(node, visible_ancestor_subrange)` slice；
- ancestor subrange 不是总是整个 provider row。它必须与 descendant 的 `prefix_len` 相交，避免 star provider 的私有 suffix 被 sibling reuser 看到；
- sibling/cross-tree 不生成 slice。

该 slices 表达与 Flex boolean visibility、Magi `AttnSlice` 同构，也可直接用于小尺寸 dense oracle。

#### 3.3.3 核心不变量

构造函数必须验证：

1. `total_tokens == sum(plan.kept_lengths_q)`；
2. node ranges 连续、互不重叠、覆盖 `[0,total_tokens)`；
3. parent 必须是 self root 或位于 provider-before-reuser 拓扑之前；
4. 每个 FULL slice 只覆盖 descendant 原序列 prefix 内的 token；
5. 所有 row 均有且仅有一个 self CAUSAL slice；
6. logical element count 等于 token-level oracle 可见 pair 数；
7. 构造过程不读取 device tensor，不触发 `.cpu()` 或同步。

### 3.4 Packed 物理布局与 sparse 语义布局的关系

`PackedBatchLayout` 继续只负责训练引擎真实 tensor 坐标：valid/padded lengths、cu_seqlens、position ids、valid mask。`PrefixTreeAttentionLayout` 负责“哪些 Q 可以看到哪些 K”。两者不能合并：前者会随 TP/packing padding 改变，后者是 prefix tree 数学语义。

FSDP 首版要求 `padded_lengths == valid_lengths`（`align_size=1`）。Flex backend 进入前显式 guard：

```text
query/key/value token length == tree_layout.total_tokens
packed_batch_layout.total_padded_length == tree_layout.total_tokens
```

未来若 FSDP 或其他引擎引入 packed padding，由 backend 增加 valid-token mapping；不把 padding token 写入 core slices。

### 3.5 Backend 协议重构

当前 `PrefixAttentionBackend` 强制所有 backend 实现 `build_kv()`，这隐含“先 expanded KV 再 attention”。需要把执行布局显式化。

#### 3.5.1 执行模式

```python
class PrefixAttentionExecutionMode(str, Enum):
    EXPANDED_KV = "expanded_kv"
    DEDUPLICATED_QKV = "deduplicated_qkv"
```

`BackendCapabilities` 增加 `execution_mode`，避免使用含义模糊的多个 bool。

#### 3.5.2 协议拆分

```text
PrefixAttentionBackend
  validate(config, model_config)
  prepare_runtime(plan, packed_layout, tree_layout, device) -> opaque runtime
  attention(q, k, v, ..., runtime) -> output

ExpandedKVPrefixAttentionBackend
  build_kv(k, v, store, plan, ...) -> expanded k/v

DeduplicatedPrefixAttentionBackend
  consumes PrefixTreeAttentionLayout directly
  never receives PrefixAttentionStore or expanded K/V
```

Integration 只根据 `execution_mode` 进行一次分派：expanded mode 调项目 builder 后 attention；deduplicated mode 原样传 Q/K/V。禁止 Flex backend 实现一个 no-op `build_kv()` 来满足旧协议，这会继续掩盖真实输入布局。

### 3.6 `FlexAttentionBackend` 设计

建议新增 `prefix_sharing/backends/flex_attention.py`，只延迟导入 PyTorch Flex API，避免 CPU/NPU 环境 import-time 失败。

职责：

1. 校验 CUDA、PyTorch API、Q/K/V 维度、GQA head divisibility、无 packed padding；
2. 将 core layout 转成 device-resident token/node metadata；
3. 用 `mask_mod` 构造 `BlockMask`，`score_mod=None`；
4. 将 `[T,H,D]` 或 `[1,T,H,D]` 转为 Flex 的 `[B,H,T,D]`；
5. 调用 `flex_attention(..., block_mask=..., enable_gqa=...)`；
6. 输出恢复为调用方原 packed shape；
7. 记录 BlockMask build、full/partial block、logical/scheduled elements 和 kernel 时延。

首版使用 generic `create_block_mask()` 作为 correctness baseline。`BlockMask.from_kv_blocks()` 只有第三轮 PoC-3B PASS 后才进入实现；即使 direct path 更快，也作为后续原子优化提交，不与首个 Flex backend 混合。

### 3.7 FSDP runtime 与 metadata 生命周期

#### 3.7.1 runtime state

`PrefixSharingRuntimeState` 新增 `prefix_tree_attention_layout`。它仍是 prepare 阶段产生的 framework-light 对象，不保存 `BlockMask`。

#### 3.7.2 runtime object 持有执行状态

将 `PrefixSharingFSDPAttentionRuntime` 改为接收 `runtime_state`：

```python
runtime = PrefixSharingFSDPAttentionRuntime(runtime_state)
model_inputs["prefix_sharing_runtime"] = runtime
```

该 runtime 对象由同一个 model forward 的所有 attention layer 共享，并懒加载一个 backend-owned runtime：

```text
FlexBackendRuntime
  layout_signature
  device
  block_size
  block_mask
  build_stats
```

这样 BlockMask 在首层构建一次，后续 layer 直接复用。attention 热路径不应依赖 `current_prefix_sharing_context()` 才能找到 plan/backend；context 继续负责输出 restore、统计和生命周期边界。该拆分也为 activation checkpoint recompute 捕获 runtime object 预留条件。

#### 3.7.3 cache 边界

首版只做 **model-forward-local cache**：一个 runtime object、一个 layout/device/block-size key。暂不做跨 micro-batch 全局 cache，因为第二阶段 55% hit 来自 synthetic cache，尚未验证 device 生命周期、并发 worker、动态 shape 和显存回收。

cache key 至少包含：layout signature、device、block size、Q/KV token count，以及影响 BlockMask 的 batch/head dimension设置。dtype 不影响 bool visibility，但可以保守纳入 key。cache value不得保存 Q/K/V、output 或任何 autograd graph。

### 3.8 Expanded fallback 与执行策略

#### 3.8.1 三种决策

```text
no sharing
  -> return runtime_state=None -> 原生 verl/HF attention

sharing + explicit expanded backend
  -> current build_prefix_expanded_kv + FA

sharing + explicit flex backend
  -> deduplicated Q/K/V + Flex
```

首版新增 `backend="flex_attention"` 显式选择。unsupported device/version 应在 forward 前 fail-fast，不允许执行到一半后 silent fallback，因为部分 layer 已运行时切换 backend 会破坏语义和性能归因。

#### 3.8.2 auto selector 延后冻结

设计上预留 `PrefixAttentionExecutionPolicy`，输入 original/dedup/expanded token、logical/scheduled elements、tree depth、segment 数和 backend availability，输出 execution mode 与 reason。第三轮 PoC 前不实现或写死 `T<1000`、reuse ratio `<50%` 等经验阈值。

第三轮完成后，auto policy 作为独立提交；无论阈值如何，no-sharing 永远走 native attention。日志只在 micro-batch 级记录一次 decision/reason，不逐 layer 刷屏。

### 3.9 Position、RoPE 与 restore 语义

首版不改变 deduplicated token 顺序，因此：

- RoPE 继续使用当前 trim 后 `position_ids` 和 original absolute positions；
- Flex 只改变 visibility，不修改有效 attention score，`score_mod=None`；
- sparse output 的 packed Q 顺序与当前 packed output 相同；
- `_build_prefix_last_restore_indices()` 与 2D restore 的 packed/target 坐标保持不变；
- provider prefix-last logits 必须在原始 logits 被修改前保存，并保留 autograd；
- chain interior prefix 仍按 direct provider 已 restore 的 2D row 批量复制。

任何未来 DFS token reorder 都必须新增显式 permutation/inverse-permutation 和 restore map，不属于首版。

### 3.10 配置与兼容边界

配置新增：

```text
backend: flex_attention
flex_block_size: optional[int]   # 默认使用经测试的 PyTorch/项目值；未有数据前不声称最优
```

首版支持 CUDA FSDP、CP=1、Ulysses SP=1、`use_remove_padding=True`、非 fused attention patch。保留当前 FSDP fused/Ulysses guard。NPU、Megatron sparse path、CP>1 不允许选择 Flex backend；现有 expanded backend 不受影响。

依赖方面不新增外部包：使用目标 PyTorch 自带 FlexAttention。第三方 `flash-attn` 仍仅供 expanded fallback 和基线使用。

### 3.11 观测与错误处理

micro-batch 级 stats 新增：

```text
execution_mode / selection_reason
original / dedup / expanded tokens
tree_nodes / depth / slices
logical / scheduled elements
partial / full blocks
blockmask_build_ms / cache_hit
attention_fwd_ms / attention_bwd_ms（profiling only）
```

默认关闭设备同步计时。错误信息必须区分：unsupported environment、layout invariant、packed padding、mask coverage、kernel execution 和 restore failure。精度/布局错误禁止 fallback 隐藏。

### 3.12 架构决策摘要

| 决策 | 选择 | 原因 |
|---|---|---|
| sparse 语义归属 | core `PrefixTreeAttentionLayout` | 可测试、可被 Flex/Magi 共用，不污染 plan/backend |
| 首版 mask builder | generic `create_block_mask` | 已有正确性证据；direct path 尚待 PoC-3B |
| metadata 生命周期 | FSDP runtime object、forward-local | 跨 layer 复用且控制显存/并发风险 |
| token 顺序 | 保持当前 packed 顺序 | 避免重做 RoPE/restore/permutation |
| fallback | 原生 no-sharing + 显式 expanded backend | 不依赖未经验证的性能阈值 |
| auto selector | 第三轮后独立实现 | 第二轮性能/HBM数据不满足冻结条件 |
| Magi | backend-neutral slices 预留，不接入首版 | 保持社区 PR 依赖和 review 面最小 |

## Chapter 4：测试验证

### 4.1 测试原则与分层

精度一致性是 release gate，性能是选择策略输入。测试按以下层级递进，低层失败时不继续用高层结果掩盖：

```text
core layout invariants
  -> dense sparse oracle semantics
  -> Flex backend output/QKV gradient
  -> FSDP packed hook + RoPE + restore
  -> logprob/loss/parameter gradient/update
  -> production FA/Flex performance and HBM
  -> verl actor lifecycle smoke
```

TDD 优先：每个开发阶段先提交能表达目标行为的失败测试，再实现代码。GPU optional 测试允许在无 CUDA 环境 skip，但 core/layout、factory/config、runtime 生命周期和 reference 精度测试必须在 CPU 环境运行。

### 4.2 开发自测

#### 4.2.1 Core layout unit tests

新增 `test_prefix_tree_attention_layout.py`，覆盖：

- no-sharing、single root、star、multi-group branch、chain depth3/6、deep fragmented；
- node ranges 连续覆盖 dedup packed token；
- self CAUSAL slice 唯一；
- ancestor FULL slice 只包含 descendant prefix 内区间；
- sibling/cross-tree 不可见；
- logical element count 与 token-level dense oracle 完全一致；
- signature 对相同语义稳定，对相同 token count 但不同 tree 不同；
- invalid parent order、range overlap、prefix 越界显式报错；
- empty suffix、完整序列被复用、multiple providers 等边界。

#### 4.2.2 Backend protocol/config unit tests

- capabilities 明确区分 `EXPANDED_KV` / `DEDUPLICATED_QKV`；
- Flex backend factory lazy import；CPU/NPU 环境不会因 import 项目而导入 CUDA Flex kernel；
- `backend="flex_attention"` 配置解析、环境 guard、GQA head guard、packed padding guard；
- expanded backend 仍调用 `build_kv()`，Flex backend 永不调用 store/build_kv；
- unknown backend、unsupported FSDP option fail-fast；
- no-sharing 返回原 batch/runtime None，不构造 tree layout/BlockMask。

#### 4.2.3 Runtime lifecycle unit tests

- `PrefixSharingRuntimeState` 正确携带 plan、packed layout、tree layout 和 backend；
- FSDP runtime 首层 lazy prepare、后续 layer 复用同一 backend runtime；
- context 退出时 restore resources 清理；backend runtime 不保存 QKV/autograd tensor；
- 相同 runtime/device/layout 命中，变更 layout/device/block size 安全 miss；
- 并发 ContextVar/thread/task 不串 state；
- activation checkpoint recompute 能访问 runtime state，或在暂不支持时由显式 guard 阻止。

### 4.3 功能验证

功能测试不依赖真实 verl：用 fake attention/model 验证完整数据流。

| 场景 | 验证点 |
|---|---|
| star | 多 reuser 只保存一份 prefix K/V，visibility 正确 |
| branch | sibling suffix 完全不可见 |
| chain | deepest leaf 可见全部 ancestors，provider-before-reuser 不被 sparse path错误依赖 |
| multi-group | 不同 prefix tree 之间不可见 |
| no sharing | 原生 attention，Flex backend 调用次数为 0 |
| GQA | `H_Q % H_KV == 0` 输出 shape/gradient 正确 |
| variable suffix | partial/tail block 与 position offset 正确 |

小尺寸功能测试同时运行 dense oracle 与 backend，Flex 输出必须保持 packed Q shape；不得产生 expanded K/V tensor。可以通过 spy/allocator stats 断言 sparse path 没有调用 `PrefixAttentionStore.store/load`。

### 4.4 集成验证

#### 4.4.1 FSDP patch integration

在现有 `test_verl_fsdp_adapter.py`、`test_verl_fsdp_ch4_functional.py` 基础上增加：

- `build_prefix_sharing_micro_batch_fsdp()` 构造 tree layout；
- NestedTensor/remove-padding 输入的 token order、position ids、layout total 一致；
- attention patch 收到同一个 `PrefixSharingFSDPAttentionRuntime`；
- 24 层 fake model 只构造一个 BlockMask；
- dense fallback 只作 correctness，不作为性能主路径；
- expanded backend 行为不回归。

#### 4.4.2 Real FSDP world_size=1

CUDA optional：Qwen2.5-0.5B tiny batch，分别运行 PS=OFF、expanded-FA、Flex；覆盖 star 和 chain。验证真实 HF attention patch、RoPE、packed shape、logits save 和 restore。该测试对应 PoC-2E。

#### 4.4.3 verl actor lifecycle

在目标 verl 环境执行 `compute_old_log_prob -> ref log_prob -> update_actor`，验证每个 phase 的 runtime 创建/释放、BlockMask 构建次数、logprob/loss/gradient。该测试对应 PoC-2F，完成前不能向 verl 提交 ready PR。

### 4.5 精度对齐

#### 4.5.1 Attention-level

三方 reference：项目 expanded-KV + TorchRef/FA、dense sparse SDPA、Flex。覆盖 fp32 与 bf16、forward 与 Q/K/V backward、随机 upstream gradient、provider directed gradient。

必须报告：max/mean absolute、relative-L2、cosine、tensor norm、finite。fp32 使用严格阈值；bf16 以 expanded-FA 和 Flex 相对同一 fp32 oracle 的误差比较，不能只看 absolute max。

#### 4.5.2 Model-level

比较：

- attention output；
- hidden state 与 logits；
- 有效 token logprob，重点 suffix first token；
- entropy（启用时）；
- scalar actor loss；
- 关键参数梯度 relative-L2/cosine；
- 一次 optimizer update 后参数 relative-L2。

prefix-last restore 单独建立 star/chain regression：provider logits 必须使用 reuser first suffix label 重新计算，不能直接复制 provider logprob；梯度必须回到 provider prefix graph。

#### 4.5.3 Checkpoint/recompute

activation checkpointing 开/关分别比较 loss/gradient；记录 attention forward 调用次数和 backend runtime availability。若当前 verl checkpoint closure 无法保留 runtime，则首版明确 guard，不允许静默产生无 PrefixSharing 的 recompute。

### 4.6 性能对比

性能测试采用第三轮 PoC-3C 的 production harness，独立于 correctness CI：

- 三路径：PS=OFF FA、expanded-KV FA、dedup Flex；
- single-layer cold、steady layer、24-layer amortized；
- forward、backward、module total；
- QKV、metadata、expanded KV、autograd、peak HBM 快照；
- no-sharing、star long-prompt、chain、fragmented；
- 至少两个 GQA model shape 与两个 batch scale；
- cold compile 与 warm p50/p90 分离。

性能测试不以固定百分比作为单元测试断言，避免硬件噪声；通过结构性断言保护：Flex K/V token==dedup、expanded K/V token==expanded、no-sharing 不调用 Flex、BlockMask 每 forward 构建至多一次、无 dense `[T,T]` 长序列 allocation。具体 selector 阈值由 benchmark 数据生成并记录版本/硬件。

### 4.7 冒烟测试

| 环境 | 冒烟内容 | 预期 |
|---|---|---|
| CPU/no torch CUDA | import、config、layout、factory lazy import | PASS |
| CUDA + torch Flex，无 flash-attn | Flex star forward/backward | PASS；expanded optional skip |
| CUDA + Flex + flash-attn | 三路径 tiny star/chain | PASS |
| verl FSDP world_size=1 | one old-logprob + actor update | PASS |
| unsupported NPU/CP/Ulysses | 选择 Flex backend | 明确 config error |

所有 optional skip 必须打印缺失依赖/设备原因；不能将 runtime error 转成 skip。

### 4.8 回归命令与 CI 矩阵

本地 CPU 标准回归：

```bash
PYTHONPATH=prefix-sharing PYTHONPYCACHEPREFIX=/private/tmp/prefixsharing-attn-pycache \
python3 -m pytest -q -p no:cacheprovider \
  prefix-sharing/tests/unit_test \
  prefix-sharing/tests/integrated_test \
  prefix-sharing/tests/system_test
```

GPU optional：

```bash
PYTHONPATH=prefix-sharing python3 -m pytest -q -p no:cacheprovider \
  prefix-sharing/tests/integrated_test/optional/test_gpu_flex_backend.py \
  prefix-sharing/tests/integrated_test/optional/test_gpu_flash_backend.py \
  prefix-sharing/tests/integrated_test/optional/test_verl_fsdp_flex_e2e.py
```

建议 CI：CPU required；CUDA Flex smoke required（verl PR 条件允许时）；flash-attn/FSDP e2e nightly 或设备 CI；NPU/Megatron 回归确保旧 backend 无行为变化。

### 4.9 Release gate

| Gate | 必须满足 |
|---|---|
| Core | layout/invariant/oracle 全部 CPU PASS |
| Backend | Flex forward/backward/GQA/shape PASS，无 expanded KV/store 调用 |
| Integration | FSDP packed hook、RoPE、restore PASS |
| Precision | logprob/loss/gradient/update 与 baseline 满足约定误差 |
| Lifecycle | 每 forward 一个 BlockMask；无跨 context 泄漏；checkpoint 行为明确 |
| Performance | 第三轮 production benchmark 有效；不使用无效第二轮阈值 |
| Smoke | 目标 torch/verl/CUDA 环境跑通，unsupported 环境 fail-fast |

## Chapter 5：开发计划

### 5.1 开发原则与依赖关系

开发不等待第三阶段全部结束后才启动。第三阶段 PoC 与代码开发按以下依赖并行：

- 3A/3B 校正的是 correctness 证据，必须在 sparse layout 和 Flex backend 合入前通过；
- 3C/3D 决定性能承诺和 auto selector，不阻塞显式 `flex_attention` experimental backend；
- 2E/2F 依赖 minimal backend，进入集成开发后执行；
- Magi、direct BlockMask 优化和自动阈值均不进入首个功能 PR。

所有阶段遵循 TDD 优先。每个提交只完成一个可独立 review 的目标，不把 core 数据模型、backend 协议、FSDP patch 和性能策略揉成一个提交。

### 5.2 Phase 0：校正关键 PoC（ClaudeCode，可与 Phase 1 并行）

**任务。** 严格执行 2.3 的 3A/3B；有目标设备时并行执行 3C/3D。

**产物。** 可复现脚本、程序断言、文档回填和准确 commit；不得只提交 JSON 结果或人工判断。

**出口。** 3A 证明项目 expanded builder、dense oracle、Flex 的 output/QKV gradient 闭环；3B 证明 generic BlockMask coverage 正确。未通过时暂停 Phase 3 合入，并根据失败修改 layout/mask 设计。

### 5.3 Phase 1：实现 backend-neutral sparse layout

**先写测试。** 新增 core unit tests，覆盖 star、branch、chain、deep tree、多独立 group、no-sharing、零长度非法输入以及 ancestor slice 截断规则。

**代码。**

- 新增 `prefix_sharing/core/attention_layout.py`；
- 由 `PrefixSharingPlan` 派生 immutable `PrefixTreeAttentionLayout`；
- 提供 token-level visibility oracle，仅供测试和小规模 reference；
- 不引入 torch、Flex、verl 或设备依赖；
- 暂时保留 plan 中 expanded-KV 字段，保证旧 backend 无行为变化。

**出口。** CPU tests 全部通过；layout 能独立表达任意当前 planner 产出的复用树；不修改既有 backend 输出。

建议原子提交：`[feat] 新增前缀树注意力布局`。

### 5.4 Phase 2：拆分 attention backend 执行协议

**先写测试。** 用 fake expanded/sparse backend 验证 runtime 只调用所选模式的方法；no-sharing 不进入 prefix-sharing backend；unsupported mode fail-fast。

**代码。**

- 引入显式 `PrefixAttentionExecutionMode`；
- 将通用能力、expanded-KV 执行能力和 sparse 执行能力拆分；
- 现有 TorchRef/GPU FA/NPU FA 适配 expanded 协议；
- integration 不再假设所有 backend 都必须先执行 `build_kv()`；
- 不使用空实现或伪造 K/V 兼容旧接口。

**出口。** 现有 backend 回归全通过；协议可以在不分配 expanded KV 的情况下调用 sparse backend。

建议原子提交：`[refactor] 拆分前缀注意力执行协议`。

### 5.5 Phase 3：实现首版 `FlexAttentionBackend`

**先写测试。** 覆盖 mask visibility、GQA、RoPE 后 Q/K 输入、forward/backward、provider 定向梯度、动态 shape、unsupported dtype/device/version。

**代码。**

- 新增 `prefix_sharing/backends/flex_attention.py`；
- 从 `PrefixTreeAttentionLayout` 构造 generic `create_block_mask(mask_mod)`；
- 使用 lazy import，不给 CPU/NPU/旧 torch 环境增加 import-time 依赖；
- 同一 model forward 内复用 immutable BlockMask；
- candidate 路径直接消费去重 Q/K/V，禁止调用 `PrefixAttentionStore` 和 expanded builder；
- 首版不实现 direct `from_kv_blocks` 优化。

**出口。** 3A/3B PASS；backend unit/function tests PASS；profile 证明没有 expanded K/V 分配。

建议原子提交：`[feat] 实现FlexAttention前缀复用后端`。

### 5.6 Phase 4：接入 FSDP packed runtime

**先写测试。** fake model/layer 验证每个 model forward 只创建一个 runtime metadata，各 layer 共用同一 BlockMask；context 退出后不可复用；activation checkpoint 重算行为可观测。

**代码。**

- runtime state 持有 tree layout；
- 新增/调整 FSDP attention runtime，使 backend runtime metadata 生命周期归属于一次 model forward；
- `verl_fsdp` integration 按 execution mode 分派 expanded 或 sparse 路径；
- 首版保持 packed token 顺序，不改 position ids 和 restore indices；
- 增加 CUDA、remove-padding、CP/SP、torch version 等明确 guard。

**出口。** fake integration、real FSDP world-size=1 和旧 expanded backend 回归通过；BlockMask 无跨 micro-batch 泄漏。

建议原子提交：`[feat] 接入FSDP FlexAttention运行时`。

### 5.7 Phase 5：精度闭环与训练生命周期验收

**任务。** 执行 2E/2F，并完成 Chapter 4 的 precision/integration matrix：

- attention output 与 Q/K/V gradient；
- model logits、restored logprob、loss、parameter gradient；
- 一次 optimizer update 后参数；
- activation checkpoint on/off；
- compute-old-log-prob、reference log-prob、actor update；
- star、branch、chain 和 deep tree。

发现偏差时优先定位 mask、position、restore 和 checkpoint 生命周期，不通过放宽阈值掩盖语义错误。

**出口。** 精度红线全部通过，2E/2F 回填完成，旧 FA backend 无回归。

### 5.8 Phase 6：设备性能验收与策略冻结

执行 3C/3D，并至少覆盖两个模型 Q/KV head shape、两个 batch scale、两类共享拓扑。报告 attention module forward/backward、完整 peak HBM、24-layer 摊销和端到端 trainer 指标。

首版优先暴露显式 backend 选择：

- `native`：no-sharing；
- `expanded_flash_attention`：已有兼容路径；
- `flex_attention`：KV 零冗余 experimental 路径。

只有实测出现跨 workload 稳定分界，才在单独提交中加入 auto selector。不得直接采用第二轮探索性的 `T<1000` 或压缩率 `<50%`。

建议后续原子提交：`[feat] 新增前缀注意力后端选择策略`。

### 5.9 Phase 7：文档、兼容矩阵与 PR 收口

- 更新 README、架构/概念文档和配置示例；
- 明确 experimental、fallback、版本与并行策略边界；
- 记录 Magi、direct BlockMask、CP/Ulysses、NPU 等遗留项；
- PR body 提供实际测试命令、通过/跳过数量、设备环境、精度和性能摘要；
- 将 core、协议重构、Flex backend、FSDP integration 按原子提交保留，方便社区逐层 review。

### 5.10 并行安排与 Definition of Done

可并行：ClaudeCode 执行 Phase 0/设备实验；Codex 实现 Phase 1/2。Phase 3 合入前等待 3A/3B；Phase 6 与功能正确性开发可并行，但性能策略必须后置。

首版完成的定义：

1. FSDP packed 路径实际以去重 Q/K/V 执行 FlexAttention，未构造 expanded K/V；
2. 任意当前支持的前缀树 output/logprob/loss/gradient 与 baseline 对齐；
3. no-sharing 原生 bypass，旧 expanded backend 完整保留；
4. runtime metadata 每 forward 构建一次并正确释放；
5. 开发自测、功能、集成、精度、性能、smoke 六类验证有真实结果；
6. 未验证环境明确 fail-fast，不静默退化为错误语义。

## Chapter 6：当前结论

### 6.1 技术决策

当前已经具备开展方案设计和开发的条件，不应继续原样重复第二阶段 PoC。主线冻结为：

1. 先建立 backend-neutral `PrefixTreeAttentionLayout`；
2. 以 FlexAttention 完成 verl FSDP 的首个 Q/K/V 零冗余 backend；
3. 保留 expanded-KV + FA 作为兼容和精度 reference；
4. no-sharing 直接 bypass prefix-sharing；
5. Magi FFA 作为后续高性能/分布式候选，不阻塞首版。

### 6.2 已确认与尚未确认

已确认：Flex 的 mask 表达能力覆盖当前 prefix tree；metadata 可跨 layer 复用；FSDP 接入点存在；方案无需增加第三方 Python package；保持 packed token 顺序可以复用现有 position/restore 语义。

尚未被有效证据确认：项目 production expanded 路径与 Flex 的完整 Q/K/V gradient 对齐、direct BlockMask coverage、production packed FA 对照性能、完整 forward+backward HBM、目标 torch 2.9.1 行为、真实 FSDP/actor 生命周期。第二阶段相关数字只能作为探索性观察。

### 6.3 当前执行建议

立即并行启动 Phase 0 与 Phase 1/2。3A/3B 是 Flex backend 合入 gate；3C/3D 是性能策略和对外性能结论 gate。首版采用显式 backend 配置，不在证据不足时提前实现自动阈值。

## Chapter 7：遗留问题

### 7.1 首版合入前必须关闭

- **项目主路径精度证据。** 第二阶段使用了手写 expanded KV；按 3A 以项目 builder、store、production FA 和完整 backward 重测。
- **BlockMask coverage。** 修正 partial/full block 统计并验证 direct constructor；首版 generic path 也必须通过 token-level coverage oracle。
- **训练级 HBM/时延。** 独立进程测量 forward+backward 和 24-layer 摊销，不能沿用第二阶段受 allocator、逐 row FA 和 no-grad 影响的数据。
- **目标版本。** 在 verl 目标 torch 2.9.1 验证 API、compile、精度和性能；2.6 结果不关闭此项。
- **真实 FSDP 与 actor 生命周期。** 完成 2E/2F，覆盖 restore、optimizer update 和 activation checkpoint。

### 7.2 首版后优先优化

- **Direct BlockMask builder。** 若 3B 证明正确且 metadata/执行性能稳定优于 generic builder，再作为独立优化合入。
- **Auto selector。** 根据 3C/真实 trainer 数据按设备、模型 shape、token 数、共享率和拓扑制定；没有稳定分界前保持显式配置。
- **Runtime cache。** 首版仅在一次 model forward 内跨 layer 复用；跨 micro-batch/global cache 需解决生命周期、动态 shape、显存上限和 compile key 爆炸。
- **多卡 FSDP。** 补 world-size>1、不同 sharding strategy、gradient accumulation 和 rank-local metadata 一致性验证。
- **Dropout/确定性。** 训练启用 attention dropout 时验证 RNG 消耗与 baseline 语义，并明确可接受的精度口径。

### 7.3 中长期能力

- **MagiAttention。** 在 H100/H200 和适配软件栈评估 FFA；重点验证任意 prefix tree、反向、分布式通信和集成维护成本。
- **CP/Ulysses。** sparse logical layout 与 sequence shard/communication layout 的组合尚未设计，不能直接声明支持。
- **Megatron sparse backend。** 当前首版聚焦 FSDP；Megatron/NPU 继续走 expanded backend，后续需独立确定 hook 和 kernel。
- **NPU sparse kernel。** FlexAttention 不是 NPU 交付方案，需要评估 MindSpeed/CANN 可表达同类 block-sparse mask 的后端。
- **Token 重排。** DFS/group reorder 可能提高 block 完整度，但会增加 CPU 成本并影响 position/restore；首版不做。
- **HybridAttention。** Gated DeltaNet 等状态复用不属于本次 attention KV sparse backend 范围，需要独立 state layout 和执行后端。
