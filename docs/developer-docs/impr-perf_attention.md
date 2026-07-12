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

### 2.1 实验原则、范围与产物

#### 2.1.1 实验范围

本轮只验证 attention backend 及其最薄的 PrefixSharing QKV 接入面。所有性能结论必须明确所属层级：

| 层级 | 本轮回答的问题 | 不应据此推出的结论 |
|---|---|---|
| semantic microbenchmark | sparse tree mask 是否与当前 expanded-KV 语义、梯度一致 | 完整模型训练吞吐 |
| attention-module microbenchmark | layout/BlockMask/attention/restore 合计成本、HBM 和动态 shape 行为 | MLP、embedding、optimizer 的收益 |
| FSDP smoke | `use_remove_padding=True` 的 QKV hook 能否接入且 logprob/loss 正确 | 分布式扩展、CP、生产吞吐 |
| full RL training | old-log-prob、ref-log-prob、actor update 的端到端收益 | 某一 attention kernel 的绝对性能 |

首轮不测试 NPU；PyTorch FlexAttention 是 CUDA/Triton 路线，NPU 的 FA/build-kv 路径不应被拿来与它做 backend 横向归因。也不测试 CP、Ulysses SP、Magi distributed dispatch 或 activation checkpointing 的性能；它们留给完成 Flex FSDP 基线后的专门阶段。

#### 2.1.2 必须固定的比较对象

所有数据集、Q/K/V 随机种子、dtype、head shape、warm-up 和 iteration 数都必须一致。至少比较下列三条路径，不能只比较 Flex 与原生 FA：

| 标识 | 输入 token / KV 形态 | 作用 |
|---|---|---|
| `ps_off_fa` | 原始完整 batch；每条序列独立 causal FA | 业务 baseline，衡量 PrefixSharing 总收益 |
| `ps_on_expanded_fa` | 当前 trimmed Q + `build_kv()` expanded K/V + varlen FA | 当前 PrefixSharing backend，隔离 KV 拼接的成本与显存 |
| `ps_on_dedup_flex` | trimmed、去重后的 Q/K/V + prefix-tree `BlockMask` + FlexAttention | 目标方案，验证零冗余和稀疏 attention 性能 |

其中 `ps_on_expanded_fa` 与 `ps_on_dedup_flex` 都必须使用同一个 `PrefixSharingPlan`。否则 token 去重比例不同，任何速度或 HBM 差异都没有解释力。若 `flash-attn` 在目标环境不可用，允许先用 SDPA reference 完成精度门槛，但不得把该结果写成 FA 性能比较。

#### 2.1.3 结果回填规则与最小记录字段

**`docs/developer-docs/impr-perf_attention.md` 是本轮 PoC 唯一的正式结果载体。** ClaudeCode 不应把 `preflight.json`、`semantic.jsonl`、`attention_perf.jsonl` 等散落在临时目录后只在聊天中概述；每一组实验完成后，直接回填本章 2.10 对应的小节、表格和结论。这样硬件环境、命令、原始关键数字、失败原因和技术决策始终在同一份可 review 的文档中，形式与 `impr-perf.md` 的历轮实验一致。

临时 JSON/JSONL、profiler trace 或完整 traceback 可以在服务器用于解析和排障，但它们只是中间产物：

1. 在 2.10.1 追加环境行，并在表格下的“执行记录”代码块粘贴 preflight 的完整输出和实际命令；
2. 精度/梯度结果逐 case 回填 2.10.2，BlockMask/dynamic-shape 回填 2.10.3，attention/HBM 回填 2.10.4；
3. FSDP 与 Magi 的成功、skip 或失败均回填 2.10.5，不能把“不具备 Magi 环境”静默省略；
4. 每张表后的两三句结论必须解释数据对 Flex 默认启用、fallback 或 Magi P1 的影响；
5. 遇到失败，在相应表格后保留精简 traceback、最小复现参数与准确命令。超大 trace 可暂留服务器，但文档必须说明保存位置和不回填全文的原因。

每条回填记录至少包含：`git_commit`、`hostname`、`gpu_name`、`compute_capability`、`driver`、`cuda_runtime`、`torch_version`、`flash_attn_version`、`magi_version`、`case`、`mode`、`dtype`、`q_heads`、`kv_heads`、`head_dim`、`original_tokens`、`dedup_tokens`、`expanded_kv_tokens`、`warmup`、`iterations`、`result`。性能记录再写入 `p50_ms`、`p90_ms`、`peak_allocated_mb`、`peak_reserved_mb`；精度记录写入 `max_abs`、`mean_abs`、`max_rel`、`loss_abs`、`grad_max_abs`、`finite`。

### 2.2 预检：先确认实验解释成立

#### 2.2.1 GPU/FlexAttention 预检

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

将输出原样粘贴到 2.10.1 的“执行记录”代码块，并把主要字段填入环境表。Flex 首版的有效前提是 CUDA 可用、`torch.nn.attention.flex_attention` 可导入，且目标 verl 环境的 `torch` 是计划接入时实际会使用的版本。`torch==2.9.1` 是当前 verl 快照的目标版本；若服务器版本不同，可以做探索性实验，但必须标注为“非首版目标环境”，不能替代最终验收。

本地研究环境已验证 `torch==2.8.0` 的 CPU build 可以导入 `flex_attention`、`create_block_mask` 和 `BlockMask.from_kv_blocks`，并能运行一个普通 causal smoke test；它只证明 API 基本可用，不能证明 GPU kernel 性能或显存。

#### 2.2.2 MagiAttention 条件预检

Magi 是可选对照，不是 Flex 首版依赖。只在下面条件成立时做 2.8：

1. 有单卡 CUDA GPU，优先 H100/H200；
2. 能创建**独立**环境，不污染 verl/PyTorch/flash-attn 运行环境；
3. 按 MagiAttention `529fb0a` 的安装文档完成其依赖检查；
4. 安装后其官方 quickstart 的单卡 FFA 调用先通过。

Ampere（例如 A100）在该版本需要额外的 `flash_attn_cute`/FFA_FA 路线，且官方已提示 CUDA 版本低于 13 时可能需要显式允许并可能明显降速；4090/Ada 的 PrefixSharing FFA 支持未在本次资料中得到充分验证。A16、4090 或 A100 不满足官方安装/架构条件时，在 2.10.5 写入一条 `skipped` 记录和完整预检信息即可，**不升级或替换 verl 的依赖来强行完成 PoC**。

建议流程：先使用 Magi 官方仓库和 commit `529fb0a` 的安装说明、quickstart 完成独立 smoke；只有该 smoke 成功后，才把本章 2.4 的相同 slices 映射给 FFA。此轮只测试 CP=1 的 FFA kernel，不调用 `dispatch()`、`undispatch()`，不评估完整 distributed runtime。

### 2.3 统一 workload 与计数口径

#### 2.3.1 必测拓扑

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

#### 2.3.2 四类 token/算力指标不能混用

```text
original_tokens       = PS=OFF 进入模型的 token 总数
dedup_tokens          = trimmed Q/K/V 的物理 token 总数
expanded_kv_tokens    = build_kv 后物理 K/V token 总数
logical_pairs         = prefix-tree 中语义上可见的 QK pair 数
scheduled_block_pairs = Flex/Magi 实际调度的 tile 覆盖面积
```

`dedup_tokens < original_tokens` 才是全模型 projection/MLP 显存与计算节省的来源；`expanded_kv_tokens - dedup_tokens` 是当前 build_kv 引入的 KV 冗余；`scheduled_block_pairs / logical_pairs` 则衡量稀疏 tile 对 attention 算力的放大。报告时必须同时给出这些数字，不能把“token 节省”误写成“attention FLOPs 节省”。

### 2.4 PoC-A：树形 mask、Flex 与当前实现的精度契约

#### 2.4.1 参考实现与被测实现

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

#### 2.4.2 已有本地证据与仍需 device 验证的部分

Chapter 1.3.5 已验证 expanded-KV 与去重 sparse 语义在 CPU float64 下的 branch/chain output 和 Q/K/V gradient 对齐。该结果支持本节的 layout 方向，但尚未运行 PrefixTree `BlockMask` 的完整 device 版本。

本机无 CUDA；因此 2.4 的 GPU 测试是本轮第一个硬门槛。若 Flex 在 `star_aligned` 都无法满足精度契约，停止后续性能解读，先修正 tree range、position/RoPE 或 GQA 适配。

### 2.5 PoC-B：BlockMask 构建策略、metadata 与动态 shape

Flex 是否可用不能只看 `flex_attention()` 本体。对同一条 `PrefixTreeAttentionLayout`，分别测量以下两种构造方式：

| 构造方式 | 做法 | 目的 |
|---|---|---|
| `generic_mask_mod` | 用 device-resident token/node/position metadata 定义 `mask_mod`，调用 `create_block_mask()` | 正确性基线，确认 PyTorch 通用路径的实际成本 |
| `direct_block_metadata` | 从 tree ranges 直接生成 `BlockMask.from_kv_blocks()` 所需 block indices；partial block 仍通过严格的 `mask_mod` 表达 | 验证能否避免通用 builder 的 dense/全域扫描开销 |

第二种是优化候选，不是预设结论。只有它在所有 2.4 小型 case 与 generic path 输出/梯度一致、且不会误把 partial block 当作 full block 时，才有资格进入方案设计。若当前 PyTorch API 无法无歧义地表达 partial block，保留 generic path 并记录为 P1 实现风险，不能以不正确的 full-block 标记换取速度。

每个 workload、每种 block size（`64`、`128`；若 GPU/torch 支持再加 `256`）记录：

- host layout build 时间；
- `BlockMask` build 时间和前后 `torch.cuda.max_memory_allocated()`；
- 首次调用与 warm cache 调用分开记录；
- block 数、full/partial block 数、`scheduled_block_pairs/logical_pairs`；
- 是否发生 `.cpu()`、`.tolist()`、隐式 device synchronize，或出现 dense `[T,T]` 临时 allocation；
- 相同 token count、不同 tree shape 的编译次数和 latency。

具体执行要求：每个 case 至少 warm-up 20 次、计时 100 次；每个 iteration 前后用 `torch.cuda.synchronize()`，使用 `time.perf_counter_ns()`；计时循环内不 print、不创建 planner、不生成随机数。`torch.cuda.reset_peak_memory_stats()` 必须在每个 mode 前调用。first-run JIT/compile 另记为 `cold_ms`，不得混入 p50/p90。

动态 shape 至少按如下序列循环 50 个 micro-batch：`[star_aligned 1024/128, branch 769/127, chain 1024 depth=8, star_aligned 1024/128]`。记录第一个和第二次相同 shape 的 latency；若第二次仍接近 cold latency，说明 shape/closure 造成 compile cache 未命中，不能直接将该设计作为 RL 默认路径。

### 2.6 PoC-C：Flex attention module 的速度、显存与 fallback 边界

#### 2.6.1 计时方法

本节比较 2.1.2 的三条路径。独立 attention microbenchmark 必须拆出下列阶段，而不是只报告总时长：

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

#### 2.6.2 显存验证

显存要同时看理论和运行时：

| 指标 | 计算/采集方式 | 应有现象 |
|---|---|---|
| KV 物理元素 | 检查实际传给 attention 的 K/V 第一序列维度 | Flex 应为 `dedup_tokens`，不是 `expanded_kv_tokens` |
| KV bytes | `2 * tokens * kv_heads * head_dim * element_size` | Flex 相比 expanded FA 消除 expanded 部分的 K/V bytes |
| peak allocated/reserved HBM | forward+backward 每个 mode 的 CUDA peak | Flex 不能因 BlockMask/dense 临时量抵消大部分 KV 节省 |
| activation scaling | 固定 shape 增大 B/P/R 后 peak 的斜率 | 有 sharing 时 Flex 应随 dedup token 增长，而不是 expanded KV token 增长 |

对 `star_aligned B=64,P=2048,R=256` 以及 `chain/deep_fragmented` 各至少跑一次 forward+backward 的 peak HBM。只量 forward 会漏掉 autograd 保存的 K/V、LSE 和 BlockMask 相关状态，不能用来宣称训练显存收益。

#### 2.6.3 判读与 fallback

以下结论分别成立，不能互相替代：

| 观察 | 可作出的结论 | 后续动作 |
|---|---|---|
| Flex HBM 显著低于 expanded FA，但 attention 较慢 | KV 零冗余成功，kernel/metadata 仍需优化 | 保留 Flex，设置规模/利用率阈值和 fallback |
| `star_aligned` 快、`deep_fragmented` 慢 | layout fragmentation 是主变量 | DFS 排序、block size/threshold 优化；Magi FFA 对照有价值 |
| BlockMask build 接近/超过 attention | backend 可行但 runtime 设计不成熟 | 优先 direct metadata/cache，不先改 kernel |
| no-sharing/低共享率 Flex 慢 | 预期现象，不是否定 PrefixSharing | no-sharing 直走原生 FA，低收益走 expanded FA 或禁用 |
| Flex 与 expanded FA 都慢于 PS=OFF | 需要检查 projection trimming、mask 构建和 workload 是否真的存在净复用 | 不得发布为默认 backend |

第一阶段不预设一个统一的“必须快 X%”阈值。达到上线候选的最低条件是：2.4 精度通过、K/V 物理零冗余得到验证、没有与 token 数平方同阶的长期 dense metadata/HBM、在至少一个目标高复用 RL workload 的 module end-to-end 或完整训练上不劣于 `ps_on_expanded_fa`。是否默认启用 Flex，再由 PS=OFF 的端到端数据和低收益 fallback 边界决定。

### 2.7 PoC-D：FSDP 接入可行性 smoke

本实验只在 PoC-A 至 C 通过后进行。其目标是确认当前 `PrefixSharingFSDPAttentionRuntime` 的真实高性能入口可以被替换，而不是在 dense debug fallback 上获得虚假的性能结论。

1. 固定 `use_remove_padding=True`，确认 attention hook 收到 `[1, T, H, D]` packed Q/K/V；记录 `T == dedup_tokens`；
2. 在该 hook 内临时以 `PrefixTreeAttentionLayout -> BlockMask -> flex_attention` 替换 `build_kv()+FA`，不改变 planner、trim、position ids 或 restore；
3. 跑一个 `star_aligned` 和一个 `chain` 的实际 tiny causal LM forward/backward；比较 PS=OFF、expanded-FA、dedup-Flex 的 logits、logprob、loss、QKV/参数梯度；
4. 再跑一次 old-log-prob、ref-log-prob、actor update 的最小 verl 流程，确认同一 micro-batch 的 BlockMask 只在必要处构造，且 prefix-last restore 仍能访问正确的 provider logits；
5. 记录 FSDP world size=1 的结果。多卡 FSDP、CP、Ulysses SP 不在本轮通过条件内。

严禁使用当前 dense `[B,L,H,D] -> _pack_dense_qkv()` fallback 的总时间证明性能：这条 debug/correctness 路径已经完成 QKV projection，不能反映 remove-padding 下去重 token 对全模型的节省。

### 2.8 PoC-E：Magi FFA 条件对照

Magi 的价值是回答“若 Flex 的碎片化性能不足，AttnSlice kernel 是否值得成为第二阶段 backend”，不是取代本轮的 Flex 决策。执行条件见 2.2.2。

若条件满足，流程如下：

1. 在独立环境 clone/checkout MagiAttention `529fb0a`，按其安装文档完成官方 FFA quickstart 和单卡 backward smoke；先保存版本、CUDA、架构、安装命令与 quickstart 结果；
2. 对 2.3 的 `star_aligned`、`branch`、`chain`、`deep_fragmented`，从同一份 `PrefixTreeAttentionLayout` 导出 slices：每个 node 一个 `(node_q_range,node_k_range,CAUSAL)`，每个严格 ancestor 一个 `(node_q_range,ancestor_k_range,FULL)`；
3. 先以小尺寸 dense sparse oracle 验证 FFA output 与 Q/K/V grad；再在 bf16 运行 2.6 的 forward/backward/HBM protocol；
4. 仅与同 shape、同 token count、同 warm-up 的 `ps_on_dedup_flex` 和 `ps_on_expanded_fa` 比较；Magi FFA 的 cold JIT、安装/编译时间单列，不能混入 attention p50；
5. 输出 `slice_count`、FULL/CAUSAL slice 数、slice 覆盖的 logical pairs、kernel forward/backward、peak HBM 与精度结果。

Magi FFA 的结论规则：若它在目标架构的 `deep_fragmented` 上相对 Flex 稳定占优，且精度通过，记录为 P1 高性能 backend；若只在安装复杂的环境中可跑、或对 star/chain 没有净收益，则保留研究记录而不引入 PrefixSharing 依赖。任何 Magi 失败均不影响 Flex 首版继续推进。

### 2.9 推荐执行顺序与失败处置

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

### 2.10 回填模板与本轮决策门槛

执行者把每张表和简短结论直接追加在本节，历史失败也保留。每次回填都在条目中写明日期；不要覆写旧结果，以便后续判断版本、硬件或方案变动造成的差异。

#### 2.10.1 环境与可用性

| 日期 | 机器/GPU | compute capability | CUDA / torch / flash-attn | Flex | Magi FFA | 结论 |
|---|---|---|---|---|---|---|
| 待回填 |  |  |  |  |  |  |

**执行记录（待回填）**

```text
日期：
实际命令：
git commit：
完整 preflight 输出：
```

#### 2.10.2 精度与梯度

| case | dtype / shape | expanded vs dense oracle | Flex vs oracle output | Flex vs oracle Q/K/V grad | logits/logprob/loss | 结论 |
|---|---|---:|---:|---:|---:|---|
| 待回填 |  |  |  |  |  |  |

**精度实验记录与异常（待回填）**

```text
日期：
实际命令：
最小失败复现（若有）：
```

#### 2.10.3 BlockMask 与动态 shape

| case | constructor | block size | cold ms | warm p50 ms | peak HBM MB | full/partial blocks | scheduled/logical | cache 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 待回填 |  |  |  |  |  |  |  |  |

**BlockMask / dynamic-shape 实验记录（待回填）**

```text
日期：
实际命令：
shape 序列与 warm-up/iterations：
```

#### 2.10.4 Attention module 与显存

| case | mode | original/dedup/expanded tokens | fwd p50 ms | bwd p50 ms | module p50 ms | peak HBM MB | KV physical tokens | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| 待回填 |  |  |  |  |  |  |  |  |

**Attention / HBM 实验记录（待回填）**

```text
日期：
实际命令：
计时方法与同步方式：
```

#### 2.10.5 FSDP 与 Magi（条件执行）

| 项目 | workload | 精度 | p50 / HBM | 状态 | 对下一阶段的影响 |
|---|---|---|---|---|---|
| FSDP remove-padding smoke | 待回填 |  |  |  |  |
| Magi FFA CP=1 | 待回填或 skipped |  |  |  |  |

**FSDP / Magi 实验记录（待回填）**

```text
日期：
实际命令：
Magi 若 skipped：GPU 架构、CUDA/安装条件与 skip 原因：
```

本轮最终决策按证据分级：

- **Flex 进入方案设计/开发**：PoC-A 通过；PoC-B 未出现不可接受的 dense metadata 或持续 compile churn；PoC-C 证实 K/V 物理零冗余，并至少在一类高复用目标 workload 上不劣于当前 expanded-FA。
- **Flex 可做但必须默认 fallback**：精度和显存通过，但 `deep_fragmented`、短序列或低复用率明显变慢；将相应统计量变成 backend selector 的输入。
- **先不实现 Flex backend**：GPU 上无法守住精度，或 BlockMask 的稳定 HBM/latency 成本抵消 KV 节省且无可行的 metadata 改进方向；此时保留 layout 研究，优先 Magi FFA 或专用 kernel PoC。
- **Magi 进入下一阶段**：只在独立环境、同口径测试中，精度通过且在 Flex 弱项上有稳定净收益时成立；绝不因“能安装”而升级优先级。

## Chapter 3：方案设计

待 PoC 结论回填后补充。重点包括：backend-neutral tree layout、Flex BlockMask 构造、backend 选择/fallback、restore map、配置与分层设计。

## Chapter 4：测试验证

待 PoC 结论回填后补充。将把通过的 PoC 固化为 CPU/unit、GPU optional、FSDP integration 三层自动化测试，避免只依赖一次性 benchmark。

## Chapter 5：开发计划

待 PoC 结论回填后补充。将按可独立 review 的原子阶段拆分，不把 planner/layout、Flex backend、性能策略和 Magi PoC 混在同一开发提交中。

## Chapter 6：当前结论

当前主线建议：先实现 backend-neutral 去重布局，以 FlexAttention 完成 verl FSDP 首版；保留 expanded-KV + FA fallback，并把 Magi FFA 作为下一阶段高性能候选。

## Chapter 7：遗留问题

当前遗留问题见 1.10。待 PoC 与方案设计完成后，将未解决的硬件、版本、CP、NPU 和专用 kernel 问题统一收敛到本章。
