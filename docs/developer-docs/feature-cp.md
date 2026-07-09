# PrefixSharing 支持 Context Parallel 的研究分析与设计方案

本文档用于记录 PrefixSharing 在 Megatron / MindSpeed 运行时支持上下文并行（Context Parallel, CP）的研究、设计、测试和开发计划。

当前目标场景优先限定为：

- 训练引擎：`verl + Megatron/MCore`
- 注入方式：`prefix-sharing` setup 模块 monkey patch
- 依赖版本：`verl 0.8.0`、`Megatron-Core 0.16.1`、`MindSpeed 0.16.0`
- CP 算法：首版本只支持 `context_parallel_algo="kvallgather_cp_algo"`
- 数据格式：首版本只支持 `THD` / `use_remove_padding=True`
- 精度要求：logprob / loss / gradient 与 baseline 保持一致；精度一致性优先级高于性能收益

本文档先固化源码研究和设计方案，再进入代码开发和实机验证。

## Chapter 1：研究分析

### 1.1 目标重新界定

PrefixSharing 当前在 Megatron 路径中的核心范式是：

```text
One-Forward + prefix activation reuse + Prefix-Last Restore
```

其中：

- `PrefixSharingPlanner` 在同一 micro-batch 内识别 provider / reuser 关系；
- prepare 阶段物理裁剪 reuser 的共享 prefix，只保留 provider 完整序列和 reuser suffix；
- attention runtime 在每层复用 provider prefix activation，使 reuser suffix 的上下文语义等价于完整序列；
- vocab/logprob 阶段保存 provider prefix-last logits；
- forward_step 出口执行 Prefix-Last Restore，恢复 reuser 第一个 suffix token 对应的 logprob；
- prefix activation / KV 缓存不允许 `detach()`，必须保留 provider prefix 计算图，保证梯度语义一致。

本次 CP 适配不是重新定义 PrefixSharing，而是在 Megatron/MindSpeed 的 Context Parallel 运行方式下维持上述语义。

首版本目标明确收敛为：

```text
THD + use_remove_padding=True + context_parallel_algo="kvallgather_cp_algo"
```

不支持范围：

- `megatron_cp_algo`、`ulysses_cp_algo`、`hybrid_cp_algo`、`adaptive_cp_algo`、`hybrid_adaptive_cp_algo`；
- dynamic context parallel；
- BSHD CP；
- inter micro-batch sharing；
- virtual PP；
- RoPE fusion / fused QKV RoPE / fused actor kernel。

首版设计必须解决的问题不是“CP 与 PrefixSharing 理论上是否冲突”，而是：

1. CP 会把输入 token 在 `preprocess_thd_engine()` 阶段切成 CP-local packed token；
2. 当前 PrefixSharing attention patch 假设 Q/K/V 是 global packed token；
3. 当前 PrefixSharing logits 保存用 global packed index；
4. CP 下 logits_processor 阶段的 logits 仍是 CP-local；
5. postprocess 后才恢复 global NestedTensor 语义。

因此 CP 适配的核心是补齐 global packed 坐标与 CP-local runtime 坐标之间的映射，并尽量复用 Megatron/MindSpeed 已有 CP attention 和 Verl 已有 THD postprocess/restore 范式。

### 1.2 当前分支 PrefixSharing 代码结构摸查

当前分支主要代码结构如下：

- `prefix-sharing/prefix_sharing/core/config.py`
  - `PrefixSharingConfig.validate()` 当前通过 `supported_cp_size=1` 拒绝 `context_parallel_size != 1`。
  - `validate_for_engine()` 用于 v080 setup patch 的 forward_step 入口，当前只校验 `use_remove_padding=True`、backend、detector 等基础约束，尚未完整读取 `context_parallel_algo`。
  - CP 适配需要把“拒绝所有 CP”改为“只允许 THD + `kvallgather_cp_algo`”。

- `prefix-sharing/prefix_sharing/integrations/parallel_info.py`
  - 已有 `MegatronParallelInfo`，可从 Megatron `parallel_state` 读取 `tp_rank/tp_size`、`pp_rank/pp_size`、`cp_rank/cp_size`。
  - 当前 `cp_rank/cp_size` 主要用于日志和 padding 对齐，尚未参与 CP-local index 映射。

- `prefix-sharing/prefix_sharing/backends/packed_layout.py`
  - `PackedBatchLayout` 当前表达 global packed padded layout：`valid_lengths`、`padded_lengths`、`cu_seqlens`、`max_seqlen`、`packed_position_ids`、`valid_token_mask`。
  - 该类已经适合继续作为 packed token 坐标入口，但缺少 CP-local 视图。
  - CP 适配不应新建一套完全平行 layout，而应在 `PackedBatchLayout` 上扩展可选 CP-local view。

- `prefix-sharing/prefix_sharing/integrations/context.py`
  - `PrefixSharingRuntimeContext` 从 runtime state 中读取 plan/layout/parallel_info/store/backend。
  - `_build_prefix_last_restore_indices()` 目前用 `packed_batch_layout.packed_index(provider, provider_offset)` 生成 provider prefix-last 的 1D packed index。
  - 该 index 当前是 global packed index。CP>1 时，vocab logits 保存点看到的是 CP-local logits，因此这里需要派生 CP-local save spec，不能继续直接用 global index 读取 logits。

- `prefix-sharing/prefix_sharing/setup/patches/verl080_mcore0161_ms0160/forward_step.py`
  - patch `MegatronEngineWithLMHead.forward_step`。
  - 入口读取 batch、读取 PrefixSharingConfig、调用 `build_prefix_sharing_micro_batch_verl080()` 裁剪 batch 并构建 runtime state。
  - 在 `prefix_sharing_runtime_context(ps_state)` 内调用原始 `forward_step`。
  - 原始 `forward_step` 返回后、context 仍激活时，调用 `restore_via_2d_unfold_verl080()`。
  - 因此 Prefix-Last Restore 的写入点位于原始 forward_step 之后。

- `prefix-sharing/prefix_sharing/setup/patches/verl080_mcore0161_ms0160/attention.py`
  - context 激活时先调用 `self.get_query_key_value_tensors(...)` 提取 Q/K/V。
  - THD 下对 Q/K/V 做 `squeeze(1)`。
  - 随后委托 `prefix_attention()`。
  - 这意味着当前 PrefixSharing path 会绕开原始 `Attention.forward()` 中后续 RoPE、core attention、linear projection 逻辑。

- `prefix-sharing/prefix_sharing/integrations/megatron_runtime.py`
  - `prefix_attention()` 当前完整接管 RoPE、KV expansion、attention 计算和 output projection。
  - 进入后调用 `ensure_global_packed_token_lengths()`，要求 `query/key/value.shape[0] == packed_batch_layout.total_padded_length`。
  - 该 guard 与 CP>1 的 Verl THD preprocess 行为冲突，因为 CP>1 时每个 rank 输入模型的是 `total_padded_length / cp_size` 级别的 local token。

- `prefix-sharing/prefix_sharing/backends/torch_ref.py`
  - `build_kv()` 通过 `layout.padded_lengths` split K/V，并按 provider-before-reuser 顺序 store/load。
  - `attention()` 自己按 row 做 causal attention。
  - 该实现适合 CP=1 的 reference path，不适合直接重写 `kvallgather_cp_algo` 的 CP attention 通信。

- `prefix-sharing/prefix_sharing/setup/patches/verl080_mcore0161_ms0160/vocab_logprobs.py`
  - patch `vocab_parallel_log_probs_from_logits()`。
  - 在原函数前保存 provider prefix-last logits，因为原 Megatron cross entropy 会 in-place 修改 logits。
  - 当前保存逻辑使用 `ctx.prefix_last_restore_indices[*].provider_1d_pos` 直接索引 `logits.view(-1, vocab_per_tp)`。
  - CP>1 时该 `logits` 是 CP-local，因此保存逻辑必须做 global→local index 映射。

### 1.3 Verl THD CP preprocess / postprocess 源码结论

源码位置：`dependency/verl_cdd9014f/verl/models/mcore/util.py::preprocess_thd_engine()`。

已确认事实：

1. CP>1 时，Verl THD preprocess 使用 Megatron CP rank/size：

```python
cp_size = mpu.get_context_parallel_world_size()
cp_rank = mpu.get_context_parallel_rank()
align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
```

2. 每条序列先 padding 到 `align_size` 的倍数：

```python
pad_size = (align_size - seqlens_in_batch % align_size) % align_size
seqlens_in_batch_padded = seqlens_in_batch + pad_size
```

3. CP>1 时，送入模型的 packed token 长度是全局 padded token 长度除以 CP size：

```python
shape[0] = sum(seqlens_in_batch_padded_cpu) // cp_size
```

4. 每条序列被拆成 `CP * 2` 个 chunk，每个 rank 持有前后对称的两个 chunk：

```text
rank r 持有：
- front chunk: [half * r, half * (r + 1))
- back chunk:  [padded_len - half * (r + 1), padded_len - half * r)
```

其中：

```python
seqlen = seqlen_padded_i // cp_size
half_seqlen = seqlen // 2
```

5. `PackedSeqParams.cu_seqlens_q_padded` / `cu_seqlens_kv_padded` 仍保存 global padded row 边界：

```python
PackedSeqParams(
    qkv_format="thd",
    cu_seqlens_q=cu_seqlens_padded,
    cu_seqlens_kv=cu_seqlens_padded,
    cu_seqlens_q_padded=cu_seqlens_padded,
    cu_seqlens_kv_padded=cu_seqlens_padded,
)
```

6. `postprocess_thd_engine()` 在 CP>1 时先 all-gather 每个 CP rank 的 local output：

```python
output_list = [torch.empty_like(output) for _ in range(cp_size)]
torch.distributed.all_gather(output_list, output.detach(), group=cp_group)
output_list[cp_rank] = output
```

7. all-gather 后按与 preprocess 相反的 zigzag chunk 规则恢复每条原始序列，再构造 `torch.nested.as_nested_tensor(output_new, layout=torch.jagged)`。

关键结论：

- CP 虽然从算法概念上是 attention 上下文并行，但在 Verl THD 实现中，CP 已经前移到了模型输入 preprocess。
- 因此模型 forward、LM head、logits_processor 在 CP>1 时都处理 CP-local token。
- 只有 `postprocess_thd_engine()` 之后，log_probs / entropy 才恢复成 global NestedTensor 语义。

### 1.4 Megatron Engine logits_processor 与 Prefix-Last Restore 源码结论

源码位置：

- `dependency/verl_cdd9014f/verl/workers/engine/megatron/transformer_impl.py::MegatronEngineWithLMHead.forward_step()`
- `dependency/verl_cdd9014f/verl/models/mcore/model_forward.py::forward_model_engine()`
- `prefix-sharing/prefix_sharing/setup/patches/verl080_mcore0161_ms0160/vocab_logprobs.py`
- `prefix-sharing/prefix_sharing/integrations/verl_mcore.py::restore_via_2d_unfold_verl080()`

已确认调用顺序：

```text
MegatronEngineWithLMHead.forward_step
  -> forward_fn = get_mcore_engine_forward_fn(...)
  -> forward_model_engine(..., data_format="thd")
      -> preprocess_thd_engine(input_ids)
      -> model(input_ids=input_ids_rmpad, packed_seq_params=packed_seq_params)
      -> preprocess_thd_engine(label/temperature, need_roll=True)
      -> logits_processor(output_orig, label, temperature)
          -> vocab_parallel_log_probs_from_logits(logits_bak, label)
             [PrefixSharing vocab patch 在这里保存 provider prefix-last logits]
      -> postprocess_thd_engine(log_probs/entropy)
  -> original_forward_step returns output dict
  -> PrefixSharing forward_step patch calls restore_via_2d_unfold_verl080(output_dict)
```

源码中 `logits_processor` 注释明确说明：

```python
# logits_processor_func return tensors with shape (1, total_nnz/cp_size)
```

这说明 CP>1 时，`vocab_parallel_log_probs_from_logits()` patch 所见 logits 是 CP-local，不是 global packed logits。

但 `restore_via_2d_unfold_verl080()` 的写入点在原始 `forward_step` 返回之后，此时 `postprocess_thd_engine()` 已经执行完成，`output_dict["log_probs"]` 是 NestedTensor。该函数再展开成 `[B, L_max]` 的 2D 视图，执行已有 restore 逻辑，然后压回 NestedTensor。

因此 Prefix-Last Restore 在 CP 下应拆成两个问题：

1. **保存 provider prefix-last logits**：发生在 `postprocess_thd_engine()` 前，必须使用 CP-local logits index。
2. **写回 reuser prefix-last logprob**：发生在 `postprocess_thd_engine()` 后，可以继续复用现有 2D/Nested restore 范式。

这也解释了为什么 logits 本身不是 attention，却仍然涉及 `cp_rank`：CP 在 Verl THD 中切分的是模型输入 token，LM head 只会对本 CP rank 的 hidden states 产生 local logits。

### 1.5 MindSpeed `kvallgather_cp_algo` attention 源码结论

源码位置：

- `dependency/MindSpeed_core_r0.16.0/mindspeed/features_manager/context_parallel/context_parallel_feature.py`
- `dependency/MindSpeed_core_r0.16.0/mindspeed/te/pytorch/attention/dot_product_attention/context_parallel.py`
- `dependency/MindSpeed_core_r0.16.0/mindspeed/te/pytorch/attention/dot_product_attention/kvallgather_context_parallel.py`
- `dependency/MindSpeed_core_r0.16.0/mindspeed/te/pytorch/attention/dot_product_attention/dot_product_attention.py`

已确认事实：

- MindSpeed 为 `context_parallel_algo="kvallgather_cp_algo"` 提供独立 CP strategy。
- 该路径面向 causal attention。
- CP attention 内部会 all-gather K/V，并基于 `cp_rank/cp_size`、global `cu_seqlens`、local chunk range 计算当前 rank 的 attention。
- THD 路径会使用 NPU fusion attention，并传入 `actual_seq_qlen` / `actual_seq_kvlen`。

关键结论：

- `kvallgather_cp_algo` 的正确性依赖 MindSpeed 内部的 CP-aware K/V all-gather、zigzag chunk 还原、causal range 推导和 backward 通信。
- PrefixSharing 不应在 backend 中重新实现完整 CP attention。
- CP 适配应优先让 PrefixSharing 只负责 prefix-sharing 元数据、KV/activation 可见性或后续 mask 机制，并尽量回到原 Megatron/MindSpeed CP attention 主体。

### 1.6 PrefixSharing 与 CP 的语义关系

PrefixSharing 与 CP 没有理论冲突。

PrefixSharing 关注的是同一 micro-batch 内的语义复用关系：

```text
reuser suffix 应该看到与完整 baseline 相同的 prefix 历史
```

CP 关注的是长序列 attention 的并行计算：

```text
把一个长上下文的 token / attention 计算分布到多个 CP rank
```

二者冲突只会发生在实现层面：

- reuser 所需 prefix token 可能分布在不同 CP rank 的 local chunks；
- provider prefix-last logits 只存在于持有该 token 的 CP rank；
- 当前 PrefixSharing 使用 global packed index，而 CP runtime 输入/输出中间阶段是 CP-local；
- 当前 PrefixSharing attention path 绕开了 MindSpeed CP attention。

因此 CP 适配的语义目标是：

1. 对每个 CP rank，仅处理该 rank 本地实际持有的 token/activation；
2. 使用 global layout 维持 batch row、原始 position、restore target 的稳定语义；
3. 使用 CP-local view 完成本 rank 的 local token 索引；
4. attention 主体尽量复用 MindSpeed CP-aware path；
5. restore 写回继续在 postprocess 后的 global Nested/2D 语义中完成。

### 1.7 当前实现的 CP 适配堵点

1. **配置层堵点**
   - 当前 `PrefixSharingConfig.validate()` 拒绝所有 CP。
   - 需要改为允许 `context_parallel_size > 1` 且 `context_parallel_algo="kvallgather_cp_algo"`。
   - 其他 CP 算法、dynamic CP、BSHD CP 需要明确 guard。

2. **layout 坐标堵点**
   - 当前 `PackedBatchLayout` 只有 global packed layout。
   - CP>1 时，runtime tensor 是 CP-local packed token，不能继续用 global `cu_seqlens` 直接 split Q/K/V/logits。
   - 需要扩展 CP-local view，支持 global packed index 与 local packed index 双向映射。

3. **attention patch 堵点**
   - 当前 `attention.py` 提取 Q/K/V 后直接进入 `prefix_attention()`。
   - `prefix_attention()` 自己做 RoPE、KV expansion、attention、linear projection，且当前实现基于 global packed layout。
   - CP>1 时 runtime Q/K/V 是 CP-local packed token，直接复用现有 global packed path 会出现 split、RoPE、KV store/load 和 output shape 错位。
   - 首版方案不是等待尚未 ready 的 mask path，而是把当前 expanded KV concat 路径改造成 CP-local expanded KV concat，并通过精度实验确认它与 `kvallgather_cp_algo` 场景兼容。

4. **KV 复用方式演进堵点**
   - 当前代码采用 KV concat / expanded KV 方式。
   - 首版 CP 必须基于该方式先打通，因为 attention-mask / visibility 方案当前尚未 ready，不能作为首版依赖。
   - 后续再切换到“依赖 attention-mask 或等价可见性机制”的 KV 复用，以降低 prefix/suffix 额外 KV concat 带来的显存放大。

5. **Prefix-Last logits 保存堵点**
   - Restore 写入点可以继续复用 postprocess 后 Nested/2D 范式。
   - 但 provider prefix-last logits 的保存点在 CP-local logits 阶段。
   - 当前 `provider_1d_pos` 是 global packed index，CP>1 时必须转换为 local logits index。
   - 如果 provider prefix-last 不在当前 CP rank，需要明确该 rank 不保存，或通过 CP group 传递 saved logits。

6. **RoPE / position 堵点**
   - 当前 PrefixSharing 自己使用 `packed_position_ids` 手动 index RoPE freq。
   - CP path 若回到原 MindSpeed attention，则应尽量复用原 attention 的 RoPE/position 处理。
   - 若仍需 PrefixSharing 参与 RoPE，必须使用 CP-local position ids，而不是 global packed position ids 直接作用于 local tensor。

7. **测试条件堵点**
   - 本地单元测试可验证 layout/index/guard。
   - CP attention correctness、NPU fusion attention、真实 `kvallgather_cp_algo` 通信需要设备实验。
   - 文档中的测试验证章节需要明确哪些由开发自测覆盖，哪些由后续设备侧 ClaudeCode / NPU 环境完成。

### 1.8 源码依据汇总

| 代码位置 | 已确认事实 | 对 CP 方案的影响 |
|----------|------------|------------------|
| `dependency/verl_cdd9014f/verl/models/mcore/util.py::preprocess_thd_engine()` | CP>1 时每条序列 padding 到 `tp_size * cp_size * 2`，并切成 CP-local zigzag chunks | layout 必须同时表达 global padded row 与 local CP chunk |
| `dependency/verl_cdd9014f/verl/models/mcore/util.py::postprocess_thd_engine()` | CP>1 时 all-gather local output 并恢复 NestedTensor | restore 写入阶段可复用 postprocess 后 Nested/2D 范式 |
| `dependency/verl_cdd9014f/verl/models/mcore/model_forward.py::forward_model_engine()` | `logits_processor()` 在 `postprocess_thd_engine()` 前执行 | vocab logits 保存发生在 CP-local 阶段 |
| `dependency/verl_cdd9014f/verl/workers/engine/megatron/transformer_impl.py::logits_processor()` | 注释说明 logits/log_probs tensor shape 为 `(1, total_nnz/cp_size)` | provider prefix-last logits 保存必须做 global→local index 映射 |
| `prefix-sharing/.../patches/verl080_mcore0161_ms0160/forward_step.py` | restore 在原始 forward_step 返回后、context 仍激活时执行 | restore 写回点已经位于 postprocess 后，适合形态B |
| `prefix-sharing/.../patches/verl080_mcore0161_ms0160/attention.py` | context 激活时提取 Q/K/V 后绕开原 forward 主体 | CP path 需要调整 attention patch 责任边界 |
| `prefix-sharing/prefix_sharing/integrations/megatron_runtime.py::prefix_attention()` | 当前要求 Q/K/V 长度等于 global total padded length | CP-local runtime 下必须改 guard/路径 |
| `prefix-sharing/prefix_sharing/backends/packed_layout.py::PackedBatchLayout` | 当前只表达 global packed padded layout | 应扩展 CP-local view，不建议另起平行 layout |
| `prefix-sharing/prefix_sharing/setup/patches/.../vocab_logprobs.py` | 当前用 global provider_1d_pos 直接索引 logits | CP>1 会越界或错取，需要 CP-local save spec |
| `dependency/MindSpeed_core_r0.16.0/.../kvallgather_context_parallel.py` | MindSpeed CP attention 内部实现 K/V all-gather 与 local/global range 推导 | PrefixSharing 不应重写完整 CP attention |

## Chapter 2：方案设计

### 2.1 总体方案

首版方案采用“受控放开 + CP-local layout + 复用原 CP attention + postprocess 后 restore”的路径。

总体原则：

1. 只支持 `THD + context_parallel_algo="kvallgather_cp_algo"`。
2. PrefixSharing 不在 backend 中重写 MindSpeed CP attention。
3. `PackedBatchLayout` 继续作为统一 packed layout 入口，并扩展 CP-local view。
4. Prefix-Last Restore 的写回点继续复用当前 v080 2D/Nested restore 范式。
5. provider prefix-last logits 保存阶段引入 CP-local index 解析。
6. 首版 CP attention 以 expanded KV concat 打通为主，attention-mask / visibility 复用作为后续演进方向，不作为首版阻塞项。

目标运行链路：

```text
forward_step patch
  -> build_prefix_sharing_micro_batch_verl080
      -> plan
      -> trim batch
      -> PackedBatchLayout(global + optional CP local view)
      -> runtime state/context
  -> original Megatron forward_step under PrefixSharing context
      -> Verl preprocess_thd_engine does CP-local token split
      -> model forward
      -> attention patch
          -> CP path: PrefixSharing prepares reuse metadata / visibility info
          -> return to original Megatron/MindSpeed CP-aware attention path
      -> logits_processor
          -> vocab patch saves provider prefix-last logits using CP-local save spec
      -> postprocess_thd_engine restores NestedTensor
  -> restore_via_2d_unfold_verl080
      -> unfold NestedTensor to 2D
      -> restore prefix interior + prefix-last
      -> fold back NestedTensor
```

### 2.2 配置、guard 与支持边界

配置层新增/调整：

- `PrefixSharingConfig.validate()` / `validate_for_engine()` 支持读取：
  - `context_parallel_size`
  - `context_parallel_algo`
  - `dynamic_context_parallel`
  - `use_remove_padding`
  - `use_fused_kernels`
- 当 `context_parallel_size > 1` 时必须满足：
  - `use_remove_padding=True`
  - `context_parallel_algo == "kvallgather_cp_algo"`
  - `dynamic_context_parallel=False`
  - `data_format == "thd"` 或 engine 侧等价条件
- 继续拒绝：
  - non-THD CP
  - BSHD CP
  - dynamic CP
  - CP algorithms other than `kvallgather_cp_algo`
  - fused kernels
  - RoPE fusion / fused QKV RoPE

运行时 guard：

- attention patch 进入 CP path 时，若 `ctx.parallel_info.cp_size > 1` 且 `packed_seq_params.qkv_format != "thd"`，直接报错。
- CP path 不再使用 `ensure_global_packed_token_lengths()` 检查 Q/K/V 等于 global total padded length；改为检查 local length 是否等于 `ctx.packed_batch_layout.context_parallel.local_total_padded_length`。
- vocab logits 保存时，若 CP-local save spec 指向当前 rank 之外，不允许直接用 global index 索引本地 logits。
- restore 写回前继续检查 output 为 NestedTensor 或可展开为 2D 的受支持形态。

### 2.3 Runtime layout 与坐标体系

继续复用 `PackedBatchLayout`，新增可选 CP-local 视图，例如：

```python
@dataclass(frozen=True)
class ContextParallelPackedView:
    cp_rank: int
    cp_size: int
    local_padded_lengths: list[int]
    local_cu_seqlens: list[int]
    local_total_padded_length: int
    local_position_ids: Tensor | None
    local_valid_token_mask: Tensor | None
    global_indices_by_local: Tensor
    local_indices_by_global: Tensor | None
```

建议挂载方式：

```python
@dataclass(frozen=True)
class PackedBatchLayout:
    ...existing fields...
    context_parallel: ContextParallelPackedView | None = None
```

职责边界：

- `PackedBatchLayout` 继续表达 global packed row 语义：batch row、global padded length、restore target、global position id。
- `ContextParallelPackedView` 只表达当前 CP rank 的 local runtime token 视图。
- `PrefixSharingPlan` 不加入 CP padding / CP rank 字段，继续只表达 prefix-sharing 逻辑语义。

CP-local view 构造规则应严格对齐 `preprocess_thd_engine()`：

对于每个 row：

```text
padded_len = global padded length
chunk_len_per_rank = padded_len // cp_size
half = chunk_len_per_rank // 2
rank r local chunks:
  front_global = [row_start + half*r, row_start + half*(r+1))
  back_global  = [row_start + padded_len - half*(r+1), row_start + padded_len - half*r)
local row layout = front chunk + back chunk
```

同时需要记录 padding token：

- global valid token 范围是 `[row_start, row_start + valid_len)`；
- local chunk 中落在该范围内的是 valid token；
- 超出 valid_len 但在 padded_len 内的是 padding token；
- store/load/restore 不能把 padding 当作有效 prefix token。

必须提供的 helper：

```python
layout.context_parallel.global_to_local(global_index) -> int | None
layout.context_parallel.local_to_global(local_index) -> int
layout.context_parallel.row_local_slice(row) -> slice
layout.context_parallel.row_local_valid_mask(row) -> Tensor
```

### 2.4 Attention hook 设计

CP path 与 CP=1 path 应分流，但首版实现优先级需要明确：**CP 首版以 expanded KV concat 路径打通为主，attention-mask / visibility 路径只作为后续演进方向**。原因是 mask path 当前尚未 ready，不能作为首版 CP 的主实现依赖。

CP=1：

- 现有 `prefix_attention()` 路径继续保留。
- 当前已验证的 KV concat / expanded KV 逻辑仍作为主路径。
- 后续 attention-mask 方案 ready 后，再统一评估是否替换 CP=1 与 CP>1 的 attention 复用实现。

CP>1 首版：

- 继续采用 PrefixSharing 接管 attention 的实现形态，但必须把 global layout 改为 CP-local layout：
  - Q/K/V 输入长度按 CP-local packed token 校验；
  - RoPE 使用 CP-local position ids；
  - `build_kv()` 按 CP-local row slice 拆分当前 rank K/V；
  - store/load 只处理当前 CP rank 持有的 valid KV；
  - expanded KV 只在当前 CP rank 的 local token 视图上构造；
  - padding token 不入 store，不进入有效复用语义。
- 该路径本质上是“CP-local expanded KV concat”。它不试图重写完整 MindSpeed `kvallgather_cp_algo` 的通信逻辑，而是在当前 rank local tensor 上完成 PrefixSharing 的 KV 复用，再进入可执行 attention 路径。
- 如果首版 CP-local expanded KV 无法安全复用原 MindSpeed CP attention，则必须显式选择一个可验证的 attention execution path，并通过精度实验确认；不能 silent fallback 到 CP=1 的 global packed 假设。

CP>1 后续：

- attention-mask / visibility 方案作为性能优化和显存优化方向保留。
- 目标是未来不再构造 expanded KV，不额外 concat provider prefix KV + reuser suffix KV，而是通过 mask/metadata 控制 reuser suffix 对 provider prefix token 的可见性。
- 该方案只有在 mask path 具备端到端可运行、可测、可精度对齐之后，才能替代首版 expanded KV concat 主路径。

短期开发步骤：

**Step A：探针/guard path**

- CP>1 时，attention patch 打印或记录：
  - query/key/value shape；
  - packed_seq_params qkv_format；
  - global total padded length；
  - expected local total padded length；
  - cp_rank/cp_size；
  - 是否进入 CP-local expanded KV path。
- 如果 CP-local expanded KV path 尚未实现，明确报错或关闭 PS path，避免 silent wrong result。

**Step B：CP-local expanded KV path**

- 使用 `PackedBatchLayout.context_parallel` 拆分当前 rank local Q/K/V；
- `PrefixAttentionStore` key 增加 `cp_rank` 隔离；
- provider 存储当前 rank local valid KV；
- reuser 从同一 `tp_rank/cp_rank` 的 provider entry 加载 local prefix KV；
- 构造当前 rank local expanded KV；
- attention 输出保持当前 rank local query shape；
- 与 `postprocess_thd_engine()` 的 local output 输入形态保持一致。

**Step C：mask path 预留**

- 在 runtime context 中保留未来 mask/visibility 所需的 plan/layout 信息；
- 不把 mask path 作为首版 CP 的通过条件；
- 文档和 pending-items 中记录其为后续显存优化项。

### 2.5 KV / activation store 设计

当前 `PrefixAttentionStore` 的 key 包含：

```text
forward_id, micro_batch_id, layer_id, batch_index, prefix_state_type, tp_rank
```

CP 下需要扩展隔离维度：

```text
..., tp_rank, cp_rank
```

原因：

- CP rank 持有不同 token chunks；
- 同一 provider row 在不同 CP rank 上的 local KV 不是同一份 tensor；
- store 不能让 cp_rank=0 的 local KV 被 cp_rank=1 误读。

首版 CP 仍以 expanded KV concat 为主，因此 store 继续保存 attention KV，只是需要增加 CP rank 隔离和 CP-local valid token 语义。因此建议：

- `PrefixActivationSlotId` 增加 `cp_rank` 字段；
- `PrefixAttentionStore` 继续作为首版 CP 的主 store；
- store entry 表示当前 `tp_rank/cp_rank` 上的 local valid KV shard；
- 后续 mask path 可新增更轻的 `PrefixAttentionVisibility` 或等价 runtime metadata，但不作为首版 CP 的实现依赖；
- 具体命名后续实现前再结合代码收敛。

必须保持：

- 不 detach；
- padding token 不入 store；
- provider-before-reuser 顺序约束必须明确保留或被拓扑构建替代；
- 统计日志区分 local stored tokens 与 global semantic reused tokens。

### 2.6 Prefix-Last Restore 设计

结论：restore 写回阶段继续复用现有 v080 范式；保存 provider logits 阶段新增 CP-local save spec。

#### 2.6.1 写回阶段

保持现有流程：

```text
postprocess_thd_engine 后 output["log_probs"] 是 NestedTensor
  -> _unfold_trimmed_nested_to_2d
  -> restore_reuser_prefix_columns_2d
  -> _fold_2d_to_nested
```

理由：

- 当前 forward_step patch 的 restore 调用点已经在原始 `forward_step` 返回之后；
- 源码确认 `postprocess_thd_engine()` 在该返回之前已经执行；
- 该阶段已经回到 global row/column 语义，最接近现有 CP=1 restore 范式；
- 避免在 restore 写回阶段额外设计跨 CP rank scatter/gather。

#### 2.6.2 保存阶段

当前 `PackedPrefixLastRestoreIndex.provider_1d_pos` 是 global packed index。CP 下应增加保存用 spec：

```python
@dataclass(frozen=True)
class PrefixLastLogitsSaveIndex:
    reuse_idx_in_batch: int
    provider_idx_in_batch: int
    provider_global_packed_pos: int
    provider_local_packed_pos: int | None
    owner_cp_rank: int
    target_2d_pos: int
    label_value: int
```

生成逻辑：

1. 先按现有逻辑得到 provider prefix-last 的 global packed position；
2. 通过 `layout.context_parallel.global_to_local(global_pos)` 判断当前 CP rank 是否拥有该 token；
3. 当前 rank 拥有时，生成 `provider_local_packed_pos`；
4. 当前 rank 不拥有时，`provider_local_packed_pos=None`，不得索引本地 logits。

需要进一步设计的点：saved logits 如何在 restore 阶段可见。

候选方案：

- **方案 R1：CP group all-gather saved logits metadata/tensor**
  - 每个 rank 保存自己拥有的 provider prefix-last logits；
  - restore 前在 CP group 内 all-gather 这些 small tensor；
  - 每个 rank 都拥有完整 restore 所需 logits；
  - 实现简单，通信量很小，适合首版。

- **方案 R2：只在 owner rank restore，再依赖 postprocess 聚合**
  - 不适合当前调用点，因为 restore 已经在 postprocess 后执行，每个 rank 都有完整 NestedTensor 副本语义，不能只让 owner rank 局部改一份。

- **方案 R3：把 restore 前移到 postprocess 前 CP-local log_probs**
  - 需要处理 target token 所在 rank、provider logits 所在 rank可能不同的问题；
  - 会引入跨 rank restore 通信和 local/global target mapping；
  - 开发复杂度高，不符合“尽量复用现有 restore 范式”。

首版建议采用 R1。

注意：`postprocess_thd_engine()` 当前 all-gather 使用 `output.detach()` 创建 `output_list`，但会把当前 rank 的 `output_list[cp_rank]` 替换为未 detach 的 `output`。这是 Verl 现有实现。PrefixSharing 不能额外 detach saved logits；all-gather saved logits 时也要评估梯度路径。若 PyTorch distributed all_gather 对 autograd 不支持，需要使用支持 autograd 的 collective 或只 gather metadata 后用可保留图的方式处理。该点必须通过单测/实机精度验证确认。

### 2.7 日志与观测

保留必要探针，不增加高频冗余日志。

建议日志：

- prepare 阶段：
  - global valid/padded/cu_seqlens
  - cp_rank/cp_size
  - local_total_padded_length
- attention CP path：
  - query/key/value token length
  - expected local token length
  - qkv_format
  - whether original CP attention path is used
- vocab save：
  - global provider pos
  - owner_cp_rank
  - local_pos on current rank or skipped
- restore：
  - restored reuser count
  - saved logits gathered count
  - NestedTensor original lengths

日志默认应可通过环境变量打开，避免训练常态刷屏。

## Chapter 3：测试验证

### 3.1 开发自测：基于 `tests/` 的 UT / IT / ST

开发自测必须覆盖本地可执行测试，目标是保护逻辑和坐标变换，不依赖真实 NPU CP 通信。

#### 3.1.1 Unit Test

建议新增/更新：

- `tests/unit_test/test_config.py`
  - `CP=1` 继续通过；
  - `CP=2/4/8 + context_parallel_algo="kvallgather_cp_algo" + use_remove_padding=True` 通过；
  - `context_parallel_algo` 为其他值时报错；
  - `dynamic_context_parallel=True` 报错；
  - `use_remove_padding=False` + CP>1 报错；
  - BSHD CP 报错；
  - fused kernels / rope fusion 继续报错。

- `tests/unit_test/test_packed_layout.py`
  - 构造 `valid_lengths=[5,2]`、`tp_size=2`、`cp_size=2`，验证 global padded lengths 按 `tp*cp*2` 对齐；
  - 验证 CP rank 0/1 的 local front/back chunk；
  - 验证 `global_to_local()` / `local_to_global()`；
  - 验证 padding token 的 local valid mask；
  - 覆盖 CP=2/4/8。

- `tests/unit_test/test_runtime_context.py`
  - 验证 global restore index 仍正确；
  - 验证 CP save spec 能判断 owner cp rank；
  - 当前 rank 不拥有 provider prefix-last 时不生成 local logits slice；
  - 当前 rank 拥有时 local index 正确。

- `tests/unit_test/test_prefix_store.py`
  - `PrefixActivationSlotId` 增加 `cp_rank` 后，不同 CP rank key 隔离；
  - TP rank + CP rank 组合隔离；
  - 不影响 CP=1 旧行为。

#### 3.1.2 Integrated Test

建议新增/更新：

- `tests/integrated_test/optional/test_verl_megatron_runtime_helpers.py`
  - monkeypatch `MegatronParallelInfo(cp_size=2/4/8, cp_rank=...)`；
  - 验证 `build_prefix_sharing_micro_batch_verl080()` 生成 global layout + CP-local view；
  - 验证 trim 后 valid lengths、padded lengths、local lengths 与 Verl `preprocess_thd_engine()` 规则一致；
  - 验证 restore write path 仍能从 NestedTensor unfold/fold。

- `tests/integrated_test/optional/test_verl080_vocab_restore_cp.py`
  - mock CP-local logits；
  - 验证 `vocab_logprobs` patch 只在 owner rank 保存 logits；
  - 验证 all-gather saved logits 后，`restore_via_2d_unfold_verl080()` 可重算 prefix-last logprob；
  - 验证 saved logits 不 detach。

- `tests/integrated_test/optional/test_megatron_attention_cp_guard.py`
  - CP>1 但非 THD 报错；
  - CP>1 但 local token length 与 layout 不一致报错；
  - 未实现的 CP attention fallback 不允许 silent fallback 到错误 path。

#### 3.1.3 System Test

建议新增/更新：

- `tests/system_test/test_cp_prefix_sharing_smoke.py`
  - 使用 CPU/torch mock backend 构造 CP-local layout；
  - 跑完整 plan → layout → context → vocab save → restore 流程；
  - 不要求真实 MindSpeed CP attention，但要求核心坐标和 restore 闭环。

本地回归命令：

```bash
PYTHONPATH=prefix-sharing PYTHONPYCACHEPREFIX=/private/tmp/prefix-sharing-cp-pycache python3 -m pytest -q -p no:cacheprovider   prefix-sharing/tests/unit_test   prefix-sharing/tests/integrated_test   prefix-sharing/tests/system_test
```

### 3.2 功能验证

功能验证由具备 Megatron/MindSpeed/NPU 环境的设备侧完成。

建议矩阵：

- Qwen2.5 小模型 smoke：
  - TP=1, CP=2
  - TP=2, CP=2
  - TP=4, CP=2
- Qwen3.5/Qwen3.6 目标模型前置验证：
  - TP=4, CP=2
  - TP=4, CP=4（如资源允许）
- PP/SP 组合回归：
  - TP=4, SP=True, PP=1, CP=2
  - TP=4, SP=True, PP=2, CP=2

每个 case 检查：

- prefix-sharing path 被启用；
- `context_parallel_algo="kvallgather_cp_algo"`；
- attention hook 未走错误的 non-CP backend attention；
- logits save owner/local index 日志正常；
- restore count 与 plan 预期一致；
- 训练 step 不 hang、不 NaN、不 shape error。

### 3.3 集成验证

集成验证关注与 Verl/Megatron/MindSpeed 主流程兼容：

- `compute_log_prob` 路径：
  - old logprob 计算成功；
  - output NestedTensor 形态正确；
  - restore 后 log_probs 长度与原始 sample length 对齐。

- `update_actor` 路径：
  - forward/backward 成功；
  - loss backward 不报 autograd / collective 错误；
  - optimizer step 成功。

- activation checkpointing：
  - recompute forward 中 prefix-sharing context 正常；
  - saved logits / restore 不跨 forward 生命周期污染；
  - store 生命周期仍限定在 micro-batch context 内。

- PP/SP/TP/CP 组合：
  - PP 非 last stage restore no-op；
  - PP last stage restore 生效；
  - SP=True 时 hook token length 仍符合 CP-local THD 预期；
  - TP rank 和 CP rank store key 隔离。

### 3.4 精度对齐

精度对齐是 CP 适配的通过标准。

建议对比方式：

1. 固定随机种子、固定 micro-batch、关闭 dropout 或使用 deterministic 设置。
2. 跑 baseline：PrefixSharing disabled，CP enabled。
3. 跑 candidate：PrefixSharing enabled，CP enabled。
4. 对比：
   - post-restore `log_probs`；
   - entropy；
   - loss；
   - 关键参数梯度 norm；
   - provider prefix 相关梯度是否存在且非 None；
   - reuser suffix-first logprob 是否对齐。

精度阈值建议：

- bf16/NPU 下 `max_abs_diff`、`max_rel_diff` 阈值参考现有 TP/SP 验证标准；
- prefix-last 位置单独统计，不能只看整体均值；
- provider/reuser 分组分别统计，避免 restore 错误被非共享样本稀释。

必须验证的特殊样本：

- reuser suffix_len = 0；
- prefix_len = 1；
- provider/reuser prefix-last 分布在不同 CP rank；
- provider/reuser prefix-last 分布在同一 CP rank；
- padding 后 prefix-last 临近 chunk 边界；
- batch 内链式 provider/reuser。

### 3.5 性能对比

性能对比不作为首版功能通过门槛，但需要记录，避免 CP 适配引入明显退化。

指标：

- step time；
- forward time；
- attention time；
- CP communication time（如可从 profiler 拿到）；
- peak memory；
- saved/reused prefix token 数；
- saved logits all-gather 小通信量；
- 与 baseline 的吞吐差异。

对比组合：

```text
CP enabled + PrefixSharing disabled
CP enabled + PrefixSharing enabled
```

后续切换 attention-mask 复用方式后，需要新增：

```text
KV concat path vs attention-mask path
```

重点观察：

- KV concat path 是否显存放大明显；
- attention-mask path 是否避免 expanded KV 显存；
- CP all-gather 与 PrefixSharing metadata 通信是否引入额外瓶颈。

### 3.6 冒烟测试

冒烟测试用于快速判断代码能否进入主路径。

建议最小配置：

```text
model: small Qwen / tiny causal LM
use_remove_padding: True
tensor_model_parallel_size: 1 or 2
context_parallel_size: 2
context_parallel_algo: kvallgather_cp_algo
sequence_parallel: 可先 false，再 true
pipeline_model_parallel_size: 1
prefix_sharing.enable_prefix_sharing: True
```

检查项：

- 启动不报 config guard；
- prepare 日志出现 CP layout；
- attention path 日志显示 CP-local token length；
- vocab save 日志显示 owner/local index；
- restore 日志显示 restore count；
- 单步 forward/backward 完成；
- 输出 log_probs shape 与 baseline 一致。

## Chapter 4：开发计划

### 4.1 Phase 0：源码固化与探针确认

核心开发内容：

- 将本文 Chapter 1 的源码结论固化到文档；
- 在 CP 相关路径增加最小探针日志，确认真实设备上的 Q/K/V、logits、log_probs shape；
- 在 config 中先保守 guard：只允许 `THD + kvallgather_cp_algo`，其他 CP 算法拒绝；
- CP>1 且尚未实现 reuse 时，禁止 silent fallback 到当前 non-CP `prefix_attention()`。

自测试检查点：

- config 单测通过；
- CP 非目标算法会明确报错；
- 本地 mock CP=2/4/8 能构造 runtime state；
- 设备侧拿到 attention/logits/restore 探针日志。

### 4.2 Phase 1：扩展 `PackedBatchLayout` 的 CP-local view

核心开发内容：

- 新增 `ContextParallelPackedView` 或等价结构；
- `PackedBatchLayout.from_kept_position_rows()` 支持传入 `cp_rank/cp_size` 构造 local view；
- 实现 global/local index 映射 helper；
- 实现 local valid mask；
- 对齐 Verl `preprocess_thd_engine()` 的 zigzag chunk 规则。

自测试检查点：

- UT 覆盖 CP=2/4/8；
- local length 等于 `global_total_padded / cp_size`；
- global_to_local/local_to_global 互逆；
- padding mask 正确；
- CP=1 行为完全兼容旧 layout。

### 4.3 Phase 2：Prefix-Last logits save CP-local 化

核心开发内容：

- 从 global restore index 派生 `PrefixLastLogitsSaveIndex`；
- vocab patch 使用 local index 保存 provider prefix-last logits；
- 设计并实现 saved logits CP group small all-gather；
- 保证 saved logits 不 detach 或明确使用支持 autograd 的 collective；
- restore 写回继续复用 `restore_via_2d_unfold_verl080()`。

自测试检查点：

- owner rank 保存 logits；
- non-owner rank 不误索引；
- all-gather 后每个 rank 都具备 restore 所需 saved logits；
- prefix-last logprob 重算值与 baseline mock 对齐；
- provider prefix-last logits 的梯度路径不丢。

### 4.4 Phase 3：CP attention path 适配

核心开发内容：

- 调整 attention patch 分流：CP>1 进入 CP-local expanded KV path；
- 将当前 global packed 的 `prefix_attention()` 改造为支持 CP-local Q/K/V/layout；
- `build_kv()` 支持按 CP-local row slice store/load/expand KV；
- `PrefixAttentionStore` key 增加 `cp_rank` 隔离；
- RoPE / position ids 使用 CP-local position 视图；
- attention 输出 shape 与 CP-local query shape 对齐，保证可被 Verl `postprocess_thd_engine()` 接收；
- mask/visibility path 只做接口预留和文档记录，不作为首版 CP 主路径。

自测试检查点：

- CP>1 下 Q/K/V local length 与 layout 匹配；
- CP-local `build_kv()` 不保存 padding KV；
- 不同 `cp_rank` 的 store key 隔离；
- expanded KV local shape 与 plan/layout 预期一致；
- attention output shape 等于 local query shape；
- 设备侧 CP forward/backward 不 hang。

### 4.5 Phase 4：端到端精度闭环

核心开发内容：

- 串通 prepare → context → attention CP path → vocab save → postprocess → restore；
- 补齐统计日志；
- 补齐 PP/SP/TP/CP 组合 guard；
- 修正设备侧发现的 index 边界问题；
- 完成精度对齐脚本或实验记录模板。

自测试检查点：

- 本地 UT/IT/ST 全部通过；
- 设备侧 smoke 通过；
- baseline vs PS 的 log_probs/loss/gradient 对齐；
- prefix-last 位置单独对齐；
- provider prefix 梯度不丢。

### 4.6 Phase 5：性能验证与文档收口

核心开发内容：

- 记录 CP enabled 下 PrefixSharing on/off 性能；
- 对比 KV concat 与 attention-mask path 的显存和时间；
- 更新 README / overview / pending-items；
- 将不支持场景写入 guard 错误信息和文档。

自测试检查点：

- 性能数据有 baseline；
- 无明显 step time regression；
- pending items 已记录 BSHD CP、其他 CP algo、dynamic CP；
- PR body 包含测试结果。

## Chapter 5：当前结论

1. 首版 CP 支持范围已经确认：只支持 `THD + context_parallel_algo="kvallgather_cp_algo"`。
2. CP 在 Verl THD 中不只是 attention 内部行为；`preprocess_thd_engine()` 会先把输入 token 切成 CP-local packed token，因此模型 forward、LM head、logits_processor 都是 CP-local。
3. 首版 CP attention 以 CP-local expanded KV concat 打通为主；mask/visibility path 当前尚未 ready，只能作为后续优化方向。
4. Prefix-Last Restore 写回点可以继续沿用 postprocess 后 Nested/2D 范式；但 provider prefix-last logits 保存点位于 postprocess 前，必须使用 CP-local index。
5. `PackedBatchLayout` 可以继续作为统一 layout 入口，但必须扩展 CP-local view；不建议新建完全平行 layout。
6. 后续 KV 复用方式计划从 KV concat 转向 attention-mask / 可见性机制，但首版 CP 不能依赖尚未 ready 的 mask path；设计上需要让 CP-local expanded KV 先闭环，同时为 mask path 保留演进空间。
7. CP 适配的第一优先级是精度闭环，性能优化在精度确认后推进。

## Chapter 6：遗留问题

1. **主要遗留问题：attention-mask / visibility 复用方案尚未最终落代码**
   - 这是 CP 适配最重要的长期遗留问题之一。
   - 首版 CP 不依赖 mask path，而是以 CP-local expanded KV concat 打通为主，原因是 mask path 当前尚未 ready，不能作为首版功能闭环的前置条件。
   - 但 expanded KV concat 会带来额外 KV 显存和潜在 attention 执行路径复杂度；长期更理想的方案是通过 attention-mask / visibility metadata 控制 reuser suffix 对 provider prefix token 的可见性，避免额外 concat。
   - 后续需要单独设计 mask 数据结构、patch 点、与 MindSpeed `kvallgather_cp_algo` attention 的接口，以及对应的精度和性能验证。

2. **saved logits CP all-gather 的 autograd 方案待验证**
   - 需要确认使用何种 collective 能保留 saved provider logits 的梯度路径。
   - 如果普通 `torch.distributed.all_gather` 不支持 autograd，需要替换为 autograd-safe collective 或调整 restore 保存策略。

3. **BSHD CP 暂不支持**
   - 业务后续可能需要 BSHD CP。
   - BSHD preprocess/postprocess 与 THD 不同，需要独立设计 layout 和 restore。

4. **其他 CP 算法暂不支持**
   - Ring / Ulysses / hybrid / adaptive CP 的 token分布和 attention 通信方式不同。
   - 首版全部 guard，后续按业务需要逐个研究。

5. **dynamic context parallel 暂不支持**
   - Verl `preprocess_thd_engine(local_cp_size=...)` 有 dynamic CP 分支。
   - 首版先拒绝，避免 DP-CP group 动态变化导致 store key 和 layout 生命周期复杂化。

6. **inter micro-batch sharing 暂不支持**
   - CP 下跨 micro-batch store 生命周期、CP rank ownership、PP stage 隔离都会更复杂。
   - 本轮只支持单 micro-batch 内 sharing。

7. **设备侧实验依赖 NPU/MindSpeed 环境**
   - 本地可完成 layout/index/restore mock 测试。
   - 真实 CP attention、NPU fusion attention、跨 rank saved logits 通信和性能数据必须在设备侧验证。

---

## §3.7 4090 GPU 测试验证结果

### 3.7.1 核心限制：GPU（无 mindspeed）无法实际激活 CP>1

所有 GPU 端测试**实际均为 CP=1**，原因如下：

megatron-core v0.16.1 中，`context_parallel_size > 1` 时 CP 组与 DP 组合并为更大的 DP 组（CP 作为 DP 的扩展维度），`mpu.get_context_parallel_world_size()` 返回 `1`，不是 `2`。只有安装了 mindspeed 或其 TE patch 后，CP 才会真正独立出组、`get_context_parallel_world_size()` 才会返回 `2`。详见下文的 smoketest 探针日志。

因此 **CP>1 的实体验证必须依赖 NPU + mindspeed 环境**，4090 GPU 只能做 CP=1 的 baseline 回归和代码级正确性检查。

### 3.7.2 单元/集成测试 (本地 Mac)

| 套件 | 通过 | 跳过 |
|------|------|------|
| Unit Tests (`tests/unit_test/`) | **230** | 1 (transformers) |
| Integrated Tests (`tests/integrated_test/`) | **46** | 29 (flash_attn/npu/verl/mindspeed) |
| **CP 专项 UT** | | |
| `test_packed_layout.py` CP view | ✅ CP=2/4/8 local construction, global_to_local, local_valid_mask | |
| `test_config.py` CP guard | ✅ `context_parallel_algo="kvallgather_cp_algo"` accept/ reject, dynamic CP reject | |
| `test_runtime_context.py` CP save | ✅ owner rank save, non-owner skip, gather mechanism | |
| `test_prefix_store.py` CP isolation | ✅ `PrefixActivationSlotId.cp_rank` key isolation | |
| `test_torch_ref_backend.py` CP attention | ✅ CP-local KV build, CP-local attention shape | |

### 3.7.3 Smoketest (服务器 GPU4-5，均为 CP=1)

实际运行日志确认 `cp_size=1`（核心证据）：
```
cp_rank=0/cp_size=1    ← 全程 CP=1，CP=2 配置未生效
total_padded_length=94 expected_token_length=94  ← CP=2 时预期应为 47 (94/2)
```

**Case 1: PS=OFF (baseline, 2 卡, 实质 CP=1+DP=2)**
```
step:1 - step_time=17.5s -> throughput=5.6 tokens/s
step:2 - step_time=2.9s  -> throughput=33.0 tokens/s
```
完整跑完 2 训练步，metric 正常。

**Case 2: PS=ON (2 卡, torch_ref backend, 实质 CP=1+DP=2)**

| 项目 | 结果 |
|------|------|
| Config guard 通过 | ✅ torch_ref backend 放宽 `context_parallel_algo` 检查 |
| PS 7 patch 全部激活 | ✅ `forward_step`/`attention`/`vocab`/`no_padding` |
| Plan 检测 prefix-sharing | ✅ `prefix_len=4`, 1 provider + 1 reuser |
| 进入 `prefix_attention()` | ✅ PS attention path 触发 |
| Audit summary | ✅ `reused_valid_tokens=4, reused_valid_token_ratio=0.0408` |
| KV build (24 layers) | ✅ `store_count=2, reuse_hit_count=1, expanded_kv_tokens=98` |
| Restore | ✅ `actual_restore_count=1`（24 层一致） |
| **CP splitting 实际激活** | ❌ **`cp_size=1`** — megatron-core 无 mindspeed 时不独立 CP 组 |
| OOM | ❌ colocate 模式 2 卡 DataLoader worker killed |

**结论：** CP=2 配置虽被 prefix_sharing config guard 接受，但 megatron-core 侧的分布式初始化
并未实际建立独立的 CP 组。所有 >1 的 `context_parallel_size` 被折叠进 DP 维度。
`preprocess_thd_engine()` 读到 `cp_size=1` → 不做 token splitting。

### 3.7.4 配置修复 (commit `1c32d128` on `open-source_cp`)

放宽了 `config.py` 中 `torch_ref` backend 的 `context_parallel_algo` 检查：
- **`validate_for_engine()`**: CP>1 + `torch_ref` 不要求 `context_parallel_algo`
- **`validate()`**: 同上
- **原因**: TorchReferenceBackend 自带 position-based causal mask CP-local attention，
  不依赖 mindspeed 的 `kvallgather_cp_algo`

### 3.7.5 后续验证方向

| 优先级 | 验证项 | 条件 | 方法 |
|--------|--------|------|------|
| P0 | **CP>1 token splitting + PS path 全链路** | 需 NPU + mindspeed 环境 | TP=1 CP=2 PS=ON/OFF smoketest |
| P0 | **CP attention 精度** | CP 实际激活后 | 固定种子对比 log_probs/loss/gradient |
| P1 | TP=2 CP=2 | CP>1 工作后 | 扩展 smoketest 矩阵 |
| P1 | PP/SP 组合 | CP>1 工作后 | postprocess_thd_engine restore 兼容性 |
