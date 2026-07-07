# PrefixSharing 性能分析和优化研究

本文档基于 `PrefixSharing_perf` / `open-source_perf` 分支当前代码，按“研究分析 → 方案设计 → 测试验证 → 开发计划 → 当前结论 → 遗留问题”的功能工作流组织。目标是同时分析速度性能与显存性能，并给出可落地的优化优先级。

## 1. 研究分析

### 1.1 研究范围与边界

本轮已完成：

- 阅读当前 `prefix-sharing` 的 core / backends / integrations / setup patch 主流程代码。
- 对 core detector / planner、TorchRef backend、FSDP dense pack/scatter 做本地 CPU PoC 计时。
- 对 block causal mask 做显存规模估算。
- 拉取并合并最新 `origin/open-source` 后，重新审视已合入的 FSDP patch 与 GPU/NPU FlashAttention backend。

本轮未完成：

- 未在真实 GPU / NPU 上跑 profiler，因此涉及 device kernel、stream、HBM peak 的结论需要在目标环境复验。
- 涉及 attention 主体时，只把 GPU/NPU FlashAttention 算子作为正式优化对象；TorchRef attention 仅作为调试/reference 路线。
- TorchRef `build_kv()` 仍是正式路径热点，因为当前 GPU/NPU FlashAttention backend 也复用 TorchRef 的 KV expansion 实现。

本文中的性能包括：

- 速度性能：CPU 调度、Python 循环、device kernel、同步、日志 I/O、端到端训练耗时。
- 显存性能：HBM 峰值、临时 tensor、dense mask/bias、dense scatter、expanded KV、restore 保存信息。

### 1.2 Megatron-Based 核心流程与热点

当前 Megatron 入口主要在：

- `prefix-sharing/prefix_sharing/integrations/verl_mcore.py`
- `prefix-sharing/prefix_sharing/integrations/megatron_runtime.py`
- `prefix-sharing/prefix_sharing/integrations/context.py`
- `prefix-sharing/prefix_sharing/backends/torch_ref.py`
- `prefix-sharing/prefix_sharing/backends/flash_atten_gpu.py`
- `prefix-sharing/prefix_sharing/backends/flash_atten_npu.py`

主流程：

1. verl actor micro-batch 构造阶段读取 prefix-sharing 配置。
2. 从 `input_ids` / `attention_mask` 提取每条样本的有效 token 序列。
3. `PrefixSharingPlanner` 调用 detector 识别 provider / reuser / prefix 长度。
4. 根据 plan trim micro-batch，只保留 provider full sequence 与 reuser suffix。
5. 构造 `PackedBatchLayout` 和 `PrefixSharingRuntimeState`。
6. 进入 `prefix_sharing_runtime_context()`，创建 store、stats、restore indices。
7. Megatron attention hook 进入 prefix-sharing path：
   - 校验 packed THD token 长度。
   - 按 `packed_position_ids` 应用 RoPE。
   - backend `build_kv()` 构造 expanded KV。
   - backend `attention()` 计算 trimmed query 对 expanded KV 的注意力；正式性能路线应走 GPU/NPU FlashAttention，TorchRef attention 仅用于调试/reference。
   - 走 Megatron linear projection。
8. logprob 阶段执行 prefix-last restore，补回 reuser 第一个 suffix token 需要的 prefix-last logprob。
9. context 退出时输出 audit / layer stats。

流程级热点：

- CPU overhead：有效 token 提取中的 `nonzero`、`.detach().cpu().tolist()`、Python trie detector、plan 的 Python list/dataclass 构造。
- Device overhead：RoPE position 索引、`build_kv()` 的 split / load / cat、GPU FA varlen 输入整理、NPU FA BSH pad/stack/mask、projection 后 restore。
- Memory overhead：expanded KV、NPU FA per-sample mask 与 BSH padding、prefix-last restore 保留 provider logits autograd 路径、TP padding 后的 padded token。
- I/O overhead：热路径里存在直接 `print()`、context 退出 audit print、可选 diagnostic dump。

Megatron 路径第一阶段重点：`build_kv()`、core detector/planner 早停、热路径日志开关、GPU/NPU FA 输入整理与 mask/pad-stack 显存观测。

### 1.3 FSDP-Based 核心流程与热点

当前 FSDP 入口主要在：

- `prefix-sharing/prefix_sharing/integrations/verl_fsdp.py`
- `prefix-sharing/prefix_sharing/setup/patches/verl080_fsdp/forward_step.py`
- `prefix-sharing/prefix_sharing/setup/patches/verl080_fsdp/attention.py`
- `prefix-sharing/prefix_sharing/backends/flash_atten_gpu.py`
- `prefix-sharing/prefix_sharing/backends/flash_atten_npu.py`

主流程：

1. `FSDPEngineWithLMHead.forward_step` patch 读取 `prefix_sharing_config`，disabled 或 no-sharing 时尽量复用原 engine 流程。
2. enabled 时调用 `build_prefix_sharing_micro_batch_fsdp()`，从 dense 2D 或 NestedTensor 输入中提取有效 token 序列并调用 `PrefixSharingPlanner`。
3. 如果没有 sharing，走 `_call_original_like_engine()`，不打开 prefix-sharing context。
4. 如果有 sharing，构造 trimmed micro-batch、`PackedBatchLayout`、`PrefixSharingRuntimeState`。
5. 调用真实 engine 的 `prepare_model_inputs()` / `prepare_model_outputs()`，在 forward 期间打开 `prefix_sharing_runtime_context()`。
6. `ALL_ATTENTION_FUNCTIONS.get_interface` patch 在 context 激活时拦截 HF attention，把 `[B,H,L,D]` Q/K/V 转为 `[B,L,H,D]` 交给 `PrefixSharingFSDPAttentionRuntime`。
7. runtime 将 dense Q/K/V pack 成 packed token 或处理 `[1,T,H,D]` packed single-batch 输入。
8. backend 先复用 TorchRef `build_kv()` 构造 expanded KV，再通过正式 FA backend 执行 attention；TorchRef attention 只作为调试/reference。
9. packed attention output scatter 回 dense shape，交还 HF/verl 后续输出处理。
10. 在 raw output / model output 阶段保存 prefix-last logits，并执行 2D restore。

流程级热点：

- CPU overhead：序列提取和 planner 是 prepare 阶段主要 CPU 开销。
- Device overhead：dense Q/K/V pack、packed output scatter、HF QKV transpose、TorchRef `build_kv()`、GPU/NPU FA 输入整理。
- Memory overhead：即使 prefix-sharing 后有效 token 变少，FSDP 路径仍可能 scatter 回 dense output，保留 dense shape 的 HBM 压力；NPU FA BSH pad/stack/mask 也会引入额外 HBM；restore 在 dense 2D 输出上执行。
- 兼容性热点：NestedTensor / packed path 能否贯穿下游，决定 FSDP 显存收益上限。

FSDP 路径第一阶段重点：core 早停、减少 dense pack/scatter、尽可能把 packed/jagged 形态向后传递、避免无收益 batch 进入重流程，并针对 GPU/NPU FA backend 分别统计 varlen 输入整理与 NPU BSH mask/pad-stack 成本。

### 1.4 PoC 观测结果

本节按前文的热点分类组织 PoC 与后续实验计划。当前 Codex 侧已完成本地 CPU PoC，可用于定位 Python 调度、临时对象和粗粒度内存规模；GPU/NPU 相关实验需要后续在目标训练环境执行，并将结果回填到本文档。

#### CPU Overhead

已完成本地 PoC：

| module | case | B | L | prefix | reuser | reused tokens | detector / extract ms | planner / prepare ms | trim / pack / scatter ms | peak Python MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| core | one_provider | 8 | 256 | 128 | 7 | 896 | 1.373 | 1.413 | trim 0.012 | 0.43 |
| core | one_provider | 32 | 512 | 384 | 31 | 11904 | 10.797 | 10.939 | trim 0.046 | 1.85 |
| core | none | 32 | 512 | 0 | 0 | 0 | 25.584 | 25.939 | trim 0.081 | 6.22 |
| core | chain | 32 | 512 | 128 | 31 | 4464 | 20.438 | 20.432 | trim 0.070 | 4.64 |
| fsdp prepare | one_provider | 8 | 256 | 128 | 7 | 896 | extract 0.146 | prepare 1.715 | - | 0.47 |
| fsdp prepare | one_provider | 32 | 512 | 384 | 31 | 11904 | extract 1.080 | prepare 11.948 | - | 2.19 |
| fsdp prepare | none | 32 | 512 | 0 | 0 | 0 | extract 1.945 | prepare 27.352 | - | 6.78 |
| fsdp dense qkv | one_provider | 8 | 256 | 128 | 7 | 896 | - | - | pack 0.126 / scatter 0.195 | - |
| fsdp dense qkv | one_provider | 32 | 512 | 384 | 31 | 11904 | - | - | pack 0.500 / scatter 0.970 | - |

关键观察：

- 无 sharing batch 反而更贵，因为 detector 仍构造大量唯一 trie 节点，最后收益为 0。
- 这说明“无共享批次”很可能是 prefix detect/planner 的最坏路径之一：不仅没有计算收益，还会付出完整 CPU detect 成本。该结论需要在真实 workload 上复验；若确认，应作为 P0 优化。
- chain 场景也更贵，说明链式复用语义会放大 detector / planner 的 Python 开销。
- `trim_batch()` 本身不是主要 CPU 热点；更值得关注的是 detector/planner、FSDP prepare、dense pack/scatter。
- FSDP no-sharing prepare 同样昂贵且无收益。当前 prefix detect/planner 位于 `prefix-sharing/prefix_sharing/core/`，Megatron 与 FSDP 两条路径都调用 `PrefixSharingPlanner(config).plan(sequences)`，因此该环节是共享 core 逻辑；一次优化会同时作用于 Megatron 和 FSDP，应优先评估为 P0。
- plan 当前大量使用 Python list / dataclass 表达，读写清晰但不一定高效。该判断目前仍是性能猜想，需要通过对象数量、tracemalloc、planner p50/p90 和真实训练 prepare latency 验证；如果确认是主要开销，也应提升为 P0。

后续 GPU/NPU 环境实验指导：

1. 增加或打开 prepare 侧打点，位置必须覆盖以下函数边界：
   - Megatron：`build_prefix_sharing_micro_batch_verl070()` / `build_prefix_sharing_micro_batch_verl080()`。
   - FSDP：`build_prefix_sharing_micro_batch_fsdp()`。
   - Core：`PrefixSharingPlanner(config).plan(sequences)` 内部至少拆出 detector 与 plan construction。
2. 每个 micro-batch 记录以下字段：
   - `pipeline`: `megatron` / `fsdp`
   - `device`: `gpu` / `npu`
   - `case`: `disabled` / `enabled_no_sharing` / `one_provider` / `chain`
   - `batch_size`
   - `max_seq_len`
   - `total_valid_tokens`
   - `provider_count`
   - `reuser_count`
   - `reused_tokens`
   - `has_sharing`
   - `detector_node_count`（若当前代码还没有该字段，先记录 TODO）
   - `plan_list_field_count` / `plan_estimated_py_objects`（若当前代码还没有该字段，先记录 TODO）
3. 记录 micro-batch prepare 阶段细分耗时：
   - `extract_sequences_ms`
   - `attention_mask_nonzero_ms`
   - `to_cpu_tolist_ms`
   - `detector_ms`
   - `plan_construct_ms`
   - `planner_ms`
   - `trim_batch_ms`
   - `layout_build_ms`
   - FSDP 额外记录 `pack_dense_qkv_ms`、`scatter_output_ms`
4. 对 `.detach().cpu().tolist()` 的同步成本做 A/B 实验：
   - A：原始路径，按现有逻辑从 device tensor 提取 sequences。
   - B：在进入函数前调用一次 device synchronize，再计时 `attention_mask.nonzero()` 与 `.detach().cpu().tolist()`。
   - C：如果 batch 中可获得 CPU 侧 input ids metadata，用 CPU metadata 构造 sequences，不访问 device tensor。
   - 对比 A/B/C 的 `extract_sequences_ms`、`to_cpu_tolist_ms` 和 micro-batch prepare 总耗时。
5. 同一批 workload 分别跑：
   - prefix-sharing disabled
   - enabled 但 no-sharing batch
   - enabled 且 one-provider sharing batch
   - enabled 且 chain reuse batch
6. 每组至少记录 50 个 micro-batch，输出 p50 / p90 / p99，避免单次抖动误判。
7. 判定规则：
   - 如果 `enabled_no_sharing` 的 `planner_ms` 明显高于 sharing case，且端到端没有收益，no-sharing fast path 作为 P0。
   - 如果 `detector_ms + plan_construct_ms` 占 prepare 阶段超过 20%，core detector/planner 优化作为 P0。
   - 如果 `plan_construct_ms` 或 Python object 规模显著，list/dataclass 紧凑化进入 P0/P1，按真实占比决定。
   - 如果 `attention_mask_nonzero_ms + to_cpu_tolist_ms` 在 GPU/NPU 上占 prepare 阶段超过 5%，CPU metadata path 进入 P0/P1。

结果回填模板：

| device | pipeline | case | batch | seq | valid tokens | reused tokens | provider/reuser | nonzero ms p50/p90 | tolist ms p50/p90 | detector ms p50/p90 | plan construct ms p50/p90 | trim/layout ms p50/p90 | py objects/peak MB | conclusion |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

#### Device Overhead

已完成本地 PoC：

| case | B | L | prefix | q tokens | expanded KV tokens | build_kv ms | debug attention ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| one_provider | 8 | 256 | 128 | 1152 | 2048 | 0.185 | 3.184 |
| one_provider | 32 | 512 | 384 | 4480 | 16384 | 2.399 | 20.129 |
| chain | 32 | 512 | 128 | 11920 | 16384 | 2.334 | 41.775 |

关键观察：

- CPU PoC 中 debug attention 本体大于 `build_kv()`，但这不推翻 NPU 上 `build_kv()` 接近 40% 的实测结论。
- 当前 `build_kv()` 使用 per-row `store.load()`、`torch.cat()`、最终 `torch.cat()`，在 device 上容易产生额外 kernel、内存分配和同步压力。
- chain 场景中 q tokens 更多，attention 耗时显著增加。
- TorchRef attention 计时只用于 debug/reference 判断，不作为正式 attention 优化路线；正式 attention 只考虑 GPU/NPU FA backend。

后续 GPU/NPU 环境实验指导：

1. 只把正式 attention 算子纳入性能结论：
   - GPU：`flash_atten_gpu.py` varlen FA。
   - NPU：`flash_atten_npu.py` BSH + `npu_fusion_attention`。
   - TorchRef attention 仅用于 correctness/debug 对照。
2. 对 `prefix_attention` 或 FSDP runtime 的端到端 attention path 做分段计时：
   - `rope_ms`
   - `build_kv_ms`
   - `fa_prepare_ms`
   - `fa_kernel_ms`
   - `fa_post_ms`
   - `linear_proj_ms`（Megatron 路径）
   - `restore_ms`
   - `prefix_attention_total_ms`
3. 对 `build_kv()` 拆分计时：
   - split packed K/V row 的耗时。
   - provider store 耗时。
   - reuser load 耗时。
   - prefix copy / suffix write / `torch.cat` 或 prealloc 写入耗时。
   - final expanded K/V 拼接或 buffer finalize 耗时。
4. 对 GPU FA 拆分计时：
   - `_prepare_flash_inputs`
   - `flash_attn_varlen_func`
   - `_repad_output`
5. 对 NPU FA 拆分计时：
   - THD split
   - BSH pad/stack
   - per-sample 4D mask build
   - `npu_fusion_attention`
   - THD unpack
6. profiling 开启同步计时：
   - CUDA 使用 `torch.cuda.synchronize()`。
   - NPU 使用对应 `torch_npu.npu.synchronize()` 或环境中的 NPU synchronize API。
7. 每组 workload 记录 p50 / p90 / p99，并同时记录 `provider_count`、`reuser_count`、`valid_tokens`、`expanded_kv_tokens`。
8. 判定规则：
   - 如果 `build_kv_ms / prefix_attention_total_ms` 超过 20%，且绝对耗时明显，`build_kv()` prealloc 作为 P0。
   - 如果 `fa_prepare_ms + fa_post_ms` 接近或超过 FA kernel 耗时，FA 输入整理作为 P0。
   - 如果 NPU `mask_build_ms + pad_stack_ms` 占 attention path 超过 10%，NPU FA mask/pad-stack 优化作为 P0。

结果回填模板：

| device | pipeline | backend | case | total attention ms p50/p90 | rope ms | build_kv ms | build_kv % | fa prepare ms | fa kernel ms | fa post ms | restore/proj ms | expanded kv tokens | conclusion |
|---|---|---|---|---|---|---|---:|---|---|---|---|---:|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

#### Memory Overhead

已完成本地 PoC：

TorchRef debug mask 显存规模：

| case | total q | total kv | bool mask MB | fp16 bias MB |
|---|---:|---:|---:|---:|
| one_provider, B=8,L=256 | 1152 | 2048 | 2.25 | 4.50 |
| one_provider, B=32,L=512 | 4480 | 16384 | 70.00 | 140.00 |

FSDP dense output 与 packed output 规模：

| case | dense output MB | packed output MB |
|---|---:|---:|
| one_provider, B=8,L=256 | 4.00 | 2.25 |
| one_provider, B=32,L=512 | 32.00 | 8.75 |

关键观察：

- dense `[total_q, total_kv]` mask/bias 是明确的 HBM 风险，但它属于 TorchRef/debug attention 路线，不应作为正式 attention 主路径方案。
- 正式 NPU FA backend 当前也会构造 per-sample 4D mask `(batch_size,1,max_q,max_kv)`，并将 THD Q/K/V pad/stack 到 BSH；这部分才是正式 NPU attention 路线需要重点观测的显存项。
- dense output 比 packed output 大很多；如果最终仍 scatter 回 dense，显存收益会被削弱。
- pack/scatter CPU 时间不高，但 device 上仍可能带来显存写放大和 kernel 调度。

后续 GPU/NPU 环境实验指导：

1. 记录每个 micro-batch 的理论 token 和 tensor 规模：
   - `valid_tokens`
   - `padded_tokens`
   - `expanded_kv_tokens`
   - `provider_count`
   - `reuser_count`
   - `max_q`
   - `max_kv`
2. GPU FA 记录：
   - Q/K/V packed tensor bytes。
   - repad output bytes。
   - peak allocated / reserved memory。
3. NPU FA 记录：
   - `q_bsh/k_bsh/v_bsh` bytes。
   - per-sample 4D mask bytes。
   - output THD bytes。
   - peak HBM。
4. FSDP 记录：
   - dense Q/K/V bytes。
   - packed Q/K/V bytes。
   - dense output bytes。
   - packed output bytes。
   - restore 相关 logits/log_probs/entropy/attention_output bytes。
5. 对比 prefix-sharing disabled / enabled 的 peak HBM。如果 enabled 降低计算但 peak HBM 不降，需要优先定位是 expanded KV、FA mask/pad-stack 还是 dense scatter 抵消收益。

结果回填模板：

| device | pipeline | backend | case | peak HBM disabled | peak HBM enabled | expanded KV MB | FA mask/pad MB | dense scatter MB | conclusion |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

#### IO Overhead

已完成本地观察：

- 当前代码和 patch 中仍存在直接 `print()`、diagnostic dump、audit stats 输出路径。
- FSDP attention patch 在 `PREFIX_SHARING_DIAG_DUMP` 开启时会把 attention output 累积并落盘，用于精度对齐诊断；该能力不应在普通性能测试中开启。
- 本地 PoC 没有单独量化 stdout / 文件落盘对训练吞吐的影响。

后续 GPU/NPU 环境实验指导：

1. 分别测试以下配置：
   - 默认模式：关闭所有 diagnostic dump 和 debug print。
   - profiling 模式：只开启结构化性能数据 JSONL/CSV 低频落盘。
   - diagnostic 模式：开启 `PREFIX_SHARING_DIAG_DUMP`。
2. 每组记录：
   - forward latency
   - backward latency
   - micro-batch end-to-end latency
   - mini-batch end-to-end latency
   - stdout 行数或日志文件大小
   - diagnostic dump 文件大小
3. diagnostic 模式只用于精度排查，不纳入正式性能收益评估。
4. 如果 profiling 模式带来超过 3% 的端到端开销，需要降低落盘频率或改为 rank-local buffer 批量 flush。

结果回填模板：

| device | pipeline | mode | forward ms p50/p90 | backward ms p50/p90 | micro-batch ms p50/p90 | log size | dump size | conclusion |
|---|---|---|---|---|---|---:|---:|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

### 1.5 模块级热点归纳

#### Core

涉及代码：

- `prefix-sharing/prefix_sharing/core/prefix_detector.py`
- `prefix-sharing/prefix_sharing/core/planner.py`
- `prefix-sharing/prefix_sharing/core/batch_trim.py`
- `prefix-sharing/prefix_sharing/core/prefix_store.py`
- `prefix-sharing/prefix_sharing/core/observability.py`

热点：

- detector 对所有样本完整扫描并插入 trie，无 sharing batch 仍承担完整成本。
- plan 使用大量 Python list 和 frozen dataclass 字段，便于读写但不是低开销表示。
- `trim_batch()` 不是主要热点，真正成本更多来自 integration 层 tensor slicing、NestedTensor 构造、dense mask 重建。
- Store 使用 dataclass key + dict，本地 microbench 不是第一热点；真正成本在 backend 每 row load 后 `torch.cat()` 和 expanded KV 构造。
- audit / stats 当前有训练热路径刷屏风险。

#### Integrations

涉及代码：

- `prefix-sharing/prefix_sharing/integrations/verl_mcore.py`
- `prefix-sharing/prefix_sharing/integrations/megatron_runtime.py`
- `prefix-sharing/prefix_sharing/integrations/verl_fsdp.py`
- `prefix-sharing/prefix_sharing/setup/patches/verl080_fsdp/attention.py`
- `prefix-sharing/prefix_sharing/setup/patches/verl080_fsdp/forward_step.py`

热点：

- `attention_mask.nonzero()` 与 `.detach().cpu().tolist()` 可能触发 device 到 CPU 同步。
- Megatron `prefix_attention()` 内存在直接 `print()`。
- RoPE path 中存在 position tensor max/item、index_select、必要时扩展 RoPE cache。
- FSDP `forward_step` patch 已接入 engine 的 `prepare_model_inputs/outputs`，no-sharing 路径会回到原 engine-like 流程，性能研究需要区分 enabled-no-sharing 和 real-sharing 两类 batch。
- FSDP dense Q/K/V pack 后又 scatter 回 dense output，会削弱显存收益。
- HF attention patch 的 QKV layout 转换 `[B,H,L,D] -> [B,L,H,D]` 可能影响 contiguous 和 FA 输入整理。

#### Backends

涉及代码：

- `prefix-sharing/prefix_sharing/backends/torch_ref.py`
- `prefix-sharing/prefix_sharing/backends/block_causal_mask.py`
- `prefix-sharing/prefix_sharing/backends/flash_atten_gpu.py`
- `prefix-sharing/prefix_sharing/backends/flash_atten_npu.py`

热点：

- TorchRef `build_kv()` 逐 row split / load / cat / final cat。
- TorchRef `attention()` 逐 row 循环，GQA 通过 repeat_interleave 扩展 KV head，可能增加临时内存；该路径仅用于调试/reference。
- GPU FA 使用 varlen Q/KV 输入，attention 主体不应退回 TorchRef。
- NPU FA 使用 BSH 路线，会 split THD、pad/stack Q/K/V、构造 per-sample 4D mask，再调用 `npu_fusion_attention`；这是真实 NPU attention 路线的显存和速度观测重点。
- block causal mask 使用 dense `[total_q, total_kv]` bool mask 和 fp16 bias，显存增长快，但只应作为 TorchRef/debug 路线风险。

## 2. 方案设计

### 2.1 设计原则

1. 精度一致性优先于性能收益。
2. 保留链式复用语义，不能退化为只复用 original/raw provider。
3. 优先优化 core，因为 core 同时作用于 Megatron 和 FSDP。
4. 确定性收益优先进入第一阶段；收益不确定或依赖真实设备验证的方案进入后续阶段。
5. 观测能力默认关闭，开启后允许为了计时准确引入同步开销。

### 2.2 P0 方案

#### P0-1：Core no-sharing exact prefilter / fast path

方案：

- 按 `min_prefix_len` 提取每条样本的前缀签名。
- 如果没有任何重复签名，则不可能存在满足阈值的共享前缀，直接返回 no-sharing plan，跳过完整 trie。
- 对 no-sharing plan 走轻量构造路径，避免构造无收益的复杂中间状态。

收益：

- 同时覆盖 Megatron 与 FSDP。
- 针对 PoC 中最差的 no-sharing case，`planner_ms=25.939` 且收益为 0。

风险与约束：

- prefilter 只能做“确定无共享”的早停，不能误判有共享的 batch。
- 必须覆盖 min prefix、空序列、短序列、chain reuse 等测试。

#### P0-2：Core plan 表示与 Python 对象开销验证

方案：

- 在 `PrefixSharingPlanner.plan()` 和 `plan_from_detection()` 周围增加轻量 profiling。
- 记录 plan 构造阶段 Python 对象数量、list 字段规模、tracemalloc peak、`detector_ms`、`plan_construct_ms`。
- 对比当前 list/dataclass 表示与紧凑表示 PoC，例如 array/list-of-int 压缩、按 row 只保存必要 ranges、延迟派生部分字段。

收益：

- 如果 plan 构造在真实 workload 中占 prepare 显著比例，该优化会同时作用于 Megatron 和 FSDP。
- 与 no-sharing fast path 同属 core 优化，复用面最大。

风险与约束：

- 当前 list/dataclass 低效仍是猜想，必须先实验验证。
- 紧凑化不能牺牲 plan 的语义可读性和测试覆盖。

#### P0-3：TorchRef `build_kv()` 预分配 expanded KV buffer

方案：

- 保留旧实现作为 test reference。
- 新实现预分配 expanded key/value buffer。
- provider 按 row 顺序写入自身 expanded slot。
- reuser 从 provider expanded slot 复制 prefix，再写 suffix。
- reuser 的 expanded KV 仍可作为后续样本 provider，保留 chain reuse。

收益：

- 直接针对 NPU 实测接近 40% 的热点。
- 减少 per-row `torch.cat()`、final `torch.cat()`、临时 tensor 和 allocator 压力。

风险与约束：

- 必须保证 provider-before-reuser 顺序语义。
- 必须保证 KV 不 `detach()`，provider prefix 梯度路径保留。
- 必须验证 chain reuse 输出和梯度与 reference 一致。

#### P0-4：FA attention 输入整理与 NPU mask/pad-stack 观测/保护

方案：

- 正式 attention 路线只考虑 GPU/NPU FA backend，不把 TorchRef attention 作为性能优化目标。
- GPU FA 路线重点记录 varlen 输入整理成本：`cu_seqlens_q/cu_seqlens_kv`、pad/unpad、repad output。
- NPU FA 路线重点记录 THD split、BSH pad/stack、per-sample 4D mask `(batch_size,1,max_q,max_kv)` 的耗时和 HBM。
- 增加 attention memory guard 和 perf recorder 字段，记录 `backend,total_q,total_kv,max_q,max_kv,mask_bytes,padded_q_bytes,padded_kv_bytes`。

收益：

- 明确正式 FA 路线的 HBM 峰值来源。
- 避免把 TorchRef/debug mask 问题误判为业务主路径问题。
- 对 NPU FA 来说，per-sample mask 与 BSH pad/stack 可能是 attention 侧主要显存开销。

风险与约束：

- NPU FA mask 语义涉及 provider causal、reuser prefix all-visible、suffix causal、padding invisible，必须用 correctness test 对齐 baseline。
- GPU FA varlen 路线要确认不同 Q/KV 长度与 prefix causal 语义完全一致。

#### P0-5：结构化性能观测常驻但默认关闭

方案：

- 补齐 phase latency：prepare、build_kv、attention、restore、forward、backward、micro-batch、update、mini-batch。
- 支持 JSONL/CSV 落盘。
- 支持可选 device synchronize。

收益：

- 后续优化决策有数据闭环。
- 避免只依赖单次人工日志判断。

#### P1-0：热路径日志、audit、diagnostic dump 分级

方案：

- 将 attention/context 热路径直接 `print()` 改为 logger，并受环境变量或 logger level 控制。
- audit / layer stats 默认关闭，仅 profiling/debug 模式输出。
- diagnostic dump 只在显式开关启用时执行。

优先级说明：

- 最终交付一定需要关闭或分级使用 dump/print。
- 但它不是当前性能收益不及预期的首要嫌疑，实验验证可以做，优化优先级低于 core detect/planner、`build_kv()`、FA attention 和 device sync。

### 2.3 P1 方案

1. 避免 integration 中 device-to-CPU token extraction。
   - 在 data pipeline 中携带 CPU token metadata。
   - planner 直接读取 CPU 侧 token list，不访问 device `input_ids`。
   - 如果 1.4 实验证明 `attention_mask.nonzero()` / `.detach().cpu().tolist()` 触发显著同步，该项提升为 P0。

2. FSDP packed/jagged-native output 和 restore。
   - 减少 scatter 回 dense 的次数。
   - restore 尽量在 packed/jagged 形态完成。

3. Core plan 紧凑化。
   - 将部分 Python list 转为紧凑 array/tensor-like 表示。
   - 降低 Python 对象和内存占用。

4. FA backend 输入整理优化。
   - GPU FA 重点优化 varlen prepare/repad。
   - NPU FA 重点优化 THD split、BSH pad/stack、per-sample mask 构造。
   - TorchRef attention 只保留 correctness/debug 用途，不作为正式性能路线。

5. RoPE indexed frequency 缓存。
   - 按 layout/layer 缓存 position-indexed RoPE 结果。
   - 必须严格处理 device、dtype、seq length、TP padding。

### 2.4 P2 方案

1. Prefix store key 编码优化。
2. Cross micro-batch plan/cache。
3. 多 stream async prefetch / overlap。
4. FSDP runtime wrapper 对象缓存。
5. import-time auto-detect 日志降噪。

## 3. 测试验证

### 3.1 已完成的本地 PoC

已完成以下本地 PoC：

- Core detector / planner / trim CPU 计时。
- TorchRef backend `build_kv()` / debug attention CPU 计时。
- TorchRef block causal mask 显存规模估算。
- FSDP dense prepare / pack / scatter CPU 计时。

这些 PoC 主要用于识别 Python 和临时对象开销，不作为 GPU/NPU 性能结论。

### 3.2 P0-1 Core Prefilter 测试

正确性测试：

- no-sharing batch：返回 no-sharing plan，且不进入完整 trie。
- one-provider batch：不得误判为 no-sharing。
- chain reuse batch：不得破坏链式复用。
- 短序列和空序列：按 `min_prefix_len` 正确处理。
- prefix 长度刚好等于 `min_prefix_len`：必须识别。

性能测试：

- 对比 prefilter 前后 no-sharing case 的 planner latency。
- 记录 batch size、sequence length、unique prefix 数量。

通过标准：

- 所有 existing plan correctness test 通过。
- no-sharing prepare/planner latency 明显下降。
- 有 sharing 的 batch 输出 plan 与旧实现一致。

### 3.3 P0-2 Core Plan 表示与 Python 对象开销测试

正确性测试：

- plan 紧凑化 PoC 生成的 provider/reuser、prefix_len、keep ranges、restore spec 与当前实现完全一致。
- no-sharing、one-provider、chain reuse、短序列、空序列全部覆盖。
- Megatron 与 FSDP integration 调用 `PrefixSharingPlanner(config).plan(sequences)` 的行为不变。

性能测试：

- 记录 `detector_ms`、`plan_construct_ms`、`planner_total_ms`。
- 记录 plan 内部 list 字段长度总和、Python object 估算数量、tracemalloc peak。
- 在真实 GPU/NPU workload 中记录 prepare 阶段 p50 / p90 / p99。
- 对比当前 list/dataclass 表示和紧凑表示 PoC。

通过标准：

- 如果 `plan_construct_ms` 或 Python object 规模显著，并且紧凑表示能稳定降低 prepare latency，则 plan 表示优化进入 P0。
- 如果 detector 才是主因，优先做 no-sharing prefilter 和 detector 优化，plan 表示优化降为 P1。

### 3.4 P0-3 `build_kv()` 预分配测试

正确性测试：

- provider / reuser / chain reuser 场景输出与旧 reference 一致。
- prefix KV 不 `detach()`，provider prefix 梯度路径保留。
- TP padding 下 valid token 与 padding token 行为一致。
- 多 layer、多 tp rank key 隔离行为一致。

性能测试：

- 在 CPU、GPU/NPU 上分别记录：
  - `build_kv_ms`
  - allocator 临时分配次数或近似指标
  - HBM peak
  - expanded KV token 数
  - provider/reuser 数

通过标准：

- 输出、logprob、loss、grad 与旧 reference 一致。
- NPU/GPU 上 `build_kv()` latency 稳定下降。

### 3.5 P0-4 FA Attention 输入整理与 NPU Mask/Pad-Stack 测试

正确性测试：

- GPU FA varlen 输出与 TorchRef/debug baseline 对齐。
- NPU FA BSH mask 输出与 TorchRef/debug baseline 对齐。
- provider causal、reuser prefix all-visible、suffix causal、padding invisible 全部覆盖。
- GQA 场景下 Q heads 与 KV heads 不同的输出一致性覆盖。

性能测试：

- 记录 `backend,total_q,total_kv,max_q,max_kv,mask_bytes,padded_q_bytes,padded_kv_bytes,call_count`。
- GPU FA 记录 varlen prepare、kernel、repad output 三段耗时。
- NPU FA 记录 split、pad/stack、mask build、kernel、unpack 五段耗时。
- 结合 memory profiler 记录 HBM peak。

通过标准：

- 如果 NPU FA mask/pad-stack 在主路径出现且 HBM 或 latency 占比显著，应进入 FA 输入整理优化。
- GPU/NPU FA 输出、logprob、loss、grad 必须与 reference 一致。

### 3.6 P0-5 性能观测测试

功能测试：

- profiling disabled 时 no-op。
- profiling enabled 时 JSONL/CSV 字段完整。
- sync 开关能控制 device synchronize。
- 每个 rank 独立落盘，不互相覆盖。

字段建议：

- rank、tp/pp/cp 信息
- step、mini-batch、micro-batch
- phase、duration_ms
- valid/padded/expanded token 数
- provider/reuser 数
- backend、implementation version
- FA backend、mask bytes、padded Q/KV bytes

### 3.7 P1 日志 Gate / Diagnostic Dump 测试

正确性测试：

- 默认模式下不输出热路径 debug print。
- debug/profiling 开启时能输出必要 audit。
- `PREFIX_SHARING_DIAG_DUMP` 开启时 dump 文件完整，关闭时不落盘。

性能测试：

- 在高 layer 数、多个 micro-batch 下对比默认日志关闭、profiling、diagnostic 三种模式的 stdout 量、dump 文件大小和阶段耗时。

通过标准：

- 最终交付默认训练路径无直接 `print()` 刷屏。
- profiling/debug 模式保留定位能力。
- 如果 logging/dump 开销明显，但默认模式已关闭，则不提升为核心性能 P0。

### 3.8 结论待定项验证计划

#### `.cpu().tolist()` 是否造成显著 device sync

验证方案：

- 在 NPU/GPU 上分别记录 prepare 阶段 `extract_sequences_ms`。
- 打开/关闭 prefix-sharing，对比 actor micro-batch prepare latency。
- `PREFIX_SHARING_PROFILE_SYNC=1` 时在阶段前后 synchronize。
- 对比 CPU metadata path，不访问 device `input_ids`。

进入开发标准：

- 如果 device extraction 占 micro-batch prepare 超过 5%，进入 P1 实现 CPU metadata path。

#### FA attention 输入整理是否成为新瓶颈

验证方案：

- GPU FA：拆分记录 `_prepare_flash_inputs()`、FA kernel、`_repad_output()`。
- NPU FA：拆分记录 THD split、BSH pad/stack、4D mask build、`npu_fusion_attention`、THD unpack。
- 对比不同 batch size、prefix length、TP padding、GQA heads 下的 HBM peak 和 latency。

进入开发标准：

- 如果 FA 输入整理或 NPU mask/pad-stack 占 attention 侧耗时超过 10%，或 HBM 峰值明显抵消 prefix-sharing 收益，应进入 P1 优化。

#### Dense scatter 对 FSDP 显存收益的影响

验证方案：

- 在 FSDP attention runtime 记录 packed tokens、dense tokens、scatter bytes。
- 对比 dense scatter 版本和 packed/jagged 透传 PoC 的 HBM peak。

进入开发标准：

- 如果 dense scatter 使 HBM peak 接近 baseline，应推进 packed/jagged-native restore。

#### 多 stream async prefetch 是否值得做

验证方案：

- 完成 P0 `build_kv()` prealloc 后再做 async prefetch PoC。
- 分别在 GPU/NPU 上验证 store/load overlap、stream synchronize、autograd 依赖。

进入开发标准：

- 端到端 forward latency 稳定下降，且不引入精度或梯度风险。

## 4. 开发计划

### 4.1 第一阶段：确定性 P0 优化

目标：

- 不改变 prefix-sharing 外部语义。
- 保留链式复用。
- 先处理确定性高、跨 pipeline 复用性强的性能问题。

任务：

1. 补齐性能观测，尤其是 CPU detect/planner、device sync、`build_kv()`、GPU/NPU FA attention 分段计时。
2. 实现 core no-sharing exact prefilter / fast path。
3. 验证 plan list/dataclass 表示是否造成显著 CPU overhead；若确认，实施紧凑化。
4. 实现 TorchRef `build_kv()` 预分配 expanded KV buffer。
5. 增加 GPU/NPU FA 输入整理、NPU mask/pad-stack memory guard 和调用统计。
6. 验证 `attention_mask.nonzero()` / `.detach().cpu().tolist()` 是否触发显著 device-CPU sync；若确认，推进 CPU metadata path。
7. 跑 unit / integrated / system 回归，以及目标 NPU/GPU profile。

建议提交拆分：

- `[feat] 增加prefix-sharing性能观测`
- `[feat] 优化prefix-sharing无共享批次规划`
- `[feat] 优化PrefixSharingPlan构造开销`
- `[feat] 优化TorchRef构建expanded KV`
- `[feat] 增加FA输入整理显存观测与保护`
- `[fix] 收敛prefix-sharing热路径日志`

### 4.2 第二阶段：设备实测驱动优化

目标：

- 根据第一阶段 profiler 数据处理剩余主瓶颈。

候选任务：

1. FSDP packed/jagged-native output 和 restore。
2. GPU/NPU FA 输入整理优化，重点关注 NPU BSH pad/stack 和 per-sample mask。
3. RoPE indexed frequency cache。
4. 日志 gate / diagnostic dump 分级收敛。

进入条件：

- 第一阶段 profile 表明对应模块仍占主要耗时或显存。

### 4.3 第三阶段：长期优化

候选任务：

1. Prefix store key 编码优化。
2. Cross micro-batch sharing / plan cache。
3. 多 stream async prefetch / overlap。
4. Runtime wrapper cache。

进入条件：

- 业务主路径已稳定，且 profiler 证明这些优化能带来端到端收益。

## 5. 当前结论

### 5.1 总体判断

当前性能收益不及预期，不太可能由单一问题造成。更可能是多个因素叠加：

- 无收益 batch 仍承担完整 detector/planner 成本。
- TorchRef `build_kv()` 的小块 cat / 临时分配在 NPU 上被放大。
- 热路径日志 I/O 和 audit print 影响训练稳定性。
- NPU FA mask/pad-stack 或 FSDP dense scatter 带来显存写放大。

### 5.2 全局优先级

P0：

1. Core no-sharing exact prefilter / fast path。
2. Core plan list/dataclass 表示开销验证；若确认显著则紧凑化。
3. TorchRef `build_kv()` 预分配 expanded KV buffer。
4. GPU/NPU FA attention 分段实验与输入整理 / NPU mask/pad-stack 显存 guard。
5. `attention_mask.nonzero()` / `.detach().cpu().tolist()` device sync 实验；若确认显著则 CPU metadata path。
6. 结构化性能观测常驻但默认关闭。

P1：

1. FSDP packed/jagged-native output 和 restore。
2. GPU/NPU FA 输入整理进一步优化。
3. RoPE indexed frequency 缓存。
4. 热路径日志、audit、diagnostic dump 分级。

P2：

1. Prefix store key 编码优化。
2. Cross micro-batch plan/cache。
3. 多 stream async prefetch / overlap。
4. FSDP runtime wrapper 对象缓存。
5. import-time auto-detect 日志降噪。

### 5.3 可立即推进的结论

- Core no-sharing prefilter 是当前最稳的通用优化点；prefix detect/planner 是 Megatron 与 FSDP 共享 core，一次优化两条路径同时受益。
- plan list/dataclass 表示开销需要尽快实验验证；如果确认是 prepare 主因，应提升为 P0 实现。
- `build_kv()` prealloc 是当前最应对齐 NPU 实测瓶颈的 backend 优化点。
- `attention_mask.nonzero()` / `.detach().cpu().tolist()` 是否触发 device-CPU sync 需要实测；如果确认，CPU metadata path 应进入 P0/P1。
- FA 和 attention 相关性能实验必须作为 P0 重点执行，尤其是 GPU FA varlen prepare/repad 与 NPU FA BSH pad/stack/mask。
- 性能观测是后续所有优化的基础设施。
- dump/print 最终必须关闭或分级使用，但当前优化优先级低于 core、`build_kv()`、device sync 和 FA attention。
- attention 主体优化只考虑 GPU/NPU FA；TorchRef attention 仅保留调试/reference。
- NPU FA 的 mask/pad-stack 需要作为正式显存观测项；如果占比高，应作为显存 P0/P1 处理。

## 6. 遗留问题

### 6.1 真实设备 profile 待补

待补内容：

- NPU/GPU 上 prepare、build_kv、attention、restore、forward/backward/update 的阶段耗时。
- HBM peak、allocator 临时分配、FA 输入整理与 NPU mask/pad-stack 实际调用频次。
- TP/SP/PP 组合下的 token 数、expanded KV 数、padding token 数。

### 6.2 GPU/NPU FlashAttention 路线

当前状态：

- 正式 attention 性能路线只考虑 GPU/NPU FA。
- TorchRef attention 只用于调试/reference。
- 当前 GPU FA backend 使用 varlen Q/KV；NPU FA backend 使用 BSH pad/stack + per-sample 4D mask。

后续问题：

- GPU FA varlen 路线是否能长期稳定表达 prefix-sharing 的 expanded KV 与 suffix causal 语义。
- NPU FA BSH 路线的 pad/stack 和 mask build 是否会成为新的速度/显存瓶颈。
- 是否需要更贴近 NPU varlen 或 block-sparse mask 的专用接口。

### 6.3 FSDP Packed/Jagged 贯穿能力

当前状态：

- FSDP 路径仍可能 scatter 回 dense output。

后续问题：

- 下游 logprob / entropy / loss 是否能消费 packed/jagged 形态。
- restore 能否在 packed/jagged 形态完成。
- 与 verl FSDP 主流程的兼容边界。

### 6.4 CPU Token Metadata Path

当前状态：

- 当前 integration 通过 device tensor 提取 token 序列。

后续问题：

- verl batch 中是否已有 CPU 侧 input id metadata 可复用。
- metadata 生命周期如何与 micro-batch trim 对齐。
- 如何保证与 device tensor 内容一致。

### 6.5 Async Prefetch / 多 Stream

当前状态：

- PrefixTrain_dev 的 async/prefetch 思路可能有价值，但当前收益未验证。

后续问题：

- GPU/NPU stream 语义差异。
- autograd 依赖和同步点是否安全。
- 在 `build_kv()` prealloc 后是否仍有足够 overlap 空间。

### 6.6 Cross Micro-Batch Sharing

当前状态：

- 本轮不处理 inter micro-batch sharing。

后续问题：

- store 生命周期需要跨 micro-batch。
- key 需要包含 step / micro-batch / layer / rank / prefix identity。
- 梯度语义、activation checkpointing、PP stage 生命周期都需要重新设计。
