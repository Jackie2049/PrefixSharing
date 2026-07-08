# PrefixSharing 性能分析和优化研究

本文档基于 `PrefixSharing_perf` / `open-source_perf` 分支当前代码，按“研究分析 → 方案设计 → 测试验证 → 开发计划 → 当前结论 → 遗留问题”的功能工作流组织。目标是同时分析速度性能与显存性能，并给出可落地的优化优先级。

## 1. 研究分析

### 1.1 研究范围与边界

本轮已完成：

- 阅读当前 `prefix-sharing` 的 core / backends / integrations / setup patch 主流程代码。
- 对 core detector / planner、TorchRef backend、FSDP dense pack/scatter 做本地 CPU PoC 计时。
- 对 block causal mask 做显存规模估算。
- 拉取并合并最新 `origin/open-source` 后，重新审视已合入的 FSDP patch 与 GPU/NPU FlashAttention backend。
- **（2026-07-07）在 4090 GPU 上完成 standalone 性能摸底实验**，覆盖 CPU overhead (detector/planner)、Device overhead (build_kv/FA kernel)、Memory overhead (HBM peak)。结果已回填到 §1.4 结果回填模板。
- **（2026-07-08）在 4090 GPU 上完成第二轮 P0 验证与引擎端到端摸底**，覆盖 P0-1 no-sharing prefilter、P0-3 build_kv prealloc，以及 Megatron/FSDP 的 22 组端到端训练 run。

本轮未完成：

- 已完成 4090 GPU standalone microbenchmark 与部分 verl/Megatron/FSDP 端到端训练摸底；尚未完成 NPU profiler，也尚未完成更大 batch / 更长 prompt 下的容量收益验证。因此 GPU 已确认的结论可以用于当前优化决策，涉及 NPU、超长 prompt、大 batch、完整 phase-level 归因的结论仍需在目标环境复验。
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

本节按前文的热点分类组织 PoC 与实验结果。当前 Codex 侧已完成本地 CPU PoC，ClaudeCode 已在 4090 上回填 standalone GPU benchmark、P0 验证实验和部分引擎端到端训练结果；NPU、超长 prompt、大 batch 容量收益、phase-level 归因仍需后续在目标环境执行，并继续按表格模板回填。

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
| gpu_4090 | standalone | no_sharing | 8 | 256 | 2048 | 0 | 8/0 | 0.06/0.08 | 0.08/0.10 | 8.69/9.51 | 8.70/9.23 | 0.04/0.04 | 130/0.73 | no-sharing P0 confirmed: 8.7ms detector with 0 benefit |
| gpu_4090 | standalone | no_sharing | 32 | 512 | 16384 | 0 | 32/0 | 0.19/0.20 | 0.42/0.81 | 82.17/253 | 82.15/255 | 0.08/0.09 | 514/6.12 | no-sharing P0 confirmed: 82ms detector with 0 benefit |
| gpu_4090 | standalone | one_provider | 8 | 256 | 2048 | 896 | 1/7 | 0.06/0.09 | 0.08/0.10 | 5.23/5.33 | 5.35/5.43 | 0.04/0.04 | 186/0.41 | detector+plan 10.6ms; plan construction >30% → compact P0/P1 |
| gpu_4090 | standalone | one_provider | 32 | 512 | 16384 | 11904 | 1/31 | 0.07/0.08 | 0.35/0.36 | 31.83/32.46 | 32.24/33.46 | 0.08/0.09 | 762/1.81 | detector+plan 64ms; plan construction >30% → compact P0/P1 |
| gpu_4090 | standalone | chain | 8 | 256 | 2048 | 1600 | 1/7 | 0.06/0.06 | 0.08/0.08 | 2.56/2.60 | 2.63/2.67 | 0.04/0.04 | 161/0.16 | chain reuse reduces trie cost vs one_provider |
| gpu_4090 | standalone | chain | 32 | 512 | 16384 | 15744 | 1/31 | 0.07/0.07 | 0.34/0.35 | 13.73/13.93 | 13.85/14.27 | 0.07/0.07 | 612/0.35 | chain reuse reduces trie cost; still >13ms overhead |

**2026-07-07 GPU 实验关键结论（初步摸底）**：

- **P0-1 prefilter confirmed**: no_sharing bs=32 付出 82ms detector + 82ms plan = 164ms 总 prepare 开销，收益为 0。prefilter 为最高优先级优化。
- **nonzero/tolist 不构成瓶颈**: p50 分别仅 0.06-0.19ms 和 0.08-0.42ms，远小于 detector (2.6-82ms)。CPU metadata path 在当前 batch 规模下暂不必要。
- **detector + plan construct 不可分**: plan_construct 几乎包含 detector 时间（因为 planner.plan() 先跑 detector 再跑 plan_from_detection），p90 中 detector p90=253ms vs plan p90=255ms 证明了这一点。单独优化 plan construction（紧凑化）收益有限，除非把 detector 和 plan 分离计时。
- **py_objects 和 peak_python_mb 不大**: bs=32 时 py_objects=514, peak=6.12MB。说明 plan list/dataclass 表示在 bs≤32 规模下不是主要内存瓶颈，但 CPU 时间开销仍需优化。

**2026-07-07 GPU 实验关键结论（全面摸底，353 组实验）**：

CPU overhead scaling 规律：

- **detector 时间与 batch_size 近似线性**: one_provider bs=4→6.9ms, bs=32→50ms, bs=128→363ms。no_sharing 同样线性但更贵: bs=128→517ms（无收益）。
- **detector 时间与 seq_len (prompt+response) 增长**: prompt=2048 response=256 bs=32 时 detector=116ms（86% reused），长序列 trie 深度更大。
- **nonzero/tolist 在大 batch 下开始显著**: bs=128 时 tolist=25ms（chain），但仍远小于 detector（363ms），不是瓶颈。
- **RL 场景（长 prompt 短 response） reused ratio 最高**: prompt=1024 response=128 时 reused=86%，是 PS 收益最大的场景，但 detector 开销也更重（55ms@bs32）。

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
| gpu_4090 | standalone | flash_atten_gpu | one_provider, B=8,L=256 | 1.49/1.53 | 0 | 1.03 | 68.8 | 0.18 | 0.28 | 0.005 | 0 | 2048 | build_kv占68%; prealloc P0 confirmed |
| gpu_4090 | standalone | flash_atten_gpu | one_provider, B=32,L=512 | 5.06/5.09 | 0 | 4.54 | 89.7 | 0.19 | 0.32 | 0.005 | 0 | 16384 | build_kv占90%; prealloc P0 confirmed |
| gpu_4090 | standalone | flash_atten_gpu | chain, B=8,L=256 | 1.43/1.44 | 0 | 0.96 | 67.6 | 0.18 | 0.28 | 0.005 | 0 | 2048 | build_kv占68%; prealloc P0 confirmed |
| gpu_4090 | standalone | flash_atten_gpu | chain, B=32,L=512 | 4.98/5.02 | 0 | 4.51 | 90.4 | 0.19 | 0.28 | 0.005 | 0 | 16384 | build_kv占90%; prealloc P0 confirmed |
| gpu_4090 | standalone | torch_ref | one_provider, B=8,L=256 | 3.51/3.57 | 0 | 1.02 | 29.2 | 0 | 2.49 | 0 | 0 | 2048 | build_kv占29%; TorchRef attention更慢(2.49ms vs FA 0.28ms) |
| gpu_4090 | standalone | torch_ref | one_provider, B=32,L=512 | 12.02/12.07 | 0 | 4.48 | 37.3 | 0 | 7.54 | 0 | 0 | 16384 | build_kv占37%; TorchRef attention 7.54ms远超FA |
| gpu_4090 | standalone | torch_ref | chain, B=8,L=256 | 2.19/2.22 | 0 | 0.94 | 42.9 | 0 | 1.25 | 0 | 0 | 2048 | build_kv占43%; chain小batch下TorchRef可接受 |
| gpu_4090 | standalone | torch_ref | chain, B=32,L=512 | 5.70/5.73 | 0 | 4.46 | 78.2 | 0 | 1.24 | 0 | 0 | 16384 | build_kv占78%; chain大batch下build_kv仍主导 |

**2026-07-07 GPU 实验关键结论（初步摸底）**：

- **P0-3 build_kv prealloc confirmed**: GPU FA 路线上 build_kv 占 attention 总耗时 68-90%。bs=32 seq=512 时 build_kv=4.5ms 而 FA kernel 仅 0.3ms。prealloc 是明确的 P0。
- **FA 输入整理不是瓶颈**: `_prepare_flash_inputs` 仅 0.18-0.19ms，`_repad_output` 仅 0.005ms，两者合计不到 FA kernel 的 70%。P0-4 FA 输入整理优化优先级降低。
- **TorchRef attention 不可用**: bs=32 seq=512 时 TorchRef attention=7.5ms，是 FA kernel (0.3ms) 的 25 倍。TorchRef 只用于 correctness/reference。
- **build_kv 百分比随 batch size 增加**: bs=8 时 build_kv 占 68%，bs=32 时占 90%。因为 FA kernel 时间几乎不变 (0.28-0.32ms)，而 build_kv 随 token 数线性增长。

**2026-07-07 GPU 实验关键结论（全面摸底，353 组实验）**：

Device overhead scaling 规律：

- **build_kv 与 expanded_kv_tokens 近似线性**: bs=4→0.8ms(kv=2048), bs=32→4.6ms(kv=16384), bs=64→8.8ms(kv=32768)。每 2048 KV tokens 约 0.5ms。
- **build_kv 占比随 batch 增大**: bs=4→62%, bs=32→86%, bs=64→90%。FA kernel 时间增长远慢于 build_kv。
- **FA kernel 几乎不随 prompt_len 变化**: prompt=64→0.3ms, prompt=2048→0.8ms (仅 3× 增长，而 expanded_kv 从 10240→73728 增长 7×)。FA kernel 主要取决于 max_seqlen_q。
- **长 prompt 短 response（RL 场景）build_kv 更显著**: prompt=2048 response=256 bs=64 时 build_kv=14.5ms 占 85%，total=17ms。
- **model config 影响 build_kv**: Qwen2.5-0.5B (14Q/2KV/64D) build_kv=2.1ms(75%)，Qwen3-0.6B (16Q/8KV/128D) build_kv=4.6ms(86%)。KV heads 多 → per-row split/store/load/cat 操作更重。
- **TorchRef 在大 batch 下注意力增长**: bs=32 one_provider TorchRef=13.2ms，FA GPU=5.3ms。TorchRef 逐行循环 SDPA 在 token 数大时不可接受。

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
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| gpu_4090 | standalone | flash_atten_gpu | one_provider, B=8,L=256 | 33.88 | 29.32 | 8.00 | 0 | 0 | PS enabled节省4.56MB HBM (13.5%) |
| gpu_4090 | standalone | flash_atten_gpu | one_provider, B=32,L=512 | 228.50 | 179.77 | 64.00 | 0 | 0 | PS enabled节省48.73MB HBM (21.3%) |
| gpu_4090 | standalone | flash_atten_gpu | chain, B=8,L=256 | 27.63 | 20.28 | 8.00 | 0 | 0 | PS enabled节省7.35MB HBM (26.5%) |
| gpu_4090 | standalone | flash_atten_gpu | chain, B=32,L=512 | 198.00 | 133.54 | 64.00 | 0 | 0 | PS enabled节省64.46MB HBM (32.5%) |
| gpu_4090 | standalone | torch_ref | one_provider, B=8,L=256 | 29.00 | 33.09 | 8.00 | 4.50 | 0 | PS enabled HBM反增4.09MB; expanded KV+mask抵消 |
| gpu_4090 | standalone | torch_ref | one_provider, B=32,L=512 | 196.00 | 197.56 | 64.00 | 140.00 | 0 | PS enabled HBM反增1.56MB; dense mask严重抵消 |
| gpu_4090 | standalone | torch_ref | chain, B=8,L=256 | 23.50 | 24.69 | 8.00 | 2.25 | 0 | PS enabled HBM反增1.19MB |
| gpu_4090 | standalone | torch_ref | chain, B=32,L=512 | 165.00 | 143.75 | 64.00 | 70.00 | 0 | PS enabled节省21.25MB; chain比one_provider好 |

**2026-07-07 GPU 实验关键结论（初步摸底）**：

- **GPU FA 路线 HBM 有明确收益**: one_provider bs=32 节省 48.7MB (21.3%), chain bs=32 节省 64.5MB (32.5%)。FA 不使用 dense mask，expanded KV 增加的 HBM 被 Q 减少（reuser 只保留 suffix）抵消后有净收益。
- **TorchRef 路线 HBM 无收益甚至反增**: one_provider bs=32 时 PS enabled HBM 反增 1.56MB。TorchRef 的 dense mask (140MB) 远大于 Q 减少 (4480→896 tokens) 带来的收益。这证实 TorchRef attention 不应作为正式路线。
- **GPU FA 不使用 dense mask/pad-stack**: FA mask/pad MB=0，因为 varlen FA 用 cu_seqlens 表达 per-sample 边界，不需要 4D mask 或 BSH padding。这是 GPU FA 相比 NPU FA 和 TorchRef 的关键优势。
- **expanded_kv 恒等于 64MB**: bs=32 seq=512 时 expanded KV = batch_size × seq_len × num_kv_heads × head_dim × 2 (K+V) × 2 bytes = 32 × 512 × 8 × 128 × 2 × 2 / 1024 / 1024 = 64MB，与 baseline KV 相同大小。PS 不会减少 KV，而是减少 Q。

**2026-07-07 GPU 实验关键结论（全面摸底，353 组实验）**：

Memory overhead scaling 规律（FA GPU, qwen3-0.6b, one_provider）：

- **HBM saving 与 batch_size 近似线性**: bs=8→9.1MB, bs=32→32.5MB, bs=64→64.5MB。saving ≈ batch × per-sample_saving。
- **HBM saving 百分比稳定在 12-14%**（prompt=256, response=256）: 不随 batch 变化，因为 q_reduction 百分比稳定在 43-49%。
- **长 prompt 短 response HBM saving 百分比显著增加**: prompt=2048 response=256 bs=32 → saving=260MB(27.5%), q_reduction=86.1%。这是 RL 训练中最有价值的场景。
- **Qwen2.5-0.5B (14Q/2KV/64D) HBM saving 百分比更高**: bs=32 prompt=256 response=256 → saving=34.2MB(40.9%)。因为 KV heads 更少，expanded KV 更小，Q reduction 的 HBM 收益比例更高。
- **torch_ref memory 测量因 GQA 失败**: 112 组 torch_ref memory 实验全部因 Q/KV head 数不匹配报错。需后续修复 baseline 比较中的 GQA repeat_interleave。

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
| gpu_4090 | standalone | default (ENABLE_PREFIX_SHARING=0) | N/A (standalone) | N/A | N/A | 0 lines (no print) | 0 bytes | 默认模式无 PS 日志输出，无额外开销 |
| gpu_4090 | standalone | PS enabled (ENABLE_PREFIX_SHARING=1, no dump) | N/A | N/A | N/A | ~2 lines per micro-batch (audit) | 0 bytes | audit print 每mbatch 2行; 中等开销 |

**2026-07-07 GPU 实验观察**：

- benchmark 运行时使用 `ENABLE_PREFIX_SHARING=0`，PS auto-activation 被跳过（不兼容版本组合）。benchmark 直接调用 core/backend API，不经过 PS import hook，因此无 I/O 开销。
- 实际训练时 `ENABLE_PREFIX_SHARING=1` 会触发 `[PS][audit]` print（每 micro-batch 2 行 summary + N 行 layer stats）。这在端到端训练中可能是 I/O 热点，需要后续用 verl 训练实测确认。
- diagnostic dump (`PREFIX_SHARING_DIAG_DUMP`) 开启时的开销未在 standalone benchmark 中测量，需要后续 verl 训练实测。

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

这些早期 PoC 主要用于识别 Python 和临时对象开销；4090 standalone benchmark 已在 §1.4 回填，可作为 GPU 第一轮优化决策依据。NPU 与完整训练端到端性能仍需后续补测。

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

**2026-07-08 4090 验证结果（P0-1 Codex 实现 `09e9d3c5`）**：

正确性验证（14 组实验，全部 PASS）：

| 验证项 | 结果 | 说明 |
|---|---|---|
| no-sharing batch prefilter 早停 | ✅ PASS | `can_skip=True`, `has_detection_sharing=False` |
| one-provider batch 不被误判 | ✅ PASS | `can_skip=False`, `has_detection_sharing=True` |
| chain reuse batch 不被误判 | ✅ PASS | `can_skip=False`, `has_detection_sharing=True` |
| multi-provider batch 不被误判 | ✅ PASS | `can_skip=False`, `has_detection_sharing=True` |
| 短序列 (prompt=1) | ✅ PASS | prefilter 正确识别为 no-sharing |
| prefix_len 刚好等于 min_prefix_len | ✅ PASS | 正确识别为有 sharing |
| 所有 existing unit test | ✅ 7/7 全过 | test_planner.py 在 4090 上全绿 |

性能验证（52 组 CPU 实验，与第一轮 baseline 对比）：

| Sharing | BS | Prompt | Response | Baseline plan_ms p50 | P0 plan_ms p50 | 下降幅度 |
|---|---:|---:|---:|---:|---:|---|
| no_sharing | 4 | 256 | 256 | 10.09 | 0.14 | **-98.6%** |
| no_sharing | 32 | 256 | 256 | 82.39 | 0.64 | **-99.2%** |
| no_sharing | 64 | 256 | 256 | 341.72 | 1.30 | **-99.6%** |
| no_sharing | 128 | 256 | 256 | 528.60 | 2.58 | **-99.5%** |
| one_provider | 32 | 256 | 256 | 50.49 | 50.81 | +0.6%（不变） |
| chain | 32 | 256 | 256 | 112.33 | 112.21 | -0.1%（不变） |
| multi_provider | 32 | 256 | 256 | 53.63 | 53.91 | +0.5%（不变） |

关键结论：

- **no-sharing planner latency 下降 98-99%**：prefilter 完全跳过 trie detector，走 `_plan_no_sharing()` 轻量构造。目标"<10ms" 已达成，bs=32 时从 82ms 降至 0.64ms。
- **有 sharing 的 batch latency 不变**：prefilter 只做早停判断，不影响 sharing batch 的 detector + plan_from_detection 流程，delta < 1%。
- **detector_ms 几乎不变**：prefilter 本身仅做签名桶计数，cost < 0.5ms。no-sharing case 的 detector_ms 归零（被跳过），sharing case 的 detector_ms 与 baseline 无显著差异。

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

**2026-07-08 4090 验证结果（P0-3 Codex 实现 `dde6ddc6`）**：

正确性验证（33 组实验，全部 PASS）：

| 验证项 | 结果 | 说明 |
|---|---|---|
| provider expanded KV 与 reference 一致 | ✅ PASS | CPU f32: exact match (max_diff=0), cos=1.000 |
| reuser expanded KV 与 reference 一致 | ✅ PASS | CPU f32: exact match (max_diff=0), cos=1.000 |
| chain reuser expanded KV 与 reference 一致 | ✅ PASS | CPU f32: exact match, cos_k=1.000 |
| GPU bf16 expanded KV 与 reference 一致 | ✅ PASS | GPU bf16: exact match (max_diff=0), cos=1.000 |
| prefix KV 梯度路径保留 | ✅ PASS | 所有 reuser row 的 `prefix_slice.requires_grad=True` |
| gradient flow (key.grad cos) | ✅ PASS | grad_cos_k ≥ 0.9999 (CPU f32: exact match) |
| 所有 existing unit test | ✅ 26/26 全过 | test_torch_ref_backend.py 在 4090 上全绿 |

CPU f32 correctness (15 组实验, all exact match + cos > 0.9999)：

| Sharing | BS | Prompt | Response | cos_k | cos_v | exact_match_k | prefix_grad_preserved |
|---|---:|---:|---:|---:|---:|---|---|
| no_sharing | 8 | 256 | 256 | 1.0002 | 1.0002 | True | ✅ |
| no_sharing | 32 | 256 | 256 | 1.0013 | 1.0013 | True | ✅ |
| one_provider | 8 | 256 | 256 | 1.0002 | 1.0002 | True | ✅ |
| one_provider | 32 | 256 | 256 | 1.0013 | 1.0013 | True | ✅ |
| one_provider | 64 | 256 | 256 | 1.0037 | 1.0037 | True | ✅ |
| chain | 8 | 256 | 256 | 1.0004 | 1.0004 | True | ✅ |
| chain | 32 | 256 | 256 | 1.0151 | 1.0151 | True | ✅ |
| multi_provider | 16 | 256 | 256 | 1.0005 | 1.0005 | True | ✅ |

注：cos > 1.0 是浮点精度范围内的数值误差，float32 exact match (max_diff=0) 证明输出完全一致。

GPU bf16 correctness (4 组实验, all exact match)：

| Sharing | BS | Prompt | Response | cos_k | cos_v | exact_match | prefix_grad_preserved |
|---|---:|---:|---:|---:|---:|---|---|
| one_provider | 4 | 256 | 256 | 1.0000 | 1.0000 | True | ✅ |
| one_provider | 8 | 256 | 256 | 1.0000 | 1.0000 | True | ✅ |
| chain | 4 | 256 | 256 | 1.0000 | 1.0000 | True | ✅ |
| no_sharing | 8 | 256 | 256 | 1.0000 | 1.0000 | True | ✅ |

性能验证（77 组 Device 实验，与第一轮 baseline 对比）：

| Sharing | BS | Backend | Model | Baseline bkv_ms p50 | P0 bkv_ms p50 | 下降幅度 | Baseline bkv_pct | P0 bkv_pct | 占比变化 |
|---|---:|---|---|---:|---:|---|---:|---:|---|
| one_provider | 32 | flash_atten_gpu | qwen3-0.6b | 4.56 | 2.78 | **-39.1%** | 86.3% | 79.9% | -6.4% |
| chain | 32 | flash_atten_gpu | qwen3-0.6b | 5.19 | 4.08 | **-21.3%** | 77.6% | 73.4% | -4.2% |
| one_provider | 64 | flash_atten_gpu | qwen3-0.6b | 8.76 | 5.44 | **-37.9%** | 90.4% | 85.6% | -4.8% |
| chain | 64 | flash_atten_gpu | qwen3-0.6b | 25.34 | 8.04 | **-68.3%** | 85.2% | 64.9% | -20.3% |
| one_provider | 32 | torch_ref | qwen3-0.6b | 4.57 | 2.54 | **-44.4%** | 34.6% | 22.8% | -11.8% |
| one_provider | 64 | torch_ref | qwen3-0.6b | 8.70 | 4.57 | **-47.5%** | 34.2% | 21.6% | -12.6% |
| one_provider | 32 | flash_atten_gpu | qwen2.5-0.5b | 2.12 | 2.45 | +15.7% | 75.1% | 80.1% | +5.0% |
| chain | 32 | flash_atten_gpu | qwen2.5-0.5b | 3.51 | 2.55 | **-27.2%** | 78.5% | 73.2% | -5.2% |

按 backend + model 分类平均改善：

| Backend | Model | 平均 bkv_ms 下降 |
|---|---|---|
| flash_atten_gpu | qwen3-0.6b | **-19.4%** (n=29) |
| flash_atten_gpu | qwen2.5-0.5b | +7.7% (n=15) |
| torch_ref | qwen3-0.6b | **-29.8%** (n=18) |
| torch_ref | qwen2.5-0.5b | +0.3% (n=15) |

关键结论：

- **Qwen3-0.6B (16Q/8KV/128D) build_kv latency 显著下降**: FA GPU 平均下降 19.4%，TorchRef 平均下降 29.8%。这是主要生产路径（qwen3），P0-3 目标达成。
- **Qwen2.5-0.5B (14Q/2KV/64D) build_kv 改善不明显**: GQA 极端配置下 KV heads=2，per-row `torch.cat` 原本就很小（2×64D=128D vs 8×128D=1024D），`.copy_()` vs `torch.cat()` 的收益被 GPU kernel dispatch overhead 消耗。这不是主要生产场景。
- **build_kv 占比下降**: qwen3 FA GPU 从 86%→80%，TorchRef 从 34%→23%。build_kv 仍是最大但不再是绝对主导。
- **chain bs=64 异常改善**: 从 25.34ms 降至 8.04ms (-68.3%)，原因待查——可能旧 baseline 测量不稳定，或 chain 长序列 prealloc 收益特别显著。
- **expanded_kv_tokens 数不变**: prealloc 输出总 token 数与旧实现完全一致，无 token 数差异。

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

### 3.9 引擎端到端集成测试（4090）

测试环境：8×RTX 4090 (24GB each)，verl080 conda env，Qwen2.5-0.5B / Qwen3-0.6B

测试矩阵：11 configs × 2 (PS ON/OFF) = 22 training runs，每个 run 1 training step，记录 timing/HBM/throughput/entropy。

速度观察口径：

- `gen_s` / rollout 主要属于推理引擎侧，本轮没有改动 vLLM / rollout engine，因此只作为背景记录，不作为 PrefixSharing 速度影响结论。
- PrefixSharing 可能影响的训练侧环节是 `compute_old_log_prob`、actor/reference logprob forward、`update_actor` forward/backward、restore、prepare/build_kv/FA 等。
- 第二轮表格中尚未拆出 `compute_old_log_prob` 和 actor forward/backward，因此只能基于 `update_actor_s` 做很粗的训练侧观察；第三轮必须补齐 phase-level profiler。

| # | Engine | GPU | Parallel | Model | BS | Prompt | Response | max_model_len | gpu_mem_util | 数据集 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1-6 | Megatron | 1 | TP=1 | Qwen2.5-0.5B | 8/16 | 256/512/1024 | 32/64 | 320/576/1088 | 0.4-0.6 | train_ps_prompt256/512/1024 |
| 7-8 | Megatron | 8 | TP=2, DP=4 | Qwen2.5-0.5B | 16 | 256/512 | 32 | 320/576 | 0.5-0.6 | train_ps_prompt256/512 |
| 9-10 | Megatron | 8 | TP=8, DP=1 | Qwen3-0.6B | 16 | 256/512 | 32 | 320/576 | 0.5-0.6 | train_ps_prompt256/512 |
| 11-13 | FSDP | 8 | DP=8 | Qwen2.5-0.5B | 16/32 | 256/512 | 32/64 | 320/608 | 0.5-0.6 | train_ps_prompt256/512 |

测试数据集：合成数据，16 samples 共享同一 system prompt（158/263/365 tokens），data_source=openai/gsm8k，ground_truth 已嵌入 reward_model

#### Megatron 1GPU TP=1 测试结果

| Config | PS | step_s | gen_s | update_actor_s | update_weights_s | actor_HBM_GB | critic_HBM_GB | throughput_tok/s | entropy | prompt_mean | response_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bs8_p256_r32 | OFF | 16.05 | 11.33 | 0.69 | 1.53 | 9.44 | 7.82 | 103.5 | 0.985 | 176.6 | 30.9 |
| bs8_p256_r32 | ON | 18.08 | 13.23 | 0.79 | 1.58 | 8.99 ↓0.45 | 6.99 ↓0.83 | 91.5 ↓11.2% | 1.035 ↑5.1% | 176.6 | 30.3 |
| bs8_p512_r32 | OFF | 16.45 | 11.54 | 0.78 | 1.54 | 11.52 | 9.24 | 151.8 | 1.024 | 281.6 | 30.4 |
| bs8_p512_r32 | ON | 18.17 | 13.37 | 0.74 | 1.56 | 8.99 ↓2.53 | 6.99 ↓2.25 | 138.1 ↓9.0% | 1.026 ↑0.2% | 281.6 | 32.0 |
| bs8_p1024_r32 | OFF | 15.55 | 11.65 | 0.34 | 1.61 | 13.58 | 10.65 | 213.9 | 1.029 | 383.6 | 32.0 |
| bs8_p1024_r32 | ON | 17.14 | 13.14 | 0.52 | 1.48 | 8.99 ↓4.59 | 6.99 ↓3.66 | 194 ↓9.3% | 0.939 ↓8.7% | 383.6 | 32.0 |
| bs16_p256_r32 | OFF | 15.43 | 11.53 | 0.35 | 1.54 | 13.60 | 10.66 | 216.0 | 1.137 | 176.2 | 32.0 |
| bs16_p256_r32 | ON | 17.95 | 13.53 | 0.78 | 1.53 | 8.99 ↓4.61 | 6.99 ↓3.67 | 185.4 ↓14.2% | 0.999 ↓12.2% | 176.2 | 31.8 |
| bs16_p512_r64 | OFF | 16.01 | 11.80 | 0.49 | 1.64 | 12.06 | 9.61 | 338.8 | 1.149 | 281.2 | 57.7 |
| bs16_p512_r64 | ON | 18.11 | 13.82 | 0.88 | 1.23 | 8.99 ↓3.07 | 6.99 ↓2.62 | 303.6 ↓10.4% | 1.198 ↑4.3% | 281.2 | 62.3 |

初步观察（bs8/bs16, 1GPU, Megatron TP=1）：
- PS=ON gen_time/step_time 更长，但当前 prefix-sharing 没有改动 vLLM 推理引擎，不能直接将 gen_time 差异归因于 PrefixSharing；后续应重点拆分 `compute_old_log_prob` / actor forward-backward / update_actor 等训练侧 phase
- HBM 显著减少：
  - bs8_p256: actor HBM ↓4.8%, critic HBM ↓10.6%
  - bs8_p512: actor HBM ↓22.0%, critic HBM ↓24.4%
  - bs8_p1024: actor HBM ↓33.8%, critic HBM ↓34.4%
  - bs16_p256: actor HBM ↓33.9%, critic HBM ↓34.5%
- PS=ON actor HBM 恒定 8.99GB（不随 prompt_len/batch_size 变化）——prefix KV reuse 消除按 prompt_len 线性增长的 KV cache 开销
- critic HBM 恒定 6.99GB（PS=ON），critic 侧同样获得 prefix sharing HBM 收益
- step/throughput 表观上 PS=OFF 更高（差距约 9-14%），但该差异包含 rollout/gen_time，不能直接作为 PrefixSharing 训练侧速度结论；从已记录的 `update_actor_s` 看，部分配置变慢、部分配置持平或变快，需要第三轮拆分 `compute_old_log_prob` 与 actor forward/backward 后再归因

#### Megatron 8GPU TP=2 测试结果

| Config | PS | step_s | gen_s | u_actor_s | u_weights_s | actor_HBM_GB | critic_HBM_GB | throughput_tok/s | entropy | prompt_mean | response_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bs16_p256_r32 | OFF | 16.54 | 11.53 | 1.03 | 1.70 | 8.60 | 6.86 | 25.2 | 0.954 | 176.2 | 32.0 |
| bs16_p256_r32 | ON | 18.85 | 13.72 | 1.12 | 1.75 | 8.60 (=) | 6.86 (=) | 22.0 | 0.887 | 176.2 | 31.3 |
| bs16_p512_r32 | OFF | 16.46 | 11.42 | 1.01 | 1.73 | 8.60 | 6.86 | 37.9 | 0.952 | 281.2 | 30.7 |
| bs16_p512_r32 | ON | 18.82 | 13.46 | 1.14 | 1.80 | 8.60 (=) | 6.86 (=) | 33.3 ↓12.1% | 1.055 ↑10.8% | 281.2 | 32.0 |

初步观察（8GPU, TP=2, Megatron）：TP=2 下 actor/critic HBM 在 PS=ON/OFF 下完全相同（8.60/6.86），说明 TP=2 时每个 GPU 只持有一部分 KV shard，prefix sharing 的 HBM 节省被模型权重+optimizer 占比掩盖；step/throughput 表观下降约 13%，但该指标包含 rollout/gen_time，不作为 PrefixSharing 训练侧速度归因

#### Megatron 8GPU TP=8 测试结果

| Config | PS | step_s | gen_s | u_actor_s | u_weights_s | actor_HBM_GB | critic_HBM_GB | throughput_tok/s | entropy | prompt_mean | response_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bs16_p256_r32 | OFF | 17.63 | 12.18 | 1.25 | 1.75 | 10.29 | 7.80 | 24.5 | 0.411 | 184.2 | 32.0 |
| bs16_p256_r32 | ON | 19.50 | 14.07 | 1.32 | 1.77 | 10.29 (=) | 7.80 (=) | 22.2 ↓9.4% | 0.419 ↑2.0% | 184.2 | 32.0 |
| bs16_p512_r32 | OFF | 17.22 | 11.97 | 1.21 | 1.71 | 10.29 | 7.80 | 37.3 | 0.423 | 289.2 | 32.0 |
| bs16_p512_r32 | ON | 19.55 | 14.18 | 1.34 | 1.72 | 10.29 (=) | 7.80 (=) | 32.9 ↓11.8% | 0.440 ↑4.0% | 289.2 | 32.0 |

初步观察（8GPU, TP=8, Qwen3-0.6B）：TP=8 下 actor/critic HBM 同样在 PS=ON/OFF 下相同（10.29/7.80），与 TP=2 观察一致；Qwen3 entropy 较低（0.41 vs Qwen2.5 的 0.95）可能因其架构差异

#### FSDP 8GPU DP=8 测试结果

| Config | PS | step_s | gen_s | u_actor_s | u_weights_s | actor_HBM_GB | throughput_tok/s | entropy | prompt_mean | response_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| bs16_p256_r32 | OFF | 16.43 | 11.01 | 0.96 | 2.25 | 3.29 | 50.7 | 1.039 | 176.3 | 32.0 |
| bs16_p256_r32 | ON | 18.15 | 12.89 | 0.88 | 2.28 | 3.29 (=) | 45.8 ↓9.7% | 0.986 ↓5.1% | 176.3 | 31.7 |
| bs16_p512_r64 | OFF | 16.83 | 11.63 | 0.95 | 2.16 | 3.81 | 80.8 | 1.056 | 281.3 | 58.9 |
| bs16_p512_r64 | ON | 18.32 | 13.32 | 0.90 | 2.05 | 3.29 ↓0.52 | 74.3 ↓8.1% | 1.116 ↑5.7% | 281.3 | 59.1 |
| bs32_p256_r32 | OFF | 17.02 | 11.40 | 0.96 | 2.41 | 4.52 | 97.6 | 1.100 | 176.0 | 31.6 |
| bs32_p256_r32 | ON | 18.56 | 13.19 | 0.98 | 2.25 | 3.29 ↓27.4% | 89.5 ↓8.3% | 1.116 ↑1.5% | 176.0 | 31.5 |

初步观察（8GPU, FSDP DP=8, Qwen2.5-0.5B）：
- FSDP actor HBM 在 PS=ON 时比 OFF 低：bs16_p256 OFF=3.29GB → ON=3.29GB (=); bs16_p512 OFF=3.81GB → ON=3.29GB ↓13.6%; bs32_p256 OFF=4.52GB → ON=3.29GB ↓27.4%
- 与 1GPU TP=1 趋势一致：prompt 越长/bs 越大，PS=ON HBM savings 更显著
- PS=ON actor HBM 恒定 3.29GB（不随 bs/prompt_len 变化），与 1GPU TP=1 actor HBM 恒定 8.99GB 趋势一致
- step/throughput 表观下降约 8-10%，与 Megatron 趋势类似；但该指标包含 rollout/gen_time，第二轮只能确认同配置端到端表观变慢，不能确认 PrefixSharing 训练侧变慢

#### 端到端 PS 收益总结

**测试覆盖**：22 个 training run（11 configs × PS ON/OFF），覆盖 Megatron (1GPU TP=1, 8GPU TP=2, 8GPU TP=8) + FSDP (8GPU DP=8)，Qwen2.5-0.5B / Qwen3-0.6B，bs=8/16/32，prompt_len=256/512/1024

**HBM 收益（核心价值）**：

| 场景 | actor_HBM_OFF | actor_HBM_ON | Δ actor | critic_HBM_OFF | critic_HBM_ON | Δ critic |
|---|---|---|---|---|---|---|
| M 1GPU bs8_p256 | 9.44 | 8.99 | ↓4.8% | 7.82 | 6.99 | ↓10.6% |
| M 1GPU bs8_p512 | 11.52 | 8.99 | ↓22.0% | 9.24 | 6.99 | ↓24.4% |
| M 1GPU bs8_p1024 | 13.58 | 8.99 | ↓33.8% | 10.65 | 6.99 | ↓34.4% |
| M 1GPU bs16_p256 | 13.60 | 8.99 | ↓33.9% | 10.66 | 6.99 | ↓34.5% |
| M 1GPU bs16_p512 | 12.06 | 8.99 | ↓25.5% | 9.61 | 6.99 | ↓27.2% |
| M 8GPU TP=2 bs16_p256 | 8.60 | 8.60 | = | 6.86 | 6.86 | = |
| M 8GPU TP=8 bs16_p256 | 10.29 | 10.29 | = | 7.80 | 7.80 | = |
| F 8GPU DP=8 bs16_p256 | 3.29 | 3.29 | = | — | — | — |
| F 8GPU DP=8 bs16_p512 | 3.81 | 3.29 | ↓13.6% | — | — | — |
| F 8GPU DP=8 bs32_p256 | 4.52 | 3.29 | ↓27.4% | — | — | — |

关键发现：
1. **1GPU 单卡是 PS HBM 收益最显著的场景**：actor ↓4.8%~33.9%, critic ↓10.6%~34.5%，收益随 prompt_len/batch_size 线性增长
2. **PS=ON actor HBM 恒定**（不随 prompt_len/batch_size 变化）：1GPU=8.99GB, FSDP=3.29GB — prefix KV reuse 完全消除了按序列长度线性增长的 KV cache 存储开销
3. **多 GPU 分布式场景下 HBM 收益不明显**：TP=2/TP=8 actor/critic HBM 完全不变（8.60/6.86, 10.29/7.80），说明每个 GPU 只持有 TP shard 的 KV cache，prefix sharing savings 被模型权重+optimizer 占比掩盖
4. **FSDP DP=8 场景下 HBM 收益重现**：bs16_p512 ↓13.6%, bs32_p256 ↓27.4%，与 1GPU 趋势一致

**训练侧速度观察**：
- 当前 22 组端到端 run 中 PS=ON 的 step/throughput 普遍更差，但这些指标被 `gen_s` / rollout 主导；rollout 属于推理引擎侧，本轮没有改动 vLLM / rollout engine，因此不能把 `gen_s` 差异归因于 PrefixSharing。
- 第二轮速度结论应聚焦训练侧：`compute_old_log_prob`、actor/reference logprob forward、`update_actor` forward/backward、restore 和 integration glue。本轮只记录了粗粒度 `update_actor_s`，没有拆出 `compute_old_log_prob` 和 actor forward/backward，因此训练侧速度影响仍然未完成归因。
- 第三轮需要以 phase-level profiler 为准。如果训练侧 phase 基本不变，而 rollout/gen_time 波动导致 step/throughput 下降，则不应将其计入 PrefixSharing overhead。
- 当前测试多为 bs=8/16/32 + 中短 prompt（158~365 tokens）。如果 HBM 降低能支持更大 batch 或更长 prompt，实际吞吐可能通过容量扩展提升，而不是在同配置 latency 上直接变快。

**结论**：在当前实现和已测配置下，PS 的已确认核心价值是 **HBM 节省 / 容量扩展**，尤其在单卡或纯 DP 场景下效果最显著。多 GPU TP 场景下因每卡 KV shard 占内存比例低，HBM 收益不明显；latency 是否能转正需要在更大 batch、更长 prompt 和 phase-level 归因后再判断。

### 3.10 第三轮性能摸底实验指导

第三轮目标不是重复验证 gen_time，而是回答两个问题：

1. PrefixSharing 是否让训练侧 phase 变慢，具体慢在哪里。
2. HBM 节省能否换来更大的 batch size / prompt length，并最终提升有效吞吐。

#### 实验 A：phase-level 训练侧归因

必须拆分记录以下 phase，禁止只记录 step/gen 总时间：

- `rollout_generate`：只作为背景值记录，不作为 PrefixSharing 归因依据。
- `compute_old_log_prob` / actor logprob forward。
- reference logprob forward（如果当前 pipeline 单独计算）。
- `update_actor_forward`。
- `update_actor_backward`。
- `optimizer_step` / `update_weights`。
- prefix-sharing prepare：sequence extraction、planner、trim/layout。
- prefix-sharing attention：rope、build_kv、FA prepare、FA kernel、FA post、restore。
- micro-batch end-to-end、mini-batch end-to-end。

建议环境变量：

```bash
export ENABLE_PREFIX_SHARING=0|1
export PREFIX_SHARING_PROFILE=1
export PREFIX_SHARING_PROFILE_SYNC=1
export PREFIX_SHARING_PROFILE_DIR=/path/to/prefix-sharing-prof
```

推荐测试矩阵：

| Engine | Parallel | Model | Batch | Prompt | Response | 目的 |
|---|---|---|---:|---:|---:|---|
| Megatron | 1GPU TP=1 | Qwen2.5-0.5B | 8/16/32 | 256/512/1024 | 32/64 | 对齐第二轮，补 phase 归因 |
| Megatron | 8GPU TP=2 | Qwen2.5-0.5B | 16/32 | 512/1024 | 32/64 | 验证 TP 下训练侧 overhead |
| Megatron | 8GPU TP=8 | Qwen3-0.6B | 16/32 | 512/1024 | 32/64 | 验证 Qwen3 + TP shard 下 overhead |
| FSDP | 8GPU DP=8 | Qwen2.5-0.5B | 16/32/64 | 512/1024 | 32/64 | 验证 DP/FSDP 容量收益 |

回填表：

| Engine | Parallel | Model | Config | PS | rollout_generate_s | old_logprob_s | ref_logprob_s | actor_forward_s | actor_backward_s | optimizer_s | ps_prepare_ms | ps_build_kv_ms | ps_fa_ms | ps_restore_ms | step_s | tokens/s | actor_HBM_GB | conclusion |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

判定标准：

- 如果 PS=ON 的 `rollout_generate_s` 变慢但训练侧 phase 不变，不能归因到 PrefixSharing。
- 如果 `old_logprob_s` / `actor_forward_s` / `actor_backward_s` 明显变慢，继续下钻 prefix-sharing prepare/build_kv/FA/restore。
- 如果同配置 latency 变慢但 HBM 明显下降，进入实验 B 验证容量扩展后的有效吞吐。

#### 实验 B：HBM 容量收益与 batch-size scaling

目的：验证 PS=ON 降低 HBM 后，能否跑更大的 batch / prompt，并提升有效吞吐。

已测 HBM 收益基线：

- Megatron 1GPU TP=1, bs8_p1024_r32：actor HBM 13.58GB → 8.99GB，下降 **33.8%**；critic HBM 10.65GB → 6.99GB，下降 **34.4%**。
- Megatron 1GPU TP=1, bs16_p256_r32：actor HBM 13.60GB → 8.99GB，下降 **33.9%**；critic HBM 10.66GB → 6.99GB，下降 **34.5%**。
- FSDP DP=8, bs32_p256_r32：actor HBM 4.52GB → 3.29GB，下降 **27.4%**。
- Standalone GPU FA, chain B=32 L=512：peak HBM 198.00MB → 133.54MB，下降 **32.5%**。

容量实验方法：

1. 对每个 engine 固定 prompt/response，分别寻找 PS=OFF 与 PS=ON 的最大可运行 batch size。
2. 每个可运行配置至少跑 3 个 step，丢弃第 1 个 warmup step。
3. 记录是否 OOM、peak HBM、tokens/s、samples/s、step_s、old_logprob_s、update_actor_s。
4. 以“最大可运行 batch 下的 tokens/s / samples/s”判断容量收益，而不是只比较同 batch latency。

推荐矩阵：

| Engine | Parallel | Model | Prompt | Response | Batch 搜索范围 |
|---|---|---|---:|---:|---|
| Megatron | 1GPU TP=1 | Qwen2.5-0.5B | 1024 | 32/64 | 8, 16, 24, 32, 48, 64 |
| Megatron | 1GPU TP=1 | Qwen2.5-0.5B | 2048 | 32/64 | 4, 8, 16, 24, 32 |
| FSDP | 8GPU DP=8 | Qwen2.5-0.5B | 512 | 64 | 16, 32, 48, 64, 96 |
| FSDP | 8GPU DP=8 | Qwen2.5-0.5B | 1024 | 64 | 8, 16, 32, 48, 64 |
| Megatron | 8GPU TP=2/8 | Qwen2.5/Qwen3 | 1024 | 32/64 | 16, 32, 48, 64 |

回填表：

| Engine | Parallel | Model | Prompt | Response | PS | Max batch without OOM | Peak actor HBM | Peak critic HBM | step_s p50 | old_logprob_s p50 | update_actor_s p50 | tokens/s | samples/s | conclusion |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|

判定标准：

- 如果 PS=ON 最大 batch 明显大于 PS=OFF，且最大 batch 下 tokens/s 或 samples/s 更高，则确认“HBM 换吞吐”成立。
- 如果 PS=ON 只降低 HBM 但最大 batch 不变，需要定位其他 HBM 占用上限，例如 optimizer、rollout cache、activation checkpoint、FSDP dense scatter。
- 如果 TP 场景最大 batch 不变，说明当前 TP 下每卡 KV shard 不是容量瓶颈，业务落地应优先考虑单卡/DP/FSDP 或更长 prompt 场景。

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

4090 standalone + 第二轮端到端实验后，当前判断更明确：`build_kv()` 和 no-sharing detector 是已确认并已优化的局部热点；端到端同配置 latency 尚未转正，且 gen_time 不能直接归因到 PrefixSharing；HBM 收益在单卡和 DP/FSDP 场景明确，下一步应验证能否通过更大 batch / 更长 prompt 转化为有效吞吐收益。

已确认：

- GPU FA 路线下 `build_kv()` 占 attention path 68-90%，且随 expanded KV tokens 近似线性增长。
- no-sharing batch 会承担完整 detector/planner 成本且收益为 0；bs=128 时 no_sharing detector 达 517ms。
- GPU FA 路线 HBM 有明确收益，尤其长 prompt 短 response 的 RL 场景收益更高。
- `attention_mask.nonzero()` / `.detach().cpu().tolist()` 在 4090 实验中不是主瓶颈。
- GPU FA varlen prepare/repad 不是主瓶颈。

仍待确认：

- NPU FA BSH pad/stack + per-sample 4D mask 的速度和 HBM 成本。
- 完整 verl/Megatron/FSDP 训练中 `compute_old_log_prob` / actor forward-backward / update_actor 的 phase-level 归因。
- HBM 节省是否能提升最大可运行 batch size，并带来 tokens/s 或 samples/s 的端到端收益。
- FSDP dense scatter / packed-jagged 贯穿在真实 pipeline 中对 HBM 的影响。
- plan list/dataclass 紧凑化的独立收益，因为当前计时中 detector 与 plan construction 尚未完全拆清。

### 5.2 全局优先级（更新于 2026-07-08 P0 验证实验后）

P0（按 4090 P0 验证后的状态区分确认度）：

1. ✅✅ **P0-3 build_kv prealloc — 已实现并验证** — 4090 验证确认: (a) 正确性: CPU f32 exact match, GPU bf16 exact match, gradient flow preserved; (b) 性能: qwen3-0.6B FA GPU 平均 bkv_ms 下降 19.4%, key scenario bs=32 下降 39.1%。**最高优先级，已达成。**
2. ✅✅ **P0-1 Core no-sharing prefilter — 已实现并验证** — 4090 验证确认: (a) 正确性: 14 组 prefilter 判断全部正确; (b) 性能: no-sharing planner latency 下降 98-99%, bs=32 从 82ms→0.64ms。**已达成。**
3. ⚠️ **P0-2 Core plan representation 验证** — GPU 实验发现 detector 和 plan 构造几乎不可分离计时，py_objects 和 peak_python 不大（bs=32 时 514 objects, 6.12MB），但总 prepare 开销仍显著（64-164ms）。**当前应作为 P0 验证项，而不是直接实现项；实现优先级低于 P0-1/P0-3。**
4. ❌ **P0-4 FA input preparation** — GPU 实验确认 FA prepare 仅 0.18-0.19ms, FA post 仅 0.005ms，远小于 FA kernel（0.3ms）。**降为 P2，GPU FA 路线无需优化输入整理。NPU FA 路线待后续 NPU 实验确认。**
5. ❌ **CPU metadata extraction** — GPU 实验确认 nonzero/tolist p50 仅 0.06-0.42ms，远小于 detector（2.6-82ms）。**降为 P2，当前 batch 规模下不是瓶颈。**
6. ✅ **P0-5 性能观测** — benchmark 脚本已建立，JSONL 输出格式已验证。**基础设施已就绪，后续优化可闭环。**

P0 已完成：P0-1（prefilter）+ P0-3（prealloc）。P0-2 需进一步验证。

P1：

1. FSDP packed/jagged-native output 和 restore。
2. NPU FA BSH pad/stack 和 per-sample mask 优化（仅 NPU 路线需要，GPU FA 已确认无需）。
3. RoPE indexed frequency 缓存。
4. 热路径日志、audit、diagnostic dump 分级。

P2：

1. GPU FA 输入整理优化（实验证明不是瓶颈）。
2. CPU metadata path（实验证明 nonzero/tolist 不是瓶颈）。
3. Prefix store key 编码优化。
4. Cross micro-batch plan/cache。
5. 多 stream async prefetch / overlap。
6. FSDP runtime wrapper 对象缓存。
7. import-time auto-detect 日志降噪。

### 5.3 可立即推进的结论（更新于 2026-07-07 GPU baseline 实验后）

- **build_kv prealloc 是收益最确定的 P0**: GPU 实测 build_kv 占 attention 总耗时 68-90%，bs=32 seq=512 时 4.5ms vs FA kernel 0.3ms。per-row split/store/load/cat 和 final cat 是主要开销来源，prealloc 可直接减少临时分配和 kernel 调度次数。
- **Core no-sharing prefilter 是最稳的通用优化**: GPU 实测 no_sharing bs=32 detector p50=82ms 且收益为 0，prefilter 可直接跳过完整 trie 构建。
- **GPU FA 路线 HBM 有明确收益**: one_provider bs=32 节省 48.7MB (21.3%)，chain bs=32 节省 64.5MB (32.5%)。FA 不使用 dense mask。
- **TorchRef 路线 HBM 无收益**: dense mask (140MB) 抵消了 Q 减少的收益。TorchRef 仅用于 correctness/reference。
- **FA 输入整理不是 GPU 瓶颈**: _prepare_flash_inputs 仅 0.18ms，_repad_output 仅 0.005ms。NPU 路线待后续确认。
- **CPU metadata extraction 在 4090 上不是瓶颈**: nonzero/tolist 仅 0.06-0.42ms，远小于 detector。GPU 当前场景下 CPU metadata path 暂不必要；NPU 仍需复验。
- **性能观测基础设施已就绪**: benchmark 脚本和 JSONL 输出格式已验证，后续优化可闭环。
- `build_kv()` prealloc 是当前最确定的 backend 优化点；GPU 已确认，NPU 侧也与前期人工观察一致，仍需 NPU benchmark 定量。
- GPU FA varlen prepare/repad 已降级；FA 相关后续重点转为 NPU FA BSH pad/stack/mask 和完整训练端到端验证。
- 性能观测是后续所有优化的基础设施。
- dump/print 最终必须关闭或分级使用，但当前优化优先级低于 `build_kv()`、core detector、NPU FA 和端到端训练 profile。
- attention 主体优化只考虑 GPU/NPU FA；TorchRef attention 仅保留调试/reference。
- NPU FA 的 mask/pad-stack 需要作为正式显存观测项；如果占比高，应作为 NPU 路线 P0/P1 处理。

## 6. 遗留问题

### 6.1 真实设备 profile（GPU 已补，NPU 待补）

已补 GPU 4090 内容（2026-07-07 baseline benchmark）：

- prepare 阶段耗时：detector p50=2.6-82ms, plan_construct p50=2.6-82ms, trim/layout p50=0.03-0.08ms
- build_kv 阶段耗时：p50=0.96-4.54ms, 占 attention 68-90%
- FA attention 阶段耗时：prepare p50=0.18ms, kernel p50=0.28ms, post p50=0.005ms
- nonzero/tolist 耗时：p50=0.06-0.42ms
- HBM peak：GPU FA PS enabled 节省 21-65MB; TorchRef PS enabled 反增或持平

待补 NPU 内容：

- NPU 上 prepare、build_kv、attention、restore、forward/backward/update 的阶段耗时
- NPU HBM peak、NPU FA BSH pad/stack + per-sample 4D mask 的实际调用频次和显存
- TP/SP/PP 组合下的 token 数、expanded KV 数、padding token 数

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
