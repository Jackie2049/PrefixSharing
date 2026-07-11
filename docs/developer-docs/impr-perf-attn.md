# KV-0冗余 Prefix Sharing：从 build_kv 物理展开到 Attention Mask 方案

## 1. 研究分析

### 1.1 问题定义

#### 1.1.1 现有方案的瓶颈

当前 PrefixSharing 采用 **物理 KV 展开（Physical KV Expansion，流派 C）** 实现前缀复用：

```
输入: provider_full (P_len), reuser_1_suffix (S1_len), reuser_2_suffix (S2_len)
KV:   [provider K_full, reuser_1 K_suffix, reuser_2 K_suffix]      # 形状: [1, T_expanded, H, D]
Q:    [reuser_1 Q_suffix, reuser_2 Q_suffix]                        # 形状: [1, T_Q, H, D]
```

**核心冗余**：`provider K_full`（P_len tokens）被复制到 KV 中 N 次（每个 reuser 一份），导致 KV 张量尺寸从 `P + S` 增长到 `P + N * S`。

- 即使共享 prefix 只计算一次，KV 在内存中仍然是重复的
- 随着 `N`（rollout number / group size）增大，KV 展开的成本线性增长
- 带宽消耗：expanded KV 的读写带宽是实际所需 KV 的 `(P + N*S) / (P + S)` 倍

#### 1.1.2 零冗余目标

将物理 KV 展开替换为 **attention mask 方案**：

```
输入: q, k, v 均为去重扁平布局 [1, T_flat, H, D]    # T_flat < P + N*S
mask: block-causal / block-sparse, 控制各 token 可见性
   provider tokens:   standard causal
   reuser tokens:     prefix列全部可见 + suffix 列 causal
   cross-reuser:      完全不可见
```

**收益**：
- KV 零冗余：相同的 prefix KV 在张量中只存一份
- Q 零冗余：Q 张量中只保留 reuser 的 suffix tokens，不包含 prefix
- 路径保持单次 forward pass，无额外 kernel launch
- 数学严格等价（通过 mask 而非 data duplication 控制可见性）

### 1.2 技术路线的核心区别

#### 1.2.1 两个流派对比

| 维度 | 流派 C: 物理展开 (当前) | 流派 D: 扁平打包 + Mask (目标) |
|---|---|---|
| KV 冗余 | 有 (prefix 扩展 N-1 份) | 无 |
| Q 冗余 | 无 (只保留 reuser suffix) | 无 (只保留 reuser suffix) |
| Attention kernel | `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv)` | 需支持显式 mask 的 attention (`flash_attn_with_attention_mask`) 或 block-sparse kernel |
| 元数据管理 | `cu_seqlens` (cumsum 值) | `attention_mask` / `attention_bias` 矩阵 |
| 数学等价性 | 天然保证（数据驱动） | 天然保证（mask 驱动） |
| NPU 兼容性 | 差（varlen func 不是所有 NPU FA 都支持） | 好（NPU `npu_fusion_attention` 原生支持 mask） |
| 多级树 (depth>2) | 不支持（仅 1-prefix+N-suffix 扁平） | 支持 |

#### 1.2.2 与 miniFLUX / BTT 等 Mask 方案的关联

#### 1.2.3 FA varlen vs FA causal mask 性能差异

### 1.3 现有代码分析

#### 1.3.1 build_kv 路径全链路

当前 build_kv 在 `prefix_sharing/backends/torch_ref.py` 中实现，核心数据结构链路如下：

**Plan → Layout → Backend 数据流**：

```
PrefixSharingPlanner.plan()
  → PrefixSharingPlan  (frozen dataclass)
      ├── kept_lengths_q:      list[int]      # provider=full_len, reuser=suffix_len
      ├── expanded_lengths_kv: list[int]      # always original_len (full seq)
      ├── cu_seqlens_q:        list[int]      # cumsum(kept_lengths_q)
      ├── cu_seqlens_kv:       list[int]      # cumsum(expanded_lengths_kv)
      ├── q_position_offsets:  list[int]      # reuser=prefix_len, provider=0
      ├── provider_index:      list[int]      # which provider each reuser references
      ├── prefix_lens:         list[int]      # shared prefix length per row
      ├── suffix_lens:         list[int]      # unique suffix per reuser
      ├── reuse_specs:         list[PrefixReuseSpec]
      ├── prefix_last_restore: list[PrefixLastRestoreSpec]
      └── group_ids:           list[int]

  → PackedBatchLayout    (frozen dataclass)
      ├── valid_lengths:       list[int]      # valid (non-pad) tokens per row
      ├── padded_lengths:      list[int]      # padded total per row (TP alignment)
      ├── cu_seqlens:          list[int]      # cumsum(padded_lengths), len=batch+1
      ├── max_seqlen:          int
      ├── packed_position_ids: Tensor|None
      └── valid_token_mask:    Tensor|None
```

**build_kv 展开过程**（`torch_ref.py` `TorchReferenceBackend.build_kv()`）：

```
输入: key, value (float 张量, 形状 = [sum(padded_lengths), H, D])
      store (PrefixAttentionStore), plan (PrefixSharingPlan)
      packed_batch_layout (PackedBatchLayout)

对每个 batch row i:
  1. 从 layout.padded_lengths 获取 row_len = padded_lengths[i]
  2. 从 key/value 中切分 row: key_row = key[cu_start:cu_start+row_len]
  3. 如果 row 是 provider (not plan.is_reuser(i)):
     - 保留 full row: expanded_key_row = key_row[:valid_length]
     - expanded_value_row = value_row[:valid_length]
     - 存入 store: store.store(slot_id, key_tensor=expanded_key_row, value_tensor=expanded_value_row, prefix_len=valid_length)
  4. 如果 row 是 reuser (plan.is_reuser(i)):
     - 从 store 加载 provider: entry = store.load(provider_slot_id)
     - 复制 prefix: expanded_key_row[:prefix_len].copy_(entry.key_tensor[:prefix_len])
     - 复制 suffix: expanded_key_row[prefix_len:] = key_row[:suffix_len]
     - expanded_value_row 同理
     - 重新发布（支持 transitive reuse）:
       store.store(reuser_slot_id, key_tensor=expanded_key_row, value_tensor=expanded_value_row, prefix_len=prefix_len+suffix_len)

输出: expanded_key, expanded_value  # 形状 = [sum(expanded_lengths_kv), H, D]
```

**关键观察**：
- Store 中的张量**never detach()**，保持 autograd graph 连续性，保证梯度流经共享 prefix KV
- Reuser 的 KV = prefix KV copy_ + suffix KV copy_，物理上 prefix KV 在 expanded KV 中复制了一份
- grad 回流路径：`copy_()` 保证 gradient 从 expanded KV 传回 provider prefix KV：
  `expanded_key[:prefix_len].grad → entry.key_tensor[:prefix_len].grad`，**gradient 没有断开**
- transitive reuse：链式场景下，一个 reuser 的 expanded KV 可以被后续 reuser 引用作为新的 provider

**GPU FA backend 调用链**（`flash_atten_gpu.py`）：

```
1. _prepare_flash_inputs(q, k, v, plan, packed_batch_layout):
   - 确保 THD [T, H, D] 布局
   - unpad Q: 通过 packed_batch_layout.unpad() 去除 TP padding
   - 从 plan 获取 cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
   - 返回 (q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)

2. flash_attn_varlen_func(
     q, k, v,
     cu_seqlens_q=plan.cu_seqlens_q,
     cu_seqlens_kv=plan.cu_seqlens_kv,
     max_seqlen_q=plan.max_seqlen_q,
     max_seqlen_kv=plan.max_seqlen_kv,
     causal=True,
     ...)

3. _repad_output(output, pad_layout)  # 恢复 TP padding
```

**mask 是通过 `causal=True` + varlen cu_seqlens 间接实现的**：
- Q cu_seqlens 定义了 shorter 序列（reuser 只有 suffix）
- KV cu_seqlens 定义了 full 序列（reuser 有 prefix+suffix）
- Flash attention 内部对每个 (q_seq, kv_seq) 对应用 causal mask
- reuser 的短 Q 看到完整 KV，causality 自动限制 suffix→prefix 可见性为全可见

#### 1.3.2 三个 Backend 的 Mask 机制对比

| Backend | 文件 | Mask 机制 | 当前是否支持 block mask |
|---|---|---|---|
| **TorchRef** | `torch_ref.py` | `_causal_q_kv_mask(q_len, kv_len, q_start)` — bool mask，reuser 的 prefix 全可见（q_start=prefix_len 偏移），suffix 部分 causal | ✅ 原理上已支持，per-row 构造可扩展 |
| **GPU FA** | `flash_atten_gpu.py` | `flash_attn_varlen_func` + `causal=True` + varlen cu_seqlens | ❌ 不支持显式 mask，依赖 cu_seqlens 间接实现 |
| **NPU** | `flash_atten_npu.py` | `_build_per_sample_mask()` 构造 `[B,1,max_q,max_kv]` bool mask 传入 `npu_fusion_attention` | ✅ 原生支持，已有 reuser block mask 实现 |

**TorchRef `_causal_q_kv_mask` 的详细行为**（关键参考实现）：

```python
def _causal_q_kv_mask(q_len, kv_len, q_start, device):
    """q_start: provider=0 (标准 causal), reuser=prefix_len (prefix 全可见)"""
    q_pos = torch.arange(q_start, q_start + q_len)[:, None]   # [q_len, 1]
    kv_pos = torch.arange(0, kv_len)[None, :]                  # [1, kv_len]
    mask = kv_pos <= q_pos  # lower-tri with offset
    # 当 q_start = prefix_len 时:
    #   kv_pos ∈ [0, prefix_len) 全部满足 kv_pos <= q_pos → prefix 列全 1 (可见)
    #   kv_pos ∈ [prefix_len, kv_len) 与 q_pos 构成因果 → suffix 列 causal
    return mask
```

这正是目标 block-causal mask 的完美实现。问题在于 torch_ref 目前 per-row 构造后仍需要 `build_kv` 展开 KV，然后才能将此 mask 用于 per-row SDPA。如果去掉 KV 展开，直接使用去重扁平布局 + 此 mask，就是完整的零冗余方案。

**NPU `_build_per_sample_mask` 的 block-causal 实现**（已支持的目标模式）：

```python
def _build_per_sample_mask(plan, valid_lens, kv_lens, max_q, max_kv, device):
    B = len(valid_lens)
    mask = torch.ones(B, 1, max_q, max_kv, dtype=torch.bool, device=device)
    for i in range(B):
        prefix_len = plan.prefix_lens[i]
        q_val = valid_lens[i]
        kv_val = kv_lens[i]
        if plan.is_reuser(i):
            # prefix columns [0, prefix_len) → 全 False (可见)
            mask[i, :, :q_val, :prefix_len] = False
            # suffix columns [prefix_len, kv_val) → causal (~tril)
            suffix_q = q_val
            suffix_kv = kv_val - prefix_len
            tri = torch.tril(torch.ones(suffix_q, suffix_kv, dtype=torch.bool))
            mask[i, :, :suffix_q, prefix_len:kv_val] = ~tri
        else:  # provider → standard causal
            tri = torch.tril(torch.ones(q_val, kv_val, dtype=torch.bool))
            mask[i, :, :q_val, :kv_val] = ~tri
    return mask
```

**核心差异总结**：
- **TorchRef** 已经具备构造任意 block-causal mask 的**原理能力**（`_causal_q_kv_mask` 函数），但当前依赖 expanded KV
- **NPU** 已经**直接支持 block-causal mask** 的构造（`_build_per_sample_mask`），但没有去重布局—它仍然使用 expanded KV（通过 build_kv）
- **GPU FA** 通过 `causal=True` + varlen cu_seqlens 间接实现，是三者中最受限的,不支持显式 mask

所有三个 backend 当前都需要 expanded KV（物理展开）作为数据基础。"零冗余"方案的关键是：保留 mask 构造逻辑不变，但改变 Q/KV 的数据布局，使其不再复制 prefix KV。

#### 1.3.3 plan 直接支持 mask 方案的关键字段

当前 `PrefixSharingPlan` 中与 mask 方案高度相关的字段（不需要新增，可直接复用）：

| 字段 | 值 | mask 方案中的用途 |
|---|---|---|
| `kept_lengths_q` | provider=full, reuser=suffix | Q 的去重长度（各 reuser 只保留唯一 suffix） |
| `expanded_lengths_kv` | = original_len | 在 mask 方案中需要改为**去重布局**长度 |
| `q_position_offsets` | reuser=prefix_len, provider=0 | mask 中 q_start 偏移量（直接影响 causality） |
| `prefix_lens` | 各行的共享 prefix 长度 | block-causal mask 的 prefix/suffix 分界点 |
| `provider_index` | 每个 reuser 的 provider 索引 | 确定 cross-reuser 不可见的 block 范围 |
| `reuse_specs` | PrefixReuseSpec 列表 | provider→reuser 映射关系 |
| `suffix_lens` | 各 reuser 的唯一 suffix 长度 | KV 去重后各 suffix 的 token 数 |

**现有的 O(N) 全量复制在 mask 方案中变为 O(1) 布局引用**：`expanded_lengths_kv` 不再是 `[P1, P1+S1, P1+S2, ...]`（provider全量，reuser=provider+suffix），而是 `[P, S1, S2, ...]`（prefix 只存一次，每个 reuser 只存自己的 suffix）。

#### 1.3.4 PrefixAttentionStore 的角色变更

**当前角色**：在 `build_kv()` 中跨层存储/加载 provider 的 prefix KV tensors，用于物理展开：

```
provider: store -> store.store(...)  # 存入完整 expanded KV
reuser:   entry = store.load(...)    # 加载 provider 的 KV
          expanded_row[:prefix_len] = entry.key_tensor[:prefix_len]  # 复制 prefix
          expanded_row[prefix_len:] = own_suffix                     # 追加 suffix
          store.store(...)            # 重新发布（支持链式传递）
```

**在 mask 方案中的角色**：如果采用"去重扁平布局即 provider prefix KV + reuser suffix KV 物理上只存一份"的布局：

1. **无需 store**：provider prefix KV 和 reuser suffix KV 直接在 flattened Q/KV 张量中连续排列，不需要跨 layer 的 store/load 操作
2. **无 tensor 复制**：不再需要 `copy_()` 操作 — prefix KV 就是扁平张量的第一个 segment
3. **restore 路径保留**：`PrefixLastRestoreSpec` 仍然需要，但实现简化 — restored logprob 直接来自 flattened Q 输出中的对应位置

**例外——多级树（chain）场景**：对于 transitive reuse（reuser B 引用 reuser A 的 expanded KV），如果采用纯扁平布局，则需要 chain 的每个中间节点仍然存一份 KV 供下游引用。此时 store 可能仍然需要，但不再是物理复制。

#### 1.3.5 PackedBatchLayout 的角色变更

**当前角色**：管理 framework-padded 布局到 valid 布局的转换（TP padding 的 unpad/repad）。

**在 mask 方案中的角色**：
- 仍然需要 `valid_lengths` / `padded_lengths` / `cu_seqlens` 等基础字段
- 需要扩展以表达**去重扁平布局**的 Q/KV 分段结构：
  - 无变化：`valid_lengths`、`padded_lengths`、`cu_seqlens`
  - 新增或 reference plan 的：`prefix_start_idx`、`suffix_start_indices`（各 reuser suffix 在扁平 KV 中的起始位置）
  - mask 相关：`kv_segment_bounds`（各 segment 在扁平 KV 中的 [start, end)）

**PackedBatchLayout 扩展与原生的关系**：

```
现有 layout:
  Q:     [provider Q_full | reuser1 Q_suffix | reuser2 Q_suffix]
  KV:    [provider K_expanded | reuser1 K_expanded | reuser2 K_expanded]
  cu_seqlens_q:  [0, P, P+S1, P+S1+S2]
  cu_seqlens_kv: [0, P, P+S1, P+S1+S2]

mask 方案 layout:
  Q:     [reuser1 Q_suffix | reuser2 Q_suffix]        # 与现有 Q 布局相同
  KV:    [provider K_prefix | reuser1 K_suffix | reuser2 K_suffix]  # 去重！
  cu_seqlens_q:  [0, S1, S1+S2]                        # 不变（已经是 suffix-only）
  cu_seqlens_kv: [0, P_total, P_total+S1, P_total+S1+S2]  # 变化！
```

**关键点**：Q 的 cu_seqlens 不变（已经是 suffix-only），KV 的 cu_seqlens 变短了（不再复制 prefix）。这是 mask 方案对 data layout 的唯一结构性变更。

### 1.4 业界调研

#### 1.4.1 RFC #6401 (Meituan/SandAI) — 最相关方案

**PR #6689 代码分析**（已由 arvyanh 提交完整实现，基于 MagiAttention）：

Trie 构建 → 扁平布局 → AttnSlice 表达 → MagiAttention FFA 计算。

**关键区别——扁平布局的结构差异**：

| 维度 | RFC #6401 / PR #6689 | 我们的目标方案 |
|---|---|---|
| **Q 布局** | 包含 prefix + reuser suffix（全量 Q） | 仅包含 reuser suffix（trimmed Q） |
| **KV 布局** | prefix + reuser suffix（去重） | prefix + reuser suffix（去重） |
| **mask 表达** | AttnSlice rectangle spec | block-causal bool / float bias |
| **后端 kernel** | MagiAttention FFA（AttnSlice dispatch） | FA varlen / TE bias / NPU fusion_attention |
| **依赖** | `magi_attention` CUDA kernel + DeepEP 通信 | 零额外依赖，复用 FA / TE / NPU 原生 |

RFC #6401 的 Attention rectangle spec：

```
           k: prefix    k: leaf0    k: leaf1
q: prefix   causal       ✗           ✗
q: leaf0     full      causal         ✗
q: leaf1     full        ✗         causal
```

而我们的 trimmed Q 方案 mask：

```
                  k: prefix    k: reuser1    k: reuser2
q: reuser1          full        causal         ✗
q: reuser2          full          ✗          causal
```

**差异的原因**：RFC #6401 保留了 prefix Q 段，因为我们不保留——prefix Q 的 forward 结果（logits）仅在 provider 路径上需要，而 provider 已经有了完整的结果。保留 prefix Q 是为了 CP 负载均衡和 minimize 重构损失，但额外增加了 Q 的计算量。

**关键发现**：我们的方案（trimmed Q + 去重 KV + block-causal mask）实际上是 RFC #6401 的**更激进版本**——Q 也比 RFC 更紧凑（彻底去掉 prefix Q），计算量更少。

#### 1.4.2 FA varlen causal mask 的技术限制

当前 GPU FA backend 使用 `flash_attn_varlen_func` + `causal=True` 是通过 varlen cu_seqlens**间接**实现 prefix sharing mask。这依赖一个隐含假设：同一个 q_seq 对同一个 kv_seq 的 causal mask 是标准的 lower-triangular。

对于去重 KV 布局，这不再成立，因为：
- Q 中的 reuser suffix tokens 需要看见 KV 中的 **多个独立 segment**（prefix + 自己的 suffix）
- 不同 segment 之间的 causality 边界不同
- `flash_attn_varlen_func` 的 `causal=True` 只能表达**单个连续 segment 内部**的 causal，不能表达 block-sparse pattern

**因此，要使用 FA varlen 作为后端，唯一的办法是传一个 pre-computed attention_bias (float/bool mask) 给 kernel。** 不过 `flash_attn_varlen_func` 直到 FA3（实验性）才支持 `attention_bias` 参数。FA2 的 `flash_attn_varlen_func` 完全不支持显式 bias。

**这是 mask 方案在 GPU FA 后端上要面对的核心技术挑战**。

#### 1.4.3 技术路线全景与 Kernel 选择

调研结果明确了现有技术路线的全景。

**Mask 方案 vs 其他路径的比较**：

| 路径 | Mask？ | Kernel | 速度 benchmark | 是否依赖外部 kernel |
|---|---|---|---|---|
| FA3 `flash_attn_varlen_func` + `causal=True` | ❌ 仅 causal | FA3 原生 CUDA | 1x（baseline） | ❌ |
| FA3 `flash_attn_func` + `attn_mask` | ✅ 全 mask | 降级到非 fused 路径 | 显著降级 | ❌ |
| TE `attn_mask_type="arbitrary"` | ✅ bool mask | 降级到非 fused 路径 | 显著降级 | 需要 TE |
| PyTorch SDPA + `attn_mask` (bool) | ✅ | Memory-Efficient (cutlass) fallback | 显著降级 | ❌ |
| **PyTorch FlexAttention** | ✅ BlockMask | FA3-based block-sparse | **接近 FA3** | ❌ PyTorch ≥ 2.5 |
| MagiAttention AttnSlice | ✅ rectangle | FFA (FA3-based) | 接近 FA3 | **需要 magi_attention** |
| 快手 DTA | ❌ DFS stack | 标准 HF transformer forward | 2-8x 经验 | ❌ |
| PrefixGrouper | 部分（两阶段）| 任意 attention kernel | ~Gx 理论 | ❌ |

**关键发现**：
- `flash_attn_varlen_func` **没有** `attention_bias` 参数——FA3 到当前版本（3.x）的 varlen 路径完全不支持显式 mask。唯一的 bias-like 参数是 `alibi_slopes`（per-head ALiBi 斜率）和 `softcap`（tanh softcapping）。
- `flash_attn_func`（非 varlen 的 padded 版本）接受 `attn_bias` 参数 `[batch, nheads, seqlen_q, seqlen_k]`，但这是 padded 变体，损失 varlen packing 效率。
- TE 的 `attn_mask_type="arbitrary"` 确实接受 bool mask `[B, NH, Sq, Skv]`，但 fuse attention 路径只对 `"causal"` / `"no_mask"` / `"padding"` 触发，**arbitrary 降级到非 fused 后端**，性能较差。
- PyTorch SDPA + `attn_mask` 同样降级：`scaled_dot_product_attention` 在有 mask 时不调用 FA 或 memory-efficient attention，而是走 math (C++) 后端。
- **FlexAttention**（PyTorch ≥ 2.5）是唯一在 GPU 上支持任意 mask pattern + 保持 FA3 级别性能的路径：`create_block_mask(mask_mod_fn)` 产生 block-sparse `BlockMask`，FA3 kernel 内部跳过空 tile。
- PR #6689 同时支持 **MAGI 和 FlexAttention** 两个 backend：`prefix_tree_attention=magi` 或 `prefix_tree_attention=flex`。

#### 1.4.4 Full 技术方案对比折线表

**总结：对于 mask 方案，可行的 GPU kernel 只有两个选择**：

| 路径 | 性能 | 依赖 | 集成复杂度 | 推荐场景 |
|---|---|---|---|---|
| **FlexAttention** | 接近 FA3 | PyTorch ≥ 2.5（无外部依赖） | 低 | 单 GPU/FSDP，GPU first |
| **MagiAttention FFA** | 接近 FA3（+CP） | `magi_attention` core package + CUDA 编译 + DeepEP | 高 | 分布式的 CP 场景，Megatron |
| **TE `arbitrary`** | 差（非 fused） | Megatron + TE | 中 | 仅用于正确性验证原型 |
| **NPU `npu_fusion_attention`** | 好 | CANN + MindSpeed（已有） | 低 | NPU 场景 |

其他维度对比：

| 维度 | PrefixSharing build_kv | PrefixGrouper (两阶段) | DTA/快手 (DFS栈) | RFC#6401 (Magi/Mask) |
|---|---|---|---|---|
| **KV 冗余** | 有（展开N份） | 有（prefix KV broadcast） | 零（DFS栈） | 零（扁平去重） |
| **Q 冗余** | 零（trimmed） | 零（分开调用） | 零（DFS） | 有（prefix Q 保留） |
| **计算图** | 标准 backward | 3个自定义 autograd Function | 梯度注入（2-10%近似） | 标准 backward |
| **mask 使用** | cu_seqlens 间接 | 无 | 无 | AttnSlice / Flex Mask |
| **多级树** | ❌ | ❌（仅扁平） | ✅（深度任意） | ✅（深度任意） |
| **batch 并行** | ✅ | ✅ | ❌（DFS 序列化） | ✅ |
| **NPU 可用** | 可行 | 需适配 | 需重写 DFS | 需适配 Flex/Magi |
| **依赖** | 零额外 | prefix_grouper pip包 | 自研引擎 | magi_attention / Flex

## 2. 方案设计

### 2.1 总体架构

#### 2.1.1 核心思路

**从"构建 expanded KV 作为数据"转变为"构建 attention mask 作为约束"**。

这个转换的本质是改变数据冗余的方式：
- 物理路径用**数据冗余**（每个 reuser 的 KV 中包含一份 prefix KV 副本）来实现 prefix 共享
- Mask 方案用**元数据冗余**（block-causal mask 矩阵，指定每个 Q token 可以 attend 哪些 KV token）来实现 prefix 共享

两者数学等价，但 mask 方案的 KV 带宽更优（无冗余读写），而物理路径的 attention kernel 调用更简单（标准 varlen FA）。

#### 2.1.2 转换流程对比

```
当前 build_kv 方案:
  PrefixSharingPlanner.plan(input_ids)
    → PrefixSharingPlan
      → PackedBatchLayout (valid_lengths, padded_lengths, cu_seqlens)
        → TorchReferenceBackend.build_kv(key, value, store, plan, layout)
            → store.load(provider_slot)  → 复制 prefix KV
            → concat(prefix_KV, suffix_KV) → expanded_KV
              → flash_attn_varlen_func(q, expanded_k, expanded_v,
                  cu_seqlens_q, cu_seqlens_kv, causal=True)

目标 mask 方案:
  PrefixSharingPlanner.plan(input_ids)
    → PrefixSharingPlan (kept_lengths_q, expanded_lengths_kv, q_position_offsets, ...)
      → PackedBatchLayout (valid_lengths, cu_seqlens_q, cu_seqlens_kv)
        → build_flattened_qkv(input_ids, plan)  # 构造去重 Q/KV
          → q = trimmed Q (suffix only, 与现有一致)
          → kv = deduplicated KV (prefix only once + all suffixes)
            → build_block_causal_mask(plan, layout, device, dtype)
              → block_mask: [total_q, total_kv] bool / float
                → attention_with_mask(q, k, v, mask)
```

#### 2.1.3 模块变更清单

| 模块 | 当前代码 | 变更内容 | 期望状态 |
|---|---|---|---|
| `backends/torch_ref.py` | `build_kv()` 构造 expanded KV; `attention()` 调 `_causal_q_kv_mask` per-row | 新增 `build_flattened_qkv()` 替代 `build_kv()`; 扩展 `attention()` 支持扁平布局 + block mask | 两种路径共存，通过 config 切换 |
| `backends/flash_atten_gpu.py` | `_prepare_flash_inputs` → `flash_attn_varlen_func` + `causal=True` | 新增 `_prepare_flash_inputs_noexpand` → 探索 FA3 attention_bias 参数或 TE bias 替代方案 | 两种路径共存 |
| `backends/flash_atten_npu.py` | per-sample pad/stack → `_build_per_sample_mask` → `npu_fusion_attention` | 修改 pad/stack 逻辑为扁平布局，mask 构造不变 | ✅ 最直接实现路径 |
| `backends/packed_layout.py` | `PackedBatchLayout` (valid/padded/cu_seqlens) | 扩展 `kv_segment_bounds` 字段；新增 `build_block_causal_mask` 类方法 | 新增 |
| `core/planner.py` | `PrefixSharingPlan` (kept_lengths_q, expanded_lengths_kv, ...) | 新增 `deflated_lengths_kv` 或扩展 `expanded_lengths_kv` 语义支持去重布局 | 小幅扩展 |
| `core/prefix_store.py` | 在 build_kv 中用 store 存/取 expanded KV | mask 方案在简单 GRPO 场景**无需 store**；链式场景可能需要，但不再物理复制 | 可选移除 |

### 2.2 Mask 构造算法

#### 2.2.1 扁平 GRPO 场景 (1-prefix + N-suffix)

一张图说明去重布局与 mask 的关系：

```
假设: P=2 tokens [a,b], N=2 reusers, S1=1 token [c], S2=2 tokens [d,e]

去重扁平布局:
  Q:     [c | d e]                              # total Q = 1+2 = 3
  KV:    [a b | c | d e]                        # total KV = 2+1+2 = 5
  cu_seqlens_q:  [0, 1, 3]                      # reuser1 suffix len=1, reuser2 suffix len=2
  cu_seqlens_kv: [0, 2, 3, 5]                   # prefix len=2, reuser1 suffix len=1, reuser2 suffix len=2

Mask (Q=3 × KV=5):

           k:[a  b]    k:[c]    k:[d  e]        # KV 各 segment
q:[c]       [1, 1]     [1, 0]   [0,  0 ]        # reuser1: prefix 全可见, 自己 suffix causal, other 不可见
q:[d e]     [1, 1]     [0, 0]   [1,  0 ]        # reuser2: prefix 全可见, 自己 suffix causal, other 不可见
                                                  #         [1,  1 ]  ← 2D 写成 1D
```

**构造算法**（O(total_Q × total_KV) 时间，但可优化为 O(B) 分段构造）：

```python
def build_block_causal_mask(plan, total_q, total_kv, device, dtype=torch.bool):
    """
    从 plan 信息构造 block-causal mask。

    输入:
      plan.provider_index     #[N] 每个 reuser 的 provider 索引
      plan.prefix_lens        #[N] 每个 reuser 的 prefix 长度
      plan.suffix_lens        #[N] 每个 reuser 的 suffix 长度
      plan.kept_lengths_q     #[B] Q 的段长度
      plan.deflated_lengths_kv #[B] KV 的去重段长度 (新增)

    输出:
      mask                    [total_q, total_kv] bool tensor
                              True = 可见 (attend), False = 不可见 (masked out)
    """
    # Step 1: 计算各 segment 在 Q/KV 中的起始位置
    q_seg_starts = cumsum([0] + plan.kept_lengths_q[:-1])    # 各 Q segment 的起始
    kv_seg_starts = cumsum([0] + plan.deflated_lengths_kv[:-1])  # 各 KV segment 的起始

    # Step 2: 确定 KV 中第一个 segment 是 prefix
    # (plan 中第一个非-reuser 的行为 provider, 其 kept_lengths = 总 prefix 长度)
    prefix_start = 0
    prefix_len = plan.deflated_lengths_kv[0]  # 第一个 segment 总是 prefix

    mask = torch.zeros(total_q, total_kv, dtype=torch.bool, device=device)

    # Step 3: 对每个 reuser (Q 中每个 suffix segment), 填充 mask
    for ri, reuser_idx in enumerate(plan.reuser_indices):  # reuser_indices = 非-provider 行的索引
        q_start = q_seg_starts[ri]          # 当前 reuser 在 Q 中的起始
        q_len = plan.kept_lengths_q[reuser_idx]   # Q 段长度 (suffix len)
        kv_prefix_end = prefix_len           # prefix 结束位置
        kv_own_start = kv_seg_starts[ri+1]   # 自己 suffix 在 KV 中的起始（ri+1 因为第0段=prefix）
        kv_own_len = plan.deflated_lengths_kv[ri+1]

        # 3a: prefix 全可见
        mask[q_start:q_start+q_len, :kv_prefix_end] = True

        # 3b: 自己的 suffix 因果可见
        for qi in range(q_len):
            q_pos = q_start + qi
            mask[q_pos, kv_own_start:kv_own_start+qi+1] = True

        # 3c: 其他 reuser 的 suffix 全不可见（已初始化为 False，不需要额外操作）

    return mask
```

**优化路径**：`O(B)` 分段构造而非 `O(total_Q × total_KV)` 全矩阵赋值，通过 segment-level batch assign 实现。

#### 2.2.2 多级树（chain）场景

链式场景（reuser A 的 expanded KV 被 reuser B 引用作为新的 provider）：

```
链: P → A → B
   P: [a,b,c] prefix (3 tokens)
   A: [a,b,c,d] → reuser of P, suffix=[d] (1 token)
   B: [a,b,c,d,e] → reuser of A (链式间接), suffix=[e] (1 token)

物理展开布局:
   KV: [a,b,c,d | a,b,c,d | a,b,c,d,e]  ← 3 次全量复制
   Q:  [a,b,c,d | d | e]

去重扁平布局 (mask 方案):
   KV: [a,b,c | d | e]                    ← 只存一次, 无复制
   Q:  [d | e]                             ← trimmed suffix only
   cu_seqlens_q:  [0, 1, 2]
   cu_seqlens_kv: [0, 3, 4, 5]

Mask:
            k:[a,b,c]  k:[d]  k:[e]
q:[d]          [1,1,1]   [1]   [0]     # A 能看见 prefix+自己
q:[e]          [1,1,1]   [1]   [1]     # B 能看见 prefix + A + 自己 (链式)
```

**链式 masking 的关键**：`q_position_offsets[reuser]` = 该 reuser 在去重 KV 中可见的前缀段数。B 的 q_position_offset = 4（prefix a,b,c + A 的 d = 4 tokens），所以 B 的全部 tokens 对 KV[0:4] 可见。

**公式**：对于 reuser i，其可见 KV 范围 = `[0, prefix_len_i + suffix_len_i]`，但在去重布局中 prefix 段的实际长度为 sum of all ancestor segments' lengths。

#### 2.2.3 与 DTA 的对比

DTA（快手）的 push-pop 栈式 KV cache 也是去重方案，但其实现路径完全不同：

| 维度 | Mask 方案 | DTA (快手) |
|---|---|---|
| **数据组织** | 静态扁平张量，一次性构造 | 运行时推弹栈，DFS 遍历 |
| **计算图** | 标准 backward（数学严格等价） | 梯度注入（2-10% 近似偏差） |
| **内存峰值** | 总 Q/KV 张量 | block_size 控制（更灵活） |
| **tree depth** | 支持任意深度（mask 描述即可） | 支持任意深度（push-pop 天然支持） |
| **batch 并行** | 完整 batch x N 并行 | DFS 序列化到单 batch（失去 batch 并行） |
| **实现复杂度** | 低 — 纯 mask 构造 + 标准 backward | 高 — Pop/Push 状态机 + 梯度注入 |

Mask 方案的核心优势是数学严格等价（无近似偏差）和标准 backward（无需自定义 autograd）。DTA 的精确梯度有偏差，但内存控制更灵活（block_size 可调）。

#### 2.2.4 与 RFC #6401 AttnSlice 的对比

RFC #6401 使用 MagiAttention 的 AttnSlice rectangle spec 来描述 mask，好处是：

- 无需构造 `[total_q, total_kv]` 完整 mask 矩阵（节省 mask 内存）
- 利用 FFA kernel 内部的 rectangle-aware 实现（kernel 层面的 block-sparse 优化）
- 支持 CP 负载均衡（dispatch solver 根据 rectangle 大小分配 work）

我们的 mask 构造需要：
- 要么构造完整 `[total_q, total_kv]` 矩阵作为 attention_bias（GPU FA 的 `flash_attn_varlen_func` 或 `flash_attn_func`）
- 要么构造 per-sample `[B,1,max_q,max_kv]` 矩阵（NPU `npu_fusion_attention`）
- 要么利用 TE `DotProductAttention` 的 bias 参数（Megatron 路径）

**Mask 的内存容量**：`total_q × total_kv` 当 Q=1024, KV=2048 时，bool mask 仅 2MB，float16 bias 约 4MB。即使 Q=8192, KV=16384，bool mask 为 128MB，float16 bias 为 256MB——在 BF16 训练的可控范围内。

### 2.3 Mask 格式与 Kernel 适配

#### 2.3.1 GPU 路径 — 推荐：PyTorch FlexAttention

**这是当前最推荐的 GPU 实现路径**，理由：
- 唯一的「任意 mask pattern + FA3 级别性能」的组合
- 零外部依赖（PyTorch ≥ 2.5 自有）
- PR #6689 已证明可与 prefix-tree layout 配合：`prefix_tree_attention=flex`

**FlexAttention 使用方式**：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def block_causal_mask_mod(b, h, q_idx, k_idx):
    """
    q_idx / k_idx: flattened positions in Q/KV tensor.
    假设去重布局: Q = [reuser1_suffix, reuser2_suffix, ...]
                  KV = [prefix, reuser1_suffix, reuser2_suffix, ...]
    """
    # 已知每个 segment 的边界
    # ... 根据 q_idx 所在 segment 确定它是哪个 reuser / provider
    # ... 再根据 k_idx 所在 segment 确定可见性
    # Return True = attend, False = masked

block_mask = create_block_mask(
    block_causal_mask_mod,
    1, 1,           # batch=1, heads=1 (broadcast to all heads)
    total_q_tokens,
    total_kv_tokens,
    _compile=True,
    BLOCK_SIZE=128,  # FA3 tile size
)

out = flex_attention(q, k, v, block_mask=block_mask)
```

**`block_causal_mask_mod` 的逻辑**：

```python
# 假设 plan.deflated_lengths_kv = [P, S1, S2, ...]
# plan.kept_lengths_q = [S1, S2, ...]
kv_seg_bounds = cumsum([0] + plan.deflated_lengths_kv)  # [0, P, P+S1, P+S1+S2, ...]
q_seg_bounds = cumsum([0] + plan.kept_lengths_q)         # [0, S1, S1+S2, ...]

# Step 1: 确定 q_idx 属于哪个 reuser
for ri in range(num_reusers):
    q_start = q_seg_bounds[ri]
    q_end = q_seg_bounds[ri + 1]
    if q_start <= q_idx < q_end:
        reuser_idx = ri
        break

# Step 2: 确定 k_idx 属于哪个 segment
for si in range(len(kv_seg_bounds) - 1):
    k_start = kv_seg_bounds[si]
    k_end = kv_seg_bounds[si + 1]
    if k_start <= k_idx < k_end:
        seg_idx = si
        break

# Step 3: 可见性判断
if seg_idx == 0:
    # KV segment 0 = prefix → 对所有 Q 可见
    return True
elif seg_idx == reuser_idx + 1:
    # KV segment = 自己的 suffix → causal
    q_local = q_idx - q_seg_bounds[reuser_idx]       # 在 reuser Q 内的位置
    k_local = k_idx - kv_seg_bounds[seg_idx]          # 在 segment KV 内的位置
    return k_local <= q_local   # causal
else:
    # KV segment = 其他 reuser 的 suffix → 不可见
    return False
```

**`BLOCK_SIZE` 选择**：FA3 tile size 默认 128。Block-sparse mask 在 tile 级别生效——整个 tile 内只要有任何 Q-KV pair 需要 attend，该 tile 就需要计算。因此 tile size 越小 mask 越精确，但调度开销越大。128 是合理默认。

**limitations**：
- FlexAttention 在 PyTorch ≥ 2.5 可用，当前 verl080 环境（PyTorch 2.8）满足要求
- 当前不支持分布式 CP（需要手动切分 Q/KV 并按 rank 构造 mask_mod）
- `mask_mod` 在 GPU 上每个 tile launch 时执行，性能开销很小（受 BLOCK_SIZE 降低数量级）

#### 2.3.2 MagiAttention FFA 路线（替代方案）

如果分布式 CP 成为需求，MagiAttention 的 FFA + AttnSlice 是更完善的方案：

```python
from magi_attention import AttnSlice, calc_attn

# 构造 rectangle spec 而非 full mask
rects = [
    # (q_range, k_range, mask_type)
    # prefix 内部: causal
    AttnSlice((0, P), (0, P), MaskType.CAUSAL),
    # reuser0: prefix full, suffix causal
    AttnSlice((0, S1), (0, P), MaskType.FULL),      # reuser0 → prefix = full
    AttnSlice((0, S1), (P, P+S1), MaskType.CAUSAL),  # reuser0 → own suffix = causal
    # reuser1: prefix full, suffix causal
    AttnSlice((S1, S1+S2), (0, P), MaskType.FULL),       # reuser1 → prefix = full
    AttnSlice((S1, S1+S2), (P+S1, P+S1+S2), MaskType.CAUSAL),  # reuser1 → own suffix = causal
]

out = calc_attn(q, k, v, rects)
```

**与 FlexAttention 的比较**：

| 维度 | FlexAttention | MagiAttention FFA |
|---|---|---|
| 依赖 | PyTorch ≥ 2.5 内建 | magi_attention + CUDA 编译 + DeepEP |
| mask 表达 | mask_mod 函数 | AttnSlice rectangle list |
| 单 GPU 性能 | 接近 FA3 | 接近 FA3 |
| CP 支持 | ❌（需手动切分） | ✅ dispatch solver + GroupCast/GroupReduce |
| 集成复杂度 | 低 | 高 |
| NPU 兼容 | ❌ | ❌ |

**推荐策略**：第一阶段用 FlexAttention（依赖最少、集成最简）；有 CP 需求时再考虑 MagiAttention。

#### 2.3.3 TE `attention_mask` 路线（Megatron，仅用于原型验证）

TE 的 `attn_mask_type="arbitrary"` 接受 `[B, NH, Sq, Skv]` 的 bool mask：

```python
# 构造 batch-level block-causal mask
# attn_mask: True = masked out
attn_mask = torch.zeros(1, 1, total_q, total_kv, dtype=torch.bool)
attn_mask[:, :, q_slice, other_kv_slice] = True  # 设 cross-reuser 不可见
attn_mask[:, :, q_slice, future_kv_slice] = True  # 设 suffix causal 边界外不可见

output = te_core_attention(
    q, k, v,
    attention_mask=attn_mask,
    attn_mask_type="arbitrary",
    core_attention_bias_type="no_bias",
)
```

**重要**：TE `"arbitrary"` 降级到非 fused 后端，**性能不佳**。这个路径**仅用于正确性验证原型**，不用于生产性能基准。

#### 2.3.4 NPU `npu_fusion_attention` 路线

**NPU 是最自然的实现路径**。当前 `flash_atten_npu.py` 的 mask 构造逻辑（`_build_per_sample_mask`）完全可以使用，只需将数据布局从 expanded KV 改为去重 KV：

```
当前 NPU per-sample mask: [B, 1, max_q, max_kv]   # expanded KV，每个 sample 有独立 KV 行
目标 NPU per-sample mask: [B, 1, max_q', max_kv']  # 去重 KV，max_kv' < max_kv
```

NPU 的 per-sample mask 格式与 batch-level「去重 Q 对全量 KV」模式不兼容——每个 sample 的 Q 只能 attend 到自己的 sample 的 KV。NPU 的 mask 方案需要的不是 `[B,1,max_q,max_kv]` 的 per-sample mask，而是 **batch-level mask** `[1,1,total_q,total_kv]` 或者将 Q 调整格式为 `[B,1,max_q',max_kv']` 但 KV 需要去重共享。

对于 NPU 场景，更实用的迁移路径是保持当前的 BSH pad/stack 格式不变，但**缩减 mask 构造**：用 batch-level mask 替代 per-sample mask，并去掉 expanded KV 的 prefix 复制逻辑。

#### 2.3.5 kernel 适配总结

| 场景 | 推荐 kernel | Mask 格式 | 性能预期 | 依赖 |
|---|---|---|---|---|
| GPU FSDP 单机 | FlexAttention | mask_mod → BlockMask | 接近 FA3 | PyTorch ≥ 2.5 |
| GPU Megatron/CP | MagiAttention FFA | AttnSlice list | 接近 FA3 + CP 加速 | magi_attention |
| GPU Megatron （原型） | TE `arbitrary` | [B, NH, Sq, Skv] bool | 差（非 fused） | TE |
| NPU 当前路径 | `npu_fusion_attention` BSH | per-sample bool [B,1,max_q,max_kv] | native | CANN + MindSpeed |
| NPU 改进路径 | `npu_fusion_attention` BSH | **batch-level** bool [1,1,total_q,total_kv] | native | CANN + MindSpeed |
| CPU 调试 | SDPA + attn_mask | [total_q, total_kv] bool | N/A | torch |

## 3. 测试验证

### 3.1 正确性验证

#### 3.1.1 Mask 构造正确性 — Python/PyTorch 数值验证

验证方法：逐元素对比 mask 方案的 attention 输出与 baseline（独立 forward）的 attention 输出。

```
测试策略:
1. 对 random input_ids, 构造 GRPO (1-prefix + N-suffix) 场景
2. 独立 forward: N 条全序列分别过 attention，得到 N 个 attention 输出
3. Mask 方案: 去重扁平 Q/KV + block-causal mask → 一次 forward 得到 N 个 attention 输出
4. 逐元素 cos_sim ≥ 0.999（float32）或 cos_sim ≥ 0.99（bfloat16）
5. 验证 backward: 对比两种路径的梯度，相对误差 ≤ 1e-5
```

#### 3.1.2 与 build_kv 路径的数学等价性

现有 build_kv 路径已经是经过验证的"标准答案"。mask 方案需要与 build_kv 路径逐元素等价。

```
测试:
1. build_kv_path: expanded_KV + flash_attn_varlen_func + causal=True → output_ref
2. mask_path: flattened_QKV + block_causal_mask + attention_with_mask → output_mask
3. assert_allclose(output_ref, output_mask, atol=1e-5)  # float32
```

#### 3.1.3 梯度等价性

```
测试:
1. 在 float32 精度下跑几个 attention layer（不用完整模型）
2. forward 后对 output 求和作为 loss
3. backward 得到所有输入的梯度
4. build_kv 路径 vs mask 路径：梯度张量的相对误差
5. 要求: max(|grad_mask - grad_ref| / |grad_ref|) < 1e-5
```

### 3.2 性能基准

#### 3.2.1 单 batch 场景对比

核心 benchmark：对比两种路径在相同输入下的 **KV 内存峰值** 和 **attention 耗时**。

| 场景 | 参数 | 关键指标 |
|---|---|---|
| GRPO n=8, prefix=512, suffix=256 | build_kv vs mask | KV GB, step ms |
| GRPO n=8, prefix=2048, suffix=128 | build_kv vs mask | KV 节省比率 |
| Chain depth=3 | build_kv vs mask | KV 节省 + extra overhead |
| no-sharing | build_kv vs mask | 退化为 baseline 开销 |

**预期结果**：
- KV 内存节省 = `1 - (P + N*S) / (P + ΣS_i)` ≈ 在典型 GRPO 场景中 40-70%
- attention 时间差异取决于 kernel：FA varlen 路径 mask 方案可能略慢（非 causal kernel 优化）；NPU 路径可能更快（更小的 BSH 张量）

#### 3.2.2 NPU 场景

NPU 路径（`npu_fusion_attention`）的结果可预测：去重布局减小 BSH 中的 `max_q` 和 `max_kv`，直接降低 NPU 的 attention 计算量和 HBM 占用。

#### 3.2.3 与 baseline（no prefix sharing）的对比

mask 方案不应该比 no-sharing baseline 更差。验证：

```
baseline:   N 条独立 forward (no prefix sharing)
mask_path:  flatten + block-causal mask
```

期望 mask 方案的 attention 计算量 ≤ baseline 的 N 倍？

实际上 mask 方案的计算量 = **扁平 Q 长度 × 扁平 KV 长度**，而非独立的 N × (P+Si)²。

**定量分析**：
- baseline: `Σ(P+Si)²` per-sample attention FLOPs
- build_kv: `ΣSi × (P+Si)` 的 attention FLOPs + N 倍 prefix KV 复制带宽
- mask: `ΣSi × (P + ΣSi)` 的 attention FLOPs + 一次 mask 构造

当 N=8, P=512, Si=256:
- baseline: 8 × (512+256)² = 4.72M FLOPs
- build_kv: 8 × 256 × (512+256) = 1.57M + 8×512 KV 带宽
- mask: 8 × 256 × (512 + 8×256) = 4.19M FLOPs

**mask 方案的 FLOPs 比 build_kv 多**（因为 KV 去重后 reuser 之间不再互相阻挡，每个 reuser 的 Q 看到更长的 KV），但比 baseline 少（尤其在 N 大时）。这里说的 FLOPs 是指 attention score 计算中的浮点运算：`Q × K^T` 的矩阵乘法。

mask 方案在 attention FLOPs 上应该**与 baseline 对比**而不是与 build_kv 对比，因为两者的 KV 布局目标不同。Mask 方案的净收益 vs build_kv 是 **KV 带宽的零冗余**，而不是 FLOPs 节省。

## 4. 开发计划

### 4.1 第一阶段：原型验证（当前阶段）

**目标**：实现去重扁平 Q/KV + block-causal mask 的 TorchRef 路径，通过 CPU 正确性验证。

#### 4.1.1 Mask 构造工具

新增 `backends/mask_utils.py`：

- `build_block_causal_mask(plan, device, dtype) -> BoolTensor`
- `build_qkv_flattened(input_ids, plan, model) -> (q, k, v)`
- 复用现有 `PrefixSharingPlan` 字段，新增 `deflated_lengths_kv`

#### 4.1.2 TorchRef mask-based attention 原型

TorchRef backend 新增 mask 路径：

- `build_flattened_qkv()`：构造去重 Q/KV（不再调 store.load/clone）
- `flattened_attention()`：一次性 per-row SDPA with block mask（替代逐 row build_kv + per_row attention 的两步流程）
- 复用现有 `_causal_q_kv_mask` 逻辑，扩展为完整的 block-causal mask

#### 4.1.3 CPU 正确性验证

- 使用现有的 `tools/verify_p0_correctness.py` 框架，新增 mask 路径对比
- 覆盖：one_provider, chain, no_sharing, multi_provider
- 参数：BS=4/8, P=16/32, S=8/16
- 指标：cos_sim ≥ 0.999, grad_rel_err < 1e-5

### 4.2 第二阶段：GPU 路径

#### 4.2.1 TE bias 路径（Megatron）

- 验证 TE `DotProductAttention` 对 `attention_bias` block-causal 格式的支持
- 构造 `[1, 1, total_q, total_kv]` float bias 并传入
- 在 Megatron 1GPU 上跑通 1 step

#### 4.2.2 NPU fusion_attention mask 路径

- 修改 pad/stack 逻辑为去重扁平 KV
- mask 构造完全复用现有 `_build_per_sample_mask`
- 在 NPU 上跑通 1 step

#### 4.2.3 性能对比

- 对比 build_kv vs mask 方案的 KV HBM 峰值
- 对比 step_s / gen_s / attention kernel 耗时
- 覆盖 P=256/512/1024, N=8/16/32, S=64/128/256

### 4.3 第三阶段：集成与优化

#### 4.3.1 与 verl FSDP engine 集成

一个 config 开关 `share_mode`（`kv_expand` / `block_mask`）：

```
prefix_sharing:
  enabled: true
  share_mode: block_mask  # 新增模式
```

两种模式共享：`PrefixSharingPlanner.plan()`、`PackedBatchLayout` 基础字段、`PrefixLastRestoreSpec`

#### 4.3.2 与 verl Megatron engine 集成

Megatron 路径通过 TE attention bias 实现 block-causal mask，不需要 `build_kv` 流程。

#### 4.3.3 Mask 内存优化

对于大型输入（P=4096, N=32, S=512），`[total_q, total_kv]` bool mask = `16K × 20K = 320M bits = 40MB`，float16 bias = `640MB`。优化路径：

1. Bool mask + broadcast：在 kernel 内部展开为 float 时利用 sparsity
2. Segment-level mask：不为哑元行（完全不可见区域）分配实际 mask 行
3. Per-sample mask（NPU 路径）：`[B,1,max_q,max_kv]` 自然解决了 mask 内存问题 B << total_q

#### 4.3.4 多级树支持

扩展 mask 构造算法支持通用树结构，输入为 **prefix segments (hash, length)** 映射。

### 4.4 里程碑

| 里程碑 | 时间 | 交付物 |
|---|---|---|
| M1 | TBD | Mask 构造 + CPU 正确性通过 (tools/verify_p0_correctness 扩展) |
| M2 | TBD | GPU TE bias / GA mask 路径跑通 1 step (Megatron engine) |
| M3 | TBD | NPU mask 路径跑通 1 step (验证已验证的 _build_per_sample_mask 路径) |
| M4 | TBD | 完整性能对比数据，两种模式可配置切换 |
