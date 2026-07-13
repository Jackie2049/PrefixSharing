# NPU Flash Attention 数据排布示例

> **注意**：当前默认使用 **TND 单样本路径**（内存优化），BSH 路径作为 fallback。
> 下文先展示 TND 路径（推荐），再保留 BSH 路径供参考。

---

## TND 单样本路径（默认，推荐）

### 核心思路

将所有样本拼接为一条 "超级样本"，Q 和 KV 都保持 THD 格式，
planner 的 `(total_q, s_packed_length)` SS 格式 mask 直接传给内核。

**优势**：
- 无 Q padding（按实际长度存储）
- 无 K/V batch expand（避免内核 materialize）
- Mask 从 `(B, max_q, s_packed_length)` 降至 `(total_q, s_packed_length)`
- 内存降低 4-8 倍

### 数据流（同样的 3-sample 场景）

```
输入 Q (THD):   shape (8, 2, 3)     # total_q=8, 直接传入
输入 K (THD):   shape (8, 2, 3)     # s_packed_length=8, 直接传入
输入 V (THD):   shape (8, 2, 3)     # s_packed_length=8, 直接传入

Mask (SS):      shape (8, 8)        # planner 缓存的 global_custom_mask 取反

调用:
  npu_fusion_attention(
      q=q,                  # (8, 2, 3) THD
      k=k,                  # (8, 2, 3) THD
      v=v,                  # (8, 2, 3) THD
      head_num=2,
      input_layout="TND",
      atten_mask=~global_mask,    # (8, 8) SS, True=masked
      sparse_mode=1,
      actual_seq_qlen=[8],
      actual_seq_kvlen=[8],
  )

输出: (8, 2, 3) THD — 无需 unpack
```

### Mask 内容 (SS 格式)

与 planner.build_global_custom_mask() 完全一致（取反后 True=masked）：

| Q 行 | 可见的 s_packed 位置 (False=visible) |
|------|------|
| Q[0] sample0 pos=0 | [0] |
| Q[1] sample0 pos=1 | [0, 1] |
| Q[2] sample0 pos=2 | [0, 1, 2] |
| Q[3] sample0 pos=3 | [0, 1, 2, 3] |
| Q[4] sample0 pos=4 | [0, 1, 2, 3, 4] |
| Q[5] sample1 pos=2 | [0, 1, 5] |
| Q[6] sample1 pos=3 | [0, 1, 5, 6] |
| Q[7] sample2 pos=2 | [0, 1, 7] |

### 内存对比

| 项目 | BSH | TND | 节省 |
|------|-----|-----|------|
| mask | (3, 1, 5, 8) = 120 bool | (8, 8) = 64 bool | **-47%** |
| Q | (3, 5, 6) = 90 fp16 | (8, 2, 3) = 48 fp16 | **-47%** |
| K/V | expand 视图 | (8, 2, 3) 原样 | 无 materialize |

> 在实际训练中 (B=32, max_q=2048, s_packed=8192)，
> TND 路径节省 **~75-87%** 的 attention 内存。

---

## BSH Fallback 路径（以下保留原文档供参考）

## 场景设定

三个 sample 共享前缀 `[a, b]`：
- **sample 0**: provider，完整序列 `[1, 2, 3, 4, 5]`
- **sample 1**: reuser，序列 `[1, 2, 9, 10]`（共享前缀 `[1,2]`）
- **sample 2**: reuser，序列 `[1, 2, 99]`（共享前缀 `[1,2]`）

参数：`batch=3`, `n_q_heads=n_kv_heads=2`, `head_dim=3`

---

## Step 1: `PrefixSharingPlan` 元数据

| 字段 | 值 | 含义 |
|------|-----|------|
| `prefix_lens` | `[0, 2, 2]` | 每个 sample 的前缀长度（provider=0） |
| `is_provider` | `[True, False, False]` | sample 0 是 provider |
| `s_packed_length` | `8` | 去重后的总 KV 长度 |
| `s_packed_kv_ranges` | `[[(0,5)], [(0,2),(5,7)], [(0,2),(7,8)]]` | 每个 sample 的 KV 区间 |
| `s_packed_prefix_end` | `[0, 2, 2]` | prefix 在 s_packed 中的结束位置 |
| `s_packed_q_lengths` | `[5, 2, 1]` | 每个 sample 的 Q 长度（provider=全序列，reuser=suffix） |
| `s_packed_q_starts` | `[0, 5, 7]` | Q 在 packed Q 中的累加起点 |

---

## Step 2: 输入张量形状

```
q shape: (total_q=8, n_q_heads=2, head_dim=3)        # THD
k shape: (s_packed_length=8, n_kv_heads=2, head_dim=3)
v shape: (s_packed_length=8, n_kv_heads=2, head_dim=3)
```

注意：**K/V 是 deduped storage**，prefix `[1, 2]` 只在 `s_packed[0:2)` 存一份。

---

## Step 3: s_packed 内容布局

```
s_packed[0] = 1     ← shared prefix (sample 0,1,2 都用)
s_packed[1] = 2     ← shared prefix (sample 0,1,2 都用)
s_packed[2] = 3     ← sample 0 的 suffix
s_packed[3] = 4     ← sample 0 的 suffix
s_packed[4] = 5     ← sample 0 的 suffix
s_packed[5] = 9     ← sample 1 的 suffix
s_packed[6] = 10    ← sample 1 的 suffix
s_packed[7] = 99    ← sample 2 的 suffix
```

**关键点**：前缀 `[1,2]` 在 `s_packed[0:2)` 只存 1 份，被 sample 0/1/2 共用。

---

## Step 4: Q packed 布局 (THD → `total_q=8`)

```
Q[0:5] = sample 0 (provider)        Q 完整序列 [1,2,3,4,5]
Q[5:7] = sample 1 (reuser suffix)  Q 序列 [9,10]
Q[7:8] = sample 2 (reuser suffix)  Q 序列 [99]
```

---

## Step 5: padded_lengths

```
padded_lengths = [5, 2, 1]   # 各 sample 的 Q 长度不齐
max_q = 5                    # pad 到最大值
```

---

## Step 6: BSH 转换后给 `npu_fusion_attention` 的张量

```python
q_bsh shape: (batch=3, max_q=5, hidden_q=6)     # BSH
k_bsh shape: (batch=3, s_packed=8, hidden_kv=6) # BSH
v_bsh shape: (batch=3, s_packed=8, hidden_kv=6) # BSH
```

**关键点**：
- `k_bsh`/`v_bsh` 的 batch 维度只是 `expand` 出来的**影子**，所有 batch 行的 K/V 完全相同。
- batch 维度的语义通过 `atten_mask` 表达（每个 sample 看自己的 KV 区间）。

```python
# 实际 NPU backend 构造逻辑
k_bsh = k.reshape(8, 6).unsqueeze(0).expand(3, -1, -1)  # (3, 8, 6)
v_bsh = v.reshape(8, 6).unsqueeze(0).expand(3, -1, -1)
q_bsh = torch.zeros(3, 5, 6)
q_bsh[0, :5, :] = Q[0:5].reshape(5, 6)   # sample 0
q_bsh[1, :2, :] = Q[5:7].reshape(2, 6)   # sample 1
q_bsh[2, :1, :] = Q[7:8].reshape(1, 6)   # sample 2
```

---

## Step 7: atten_mask（核心！）

**Shape**: `(batch=3, 1, max_q=5, s_packed=8)`
**约定**: `True = masked (不可见)`, `False = visible`

### Sample 0 (provider)

| Q 行 (original_pos) | 可见的 s_packed 位置 |
|---|---|
| row 0 (pos=0) | [0] |
| row 1 (pos=1) | [0, 1] |
| row 2 (pos=2) | [0, 1, 2] |
| row 3 (pos=3) | [0, 1, 2, 3] |
| row 4 (pos=4) | [0, 1, 2, 3, 4] |

### Sample 1 (reuser, prefix_len=2)
- `s_packed_kv_ranges = [(0, 2), (5, 7)]`
- prefix block `(0, 2)`: causal 边界 = `q_original_pos`，可见 `[0, min(2, q_orig))`
- suffix block `(5, 7)`: causal 边界 = `q_original_pos + prefix_len + 1`

| Q 行 (original_pos) | 可见的 s_packed 位置 |
|---|---|
| row 0 (pos=2) | [0, 1, 5]  ← prefix 全可见 + suffix[0] self-attend |
| row 1 (pos=3) | [0, 1, 5, 6]  ← + suffix[1] |
| row 2..4 | 全部 masked（pad 区域） |

### Sample 2 (reuser, prefix_len=2)
- `s_packed_kv_ranges = [(0, 2), (7, 8)]`

| Q 行 (original_pos) | 可见的 s_packed 位置 |
|---|---|
| row 0 (pos=2) | [0, 1, 7]  ← prefix 全可见 + suffix[0] self-attend |
| row 1..4 | 全部 masked（pad 区域） |

---

## 数据流总结图

```
原始序列 (3 个 sample, 共享前缀 [1,2]):
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ 1 2 3 4 5   │  │ 1 2 9 10    │  │ 1 2 99      │
└─────────────┘  └─────────────┘  └─────────────┘
       ↓                ↓                ↓
     provider      reuser (pre=2)  reuser (pre=2)
       ↓                ↓                ↓
      Q[0:5]           Q[5:7]           Q[7:8]
                          (suffix only)  (suffix only)

去重后的 s_packed (长度 8):
┌───┬───┬───┬───┬───┬────┬────┬─────┐
│ 1 │ 2 │ 3 │ 4 │ 5 │ 9  │ 10 │ 99  │   ← shared prefix [1,2] 只存 1 份
└───┴───┴───┴───┴───┴────┴────┴─────┘
      ├─────┤                  └─────┬─────┘
   共享前缀                         各 sample 独占

K/V (THD): shape (8, 2, 3)
            ↓ reshape + expand
K/V (BSH): shape (3, 8, 6)   ← batch 维只是影子

Q (THD):   shape (8, 2, 3)
            ↓ split + pad
Q (BSH):   shape (3, 5, 6)   ← pad 到 max_q=5

atten_mask: shape (3, 1, 5, 8)
  - 每个 batch 行定义自己的可见 KV 区间
  - True = masked (npu_fusion_attention 约定)

最终调用:
  npu_fusion_attention(
      q=q_bsh,        # (3, 5, 6)
      k=k_bsh,        # (3, 8, 6)  ← expand 出来的相同 K
      v=v_bsh,        # (3, 8, 6)  ← expand 出来的相同 V
      atten_mask=atten_mask,  # (3, 1, 5, 8)
      layout="BSH",
      sparse_mode=1,
  )
```

## 关键设计要点

1. **dedup storage**：共享前缀只存 1 份，省显存、带宽
2. **K/V 在 BSH 维度上冗余**：每个 batch 行的 K/V 完全相同，通过 `expand`（无内存拷贝）实现
3. **batch 区分靠 mask**：`atten_mask` 精确控制每个 sample 只能 attend 到自己的 KV 区间 + causal 约束
4. **padding 处理**：长度不齐的 Q pad 到 `max_q`，pad 行的 mask 全部为 True（masked）