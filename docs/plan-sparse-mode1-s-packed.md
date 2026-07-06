# Plan: sparse_mode=1 — s_packed 去重存储 + 全局 Custom Causal Mask

## 背景

当前 prefix sharing 实现沿用稠密方案：每个 reuser 都把自己的 prefix 长度放到
`expanded_lengths_kv` 里，对应 `kept_lengths_q` 只有 suffix。这要求 attention
backend 支持 **different Q/KV lengths**，要么走两段 THD 拼接 + 局部 attention，
要么走 padded Q + gather mask。前者在 NPU `npu_fusion_attention` 上需要两次调用
（含 prefix 段、suffix 段），后者在 GPU `flash_attn_varlen_func` 上不支持
arbitrary mask。

为了把 KV 真正去重（reuser 完全不出现 shared prefix 的 K/V），引入
**sparse_mode=1**：构建一份全局去重的 **s_packed KV 序列**，再配合一份
(total_q, s_packed_length) 的 **global custom causal mask**——attention
backend 只需一次调用，且 K/V 总数从 `sum(original_lengths)` 降到
`sum(unique_tokens)`。

## 关键数据流

```
plan.input_ids              detector  →  PrefixDetectionResult
                                                    ↓
                                PrefixSharingPlan.plan_from_detection
                                                    ↓
                            build_s_packed(input_ids, prefix_lens, original_lengths)
                                                    ↓
         ┌──────────── s_packed_kv_ranges[i]   (prefix 段区间 + suffix 段区间)
         │    s_packed_length                   ← 总长度（去重后）
         │    s_packed_q_lengths[i]            ← = suffix_len
         │    s_packed_q_starts[i]             ← cumulative
         │    s_packed_prefix_end[i]           ← prefix 段在 s_packed 中的 end
                                                    ↓
                       build_global_custom_mask(device)
                                                    ↓
                              (total_q, s_packed_length) bool mask
                              （True = visible，与 _causal_q_kv_mask 同语义）
                                                    ↓
                              TorchReferenceBackend.attention
                              ┌──── per-batch q slice
                              ├──── per-batch causal（prefix 段用 prefix 位置，
                              │    suffix 段用 suffix 位置）
                              └────→ output

                              NpuFlashAttentionBackend.attention
                              ┌──── 把 global mask expand 到 (B, 1, max_q, S)
                              ├──── K/V 在 BSHD 上 broadcast 为同一份 s_packed
                              └──── 一次 npu_fusion_attention
```

## s_packed 构建语义（`build_s_packed`）

输入：
- `input_ids[i]`：第 i 个序列的 token id 序列
- `prefix_lens[i]`：第 i 个序列的 prefix 长度（0 表示 provider/standalone）
- `original_lengths[i]`：`len(input_ids[i])`

逐 input 处理：
- **Prefix 段（仅当 prefix_len > 0）**：在当前已构建的 `s_packed` 中作前缀
  匹配（线性查找足够；mini-batch input_count 个数小），匹配成功则复用该区间；
  匹配失败则把 `seq[:prefix_len]` 全部追加到 `s_packed` 末尾。
- **Suffix 段**：总是**追加**到 `s_packed` 末尾（即使当前已有同 suffix，避免破坏
  后续 reuser 的区间连续性）。
- 返回每个 input 的 KV 区间列表 `[(lo, hi), ...]`（闭区间，含 prefix/suffix
  零到多段）和去重后的 `len(s_packed)`。

**不变量**：
- `s_packed_q_lengths[i] == suffix_lens[i]`（reuser 时是 `original - prefix`，
  provider/standalone 时是 `original`）
- `sum(s_packed_q_lengths) == total_q`（Q 端未去重，仍按 batch 顺序 flat）
- 每个 input 的各段区间都满足 `kv_hi - kv_lo > 0`，相邻段不重叠
- `s_packed_prefix_end[i]` 等于第一段 prefix 区间（如果存在）的 `hi`，
  用于在 mask 构建时分段

## Global Custom Causal Mask 语义（`build_global_custom_mask`）

形状：`(total_q, s_packed_length)`，`dtype=bool`，device 由 caller 传入。
**True = visible**（与既有 `_causal_q_kv_mask` 语义一致）。

对每个 input，按 `qi in [0, q_len)`：
- 计算该 q 在 s_packed flat Q 中的实际位置 `q_s_packed_pos = q_offset + qi`
- 遍历该 input 的每一段 KV 区间 `(kv_lo, kv_hi)`：
  - **prefix 段**（`kv_hi <= prefix_end`）：visible 区间长度为
    `min(kv_hi, kv_lo + qi + 1)`，即在 prefix 内部按绝对位置产生因果
  - **suffix 段**（`kv_hi > prefix_end`）：以 qi 相对于 prefix_len 的相对位置
    触发因果，visible 区间长度为
    `min(kv_hi, kv_lo + (qi - prefix_len) + 1)`

边界：
- prefix_len == 0（standalone 或 provider）时没有 prefix 段，全段都按 suffix 段
  处理（`prefix_end = 0`，分支退化为 `kv_hi > 0` 恒真）
- 一个 input 多段（prefix 复用 + suffix 追加）时，mask 行可能有两个不连续的
  visible 区间，这是正确的

## RoPE 与位置偏移

`q_position_offsets[i]` 和 `kv_position_offsets[i]` **沿用既有字段**，未做调整：
- reuser：`q_offset = prefix_len`，`kv_offset = 0` → Q 上的位置 i 对应绝对
  位置 `prefix_len + i`，足以恢复 suffix token 的正确 RoPE
- provider/standalone：`q_offset = 0`，`kv_offset = 0` → 位置与原序列一致

新增字段不引入新的位置维度，与既有 RoPE 协议兼容。

## Backend 改造

### TorchReferenceBackend（重写）

```python
def build_kv(key, value, store, plan, *, packed_batch_layout, layer_id, ...):
    """从 framework 提供的 padded K/V 切出有效行，拼成 s_packed K/V。
    不再依赖 store chain（store 入参保留为兼容签名,实现中不读）。"""
    layout = packed_batch_layout or PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
    key_rows = _split_packed(key, layout.padded_lengths)
    value_rows = _split_packed(value, layout.padded_lengths)
    s_packed_k, s_packed_v, stored_tokens = [], [], 0
    for i in range(plan.batch_size):
        if layout.valid_lengths[i] == 0:
            continue
        prefix_len = plan.prefix_lens[i]
        suffix_k, suffix_v = key_rows[i][:layout.valid_lengths[i]], value_rows[i][:]
        if not plan.is_reuser(i):
            # Provider 整行都是 unique，append 到 s_packed
            s_packed_k.append(suffix_k); s_packed_v.append(suffix_v)
        else:
            # Reuser: 先 copy prefix 的 K/V（去重，已在 s_packed 中则跳过）
            if prefix_len > 0 and plan.s_packed_prefix_end[i] not in ({/* 复用区间 */} or {len(s_packed_k)}):
                # 这里走的是 prefix-suffix 分裂语义
                ...  # 见 build_global_custom_mask 的视角
            # 总是 append suffix
            s_packed_k.append(suffix_k[prefix_len:])  # suffix 部分
            s_packed_v.append(suffix_v[prefix_len:])
    return cat(s_packed_k), cat(s_packed_v), stored_tokens

def attention(query, key, value, plan, *, packed_batch_layout, ...):
    """基于 global custom mask 重写 attention；padded Q 也覆盖。"""
    mask = plan.build_global_custom_mask(query.device)
    # 把 query 按 plan.s_packed_q_starts 切成 batch 行；padded 区域被 build_kv
    # 的 split_packed 限到 max_q，残留 pad 也通过 mask 抑制
    outputs = []
    for i in range(plan.batch_size):
        q_i = q_rows[i]  # 按 padded length 切
        s_i, e_i = plan.s_packed_q_starts[i], plan.s_packed_q_starts[i] + plan.s_packed_q_lengths[i]
        attn = q_i @ k[s_packed].T * scale
        attn = attn.masked_fill(~mask[s_i:e_i], -inf)
        outputs.append(F.softmax(attn) @ v[s_packed])
    return pad_and_cat(outputs, layout)
```

**store 入参保留**用以兼容既有调用方签名（sparse_mode 下不读）；后续若确认
不再使用可移除。

### NpuFlashAttentionBackend（适配）

```python
def _build_s_packed_mask(plan, device):
    """把 (total_q, S) 展开为 BSHD attention 所需的 (B, 1, max_q, S)。"""
    mask = plan.build_global_custom_mask(device)  # (total_q, S)
    b, max_q, s = plan.batch_size, max(plan.s_packed_q_lengths), plan.s_packed_length
    bshd = mask.new_zeros((b, max_q, s))
    for i in range(b):
        s_i, e_i = plan.s_packed_q_starts[i], s_packed_q_starts[i]+s_packed_q_lengths[i]
        bshd[i, :plan.s_packed_q_lengths[i]] = mask[s_i:e_i]
    return bshd.unsqueeze(1)  # (B, 1, max_q, S)

def attention(query, key, value, plan, ...):
    """K/V 在 BSHD 维度上每行复用同一份 s_packed 内容；单次 npu_fusion_attention。"""
    mask = _build_s_packed_mask(plan, query.device)
    b = plan.batch_size
    s = plan.s_packed_length
    k_bshd = key.unsqueeze(1).expand(b, plan.s_packed_q_lengths_count, s, h, d)
    v_bshd = value.unsqueeze(1).expand(...)
    return npu_fusion_attention(q, k_bshd, v_bshd, ..., atten_mask=mask)
```

### GpuFlashAttentionBackend（fallback）

`flash_attn_varlen_func` 不支持 arbitrary mask，sparse_mode 1 下退回到
`TorchReferenceBackend.attention`。

## 字段一览

| 字段 | 类型 | 含义 |
|---|---|---|
| `s_packed_length` | int | 去重后 s_packed KV 序列总长 |
| `s_packed_kv_ranges[i]` | list[(lo, hi)] | input_i 在 s_packed 中的 KV 区间，可能多段 |
| `s_packed_q_lengths[i]` | int | input_i 的 Q 长度（=suffix_len） |
| `s_packed_q_starts[i]` | int | input_i 的 Q 在 s_packed flat Q 中的起始 |
| `s_packed_prefix_end[i]` | int | input_i 的 prefix 段在 s_packed 中的结束位置 |

## 测试覆盖（`tests/unit_test/test_s_packed.py`，21 个用例）

- `build_s_packed`：5 个场景（无共享 / 部分共享 / 三输入 / 零 prefix reuser / 全相同）
- 字段正确性：`test_plan_includes_s_packed_fields`、
  `test_plan_s_packed_q_starts_are_sequential`
- mask 形状与因果：`test_global_mask_shape`、`test_global_mask_provider_causal`、
  `test_global_mask_prefix_reuse_visible`、`test_global_mask_cross_input_isolation`、
  `test_global_mask_same_input_dedup`
- torch_ref s_packed KV：`test_build_kv_s_packed_shape`、
  `test_build_kv_s_packed_provider_equals_own_kv`
- torch_ref attention：`test_attention_s_packed_visibility`、
  `test_attention_s_packed_per_input_output`、
  `test_attention_s_packed_shape_aligned`
- 缓存与边界：`test_mask_cached_on_repeated_build`、
  `test_single_input_no_sharing`、`test_empty_batch`

## 风险与边界

- **prefix_match 复杂度**：`build_s_packed` 的 prefix 段是 O(N×L) 线性扫描；
  单 micro-batch N 通常 ≤ 64，L 通常 ≤ 2k，对当前规模不构成瓶颈
- **mask 大小**：(total_q, s_packed_length)，极端 batch 下可能达
  (16k, 32k) ≈ 512M bool ≈ 512 MB；后续可考虑 CSR/sparse 存储，但当前规模安全
- **FlashAttention GPU**：sparse_mode 1 下走 torch_ref 路径，性能优势
  需在 GPU 上重新评估（KV 总 token 数减少，但 mask eval 增加 FLOPs）；
  GPU s_packed mask 走 FlashAttention-2 / 3 的 `flash_attn_func` 自定义
  mask 接口的优化留给后续 PR
- **store 入参**：sparse_mode 1 下 torch_ref build_kv 不读 store，仅为
  兼容签名；若后续不再需要可一并删除

## 改动文件

| 文件 | 说明 |
|---|---|
| `prefix_sharing/core/planner.py` | +139 行：`build_s_packed` 静态方法、`build_global_custom_mask`、4 个新字段 |
| `prefix_sharing/backends/torch_ref.py` | ±240 行：build_kv/attention 重写为 s_packed + global mask |
| `prefix_sharing/backends/flash_atten_npu.py` | ±192 行：BSHD s_packed mask + 单次 npu_fusion_attention |
| `prefix_sharing/backends/flash_atten_gpu.py` | -86 行：fallback to torch_ref |
| `tests/unit_test/test_s_packed.py` | +421 行：21 个 s_packed 单元测试（新增） |
| `tests/unit_test/test_torch_ref_backend.py` | ±67 行：既有 build_kv/attention 测试更新 |

## 测试结果

193 个单元测试全部通过。
