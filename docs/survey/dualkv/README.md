# DualKV: Shared-Prompt Flash Attention

## 论文

arXiv 2605.15422 — "DualKV: Shared-Prompt Flash Attention for Efficient RL Training with Large Rollouts and Long Contexts"

**作者**：Jiading Gai, Shuai Zhang, Xiang Song, Bernie Wang, George Karypis（Amazon）

## 代码仓库

- [amazon-science/dualkv-flash-attn-for-rl](https://github.com/amazon-science/dualkv-flash-attn-for-rl)（官方，CC-BY-NC-4.0 非商用许可）
- [JiadingGai/dualkv-flash-attention](https://github.com/JiadingGai/dualkv-flash-attention)（作者个人 fork）

包含：
- 自定义 FA2 前向/反向 CUDA kernel（`flash_fwd_kernel_dualkv_training.h` 477 行，`flash_bwd_kernel_dualkv_training.h` 733 行）
- 针对 hdim=64/96/128/192/256、fp16/bf16、causal/non-causal、sm80 的 40 个编译实例
- veRL v0.7.0 集成（`monkey_patch.py` `_make_dualkv_flash_wrapper`）
- 论文实验复现脚本（kernel 级微基准 + end-to-end 训练基准）

## 核心思路

DualKV 修改 FlashAttention-2 的 CUDA kernel，将 Q/KV 分为**两个物理区域**：共享 context（prompt，仅 1 份）和 per-sequence decoded（response，N 份），在一个 kernel launch 内完成全部 attention。

配合 veRL 数据管线的 repacking 优化，micro-batch 从 `N(P+R)` token 减少到 `P + NR` token，全部 per-token 运算（norm、projection、MLP、attention）的输入 token 数都显著减少。

## 核心分析差异

DualKV 与其他 prefix sharing 方案的本质区别：

| 维度 | DualKV | PrefixGrouper / PrefixSharing |
|------|--------|-------------------------------|
| **抽象层次** | CUDA kernel 级（FA2 内部） | Python 级（attention 操作/布局） |
| **共享范围** | attention 层全部（由 repacking 带来） | attention 层全部（Q/KV 拼接） |
| **token 减少** | 所有 per-token 运算（norm+proj+MLP+attn） | 仅 attention 层（Q/KV 形状变化） |
| **启动方式** | 替换 FA kernel 调用点 | monkey-patch attention forward |
| **varlen 兼容** | 原生（k_ctx batch=1, k_dec varlen） | 原生（cu_seqlens_q ≠ cu_seqlens_kv） |

## 性能数据

| 配置 | 加速比 |
|------|--------|
| Qwen3-8B GRPO N=32, 8K ctx, 8×H100 | 1.63-2.09× policy-update |
| DAPO 同配置 | 2.47× policy-update |
| 30B MoE 16×H100 | 3.82× policy-update, 3.38× end-to-end |
| kernel fwd (FA2 vs DualKV) | 最大 6.2× |
| MFU (8B GRPO) | 36% → 76%（FA → DualKV） |

## 优缺点

**优点**：
- kernel 级实现，优化路径最短
- 因 repacking 受益的不只是 attention——全部 per-token 运算量都减少
- 支持 Ulysses SP（SP>1 时 all-to-all 重建完整序列后走 DualKV）
- 数学严格等价（无近似）

**缺点**：
- 需编译自定义 CUDA kernel，非 pip 即用
- 仅支持 hdim ≤ 256（FA2 限制），不支持 FA3/FA4
- CC-BY-NC-4.0 非商用许可
- 仅 H100/A100（sm80+），不支持 NPU
