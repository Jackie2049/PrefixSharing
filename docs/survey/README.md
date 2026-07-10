# RL Training Prefix Reuse 技术调研

调研 2025 年业界 RL 训练阶段前缀复用/共享的同类技术方案。

## 方案列表

| 相似度 | 方案 | 团队 | 论文 | 代码 | 核心思路 | 与 PrefixSharing 的核心差异 |
|-------|------|------|------|------|---------|----------------------------|
| ⭐⭐⭐⭐⭐ | PrefixGrouper (verl PR #4368) | verl 社区 | [PR #4368](prefix-grouper_verl/pr_4368.md) | [johncaged/PrefixGrouper](https://github.com/johncaged/PrefixGrouper) | verl FSDP worker 集成 PrefixGrouper 的 attention decomposition | 同 verl FSDP + GRPO 前缀复用场景，已合入 main；我们走 NestedTensor + packed layout，他们走两阶段 attention decomposition |
| ⭐⭐⭐⭐⭐ | Prefix Sharing (美团) | 美团搜索 | [verl issue #6401](prefix-sharing_meituan/verl_issue_6401.md) | [meituan-search/verl#verl_prefix_share](https://github.com/meituan-search/verl/tree/verl_prefix_share) | 同 PR #4368，verl 集成 PrefixGrouper + monkey-patch | ~365 行低侵入集成，FSDP-only；我们是 NestedTensor 路径，集成深度更深但改动量更大 |
| ⭐⭐⭐⭐ | PrefixGrouper (CASIA) | 中科院 IVA | [arxiv 2506.05433](prefix-grouper_cas/paper.pdf) | [CASIA-IVA-Lab/PrefixGrouper](https://github.com/CASIA-IVA-Lab/PrefixGrouper) | 两阶段 attention：prefix self-attn + suffix concat-attn + 3 自定义 autograd | 同属 GRPO 前缀复用核心问题，数学等价性严格保证；我们单次 forward 他们两次 pass，实现架构差异大 |
| ⭐⭐⭐ | DTA (蚂蚁) | 蚂蚁/AReaL | 同 issue #6401 | [areal-project/AReaL feat/dta](https://github.com/areal-project/AReaL/tree/feat/dta) | Trie 树 DFS 序列化 + Pop/Push 栈式 KV cache + chunked backward | 同 RL 前缀复用场景、attention 层优化；但走独立引擎路径（1,406 行），支持 FSDP2/Megatron 双后端 |
| ⭐⭐⭐ | Prefix Sharing (快手) | 快手/Kwai | [arxiv 2511.00413](prefix-sharing_kuaishou/paper.pdf) | [Whisper-6/DynamicTreeAttn](https://github.com/Whisper-6/DynamicTreeAttn) | Trie 树 + Pop/Push 栈式 KV cache + 梯度注入 chunked backward | 同前缀复用目标；他们 Trie 栈可灵活弹出分叉部分，但梯度 2-10% 偏差；我们扁平 packed 精确等价 |
| ⭐⭐ | MagiAttention | SandAI | 无独立论文 | [SandAI-org/MagiAttention](https://github.com/SandAI-org/MagiAttention) | Chunk-level CP sharding + dispatch solver workload 均衡 | 非 prefix sharing 本身，是 tree layout 下 attention 后端；我们解决算法层，他们解决分布式执行层 |
| ⭐⭐ | TreeRL | — | [arxiv 2506.11902](treerl/paper.pdf) | 无公开代码 | on-policy tree search + 前缀共享到分支点 | 搜索树层面的前缀共享（MCTS 训练），不关注 attention 计算优化；我们是 attention 层微观优化 |
| ⭐ | rStar-Math | Microsoft | [arxiv 2501.04519](rstar-math/paper.pdf) | [microsoft/rStar](https://github.com/microsoft/rStar) | MCTS 深度思考 → 天然前缀共享树 | MCTS 推理框架，前缀共享是自然产物；我们是显式的训练阶段 attention 优化 |
| ⭐ | DeepSearch | — | [arxiv 2509.25454](deepsearch/paper.pdf) | 无公开代码 | MCTS 嵌入 RLVR 训练循环 → 搜索树前缀共享 | 同 TreeRL，搜索树层面；不涉及 packed attention / KV cache 等底层优化 |

## 子目录说明

每个子目录包含：
- `paper.pdf` — 论文原文（如有）
- `*.md` — issue/PR 讨论原文
- 代码仓库通过 README 中的链接引用，不直接纳入 git（代码量大）

