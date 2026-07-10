# RL Training Prefix Reuse 技术调研

调研 2025 年业界 RL 训练阶段前缀复用/共享的同类技术方案。

## 方案列表

| 方案 | 团队 | 论文 | 代码 | 核心思路 | 与 PrefixSharing 的核心差异 |
|------|------|------|------|---------|----------------------------|
| PrefixGrouper | 中科院 IVA | [arxiv 2506.05433](prefix-grouper_cas/paper.pdf) | [CASIA-IVA-Lab/PrefixGrouper](https://github.com/CASIA-IVA-Lab/PrefixGrouper) | 两阶段 attention：prefix self-attn + suffix concat-attn | 我们是 flat packed + NestedTensor 单次 forward；他们拆为两次 attention pass，依赖三组自定义 autograd |
| Prefix Sharing (快手) | 快手/Kwai | [arxiv 2511.00413](prefix-sharing_kuaishou/paper.pdf) | [Whisper-6/DynamicTreeAttn](https://github.com/Whisper-6/DynamicTreeAttn) | Trie 树 DFS 序列化 + Pop/Push 栈式 KV cache + 梯度注入 chunked backward | 他们是 Trie 树 + 栈式 KV cache + 梯度注入，可灵活弹出分叉部分；我们是扁平 packed 布局 + 标准 backward，无法跨序列共享中间 KV |
| Prefix Sharing (美团) | 美团搜索 | [verl issue #6401](prefix-sharing_meituan/verl_issue_6401.md) | [meituan-search/verl#verl_prefix_share](https://github.com/meituan-search/verl/tree/verl_prefix_share) | verl 集成 PrefixGrouper + FSDP monkey-patch | 同 CASIA 两阶段方案，在 verl 中以 ~365 行低侵入 monkey-patch 集成；我们是 NestedTensor 路径，集成深度更深但灵活性更高 |
| DTA (蚂蚁) | 蚂蚁/AReaL | 同 issue #6401 | [areal-project/AReaL feat/dta](https://github.com/areal-project/AReaL/tree/feat/dta) | 树形训练引擎：chunked backprop + 分叉点 KV cache | 源自快手方案，AReaL 框架独立引擎（~1,406 行），支持 FSDP2/Megatron 双后端；我们是 FSDP 单一引擎内嵌方案 |
| PrefixGrouper (verl PR) | verl 社区 | [PR #4368](prefix-grouper_verl/pr_4368.md) | [johncaged/PrefixGrouper](https://github.com/johncaged/PrefixGrouper) | verl FSDP worker 集成 PrefixGrouper | 与美团方案同源，已合入 verl main；我们走的 NestedTensor + packed layout 路径是另一条技术路线 |
| MagiAttention | SandAI | 无独立论文 | [SandAI-org/MagiAttention](https://github.com/SandAI-org/MagiAttention) | 分布式 attention：chunk-level sharding + CP 负载均衡 | 他们解决 CP 下 tree layout 的 attention workload 不均衡问题，不是 prefix sharing 本身；我们是 prefix sharing 算法层 |
| TreeRL | — | [arxiv 2506.11902](treerl/paper.pdf) | 无公开代码 | on-policy tree search + 前缀共享到分支点 | 他们是搜索树层面的前缀共享（MCTS 训练框架），不关注 attention 计算优化；我们是 attention 计算层面的前缀重用 |
| rStar-Math | Microsoft | [arxiv 2501.04519](rstar-math/paper.pdf) | [microsoft/rStar](https://github.com/microsoft/rStar) | MCTS 深度思考 → 天然前缀共享树 | 他们是 MCTS 推理框架，前缀共享是自然产物而非显式优化目标；我们是显式的训练阶段 attention 优化 |
| DeepSearch | — | [arxiv 2509.25454](deepsearch/paper.pdf) | 无公开代码 | MCTS 嵌入 RLVR 训练循环 → 搜索树前缀共享 | 同 TreeRL，搜索树层面；不涉及 packed attention / KV cache 等底层优化 |

## 子目录说明

每个子目录包含：
- `paper.pdf` — 论文原文（如有）
- `*.md` — issue/PR 讨论原文
- 代码仓库通过 README 中的链接引用，不直接纳入 git（代码量大）

