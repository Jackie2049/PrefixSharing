# RL Training Prefix Reuse 技术调研

调研 2025 年业界 RL 训练阶段前缀复用/共享的同类技术方案。

## 方案列表

| 方案 | 团队 | 论文 | 代码 | 核心思路 |
|------|------|------|------|---------|
| PrefixGrouper | 中科院 IVA | [arxiv 2506.05433](prefix-grouper_cas/paper.pdf) | [CASIA-IVA-Lab/PrefixGrouper](https://github.com/CASIA-IVA-Lab/PrefixGrouper) | 两阶段 attention：prefix self-attn + suffix concat-attn |
| Prefix Sharing (快手) | 快手/Kwai | [arxiv 2511.00413](prefix-sharing_kuaishou/paper.pdf) | [Whisper-6/DynamicTreeAttn](https://github.com/Whisper-6/DynamicTreeAttn) | 树形布局前缀复用 + causal conv1d KV cache |
| Prefix Sharing (美团) | 美团搜索 | [verl issue #6401](prefix-sharing_meituan/verl_issue_6401.md) | [meituan-search/verl#verl_prefix_share](https://github.com/meituan-search/verl/tree/verl_prefix_share) | verl 集成 PrefixGrouper + FSDP monkey-patch |
| DTA (蚂蚁) | 蚂蚁/AReaL | 同 issue #6401 | [areal-project/AReaL feat/dta](https://github.com/areal-project/AReaL/tree/feat/dta) | 树形训练引擎：chunked backprop + 分叉点 KV cache |
| PrefixGrouper (verl PR) | verl 社区 | [PR #4368](prefix-grouper_verl/pr_4368.md) | [johncaged/PrefixGrouper](https://github.com/johncaged/PrefixGrouper) | verl FSDP worker 集成 PrefixGrouper |
| MagiAttention | SandAI | 无独立论文 | [SandAI-org/MagiAttention](https://github.com/SandAI-org/MagiAttention) | 分布式 attention：chunk-level sharding + CP 负载均衡 |
| TreeRL | — | [arxiv 2506.11902](treerl/paper.pdf) | 无公开代码 | on-policy tree search + 前缀共享到分支点 |
| rStar-Math | Microsoft | [arxiv 2501.04519](rstar-math/paper.pdf) | [microsoft/rStar](https://github.com/microsoft/rStar) | MCTS 深度思考 → 天然前缀共享树 |
| DeepSearch | — | [arxiv 2509.25454](deepsearch/paper.pdf) | 无公开代码 | MCTS 嵌入 RLVR 训练循环 → 搜索树前缀共享 |

## 子目录说明

每个子目录包含：
- `paper.pdf` — 论文原文（如有）
- `*.md` — issue/PR 讨论原文
- 代码仓库通过 README 中的链接引用，不直接纳入 git（代码量大）

