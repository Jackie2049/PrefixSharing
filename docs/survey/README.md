# RL Training Prefix Reuse 技术调研

调研 2025 年业界 RL 训练阶段前缀复用/共享的同类技术方案。

## 方案列表

| 相似度 | 方案 | 团队 | 论文 | 代码 | 核心思路 | 与 PrefixSharing 的核心相似点 | 与 PrefixSharing 的核心差异 |
|-------|------|------|------|------|---------|----------------------------|----------------------------|
| ⭐⭐⭐⭐⭐ | PrefixGrouper (verl PR #4368) | verl 社区 | [PR #4368](prefix-grouper_verl/pr_4368.md) | [code](prefix-grouper_verl/README.md) | verl FSDP worker 集成 PrefixGrouper 的 attention decomposition | 同属 verl FSDP 生态、同一 GRPO 前缀复用问题域、同为训练阶段优化 | 我们走 NestedTensor + packed layout 单次 forward；他们走两阶段 attention decomposition |
| ⭐⭐⭐⭐⭐ | Prefix Sharing (美团) | 美团搜索 | [verl issue #6401](prefix-sharing_meituan/verl_issue_6401.md) | [code/](prefix-sharing_meituan/code/) (verl_prefix_share branch) | 同 PR #4368，verl 集成 PrefixGrouper + monkey-patch | 同上（同源方案），同 verl 社区、同 FSDP backend、共享 prefix_grouper 外部库 | ~365 行低侵入 monkey-patch；我们是 NestedTensor 路径，改动量更大但灵活性更高 |
| ⭐⭐⭐⭐ | PrefixGrouper (CASIA) | 中科院 IVA | [arxiv 2506.05433](prefix-grouper_cas/paper.pdf) | [code/](prefix-grouper_cas/code/) | 两阶段 attention：prefix self-attn + suffix concat-attn + 3 自定义 autograd | 同为 GRPO 场景 design，数学等价性严格保证，都无需修改 transformers 源码 | 我们单次 forward 单次 mask；他们两次 attention pass + index scatter-gather |
| ⭐⭐⭐ | DTA (蚂蚁) | 蚂蚁/AReaL | 同 issue #6401 | [code/](dta_areal/code/) (feat/dta branch) | Trie 树 DFS 序列化 + Pop/Push 栈式 KV cache + chunked backward | 同关注 RL 前缀复用、attention 层优化 | 他们走独立前缀复用调度器（1,406 行），DTA 模式下平行化走 ZeRO-1（朴素 DP），FSDP/Megatron 尚未适配 |
| ⭐⭐⭐ | Prefix Sharing (快手) | 快手/Kwai | [arxiv 2511.00413](prefix-sharing_kuaishou/paper.pdf) | [code/](dynamic-tree-attn_kuaishou/code/) | Trie 树 + Pop/Push 栈式 KV cache + 梯度注入 chunked backward | 同前缀复用目标，同支持多 response 共享 prefix KV | 他们 Trie 栈可灵活弹出分叉部分，但梯度 2-10% 偏差；我们扁平 packed 精确等价 |
| ⭐⭐ | Forge (MiniMax) | MiniMax | — | 闭源（见 [README](minimax/README.md)） | Prefix Tree Merging + MagiAttention + CISPO 算法 | 同关注 RL 训练前缀树合并（inspire RFC #6401） | 闭源专有，40x 加速无法验证；我们是开源软件栈 |
| ⭐⭐ | DualKV | 亚马逊 | [arXiv 2605.15422](dualkv/README.md) | [code/](dualkv/code/) | 修改 FA2 kernel 实现共享 prompt 的 KV 原子累加 | 同为 RL 训练 prefix 冗余消除，数学严格等价 | kernel 层面实现，FA2 绑定（sm80+），非商用许可；我们是 Python 层面，跨硬件 |
| ⭐⭐ | MagiAttention | SandAI | 无独立论文 | [SandAI-org/MagiAttention](https://github.com/SandAI-org/MagiAttention) | Chunk-level CP sharding + dispatch solver workload 均衡 | 同关注 tree layout 下 attention 效率，互为上下游 | 他们解决分布式执行层 workload 均衡；我们解决算法层的 prefix sharing |
| ⭐⭐ | TreeRL | — | [arxiv 2506.11902](treerl/paper.pdf) | 无公开代码 | on-policy tree search + 前缀共享到分支点 | 同关注 RL 训练中前缀重复计算的冗余问题 | 他们搜索树层面（MCTS 训练框架）；我们是 attention 计算层面 |
| ⭐ | rStar-Math | Microsoft | [arxiv 2501.04519](rstar-math/paper.pdf) | [microsoft/rStar](https://github.com/microsoft/rStar) | MCTS 深度思考 → 天然前缀共享树 | 同涉及前缀复用概念 | MCTS 推理框架，前缀共享是自然产物；我们是显式的训练阶段 attention 优化 |
| ⭐ | DeepSearch | — | [arxiv 2509.25454](deepsearch/paper.pdf) | 无公开代码 | MCTS 嵌入 RLVR 训练循环 → 搜索树前缀共享 | 同 RLVR 训练场景 | 搜索树层面；不涉及 packed attention / KV cache 等底层优化 |

## 性能表现

各方案的实验数据集、测试环境与性能数据总览。数据来源于各论文与代码仓库公布的实验结果，metrics 保留原文报告方式。

| 方案 | 实验数据集 | 测试环境 | 速度性能 | 显存性能 |
|------|----------|--------|---------|---------|
| **PrefixGrouper (verl PR #4368)** | 未公开（GRPO math reasoning） | Qwen3-4B, 4×H800, rollout.n=4, FSDP2 | 4K ctx: update_actor **1.26×**, step **1.14×**; 8K ctx: update_actor **1.70×**, step **1.27×**; old_log_prob 4K: 1.30×, 8K: 1.56× | — |
| **Prefix Sharing (美团)** | — | — | —（verl_prefix_share 分支，与 PR #4368 同源 CASIA PG 库，未公布独立性能数据） | — |
| **PrefixGrouper (CASIA)** | — | — | —（论文提出算法框架，性能数据见 PR #4368 集成评测；理论加速比 ≈ N - (N-1)·P²/(P+S)²） | — |
| **DTA (快手)** | [SWE-smith](https://github.com/swe-smith/SWE-smith) agentic RL rollouts + [Terminal Bench 2.0](https://github.com/terminalbench/terminalbench) | Qwen3-32B (dense) / Qwen3-30B-A3B (MoE), 64×H100, Megatron-Core | 端到端 **6.2–6.3×**（think-mode data），理论上限 6.5×，内存无限场景 **8.7×**; Terminal Bench 2.0 avg@4: **28.8** vs baseline 20.9 (38% gain) | 额外张量仅 **1.2 MB**（Qwen3-32B），峰值内存 = 单条 root-to-leaf path |
| **AReaL DTA (蚂蚁)** | — | — | —（DTA 引擎代码已开源，DTA 模式下平行化走 ZeRO-1 朴素 DP，未公布独立性能 benchmark） | — |
| **RFC #6401 (美团/SandAI)** | Dataset A: 浅层树 (depth=2, branch=2, seq~12.8k, ~50% shared); Dataset B: 深层树 (depth=16, branch=2, 512 leaves, ~69% saved) | H20, TP=4 (Megatron) | Dataset A: MAGI step **3.97s** (vs FA3 6.7s, **42% faster**); Dataset B: fwd **394ms / 3.02×** (8K), 851ms / **2.99×** (16K) | Dataset A: MAGI 86GB vs FA3 mbs=4 122GB (**30% less**); Dataset B: FA3 16K backward OOM, MAGI 正常完成 |
| **Forge (MiniMax)** | 未公开 | 未公开 | 声称 **40×** 端到端加速（不可验证，无细节） | — |
| **DualKV** | [LongReason](https://arxiv.org/abs/2502.20329)（Ling et al. 2025, 长上下文 math reasoning, prompt ≤8K, response ≤2K）+ [GSM8K](https://arxiv.org/abs/2110.14168)（短 prompt ~150t） | Qwen3-8B, 8×H100, FSDP2, N=32, mb/GPU=4/8 | GRPO: policy-update **1.63–2.09×**, step **1.48–1.64×**, MFU 36%→**76%**; DAPO: policy-update **2.47×**, MFU 31%→**77%** | DualKV mb=8: **93GB** vs FA2 mb=4: 106GB†（FA2 mb=8 OOM） |
| | [GSM8K](https://arxiv.org/abs/2110.14168) + kernel microbenchmark | A100, P∈{4K–64K}, N∈{16,28}, R=2048 | kernel fwd+bwd: P=16K N=28 **3.88×**, P=32K N=16 **5.48×**, P=64K N=16 FA2 OOM DualKV 正常; PrefixGrouper 对照 2–4× faster, 3–5× less memory | P=64K N=16: FA2 OOM, DualKV **1,358ms**; memory reduction: P=16K **85%**, P=32K **86%** |
| | Qwen3-30B-A3B (MoE), 16×H100, FSDP2, N=32, mb/GPU=8 | MoE: policy-update **3.82×**, old_log_prob **3.45×**, step **3.38×**; cost: 12.6h vs 42.6h (FA2 SP=4) | FA2 SP=4: 92GB†（SP=2 OOM）; DualKV SP=1: **103GB**†（† 含 CUDA graphs 缓存） |
| | Llama-3.1-8B memory sweep, P∈{8K–96K}, 16×H100 | P=96K mb=4→16 仅增~4GB (**<5%**); FA2 需 225→775GB (超 10× 物理容量) | DualKV peak **~100GB** vs FA2 **225–775GB** |
| **MagiAttention** | — | — | —（分布式 attention 后端，非前缀复用方案本身；作为 RFC #6401 的 attention backend 贡献其性能数据） | — |
| **TreeRL** | [AIME 2024](https://artofproblemsolving.com/wiki/index.php/2024_AIME_I) / [OlympiadBench](https://github.com/OpenBMB/OlympiadBench) / GSM8K / MATH | Qwen2.5 / GLM | —（论文强调 on-policy tree search 训练框架，前缀共享是自然产物，未单独报告 prefix sharing 加速比） | — |
| **rStar-Math** | [AIME](https://artofproblemsolving.com) / [AMC](https://artofproblemsolving.com) / [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | Qwen2.5-Math 7B / Qwen2.5-7B | —（MCTS 推理框架，前缀共享为自然产物；训练阶段不直接复用前缀，为树形训练提供场景验证） | — |
| **DeepSearch** | — | — | —（ICLR 2026，论文阶段，无公开代码） | — |

*注：† 标记含 CUDA graphs 缓存占用；— 表示该方案未公布对应数据。Prefix Sharing (美团) 与 PrefixGrouper (verl PR #4368) 同源 CASIA PG 库，预计性能接近。*

## 子目录说明

每个子目录包含：
- `paper.pdf` — 论文原文（如有）
- `*.md` — issue/PR 讨论原文
- 代码仓库通过 README 中的链接引用，不直接纳入 git（代码量大）

