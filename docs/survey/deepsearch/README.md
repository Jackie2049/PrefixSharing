# DeepSearch (MCTS in RLVR)

## 论文

arxiv 2509.25454 (ICLR 2026)（见 ）

## 代码仓库

无公开代码

> 代码仓库不直接纳入 git 管理（体积大），通过链接引用。

## 核心思路

将 MCTS 直接嵌入 RLVR 训练循环，从中间推理步骤分支。搜索树 rooted at input prefix，自然产生前缀共享结构。
