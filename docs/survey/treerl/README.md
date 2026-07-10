# TreeRL (on-policy tree search)

## 论文

arxiv 2506.11902（见 ）

## 代码仓库

无公开代码

> 代码仓库不直接纳入 git 管理（体积大），通过链接引用。

## 核心思路

直接将 on-policy tree search 嵌入 RL 训练，从高不确定中间步骤分支。每个分支共享到分支点的前缀。与 DTA 的树形训练有结构相似性。
