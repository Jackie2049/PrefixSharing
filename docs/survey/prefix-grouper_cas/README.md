# 中科院 PrefixGrouper

## 论文

arxiv 2506.05433（见 ）

## 代码仓库

https://github.com/CASIA-IVA-Lab/PrefixGrouper

> 代码仓库不直接纳入 git 管理（体积大），通过链接引用。

## 核心思路

两阶段 attention：prefix self-attn + suffix concat-attn。将 GRPO 中同一 prompt 的 G 个 rollout 组组，prefix 只算一次，suffix 拼接到 prefix KV 后一次性计算。
