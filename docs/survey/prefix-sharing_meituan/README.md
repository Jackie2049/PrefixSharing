# 美团 Prefix Sharing (verl 集成)

## 论文

verl issue #6401 RFC（见 ）

## 代码仓库

https://github.com/meituan-search/verl/tree/verl_prefix_share

> 代码仓库不直接纳入 git 管理（体积大），通过链接引用。

## 核心思路

在 verl FSDP worker 中集成 PrefixGrouper，monkey-patch attention 支持 prefix_grouper 参数传递。通过 verl 配置 use_prefix_grouper=true 启用。
