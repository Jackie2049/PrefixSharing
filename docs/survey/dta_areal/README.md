# 蚂蚁 AReaL DTA (Dynamic Tree Attention)

## 论文

verl issue #6401 RFC 同页（见 ）

## 代码仓库

https://github.com/areal-project/AReaL/tree/feat/dta

> 代码仓库不直接纳入 git 管理（体积大），通过链接引用。

## 核心思路

树形训练引擎：TokenTrie 构建 DFS 序列、DTAEngine 分叉点 KV cache + chunked backpropagation。支持 prefix 共享到分支点，分支处 fork 后独立 backward。
