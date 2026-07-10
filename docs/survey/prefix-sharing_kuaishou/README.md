# 快手 Dynamic Tree Attention / Tree Training

## 论文

arxiv 2511.00413（见 ）

## 代码仓库

https://github.com/Whisper-6/DynamicTreeAttn

> 代码仓库不直接纳入 git 管理（体积大），通过链接引用。

## 核心思路

树形布局前缀复用。将多个共享前缀的序列组织为 prefix | leaf_0 | ... | leaf_{n-1} 布局，支持 causal conv1d KV cache，prefix 只前向一次。
