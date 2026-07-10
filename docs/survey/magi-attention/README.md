# MagiAttention (分布式 attention 后端)

## 论文

无独立论文（见 ）

## 代码仓库

https://github.com/SandAI-org/MagiAttention

> 代码仓库不直接纳入 git 管理（体积大），通过链接引用。

## 核心思路

chunk-level sharding + workload-balanced CP dispatch。不是前缀复用方案本身，而是前缀树布局下的负载均衡后端——prefix token 的 KV 远多于 leaf token，标准 CP 会严重失衡。
