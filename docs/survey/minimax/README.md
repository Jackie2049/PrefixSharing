# MiniMax Forge

## 技术方案

https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm

## 代码仓库

代码未公开（闭源专有框架）。

## 核心思路

Forge 提出 Prefix Tree Merging（前缀树合并）方案，将多 trajectory 合并为单棵前缀树，通过 MagiAttention 的稀疏 mask 做因果隔离。声称 40x 加速（不可验证）。

## 关键信息

- 三模块架构：智能体侧 / 中间件抽象层 / 训练和推理侧
- 核心算法 CISPO（不公开）
- 全局 L3 KV 缓存池
- 异构 PD 分离 + MTP 推测解码
