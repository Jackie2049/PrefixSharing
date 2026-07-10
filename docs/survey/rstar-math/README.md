# rStar-Math (MCTS deep thinking)

## 论文

arxiv 2501.04519（见 ）

## 代码仓库

https://github.com/microsoft/rStar

> 代码仓库不直接纳入 git 管理（体积大），通过链接引用。

## 核心思路

MCTS 深度思考：小模型做 test-time search，PRM 引导。MCTS rollout 天然产生共享前缀树。训练阶段不直接用前缀复用，但为树形训练提供场景验证。
