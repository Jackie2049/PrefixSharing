# PrefixSharing on verl FSDP 使用指南

本文面向希望在 verl FSDP 路径中试用 PrefixSharing 的用户。当前能力仍属于实验特性，已在 4090 单机 FSDP 场景完成单卡、2/4/8 卡功能和 bf16 级精度验证，但尚未作为默认主特性合入 README。

## 1. 原理简介

PrefixSharing 用于减少同一 micro-batch 内共享前缀的重复计算。

在 RL 训练中，step/tree/prompt 多样本场景经常出现多条序列拥有相同或部分相同的前缀。PrefixSharing 会在 micro-batch 内自动检测 provider/reuser 关系：

- provider：完整计算共享前缀和自己的 suffix；
- reuser：裁剪掉已经可复用的前缀，仅计算 suffix；
- attention 中复用 provider 的 prefix KV；
- 输出阶段恢复 reuser 缺失的 prefix logprob / entropy / logits，其中 prefix-last logprob 使用 provider prefix-last logits 和 reuser label 重新计算。

FSDP 路径首版保留 PrefixSharing 自己的 arbitrary-prefix planner，不把计划转换成 PrefixGrouper 的 prompt-only `group_info`。PrefixGrouper 的 prompt-only 模式仍由 PrefixGrouper 包负责；PrefixSharing 负责 arbitrary-prefix 模式。

## 2. 适用范围

当前已验证主路径：

- verl 0.8 FSDP engine；
- Qwen2.5-0.5B；
- 单机 4090；
- 单卡、2/4/8 卡纯 FSDP；
- `use_remove_padding=True` packed path；
- `gradient_checkpointing=False` 的 smoke / 精度验证；
- GRPO no-critic 测试配置。

当前不作为首版承诺：

- Ulysses SP；
- ring attention；
- fused kernels；
- Megatron / MindSpeed 路径；
- dense / 非 remove-padding FSDP 主路径；
- 性能收益 benchmark。

## 3. 配套环境

与主分支相比，本实验路径需要额外安装并导入 PrefixSharing 包，同时显式选择 FSDP patch set。

推荐环境变量：

```bash
export VERL_USE_EXTERNAL_MODULES=prefix_sharing
export PREFIX_SHARING_PATCHSET=verl080_fsdp
export ENABLE_PREFIX_SHARING=1
```

含义：

- `VERL_USE_EXTERNAL_MODULES=prefix_sharing`：让 verl worker 启动时 import PrefixSharing 包；
- `PREFIX_SHARING_PATCHSET=verl080_fsdp`：显式安装 FSDP patch set，避免自动探测误选 Megatron / MindSpeed patch；
- `ENABLE_PREFIX_SHARING=1`：启用每个 micro-batch 的 PrefixSharing 逻辑。

交互式或 notebook 环境也可以显式调用：

```python
import prefix_sharing.setup

prefix_sharing.setup.install("verl080_fsdp")
```

两种方式等价；脚本化批量测试更推荐环境变量。

## 4. 快速开始

### 4.1 如何启用

当前可用的实验入口：

```bash
export VERL_USE_EXTERNAL_MODULES=prefix_sharing
export PREFIX_SHARING_PATCHSET=verl080_fsdp
export ENABLE_PREFIX_SHARING=1
```

verl 配置中建议确保：

```yaml
actor_rollout_ref:
  model:
    use_remove_padding: true
  actor:
    strategy: fsdp
```

后续面向社区的用户入口计划对齐 PrefixGrouper：

```yaml
actor_rollout_ref:
  actor:
    use_prefix_grouper: true
    prefix_grouper:
      mode: arbitrary_prefix
      min_prefix_len: 1
      min_group_size: 2
```

语义：

- `mode: prompt_only`：保留给 PrefixGrouper 包处理；
- `mode: arbitrary_prefix`：进入 PrefixSharing 包处理；
- 缺省 `mode`：按 prompt-only 兼容处理。

### 4.2 样例脚本

下面是最小启动脚本骨架，实际训练命令按你的 verl 任务脚本替换：

```bash
#!/usr/bin/env bash
set -euo pipefail

export VERL_USE_EXTERNAL_MODULES=prefix_sharing
export PREFIX_SHARING_PATCHSET=verl080_fsdp
export ENABLE_PREFIX_SHARING=1

python3 -m verl.trainer.main_ppo \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.model.use_remove_padding=true \
  "$@"
```

确认生效：

1. 日志中出现 PrefixSharing patch set 安装信息；
2. 训练时出现 `[PS][audit]`；
3. audit 中每层 `store_count`、`reuse_hit_count` 非零；
4. `matches_expected=True`。

精度诊断可选打开：

```bash
export PREFIX_SHARING_DIAG_DUMP=/path/to/dump_on
```

ON/OFF 对比建议分别运行两次：

```bash
# OFF
ENABLE_PREFIX_SHARING=0 PREFIX_SHARING_DIAG_DUMP=/path/to/dump_off ...

# ON
ENABLE_PREFIX_SHARING=1 PREFIX_SHARING_DIAG_DUMP=/path/to/dump_on ...
```

然后使用：

```bash
python3 prefix-sharing/prefix_sharing/tools/cmp_diag_verl080.py \
  --dir-on /path/to/dump_on \
  --dir-off /path/to/dump_off \
  --tag train
```

注意：真实 rollout 在采样温度大于 0 时 ON/OFF response 可能不同，逐元素 logprob 对比需要先确认 `input_ids_train.pt` 字节级一致；否则应以 attention output / logits / restore 区域对齐作为主要判断。

## 5. 遗留问题

当前遗留项：

1. verl 用户面配置还未正式合入上游 schema。`prefix_grouper.mode=arbitrary_prefix` 是计划中的社区入口，当前实验主要依赖环境变量。
2. 性能 benchmark 尚未完成。后续需要补 baseline FSDP、PrefixGrouper prompt-only、PrefixSharing arbitrary-prefix 三方对比。
3. FSDP 首版主路径是 `use_remove_padding=True` packed path，dense / 非 remove-padding 不作为首版承诺。
4. 暂不支持 Ulysses SP、ring attention、fused kernels。
5. 文档后续开源前需要从中文开发文档整理为社区可读文档。
